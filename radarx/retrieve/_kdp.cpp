// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Differential phase processing and KDP kernel.
//
// Rays are independent, so the rays of all sweeps of a volume form one pool
// of work that threads take in small blocks from an atomic counter:
//
// 1. gate mask (finite phase, rhohv, circular phase texture from running
//    sums of cos/sin) and the phasor sum of the first valid gates per ray;
// 2. system offset per ray or per sweep (circular mean);
// 3. per ray: unfolding, gap filling, range filtering (iterative low-pass
//    filter after Hubbert and Bringi 1995, iterative KDP after Vulpiani et
//    al. 2012, or a monotone fit as assumed by Maesaka et al. 2012) and KDP
//    as half the least-squares slope over a reflectivity-dependent window.
//
// All windows use running (prefix) sums, so each pass is O(N) in the number
// of gates regardless of the window length. Every step follows the NumPy
// reference in radarx/retrieve/kdp.py, in the same order of operations.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace py = pybind11;
using DArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kPi = 3.14159265358979323846;  // M_PI is not standard (MSVC)
constexpr double kDegToRad = kPi / 180.0;
constexpr double kRadToDeg = 180.0 / kPi;
constexpr int kUnfoldMemory = 5;  // reference: mean of the previous 5 gates
constexpr int64_t kMinValid = 3;  // rays with fewer valid gates stay empty
constexpr double kSignZ = 30.0;     // rain gates for the sign test: Z [dBZ] >=
constexpr double kSignRho = 0.95;   // and rhohv >=
constexpr double kSignSigmas = 3.0; // required significance of the sign test

// Evidence on the sign convention: wrapped phase differences of adjacent
// rain gates. Within a run of rain gates the differences telescope, so the
// noise of their sum grows with the number of runs, not of gates.
struct SignStats {
    double sum = 0.0, sum2 = 0.0;
    int64_t pairs = 0, runs = 0;
};

enum Method { kHubbert = 0, kVulpiani = 1, kMonotone = 2 };
enum OffsetMode { kSweep = 0, kRay = 1, kFixed = 2 };

// Options shared by all sweeps.
struct Params {
    int method;
    double rhohv_min;
    double tex_limit;  // exp(-texture_max^2), texture_max in radians
    int64_t n_offset;
    int offset_mode;
    double offset_value;
    int n_iter;
    double delta_thr;
    double z_thr;
    double kdp_min, kdp_max;
    double min_frac;  // minimum share of valid gates in the KDP window
    double sign;      // +1, or -1 for systems whose phase decreases in range
};

// One sweep: data pointers and the windows in gates for its gate spacing.
struct Sweep {
    const double* phi = nullptr;
    const double* rho = nullptr;
    const double* z = nullptr;
    double* out = nullptr;
    double* kdp = nullptr;
    double* off = nullptr;
    uint8_t* valid = nullptr;
    int64_t nray = 0, ngate = 0;
    int64_t htex = 1, hf = 1, hk_short = 1, hk_long = 1;
    double dr = 1.0;        // gate spacing [km]
    int64_t first_ray = 0;  // index of the sweep's first ray in the pool
};

// Run fn(sweep, ray, scratch) over the rays of all sweeps with atomic block
// scheduling; make() builds one scratch per thread before the loop.
template <class Make, class Fn>
void parallel_rays(const std::vector<Sweep>& sweeps, int n_threads, Make make, Fn fn) {
    const int64_t total = sweeps.empty() ? 0 : sweeps.back().first_ray + sweeps.back().nray;
    if (total == 0) return;
    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    int nt = n_threads > 0 ? n_threads : static_cast<int>(hw);
    nt = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(nt, total)));
    const int64_t block = 4;
    std::atomic<int64_t> next{0};
    auto worker = [&]() {
        auto scratch = make();
        size_t k = 0;
        for (;;) {
            const int64_t r0 = next.fetch_add(block);
            if (r0 >= total) break;
            const int64_t r1 = std::min(total, r0 + block);
            for (int64_t r = r0; r < r1; ++r) {
                while (r >= sweeps[k].first_ray + sweeps[k].nray) ++k;
                fn(sweeps[k], r - sweeps[k].first_ray, scratch);
            }
        }
    };
    std::vector<std::thread> pool;
    for (int t = 1; t < nt; ++t) pool.emplace_back(worker);
    worker();
    for (auto& th : pool) th.join();
}

// Prefix sums with a leading zero (sequential, as numpy.cumsum).
inline void prefix(const double* a, int64_t n, double* out) {
    out[0] = 0.0;
    for (int64_t i = 0; i < n; ++i) out[i + 1] = out[i] + a[i];
}

// Prefix counts of a 0/1 mask with a leading zero.
inline void prefix_valid(const uint8_t* v, int64_t n, double* out) {
    out[0] = 0.0;
    for (int64_t i = 0; i < n; ++i) out[i + 1] = out[i] + static_cast<double>(v[i]);
}

struct MaskScratch {
    std::vector<double> c, s, b, pc, ps, pb;
    explicit MaskScratch(int64_t ng)
        : c(ng), s(ng), b(ng), pc(ng + 1), ps(ng + 1), pb(ng + 1) {}
};

// Per-thread buffers, sized once for the longest ray and the widest filter so
// that nothing is allocated while rays are processed.
struct RayScratch {
    std::vector<double> x, y0, y, f, pad, ps, p1, p2, gy, kdp, mean, fit;
    std::vector<int64_t> idx, count;
    RayScratch(int64_t ng, int64_t hmax)
        : x(ng), y0(ng), y(ng), f(ng), pad(ng + 2 * hmax), ps(ng + 2 * hmax + 1),
          p1(ng + 1), p2(ng + 1), gy(ng), kdp(ng), fit(ng) {
        idx.reserve(ng);
        count.reserve(ng);
        mean.reserve(ng);
    }
};

// Moving average of length 2h+1 with odd reflection at both ends, in place.
void boxcar(double* y, int64_t ng, int64_t h, RayScratch& s) {
    const int64_t np_ = ng + 2 * h;
    double* p = s.pad.data();
    for (int64_t j = 0; j < h; ++j) {
        const int64_t k = std::min<int64_t>(h - j, ng - 1);
        p[j] = 2.0 * y[0] - y[k];
    }
    for (int64_t i = 0; i < ng; ++i) p[h + i] = y[i];
    for (int64_t m = 0; m < h; ++m) {
        const int64_t k = std::min<int64_t>(m + 1, ng - 1);
        p[h + ng + m] = 2.0 * y[ng - 1] - y[ng - 1 - k];
    }
    prefix(p, np_, s.ps.data());
    const double* S = s.ps.data();
    const double len = static_cast<double>(2 * h + 1);
    for (int64_t i = 0; i < ng; ++i) y[i] = (S[i + 2 * h + 1] - S[i]) / len;
}

// Low-pass filter: three moving-average passes (cubic B-spline kernel).
void smooth(double* y, int64_t ng, int64_t h, RayScratch& s) {
    for (int pass = 0; pass < 3; ++pass) boxcar(y, ng, h, s);
}

// KDP = half the least-squares slope over [g - h, g + h] (clipped), per km.
void kdp_of(const double* y, const double* z, const Params& p, const Sweep& sw,
            RayScratch& s, double* kdp) {
    const int64_t ng = sw.ngate;
    for (int64_t g = 0; g < ng; ++g) s.gy[g] = y[g] * static_cast<double>(g);
    prefix(y, ng, s.p1.data());
    prefix(s.gy.data(), ng, s.p2.data());
    const double* P1 = s.p1.data();
    const double* P2 = s.p2.data();
    for (int64_t g = 0; g < ng; ++g) {
        const bool heavy = z != nullptr && z[g] >= p.z_thr;  // false for NaN
        const int64_t h = heavy ? sw.hk_short : sw.hk_long;
        const int64_t lo = std::max<int64_t>(g - h, 0);
        const int64_t hi = std::min<int64_t>(g + h, ng - 1) + 1;
        const double sy = P1[hi] - P1[lo];
        const double sgy = P2[hi] - P2[lo];
        const double n = static_cast<double>(hi - lo);
        const double gbar = 0.5 * static_cast<double>(lo + hi - 1);
        const double den = n * (n * n - 1.0) / 12.0;
        kdp[g] = 0.5 * ((sgy - gbar * sy) / den / sw.dr);
    }
}

// numpy.interp of (idx, val) at every gate: linear inside, constant outside.
void interp_gates(const std::vector<int64_t>& idx, const double* val, int64_t ng,
                  double* out) {
    const int64_t n = static_cast<int64_t>(idx.size());
    int64_t j = 0;
    for (int64_t g = 0; g < ng; ++g) {
        if (g <= idx[0]) {
            out[g] = val[0];
        } else if (g >= idx[n - 1]) {
            out[g] = val[n - 1];
        } else {
            while (idx[j + 1] < g) ++j;
            if (idx[j + 1] == g) {
                out[g] = val[j + 1];
                continue;
            }
            const double slope =
                (val[j + 1] - val[j]) / static_cast<double>(idx[j + 1] - idx[j]);
            out[g] = slope * static_cast<double>(g - idx[j]) + val[j];
        }
    }
}

// Difference wrapped into [-180, 180) degrees.
inline double wrap180(double d) { return d - 360.0 * std::floor(d / 360.0 + 0.5); }

// Sign-test statistics of one ray over adjacent rain gates.
void sign_stats(const Sweep& sw, int64_t r, const uint8_t* v, SignStats& st) {
    const int64_t ng = sw.ngate;
    const double* x = sw.phi + r * ng;
    const double* rr = sw.rho ? sw.rho + r * ng : nullptr;
    const double* zr = sw.z ? sw.z + r * ng : nullptr;
    st = SignStats();
    bool prev_rain = false, prev_pair = false;
    for (int64_t g = 0; g < ng; ++g) {
        bool rain = v[g] != 0;
        if (rr) rain = rain && rr[g] >= kSignRho;
        if (zr) rain = rain && zr[g] >= kSignZ;
        const bool pair = rain && prev_rain;
        if (pair) {
            const double d = wrap180(x[g] - x[g - 1]);
            st.sum += d;
            st.sum2 += d * d;
            ++st.pairs;
            if (!prev_pair) ++st.runs;
        }
        prev_rain = rain;
        prev_pair = pair;
    }
}

// Gate mask, the phasor sum of the first n_offset valid gates and the
// sign-test statistics of one ray.
void mask_ray(const Params& p, const Sweep& sw, int64_t r, MaskScratch& s,
              double& cr, double& sr, int64_t& rank, SignStats& st) {
    const int64_t ng = sw.ngate;
    const double* x = sw.phi + r * ng;
    const double* rr = sw.rho ? sw.rho + r * ng : nullptr;
    for (int64_t g = 0; g < ng; ++g) {
        bool b = std::isfinite(x[g]);
        if (rr) b = b && rr[g] >= p.rhohv_min;  // false for NaN
        if (b) {
            const double rad = x[g] * kDegToRad;
            s.c[g] = std::cos(rad);
            s.s[g] = std::sin(rad);
            s.b[g] = 1.0;
        } else {
            s.c[g] = s.s[g] = s.b[g] = 0.0;
        }
    }
    prefix(s.c.data(), ng, s.pc.data());
    prefix(s.s.data(), ng, s.ps.data());
    prefix(s.b.data(), ng, s.pb.data());
    uint8_t* v = sw.valid + r * ng;
    const double need = static_cast<double>(sw.htex + 1);
    rank = 0;
    cr = sr = 0.0;
    for (int64_t g = 0; g < ng; ++g) {
        const int64_t lo = std::max<int64_t>(g - sw.htex, 0);
        const int64_t hi = std::min<int64_t>(g + sw.htex, ng - 1) + 1;
        const double sc = s.pc[hi] - s.pc[lo];
        const double ss = s.ps[hi] - s.ps[lo];
        const double sn = s.pb[hi] - s.pb[lo];
        // circular std <= limit  <=>  R^2 >= exp(-limit^2), R = |sum| / n
        const bool ok = sn >= need && sc * sc + ss * ss >= sn * sn * p.tex_limit;
        v[g] = (s.b[g] != 0.0 && ok) ? 1 : 0;
        if (v[g] && rank < p.n_offset) {
            cr += s.c[g];
            sr += s.s[g];
            ++rank;
        }
    }
    sign_stats(sw, r, v, st);
}

// Unfold sign * phase of the valid gates relative to the running reference
// and remove the offset (already multiplied by sign): s.x[i] holds the value
// of valid gate s.idx[i]. Returns their count.
int64_t unfold_ray(const double* x, const uint8_t* v, int64_t ng, double sign,
                   double offset, RayScratch& s) {
    double buf[kUnfoldMemory];
    for (double& b : buf) b = offset;
    int pos = 0;
    s.idx.clear();
    for (int64_t g = 0; g < ng; ++g) {
        if (!v[g]) continue;
        double sum = 0.0;
        for (double b : buf) sum += b;
        const double ref = sum / kUnfoldMemory;
        const double xg = sign * x[g];
        const double u = xg + 360.0 * std::floor((ref - xg) / 360.0 + 0.5);
        s.x[s.idx.size()] = u - offset;
        s.idx.push_back(g);
        buf[pos] = u;
        pos = (pos + 1) % kUnfoldMemory;
    }
    return static_cast<int64_t>(s.idx.size());
}

// Hubbert and Bringi (1995): replace gates that depart from the filtered
// profile (and masked gates) by the filtered values, iterate, filter.
void filter_hubbert(const Params& p, const Sweep& sw, const uint8_t* v, RayScratch& s,
                    double* o) {
    const int64_t ng = sw.ngate;
    std::copy(s.y0.begin(), s.y0.begin() + ng, s.y.begin());
    for (int it = 0; it < p.n_iter; ++it) {
        std::copy(s.y.begin(), s.y.begin() + ng, s.f.begin());
        smooth(s.f.data(), ng, sw.hf, s);
        for (int64_t g = 0; g < ng; ++g) {
            const bool keep = v[g] && std::fabs(s.y0[g] - s.f[g]) <= p.delta_thr;
            s.y[g] = keep ? s.y0[g] : s.f[g];
        }
    }
    std::copy(s.y.begin(), s.y.begin() + ng, o);
    smooth(o, ng, sw.hf, s);
}

// Vulpiani et al. (2012): KDP from PhiDP, implausible values set to zero,
// PhiDP rebuilt by integration; leaves the last KDP in s.kdp.
void filter_vulpiani(const Params& p, const Sweep& sw, const double* zr, int64_t nv,
                     RayScratch& s, double* o) {
    const int64_t ng = sw.ngate;
    std::copy(s.y0.begin(), s.y0.begin() + ng, o);
    const int iters = std::max(p.n_iter, 1);
    double* kd = s.kdp.data();
    for (int it = 0; it < iters; ++it) {
        kdp_of(o, zr, p, sw, s, kd);
        for (int64_t g = 0; g < ng; ++g)
            if (kd[g] < p.kdp_min || kd[g] > p.kdp_max) kd[g] = 0.0;
        const double start = o[0];
        double acc = 0.0;
        for (int64_t g = 1; g < ng; ++g) {
            acc += sw.dr * (kd[g - 1] + kd[g]);
            o[g] = start + acc;
        }
    }
    // integration constant: least-squares fit to the measured gates
    double sum = 0.0;
    for (int64_t i = 0; i < nv; ++i) sum += s.x[i] - o[s.idx[i]];
    const double shift = sum / static_cast<double>(nv);
    for (int64_t g = 0; g < ng; ++g) o[g] += shift;
}

// Non-decreasing least-squares fit of s.x[0:nv] (pool-adjacent-violators)
// into s.fit.
void isotonic(int64_t nv, RayScratch& s) {
    s.mean.clear();
    s.count.clear();
    for (int64_t i = 0; i < nv; ++i) {
        s.mean.push_back(s.x[i]);
        s.count.push_back(1);
        while (s.mean.size() > 1 && s.mean[s.mean.size() - 2] > s.mean.back()) {
            const size_t m = s.mean.size();
            const int64_t n = s.count[m - 2] + s.count[m - 1];
            const double avg = (s.mean[m - 2] * static_cast<double>(s.count[m - 2]) +
                                s.mean[m - 1] * static_cast<double>(s.count[m - 1])) /
                               static_cast<double>(n);
            s.mean.pop_back();
            s.count.pop_back();
            s.mean.back() = avg;
            s.count.back() = n;
        }
    }
    int64_t i = 0;
    for (size_t b = 0; b < s.mean.size(); ++b)
        for (int64_t c = 0; c < s.count[b]; ++c) s.fit[i++] = s.mean[b];
}

// Unfold, fill, filter and differentiate one ray.
void process_ray(const Params& p, const Sweep& sw, int64_t r, RayScratch& s) {
    const int64_t ng = sw.ngate;
    const uint8_t* v = sw.valid + r * ng;
    const double* zr = sw.z ? sw.z + r * ng : nullptr;
    double* o = sw.out + r * ng;
    double* k = sw.kdp + r * ng;

    const int64_t nv = unfold_ray(sw.phi + r * ng, v, ng, p.sign, p.sign * sw.off[r], s);
    if (nv < kMinValid) {
        for (int64_t g = 0; g < ng; ++g) o[g] = k[g] = kNaN;
        return;
    }
    interp_gates(s.idx, s.x.data(), ng, s.y0.data());  // bridge masked gates

    if (p.method == kHubbert) {
        filter_hubbert(p, sw, v, s, o);
        kdp_of(o, zr, p, sw, s, s.kdp.data());
    } else if (p.method == kVulpiani) {
        filter_vulpiani(p, sw, zr, nv, s, o);
    } else {
        isotonic(nv, s);
        interp_gates(s.idx, s.fit.data(), ng, o);
        smooth(o, ng, sw.hf, s);
        kdp_of(o, zr, p, sw, s, s.kdp.data());
    }
    // KDP only at valid gates whose window holds enough valid gates
    prefix_valid(v, ng, s.p1.data());
    const double* P = s.p1.data();
    for (int64_t g = 0; g < ng; ++g) {
        const int64_t h = (zr != nullptr && zr[g] >= p.z_thr) ? sw.hk_short : sw.hk_long;
        const int64_t lo = std::max<int64_t>(g - h, 0);
        const int64_t hi = std::min<int64_t>(g + h, ng - 1) + 1;
        const bool enough = P[hi] - P[lo] >= p.min_frac * static_cast<double>(hi - lo);
        k[g] = (v[g] && enough) ? s.kdp[g] : kNaN;
    }
}

const double* optional_data(const py::object& obj, const DArray& phi, DArray& keep,
                            const char* name) {
    if (obj.is_none()) return nullptr;
    keep = obj.cast<DArray>();
    if (keep.ndim() != 2 || keep.shape(0) != phi.shape(0) || keep.shape(1) != phi.shape(1))
        throw std::invalid_argument(std::string(name) + " must have the shape of phidp");
    return keep.data();
}

}  // namespace

// Process the sweeps of a volume in one call. Per sweep: phidp (ray, gate),
// optional rhohv and dbzh of the same shape (or None), the gate spacing [km]
// and the half windows in gates (texture, filter, short and long KDP window).
// Returns lists of processed phidp, kdp and the per-ray offset.
py::tuple process_phidp(const std::vector<DArray>& phi, const std::vector<py::object>& rho,
                        const std::vector<py::object>& dbz, const std::vector<double>& dr,
                        const std::vector<int64_t>& htex, const std::vector<int64_t>& hf,
                        const std::vector<int64_t>& hk_short,
                        const std::vector<int64_t>& hk_long, int method, double rhohv_min,
                        double tex_limit, int64_t n_offset, int offset_mode,
                        double offset_value, int n_iter, double delta_thr, double z_thr,
                        double kdp_min, double kdp_max, double min_valid_fraction,
                        int phidp_sign, int n_threads) {
    const size_t ns = phi.size();
    if (rho.size() != ns || dbz.size() != ns || dr.size() != ns || htex.size() != ns ||
        hf.size() != ns || hk_short.size() != ns || hk_long.size() != ns)
        throw std::invalid_argument("need one entry per sweep in every list");
    if (method < 0 || method > 2) throw std::invalid_argument("unknown method");
    if (offset_mode < 0 || offset_mode > 2) throw std::invalid_argument("unknown offset mode");
    if (phidp_sign < -1 || phidp_sign > 1)
        throw std::invalid_argument("phidp_sign must be -1, 0 (auto) or 1");
    Params p{method, rhohv_min, tex_limit, n_offset, offset_mode, offset_value,
             n_iter, delta_thr, z_thr,     kdp_min,  kdp_max,     min_valid_fraction,
             1.0};

    std::vector<DArray> rho_keep(ns), z_keep(ns);
    std::vector<py::array_t<double>> outs, kdps, offs;
    std::vector<Sweep> sweeps(ns);
    int64_t first = 0, max_ng = 0, max_h = 1, total_gates = 0;
    for (size_t i = 0; i < ns; ++i) {
        if (phi[i].ndim() != 2) throw std::invalid_argument("phidp must be 2-D (ray, gate)");
        Sweep& sw = sweeps[i];
        sw.nray = phi[i].shape(0);
        sw.ngate = phi[i].shape(1);
        if (sw.ngate < 2) throw std::invalid_argument("need at least two gates");
        if (htex[i] < 1 || hf[i] < 1 || hk_short[i] < 1 || hk_long[i] < 1 || !(dr[i] > 0))
            throw std::invalid_argument("windows must be at least one gate, dr positive");
        sw.phi = phi[i].data();
        sw.rho = optional_data(rho[i], phi[i], rho_keep[i], "rhohv");
        sw.z = optional_data(dbz[i], phi[i], z_keep[i], "dbzh");
        sw.dr = dr[i];
        sw.htex = htex[i];
        sw.hf = hf[i];
        sw.hk_short = hk_short[i];
        sw.hk_long = hk_long[i];
        sw.first_ray = first;
        first += sw.nray;
        total_gates += sw.nray * sw.ngate;
        max_ng = std::max(max_ng, sw.ngate);
        max_h = std::max(max_h, sw.hf);
        outs.emplace_back(std::vector<py::ssize_t>{sw.nray, sw.ngate});
        kdps.emplace_back(std::vector<py::ssize_t>{sw.nray, sw.ngate});
        offs.emplace_back(std::vector<py::ssize_t>{sw.nray});
        sw.out = outs.back().mutable_data();
        sw.kdp = kdps.back().mutable_data();
        sw.off = offs.back().mutable_data();
    }
    const int64_t total_rays = first;
    std::vector<uint8_t> valid(static_cast<size_t>(total_gates));
    {
        int64_t at = 0;
        for (Sweep& sw : sweeps) {
            sw.valid = valid.data() + at;
            at += sw.nray * sw.ngate;
        }
    }

    {
        py::gil_scoped_release release;
        std::vector<double> ray_c(total_rays), ray_s(total_rays);
        std::vector<int64_t> ray_n(total_rays);
        std::vector<SignStats> ray_sign(total_rays);

        // 1. gate masks and phasor sums of the first valid gates
        parallel_rays(
            sweeps, n_threads, [&] { return MaskScratch(max_ng); },
            [&](const Sweep& sw, int64_t r, MaskScratch& s) {
                const int64_t q = sw.first_ray + r;
                mask_ray(p, sw, r, s, ray_c[q], ray_s[q], ray_n[q], ray_sign[q]);
            });

        // 2. sign convention: the phase must increase in range through rain.
        // Pooled over all rays of all sweeps, the phase is flipped only if
        // it decreases with a significance of kSignSigmas.
        if (phidp_sign != 0) {
            p.sign = phidp_sign;
        } else {
            SignStats t;
            for (int64_t q = 0; q < total_rays; ++q) {
                t.sum += ray_sign[q].sum;
                t.sum2 += ray_sign[q].sum2;
                t.pairs += ray_sign[q].pairs;
                t.runs += ray_sign[q].runs;
            }
            p.sign = 1.0;
            if (t.pairs > 0) {
                const double noise =
                    std::sqrt(static_cast<double>(t.runs) * t.sum2 / static_cast<double>(t.pairs));
                if (t.sum < -kSignSigmas * noise) p.sign = -1.0;
            }
        }

        // 3. system offset per sweep (circular mean) or per ray
        for (Sweep& sw : sweeps) {
            double ct = 0.0, st = 0.0;
            bool any = false;
            for (int64_t r = 0; r < sw.nray; ++r) {
                const int64_t q = sw.first_ray + r;
                ct += ray_c[q];
                st += ray_s[q];
                any = any || ray_n[q] > 0;
            }
            const double sweep = any ? std::atan2(st, ct) * kRadToDeg : 0.0;
            for (int64_t r = 0; r < sw.nray; ++r) {
                const int64_t q = sw.first_ray + r;
                if (p.offset_mode == kFixed) sw.off[r] = p.offset_value;
                else if (p.offset_mode == kSweep || ray_n[q] == 0) sw.off[r] = sweep;
                else sw.off[r] = std::atan2(ray_s[q], ray_c[q]) * kRadToDeg;
            }
        }

        // 4. unfold, filter and differentiate every ray
        parallel_rays(
            sweeps, n_threads, [&] { return RayScratch(max_ng, max_h); },
            [&](const Sweep& sw, int64_t r, RayScratch& s) { process_ray(p, sw, r, s); });
    }
    py::list out_list, kdp_list, off_list;
    for (size_t i = 0; i < ns; ++i) {
        out_list.append(outs[i]);
        kdp_list.append(kdps[i]);
        off_list.append(offs[i]);
    }
    return py::make_tuple(out_list, kdp_list, off_list, static_cast<int>(p.sign));
}

PYBIND11_MODULE(_kdp, m) {
    m.doc() = "Compiled differential phase processing and KDP kernel for radarx.";
    m.def("process_phidp", &process_phidp, py::arg("phidp"), py::arg("rhohv"),
          py::arg("dbzh"), py::arg("dr"), py::arg("htex"), py::arg("hf"),
          py::arg("hk_short"), py::arg("hk_long"), py::arg("method"),
          py::arg("rhohv_min"), py::arg("tex_limit"), py::arg("n_offset"),
          py::arg("offset_mode"), py::arg("offset_value"), py::arg("n_iter"),
          py::arg("delta_threshold"), py::arg("z_threshold"), py::arg("kdp_min"),
          py::arg("kdp_max"), py::arg("min_valid_fraction") = 0.5,
          py::arg("phidp_sign") = 0, py::arg("n_threads") = 0);
}
