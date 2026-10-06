// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Non-meteorological echo classification kernel.
//
// The rays of all sweeps of a volume form one pool of work that threads take
// in small blocks from an atomic counter:
//
// 1. per ray, one fused pass: no-data masking and running (prefix) sums of
//    every quantity along the ray (echo, rhohv, zdr, zdr^2, cos/sin phidp,
//    squared reflectivity differences, spin changes); then per gate the
//    window features, their trapezoidal memberships and the weighted mean
//    score (fuzzy logic after Gourley et al. 2007 and Krause 2016; texture
//    and spin of the reflectivity after Steiner and Smith 2002);
// 2. per ray, the mean score over the 3 x 3 neighbouring gates and the
//    meteorological / non-meteorological decision;
// 3. per sweep (in parallel), 8-connected regions of meteorological gates by
//    union-find; regions smaller than min_size become speckle.
//
// Prefix sums make every window O(1), so each pass is O(N) in the number of
// gates. Every step follows the NumPy reference in radarx/retrieve/qc.py, in
// the same order of operations.

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
using UArray = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;

namespace {

constexpr double kPi = 3.14159265358979323846;  // M_PI is not standard (MSVC)
constexpr double kDegToRad = kPi / 180.0;
constexpr double kRadToDeg = 180.0 / kPi;
constexpr int kFeatures = 6;  // rhohv, zdr, zdr_texture, phidp_texture, dbz_texture, spin
constexpr float kNaNf = std::numeric_limits<float>::quiet_NaN();

enum Class : int8_t { kNoEcho = 0, kMeteo = 1, kNonMeteo = 2, kSpeckle = 3 };

struct Params {
    double floor_z, floor_zdr, floor_rho, floor_phi;  // no-data floors
    double snr_min;
    double lim[kFeatures][4];
    double w[kFeatures];
    double spin_thr;
    float threshold;
    int64_t min_size;
};

struct Sweep {
    const double* z = nullptr;
    const double* zdr = nullptr;
    const double* rho = nullptr;
    const double* phi = nullptr;
    const double* snr = nullptr;
    const uint8_t* links = nullptr;
    float* raw = nullptr;
    float* score = nullptr;
    int8_t* cls = nullptr;
    int64_t nray = 0, ngate = 0, h = 1;
    int64_t first_ray = 0;
};

// Run fn(sweep, ray, scratch) over the rays of all sweeps with atomic block
// scheduling; make() builds one scratch per thread before the loop.
template <class Make, class Fn>
void parallel_rays(const std::vector<Sweep>& sweeps, int nt, Make make, Fn fn) {
    const int64_t total = sweeps.empty() ? 0 : sweeps.back().first_ray + sweeps.back().nray;
    if (total == 0) return;
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

// Trapezoidal membership: 0 at or below a and at or above d, 1 in [b, c].
inline double trapezoid(double x, const double* l) {
    if (x <= l[0] || x >= l[3]) return 0.0;
    if (x < l[1]) return (x - l[0]) / (l[1] - l[0]);
    if (x > l[2]) return (l[3] - x) / (l[3] - l[2]);
    return 1.0;
}

// Window [lo, hi) sum of a prefix array (zero for an empty window).
inline double span(const double* p, int64_t lo, int64_t hi) {
    return hi > lo ? p[hi] - p[lo] : 0.0;
}

// Per-thread prefix arrays (ngate + 1 each).
struct RayScratch {
    enum { kEcho, kPair, kDz2, kTrip, kFlip, kZn, kZ1, kZ2, kRn, kR1, kPn, kPc, kPs, kN };
    std::vector<double> buf;
    std::vector<uint8_t> echo;
    std::vector<double> dz;
    int64_t stride;
    explicit RayScratch(int64_t ng)
        : buf(static_cast<size_t>(kN * (ng + 1))), echo(ng), dz(ng), stride(ng + 1) {}
    double* p(int k) { return buf.data() + k * stride; }
};

inline bool valid(const double* x, int64_t i, double floor) {
    return x != nullptr && std::isfinite(x[i]) && x[i] > floor;
}

// The fields of one ray (nullptr where a field is missing).
struct Ray {
    const double *z, *zdr, *rho, *phi, *snr;
    int64_t ng;
};

// Echo flags and the reflectivity difference to the next gate.
void mark_echo(const Params& P, const Ray& ray, RayScratch& s) {
    uint8_t* echo = s.echo.data();
    for (int64_t g = 0; g < ray.ng; ++g) {
        bool e = std::isfinite(ray.z[g]) && ray.z[g] > P.floor_z;
        if (ray.snr != nullptr) e = e && ray.snr[g] >= P.snr_min;
        echo[g] = e ? 1 : 0;
    }
    for (int64_t g = 0; g < ray.ng; ++g) {
        s.dz[g] = (g + 1 < ray.ng)
                      ? (echo[g + 1] ? ray.z[g + 1] : 0.0) - (echo[g] ? ray.z[g] : 0.0)
                      : 0.0;
    }
}

// Prefix sums of the reflectivity quantities: echo, pairs of adjacent echo
// gates and their squared difference, triples and spin changes.
void reflectivity_prefixes(const Params& P, const Ray& ray, RayScratch& s) {
    const uint8_t* echo = s.echo.data();
    const double* dz = s.dz.data();
    double* pe = s.p(RayScratch::kEcho);
    double* pp = s.p(RayScratch::kPair);
    double* pd = s.p(RayScratch::kDz2);
    double* pt = s.p(RayScratch::kTrip);
    double* pf = s.p(RayScratch::kFlip);
    const double t = P.spin_thr;
    for (int64_t g = 0; g < ray.ng; ++g) {
        const bool e = echo[g] != 0;
        const bool pair = e && g + 1 < ray.ng && echo[g + 1];
        const bool trip = pair && g >= 1 && echo[g - 1];
        const double d1 = trip ? dz[g - 1] : 0.0, d2 = dz[g];
        const bool flip = trip && d1 * d2 < 0.0 && std::fabs(d1) >= t && std::fabs(d2) >= t;
        pe[g + 1] = pe[g] + (e ? 1.0 : 0.0);
        pp[g + 1] = pp[g] + (pair ? 1.0 : 0.0);
        pd[g + 1] = pd[g] + (pair ? dz[g] * dz[g] : 0.0);
        pt[g + 1] = pt[g] + (trip ? 1.0 : 0.0);
        pf[g + 1] = pf[g] + (flip ? 1.0 : 0.0);
    }
}

// Prefix sums of the polarimetric quantities at echo gates with data.
void polarimetric_prefixes(const Params& P, const Ray& ray, RayScratch& s) {
    const uint8_t* echo = s.echo.data();
    double* zn = s.p(RayScratch::kZn);
    double* z1 = s.p(RayScratch::kZ1);
    double* z2 = s.p(RayScratch::kZ2);
    double* rn = s.p(RayScratch::kRn);
    double* r1 = s.p(RayScratch::kR1);
    double* pn = s.p(RayScratch::kPn);
    double* pc = s.p(RayScratch::kPc);
    double* ps = s.p(RayScratch::kPs);
    for (int64_t g = 0; g < ray.ng; ++g) {
        const bool e = echo[g] != 0;
        const bool okd = e && valid(ray.zdr, g, P.floor_zdr);
        const double vd = okd ? ray.zdr[g] : 0.0;
        zn[g + 1] = zn[g] + (okd ? 1.0 : 0.0);
        z1[g + 1] = z1[g] + vd;
        z2[g + 1] = z2[g] + vd * vd;
        const bool okr = e && valid(ray.rho, g, P.floor_rho);
        double vr = okr ? ray.rho[g] : 0.0;
        if (vr > 1.0) vr = 2.0 - vr;  // values above 1 only occur at low SNR
        rn[g + 1] = rn[g] + (okr ? 1.0 : 0.0);
        r1[g + 1] = r1[g] + vr;
        const bool okp = e && valid(ray.phi, g, P.floor_phi);
        const double rad = (okp ? ray.phi[g] : 0.0) * kDegToRad;
        pn[g + 1] = pn[g] + (okp ? 1.0 : 0.0);
        pc[g + 1] = pc[g] + (okp ? std::cos(rad) : 0.0);
        ps[g + 1] = ps[g] + (okp ? std::sin(rad) : 0.0);
    }
}

// Weighted sum of memberships (num) and of weights (den) over the window
// [lo, hi), in the order of FEATURES in qc.py.
struct Score {
    const Params& P;
    double num = 0.0, den = 0.0;
    explicit Score(const Params& params) : P(params) {}
    void add(int k, double x) {
        if (P.w[k] <= 0.0) return;
        num = num + P.w[k] * trapezoid(x, P.lim[k]);
        den = den + P.w[k];
    }
};

void polarimetric_features(const Ray& ray, RayScratch& s, int64_t lo, int64_t hi,
                           Score& sc) {
    if (ray.rho != nullptr) {
        const double n = span(s.p(RayScratch::kRn), lo, hi);
        if (n >= 1) sc.add(0, span(s.p(RayScratch::kR1), lo, hi) / n);
    }
    if (ray.zdr != nullptr) {
        const double n = span(s.p(RayScratch::kZn), lo, hi);
        const double mean = span(s.p(RayScratch::kZ1), lo, hi) / n;
        if (n >= 1) sc.add(1, mean);
        if (n >= 3) {
            const double var = span(s.p(RayScratch::kZ2), lo, hi) / n - mean * mean;
            sc.add(2, std::sqrt(std::max(var, 0.0)));
        }
    }
    if (ray.phi != nullptr) {
        const double n = span(s.p(RayScratch::kPn), lo, hi);
        if (n >= 3) {
            const double c = span(s.p(RayScratch::kPc), lo, hi);
            const double sn = span(s.p(RayScratch::kPs), lo, hi);
            const double r2 = std::min(std::max((c * c + sn * sn) / (n * n), 1e-300), 1.0);
            sc.add(3, std::sqrt(-std::log(r2)) * kRadToDeg);
        }
    }
}

void reflectivity_features(RayScratch& s, int64_t lo, int64_t hi, Score& sc) {
    const double npair = span(s.p(RayScratch::kPair), lo, hi - 1);
    if (npair >= 1) sc.add(4, span(s.p(RayScratch::kDz2), lo, hi - 1) / npair);
    const double ntrip = span(s.p(RayScratch::kTrip), lo + 1, hi - 1);
    if (ntrip >= 1) sc.add(5, 100.0 * span(s.p(RayScratch::kFlip), lo + 1, hi - 1) / ntrip);
}

// Step 1: features, memberships and the raw score of one ray.
void features_ray(const Params& P, const Sweep& sw, int64_t r, RayScratch& s) {
    const int64_t ng = sw.ngate, off = r * ng, h = sw.h;
    const Ray ray{sw.z + off,
                  sw.zdr ? sw.zdr + off : nullptr,
                  sw.rho ? sw.rho + off : nullptr,
                  sw.phi ? sw.phi + off : nullptr,
                  sw.snr ? sw.snr + off : nullptr,
                  ng};
    for (int k = 0; k < RayScratch::kN; ++k) s.p(k)[0] = 0.0;
    mark_echo(P, ray, s);
    reflectivity_prefixes(P, ray, s);
    polarimetric_prefixes(P, ray, s);
    const double* pe = s.p(RayScratch::kEcho);
    float* raw = sw.raw + off;
    int8_t* cls = sw.cls + off;
    for (int64_t g = 0; g < ng; ++g) {
        if (!s.echo[g]) {
            raw[g] = kNaNf;
            cls[g] = kNoEcho;
            continue;
        }
        cls[g] = kSpeckle;  // until the score says otherwise
        const int64_t lo = std::max<int64_t>(g - h, 0);
        const int64_t hi = std::min<int64_t>(g + h, ng - 1) + 1;
        Score sc(P);
        polarimetric_features(ray, s, lo, hi, sc);
        reflectivity_features(s, lo, hi, sc);
        const bool isolated = span(pe, lo, hi) < static_cast<double>(h + 1);
        raw[g] = (!isolated && sc.den > 0.0) ? static_cast<float>(sc.num / sc.den) : kNaNf;
    }
}

// Previous and next linked ray (-1 if none), as _neighbour_rays in qc.py.
inline void neighbours(const Sweep& sw, int64_t r, int64_t& prv, int64_t& nxt) {
    const int64_t n = sw.nray;
    nxt = sw.links[r] ? (r + 1) % n : -1;
    if (nxt == r) nxt = -1;
    const int64_t q = (r - 1 + n) % n;
    prv = sw.links[q] ? q : -1;
    if (prv == r || prv == nxt) prv = -1;
}

// Step 2: 3 x 3 mean of the raw score and the decision.
void smooth_ray(const Params& P, const Sweep& sw, int64_t r) {
    const int64_t ng = sw.ngate;
    int64_t prv, nxt;
    neighbours(sw, r, prv, nxt);
    const int64_t rays[3] = {prv, r, nxt};
    const float* self = sw.raw + r * ng;
    float* score = sw.score + r * ng;
    int8_t* cls = sw.cls + r * ng;
    for (int64_t g = 0; g < ng; ++g) {
        if (std::isnan(self[g])) {
            score[g] = kNaNf;
            continue;
        }
        double total = 0.0, count = 0.0;
        for (int64_t q : rays) {
            if (q < 0) continue;
            const float* row = sw.raw + q * ng;
            for (int64_t dg = -1; dg <= 1; ++dg) {
                const int64_t j = g + dg;
                if (j < 0 || j >= ng) continue;
                if (std::isfinite(row[j])) {
                    total = total + static_cast<double>(row[j]);
                    count = count + 1.0;
                }
            }
        }
        const float v = static_cast<float>(total / count);
        score[g] = v;
        cls[g] = v >= P.threshold ? kMeteo : kNonMeteo;
    }
}

// Union-find with path halving and union by size.
inline int64_t find(std::vector<int64_t>& parent, int64_t x) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    return x;
}

inline void unite(std::vector<int64_t>& parent, std::vector<int64_t>& size, int64_t a,
                  int64_t b) {
    a = find(parent, a);
    b = find(parent, b);
    if (a == b) return;
    if (size[a] < size[b]) std::swap(a, b);
    parent[b] = a;
    size[a] += size[b];
}

// Step 3: meteorological regions smaller than min_size become speckle.
void despeckle(const Params& P, const Sweep& sw, std::vector<int64_t>& parent,
               std::vector<int64_t>& size) {
    const int64_t ng = sw.ngate, n = sw.nray * ng;
    int8_t* cls = sw.cls;
    for (int64_t i = 0; i < n; ++i) {
        parent[i] = i;
        size[i] = 1;
    }
    for (int64_t r = 0; r < sw.nray; ++r) {
        int64_t prv, nxt;
        neighbours(sw, r, prv, nxt);
        for (int64_t g = 0; g < ng; ++g) {
            const int64_t a = r * ng + g;
            if (cls[a] != kMeteo) continue;
            if (g + 1 < ng && cls[a + 1] == kMeteo) unite(parent, size, a, a + 1);
            if (nxt < 0) continue;
            for (int64_t dg = -1; dg <= 1; ++dg) {
                const int64_t j = g + dg;
                if (j < 0 || j >= ng) continue;
                const int64_t b = nxt * ng + j;
                if (cls[b] == kMeteo) unite(parent, size, a, b);
            }
        }
    }
    for (int64_t i = 0; i < n; ++i) {
        if (cls[i] == kMeteo && size[find(parent, i)] < P.min_size) cls[i] = kSpeckle;
    }
}

const double* optional_data(const py::object& obj, const DArray& ref, DArray& keep,
                            const char* name) {
    if (obj.is_none()) return nullptr;
    keep = obj.cast<DArray>();
    if (keep.ndim() != 2 || keep.shape(0) != ref.shape(0) || keep.shape(1) != ref.shape(1))
        throw std::invalid_argument(std::string(name) + " must have the shape of dbzh");
    return keep.data();
}

}  // namespace

// Classify the sweeps of a volume in one call. Per sweep: reflectivity
// (ray, gate), optional zdr, rhohv, phidp and snr of the same shape (or
// None), the ray links (1 where ray i touches ray i + 1, the last one the
// first) and the half window in gates. Returns lists of the score (float32)
// and the class (int8) of every gate.
py::tuple classify(const std::vector<DArray>& dbz, const std::vector<py::object>& zdr,
                   const std::vector<py::object>& rho, const std::vector<py::object>& phi,
                   const std::vector<py::object>& snr, const std::vector<UArray>& links,
                   const std::vector<int64_t>& h, const DArray& floors, double snr_min,
                   const DArray& limits, const DArray& weights, double spin_threshold,
                   double threshold, int64_t min_size, int n_threads) {
    const size_t ns = dbz.size();
    if (zdr.size() != ns || rho.size() != ns || phi.size() != ns || snr.size() != ns ||
        links.size() != ns || h.size() != ns)
        throw std::invalid_argument("need one entry per sweep in every list");
    if (floors.size() != 4) throw std::invalid_argument("need four no-data floors");
    if (limits.ndim() != 2 || limits.shape(0) != kFeatures || limits.shape(1) != 4)
        throw std::invalid_argument("limits must have shape (6, 4)");
    if (weights.size() != kFeatures) throw std::invalid_argument("need six weights");
    Params P{};
    const double* fl = floors.data();
    P.floor_z = fl[0];
    P.floor_zdr = fl[1];
    P.floor_rho = fl[2];
    P.floor_phi = fl[3];
    P.snr_min = snr_min;
    for (int k = 0; k < kFeatures; ++k) {
        for (int j = 0; j < 4; ++j) P.lim[k][j] = limits.at(k, j);
        P.w[k] = weights.at(k);
    }
    P.spin_thr = spin_threshold;
    P.threshold = static_cast<float>(threshold);
    P.min_size = min_size;

    std::vector<DArray> keep(4 * ns);
    std::vector<py::array_t<float>> scores;
    std::vector<py::array_t<int8_t>> classes;
    std::vector<Sweep> sweeps(ns);
    int64_t first = 0, max_ng = 1, max_n = 1, total = 0;
    for (size_t i = 0; i < ns; ++i) {
        if (dbz[i].ndim() != 2) throw std::invalid_argument("dbzh must be 2-D (ray, gate)");
        Sweep& sw = sweeps[i];
        sw.nray = dbz[i].shape(0);
        sw.ngate = dbz[i].shape(1);
        if (h[i] < 1) throw std::invalid_argument("the half window must be at least one gate");
        if (links[i].ndim() != 1 || links[i].shape(0) != sw.nray)
            throw std::invalid_argument("need one link per ray");
        sw.z = dbz[i].data();
        sw.zdr = optional_data(zdr[i], dbz[i], keep[4 * i], "zdr");
        sw.rho = optional_data(rho[i], dbz[i], keep[4 * i + 1], "rhohv");
        sw.phi = optional_data(phi[i], dbz[i], keep[4 * i + 2], "phidp");
        sw.snr = optional_data(snr[i], dbz[i], keep[4 * i + 3], "snr");
        sw.links = links[i].data();
        sw.h = h[i];
        sw.first_ray = first;
        first += sw.nray;
        max_ng = std::max(max_ng, sw.ngate);
        max_n = std::max(max_n, sw.nray * sw.ngate);
        total += sw.nray * sw.ngate;
        scores.emplace_back(std::vector<py::ssize_t>{sw.nray, sw.ngate});
        classes.emplace_back(std::vector<py::ssize_t>{sw.nray, sw.ngate});
        sw.score = scores.back().mutable_data();
        sw.cls = classes.back().mutable_data();
    }
    std::vector<float> raw(static_cast<size_t>(total));
    {
        int64_t at = 0;
        for (Sweep& sw : sweeps) {
            sw.raw = raw.data() + at;
            at += sw.nray * sw.ngate;
        }
    }
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const int nt = n_threads > 0 ? n_threads : static_cast<int>(hw);

    {
        py::gil_scoped_release release;
        // 1. features and raw score of every gate
        parallel_rays(
            sweeps, nt, [&] { return RayScratch(max_ng); },
            [&](const Sweep& sw, int64_t r, RayScratch& s) { features_ray(P, sw, r, s); });
        // 2. 3 x 3 mean and decision
        parallel_rays(
            sweeps, nt, [] { return 0; },
            [&](const Sweep& sw, int64_t r, int&) { smooth_ray(P, sw, r); });
        // 3. speckle filter, one sweep per task
        if (P.min_size > 1 && ns > 0) {
            const int ntp = std::max(1, std::min(nt, static_cast<int>(ns)));
            std::atomic<size_t> next{0};
            auto worker = [&]() {
                std::vector<int64_t> parent(static_cast<size_t>(max_n));
                std::vector<int64_t> size(static_cast<size_t>(max_n));
                for (;;) {
                    const size_t i = next.fetch_add(1);
                    if (i >= ns) break;
                    despeckle(P, sweeps[i], parent, size);
                }
            };
            std::vector<std::thread> pool;
            for (int t = 1; t < ntp; ++t) pool.emplace_back(worker);
            worker();
            for (auto& th : pool) th.join();
        }
    }
    py::list score_list, class_list;
    for (size_t i = 0; i < ns; ++i) {
        score_list.append(scores[i]);
        class_list.append(classes[i]);
    }
    return py::make_tuple(score_list, class_list);
}

PYBIND11_MODULE(_qc, m) {
    m.doc() = "Compiled non-meteorological echo classification kernel for radarx.";
    m.def("classify", &classify, py::arg("dbzh"), py::arg("zdr"), py::arg("rhohv"),
          py::arg("phidp"), py::arg("snr"), py::arg("links"), py::arg("h"),
          py::arg("floors"), py::arg("snr_min"), py::arg("limits"), py::arg("weights"),
          py::arg("spin_threshold"), py::arg("threshold"), py::arg("min_size"),
          py::arg("n_threads") = 0);
}
