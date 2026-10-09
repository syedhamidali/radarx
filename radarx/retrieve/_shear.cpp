// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Linear least-squares derivative (LLSD) kernel.
//
// After Smith and Elmore (2004), "The use of radial velocity derivative to
// diagnose rotation and divergence", 11th Conf. on Aviation, Range, and
// Aerospace Meteorology, P5.6 (extended abstract): local plane fit
// v = u0 + ur dr + us s with s = r dtheta in a window of nearly constant width in
// metres (they use 3 gates deep and about 2500 m wide). They solve a diagonal
// system for symmetric windows; this kernel solves the full 2 x 2 system after
// removing the weighted means, so asymmetric windows (missing data, sector
// edges) are handled. Gaussian weights, min_valid_fraction and the minimum of
// 3 valid gates are radarx choices.
//
// For every gate, the radial velocity v of the gates in a window of fixed
// physical size is fitted by weighted least squares with the plane
//
//     v = a + b * s + c * dr,   s = r_k * dtheta,   dr = r_k - r_0,
//
// where dtheta is the azimuth difference to the centre ray (actual ray
// azimuths, wrapped at 360 deg) and r_k the range of the neighbouring gate.
// b is the azimuthal shear and c the radial divergence (both s^-1).
//
// The window holds every ray with |dtheta| * r_0 <= half_az and every gate
// with |r_k - r_0| <= half_r, but at least one neighbour on each side (the
// half-widths are at least 1.5 ray/gate spacings) and at most 90 deg.
//
// Uniform weights use cumulative sums along range, so a gate costs one O(1)
// lookup per ray in its window. Gaussian weights sum the (short) range window
// directly. The normal equations are solved in closed form after removing the
// weighted means, which leaves a 2x2 system for b and c.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

namespace py = pybind11;
using DArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kPi = 3.14159265358979323846;  // M_PI is not standard (MSVC)
constexpr double kDeg = kPi / 180.0;

double median(std::vector<double> v) {
    const size_t n = v.size(), h = n / 2;
    std::nth_element(v.begin(), v.begin() + h, v.end());
    double upper = v[h];
    if (n % 2) return upper;
    double lower = *std::max_element(v.begin(), v.begin() + h);
    return 0.5 * (lower + upper);
}

// Azimuth difference b - a in radians, wrapped to [-pi, pi).
inline double wrap(double d) {
    d = std::fmod(d + kPi, 2.0 * kPi);
    if (d < 0) d += 2.0 * kPi;
    return d - kPi;
}

// Weighted sums of the plane fit.
struct Sums {
    double n_total = 0, n_valid = 0;
    double w = 0, ws = 0, wr = 0, wss = 0, wsr = 0, wrr = 0, wv = 0, wvs = 0, wvr = 0;
};

// Closed-form solution for (shear, divergence); NaN if not determined.
inline void solve(const Sums& S, double min_fraction, double& shear, double& div) {
    shear = kNaN;
    div = kNaN;
    if (S.n_valid < 3 || S.n_valid < min_fraction * S.n_total || S.w <= 0) return;
    const double ms = S.ws / S.w, mr = S.wr / S.w;
    const double css = S.wss - S.ws * ms, crr = S.wrr - S.wr * mr, csr = S.wsr - S.ws * mr;
    const double cvs = S.wvs - S.wv * ms, cvr = S.wvr - S.wv * mr;
    if (!(css > 0) || !(crr > 0)) return;
    const double det = css * crr - csr * csr;
    if (!(det > 1e-10 * css * crr)) return;
    shear = (cvs * crr - cvr * csr) / det;
    div = (cvr * css - cvs * csr) / det;
}

}  // namespace


namespace {

// Geometry and buffers of one sweep, prepared before the parallel passes.
struct Plan {
    const double* v = nullptr;  // (nray, ngate) velocity, row-major
    float* out = nullptr;       // (2, nray, ngate) shear, divergence
    int64_t nray = 0, ngate = 0;
    const double* rg = nullptr;     // gate ranges [m]
    std::vector<double> az;         // sorted ray azimuths [rad]
    std::vector<int64_t> ray;       // original ray of each sorted azimuth
    std::vector<double> half_angle; // per gate [rad]
    std::vector<int64_t> k0, k1;    // range window per gate
    double half_r = 0.0;
    std::vector<double> P;          // cumulative sums (uniform weights)
};

// Shape checks; keeps pointers to the velocity and range arrays.
void check_inputs(Plan& pl, const DArray& data, const DArray& azimuth, const DArray& range) {
    if (data.ndim() != 2) throw std::invalid_argument("data must be 2-D (ray, gate)");
    pl.nray = data.shape(0);
    pl.ngate = data.shape(1);
    if (azimuth.size() != pl.nray || range.size() != pl.ngate)
        throw std::invalid_argument("azimuth/range do not match data shape");
    if (pl.nray < 3 || pl.ngate < 3) throw std::invalid_argument("sweep too small");
    pl.v = data.data();
    pl.rg = range.data();
    for (int64_t g = 1; g < pl.ngate; ++g)
        if (!(pl.rg[g] > pl.rg[g - 1])) throw std::invalid_argument("range must increase");
}

// Rays sorted by azimuth (radians); returns the median azimuth step.
double sort_azimuths(Plan& pl, const double* azimuth) {
    const int64_t nray = pl.nray;
    std::vector<std::pair<double, int64_t>> rays(nray);
    for (int64_t q = 0; q < nray; ++q) {
        const double a = std::fmod(azimuth[q], 360.0);
        rays[q] = {(a < 0 ? a + 360.0 : a) * kDeg, q};
    }
    std::stable_sort(rays.begin(), rays.end(),
                     [](const auto& p, const auto& q) { return p.first < q.first; });
    pl.az.resize(nray);
    pl.ray.resize(nray);
    for (int64_t q = 0; q < nray; ++q) {
        pl.az[q] = rays[q].first;
        pl.ray[q] = rays[q].second;
    }
    std::vector<double> steps(nray - 1);
    for (int64_t q = 0; q + 1 < nray; ++q) steps[q] = pl.az[q + 1] - pl.az[q];
    return median(steps);
}

// Per gate: half-angle of the window and its range extent [k0, k1].
void set_windows(Plan& pl, double window_range, double window_azimuth, double az_step) {
    const int64_t ngate = pl.ngate;
    const double* rg = pl.rg;
    std::vector<double> dsteps(ngate - 1);
    for (int64_t g = 0; g + 1 < ngate; ++g) dsteps[g] = rg[g + 1] - rg[g];
    pl.half_r = std::max(0.5 * window_range, 1.5 * median(dsteps));
    pl.half_angle.resize(ngate);
    pl.k0.resize(ngate);
    pl.k1.resize(ngate);
    for (int64_t g = 0, lo = 0, hi = 0; g < ngate; ++g) {
        const double r0 = rg[g];
        const double ha = r0 > 0 ? 0.5 * window_azimuth / r0 : kPi;
        pl.half_angle[g] = std::max(std::min(ha, 0.5 * kPi), 1.5 * az_step);
        while (rg[lo] < r0 - pl.half_r) ++lo;
        hi = std::max(hi, g);
        while (hi + 1 < ngate && rg[hi + 1] <= r0 + pl.half_r) ++hi;
        pl.k0[g] = lo;
        pl.k1[g] = hi;
    }
}

void prepare(Plan& pl, const DArray& data, const DArray& azimuth, const DArray& range,
             double window_range, double window_azimuth) {
    check_inputs(pl, data, azimuth, range);
    const double az_step = sort_azimuths(pl, azimuth.data());
    set_windows(pl, window_range, window_azimuth, az_step);
}

// Cumulative sums along range of one sorted ray: count, r, r^2, v, v*r over
// valid gates, interleaved; entry g holds the sum over gates < g.
void prefix_ray(Plan& pl, int64_t q) {
    const int64_t ngate = pl.ngate;
    const double* row = pl.v + pl.ray[q] * ngate;
    double* p = pl.P.data() + 5 * q * (ngate + 1);
    double c[5] = {0, 0, 0, 0, 0};
    for (int64_t g = 0; g < ngate; ++g) {
        for (int m = 0; m < 5; ++m) p[5 * g + m] = c[m];
        const double x = row[g];
        if (std::isnan(x)) continue;
        const double r = pl.rg[g];
        c[0] += 1.0;
        c[1] += r;
        c[2] += r * r;
        c[3] += x;
        c[4] += x * r;
    }
    for (int m = 0; m < 5; ++m) p[5 * ngate + m] = c[m];
}

// The window of one centre gate.
struct Window {
    double r0, ha, ngates, inv_ha2, inv_hr2;
    int64_t a, b;  // gate range [a, b]
};

// Sums of sorted ray q (azimuth offset dt) from its cumulative sums (uniform).
inline void add_uniform(const Plan& pl, const Window& w, int64_t q, double dt, Sums& S) {
    const double* p = pl.P.data() + 5 * q * (pl.ngate + 1);
    const double* pa = p + 5 * w.a;
    const double* pb = p + 5 * (w.b + 1);
    const double n = pb[0] - pa[0], sr = pb[1] - pa[1], srr = pb[2] - pa[2];
    const double sv = pb[3] - pa[3], svr = pb[4] - pa[4];
    const double r0 = w.r0;
    S.n_total += w.ngates;
    S.n_valid += n;
    S.w += n;
    S.ws += dt * sr;
    S.wr += sr - r0 * n;
    S.wss += dt * dt * srr;
    S.wsr += dt * (srr - r0 * sr);
    S.wrr += srr - 2.0 * r0 * sr + r0 * r0 * n;
    S.wv += sv;
    S.wvs += dt * svr;
    S.wvr += svr - r0 * sv;
}

// Sums of sorted ray q (azimuth offset dt) gate by gate (Gaussian weights).
inline void add_gaussian(const Plan& pl, const Window& w, int64_t q, double dt, Sums& S) {
    const double* row = pl.v + pl.ray[q] * pl.ngate;
    const double wa = std::exp(-2.0 * dt * dt * w.inv_ha2);
    S.n_total += w.ngates;
    for (int64_t k = w.a; k <= w.b; ++k) {
        const double x = row[k];
        if (std::isnan(x)) continue;
        const double dr = pl.rg[k] - w.r0, s = pl.rg[k] * dt;
        const double wt = wa * std::exp(-2.0 * dr * dr * w.inv_hr2);
        S.n_valid += 1.0;
        S.w += wt;
        S.ws += wt * s;
        S.wr += wt * dr;
        S.wss += wt * s * s;
        S.wsr += wt * s * dr;
        S.wrr += wt * dr * dr;
        S.wv += wt * x;
        S.wvs += wt * x * s;
        S.wvr += wt * x * dr;
    }
}

template <bool Gaussian>
inline void add_ray(const Plan& pl, const Window& w, int64_t q, double dt, Sums& S) {
    if constexpr (Gaussian)
        add_gaussian(pl, w, q, dt, S);
    else
        add_uniform(pl, w, q, dt, S);
}

// Sums over the window of sorted ray i: the centre ray, then outwards in
// azimuth on both sides (the window is a contiguous arc).
template <bool Gaussian>
inline Sums window_sums(const Plan& pl, const Window& w, int64_t i) {
    const int64_t nray = pl.nray;
    Sums S;
    add_ray<Gaussian>(pl, w, i, 0.0, S);
    int64_t used = 1;
    for (int64_t d = 1; used < nray; ++d, ++used) {
        const int64_t q = i + d < nray ? i + d : i + d - nray;
        const double dt = wrap(pl.az[q] - pl.az[i]);
        if (dt < 0 || dt > w.ha) break;
        add_ray<Gaussian>(pl, w, q, dt, S);
    }
    for (int64_t d = 1; used < nray; ++d, ++used) {
        const int64_t q = i - d >= 0 ? i - d : i - d + nray;
        const double dt = wrap(pl.az[q] - pl.az[i]);
        if (dt > 0 || dt < -w.ha) break;
        add_ray<Gaussian>(pl, w, q, dt, S);
    }
    return S;
}

// Shear and divergence for every gate of sorted ray i.
template <bool Gaussian>
void fit_ray(const Plan& pl, int64_t i, double min_fraction) {
    const int64_t ngate = pl.ngate;
    const int64_t ri = pl.ray[i];
    const double* centre = pl.v + ri * ngate;
    float* out_s = pl.out + ri * ngate;
    float* out_d = pl.out + pl.nray * ngate + ri * ngate;
    const double inv_hr2 = 1.0 / (pl.half_r * pl.half_r);
    for (int64_t g = 0; g < ngate; ++g) {
        double shear = kNaN, div = kNaN;
        if (!std::isnan(centre[g])) {
            const double ha = pl.half_angle[g];
            const Window w{pl.rg[g], ha, static_cast<double>(pl.k1[g] - pl.k0[g] + 1),
                           1.0 / (ha * ha), inv_hr2, pl.k0[g], pl.k1[g]};
            solve(window_sums<Gaussian>(pl, w, i), min_fraction, shear, div);
        }
        out_s[g] = static_cast<float>(shear);
        out_d[g] = static_cast<float>(div);
    }
}

// Runs body(plan, ray) over every ray of the sweeps [k_begin, k_end) on nt
// threads, handing out chunks of rays from an atomic counter.
template <class Body>
void run_rays(std::vector<Plan>& plans, const std::vector<int64_t>& start, size_t k_begin,
              size_t k_end, int nt_req, Body&& body) {
    const int64_t first = start[k_begin], total = start[k_end] - first;
    const int nt = static_cast<int>(std::min<int64_t>(nt_req, total));
    std::atomic<int64_t> next{0};
    constexpr int64_t chunk = 4;
    auto worker = [&]() {
        size_t k = k_begin;
        for (int64_t u0; (u0 = next.fetch_add(chunk)) < total;) {
            const int64_t u1 = std::min(total, u0 + chunk);
            for (int64_t u = first + u0; u < first + u1; ++u) {
                while (u >= start[k + 1]) ++k;
                body(plans[k], u - start[k]);
            }
        }
    };
    std::vector<std::thread> pool;
    for (int t = 1; t < nt; ++t) pool.emplace_back(worker);
    worker();
    for (auto& th : pool) th.join();
}

// End of the group of sweeps starting at k_begin whose cumulative sums fit
// in about 512 MB (a whole NEXRAD volume needs several times that).
size_t group_end(const std::vector<Plan>& plans, size_t k_begin) {
    constexpr double budget = 512.0 * 1024 * 1024;
    size_t k_end = k_begin;
    double bytes = 0.0;
    while (k_end < plans.size()) {
        const double b = 40.0 * plans[k_end].nray * (plans[k_end].ngate + 1);
        if (k_end > k_begin && bytes + b > budget) break;
        bytes += b;
        ++k_end;
    }
    return k_end;
}

}  // namespace

// Batched over sweeps: data[k] is (nray_k, ngate_k) radial velocity with NaN
// for missing gates. Returns one (2, nray_k, ngate_k) float32 array per sweep
// holding the azimuthal shear and radial divergence. The rays of all sweeps
// (in groups bounded by memory) form one pool of work shared by the threads.
std::vector<py::array_t<float>> llsd(const std::vector<DArray>& data,
                                     const std::vector<DArray>& azimuth,
                                     const std::vector<DArray>& range, double window_range,
                                     double window_azimuth, bool gaussian,
                                     double min_valid_fraction, int n_threads) {
    const size_t nk = data.size();
    if (azimuth.size() != nk || range.size() != nk)
        throw std::invalid_argument("need the same number of data/azimuth/range arrays");
    std::vector<Plan> plans(nk);
    std::vector<py::array_t<float>> results;
    std::vector<int64_t> start(nk + 1, 0);  // first global ray of each sweep
    for (size_t k = 0; k < nk; ++k) {
        prepare(plans[k], data[k], azimuth[k], range[k], window_range, window_azimuth);
        results.emplace_back(std::vector<int64_t>{2, plans[k].nray, plans[k].ngate});
        plans[k].out = results.back().mutable_data();
        start[k + 1] = start[k] + plans[k].nray;
    }

    {
        py::gil_scoped_release release;
        const int nt = n_threads > 0
                           ? n_threads
                           : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
        for (size_t k_begin = 0; k_begin < nk;) {
            const size_t k_end = group_end(plans, k_begin);
            if (!gaussian) {
                for (size_t k = k_begin; k < k_end; ++k)
                    plans[k].P.resize(static_cast<size_t>(5 * plans[k].nray * (plans[k].ngate + 1)));
                run_rays(plans, start, k_begin, k_end, nt,
                         [](Plan& pl, int64_t q) { prefix_ray(pl, q); });
            }
            if (gaussian)
                run_rays(plans, start, k_begin, k_end, nt, [&](Plan& pl, int64_t i) {
                    fit_ray<true>(pl, i, min_valid_fraction);
                });
            else
                run_rays(plans, start, k_begin, k_end, nt, [&](Plan& pl, int64_t i) {
                    fit_ray<false>(pl, i, min_valid_fraction);
                });
            for (size_t k = k_begin; k < k_end; ++k) std::vector<double>().swap(plans[k].P);
            k_begin = k_end;
        }
    }
    return results;
}

PYBIND11_MODULE(_shear, m) {
    m.doc() = "Compiled linear least-squares derivative (LLSD) kernel for radarx.";
    m.def("llsd", &llsd, py::arg("data"), py::arg("azimuth"), py::arg("range"),
          py::arg("window_range"), py::arg("window_azimuth"), py::arg("gaussian") = false,
          py::arg("min_valid_fraction") = 0.5, py::arg("n_threads") = 0);
}
