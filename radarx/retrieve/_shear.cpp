// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Linear least-squares derivative (LLSD) kernel.
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

void prepare(Plan& pl, const DArray& data, const DArray& azimuth, const DArray& range,
             double window_range, double window_azimuth) {
    if (data.ndim() != 2) throw std::invalid_argument("data must be 2-D (ray, gate)");
    const int64_t nray = data.shape(0), ngate = data.shape(1);
    if (azimuth.size() != nray || range.size() != ngate)
        throw std::invalid_argument("azimuth/range do not match data shape");
    if (nray < 3 || ngate < 3) throw std::invalid_argument("sweep too small");
    const double* rg = range.data();
    for (int64_t g = 1; g < ngate; ++g)
        if (!(rg[g] > rg[g - 1])) throw std::invalid_argument("range must increase");
    pl.v = data.data();
    pl.rg = rg;
    pl.nray = nray;
    pl.ngate = ngate;

    // Rays sorted by azimuth (radians).
    std::vector<std::pair<double, int64_t>> rays(nray);
    for (int64_t q = 0; q < nray; ++q) {
        double a = std::fmod(azimuth.data()[q], 360.0);
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
    const double az_step = median(steps);
    std::vector<double> dsteps(ngate - 1);
    for (int64_t g = 0; g + 1 < ngate; ++g) dsteps[g] = rg[g + 1] - rg[g];
    const double r_step = median(dsteps);

    // Per gate: half-angle of the window and its range extent [k0, k1].
    pl.half_r = std::max(0.5 * window_range, 1.5 * r_step);
    pl.half_angle.resize(ngate);
    pl.k0.resize(ngate);
    pl.k1.resize(ngate);
    for (int64_t g = 0, lo = 0, hi = 0; g < ngate; ++g) {
        const double r0 = rg[g];
        const double ha = r0 > 0 ? 0.5 * window_azimuth / r0 : kPi;
        pl.half_angle[g] = std::max(std::min(ha, 0.5 * kPi), 1.5 * az_step);
        while (rg[lo] < r0 - pl.half_r) ++lo;
        if (hi < g) hi = g;
        while (hi + 1 < ngate && rg[hi + 1] <= r0 + pl.half_r) ++hi;
        pl.k0[g] = lo;
        pl.k1[g] = hi;
    }
}

// Cumulative sums along range of one sorted ray: count, r, r^2, v, v*r over
// valid gates, interleaved; entry g holds the sum over gates < g.
void prefix_ray(Plan& pl, int64_t q) {
    const int64_t ngate = pl.ngate;
    const double* row = pl.v + pl.ray[q] * ngate;
    double* p = pl.P.data() + 5 * q * (ngate + 1);
    double c0 = 0, c1 = 0, c2 = 0, c3 = 0, c4 = 0;
    for (int64_t g = 0; g < ngate; ++g) {
        p[5 * g] = c0;
        p[5 * g + 1] = c1;
        p[5 * g + 2] = c2;
        p[5 * g + 3] = c3;
        p[5 * g + 4] = c4;
        const double x = row[g];
        if (!std::isnan(x)) {
            const double r = pl.rg[g];
            c0 += 1.0;
            c1 += r;
            c2 += r * r;
            c3 += x;
            c4 += x * r;
        }
    }
    double* e = p + 5 * ngate;
    e[0] = c0;
    e[1] = c1;
    e[2] = c2;
    e[3] = c3;
    e[4] = c4;
}

// Shear and divergence for every gate of sorted ray i.
void fit_ray(const Plan& pl, int64_t i, bool gaussian, double min_fraction) {
    const int64_t nray = pl.nray, ngate = pl.ngate, np1 = ngate + 1;
    const double* rg = pl.rg;
    const int64_t ri = pl.ray[i];
    const double* centre = pl.v + ri * ngate;
    float* out_s = pl.out + ri * ngate;
    float* out_d = pl.out + nray * ngate + ri * ngate;
    const double inv_hr2 = 1.0 / (pl.half_r * pl.half_r);
    for (int64_t g = 0; g < ngate; ++g) {
        double shear = kNaN, div = kNaN;
        if (!std::isnan(centre[g])) {
            const double r0 = rg[g], ha = pl.half_angle[g];
            const int64_t a = pl.k0[g], b = pl.k1[g];
            const double ngates = static_cast<double>(b - a + 1);
            const double inv_ha2 = 1.0 / (ha * ha);
            Sums S;
            auto add_ray = [&](int64_t q, double dt) {
                S.n_total += ngates;
                if (!gaussian) {
                    const double* pa = pl.P.data() + 5 * (q * np1 + a);
                    const double* pb = pl.P.data() + 5 * (q * np1 + b + 1);
                    const double n = pb[0] - pa[0], sr = pb[1] - pa[1], srr = pb[2] - pa[2],
                                 sv = pb[3] - pa[3], svr = pb[4] - pa[4];
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
                } else {
                    const double* row = pl.v + pl.ray[q] * ngate;
                    const double wa = std::exp(-2.0 * dt * dt * inv_ha2);
                    for (int64_t k = a; k <= b; ++k) {
                        const double x = row[k];
                        if (std::isnan(x)) continue;
                        const double dr = rg[k] - r0, s = rg[k] * dt;
                        const double w = wa * std::exp(-2.0 * dr * dr * inv_hr2);
                        S.n_valid += 1.0;
                        S.w += w;
                        S.ws += w * s;
                        S.wr += w * dr;
                        S.wss += w * s * s;
                        S.wsr += w * s * dr;
                        S.wrr += w * dr * dr;
                        S.wv += w * x;
                        S.wvs += w * x * s;
                        S.wvr += w * x * dr;
                    }
                }
            };
            add_ray(i, 0.0);
            // Walk outwards in azimuth; the window is a contiguous arc.
            int64_t used = 1;
            for (int64_t d = 1; used < nray; ++d, ++used) {
                const int64_t q = i + d < nray ? i + d : i + d - nray;
                const double dt = wrap(pl.az[q] - pl.az[i]);
                if (dt < 0 || dt > ha) break;
                add_ray(q, dt);
            }
            for (int64_t d = 1; used < nray; ++d, ++used) {
                const int64_t q = i - d >= 0 ? i - d : i - d + nray;
                const double dt = wrap(pl.az[q] - pl.az[i]);
                if (dt > 0 || dt < -ha) break;
                add_ray(q, dt);
            }
            solve(S, min_fraction, shear, div);
        }
        out_s[g] = static_cast<float>(shear);
        out_d[g] = static_cast<float>(div);
    }
}

}  // namespace

// Batched over sweeps: data[k] is (nray_k, ngate_k) radial velocity with NaN
// for missing gates. Returns one (2, nray_k, ngate_k) float32 array per sweep
// holding the azimuthal shear and radial divergence. The rays of all sweeps
// form one pool of work shared by the threads.
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
        unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        const int nt_req = n_threads > 0 ? n_threads : static_cast<int>(hw);

        // Dynamic scheduling over all rays of the sweeps [k_begin, k_end).
        auto run = [&](size_t k_begin, size_t k_end, auto&& body) {
            const int64_t first = start[k_begin], total = start[k_end] - first;
            const int nt = static_cast<int>(std::min<int64_t>(nt_req, total));
            std::atomic<int64_t> next{0};
            const int64_t chunk = 4;
            auto worker = [&]() {
                size_t k = k_begin;
                for (;;) {
                    const int64_t u0 = next.fetch_add(chunk);
                    if (u0 >= total) break;
                    const int64_t u1 = std::min(total, u0 + chunk);
                    for (int64_t u = first + u0; u < first + u1; ++u) {
                        while (u >= start[k + 1]) ++k;
                        while (u < start[k]) --k;
                        body(plans[k], u - start[k]);
                    }
                }
            };
            std::vector<std::thread> pool;
            for (int t = 1; t < nt; ++t) pool.emplace_back(worker);
            worker();
            for (auto& th : pool) th.join();
        };

        // Sweeps are processed in groups whose cumulative sums fit in about
        // 512 MB (a whole NEXRAD volume needs several times that); each group
        // is one pool of rays shared by all threads.
        const double budget = 512.0 * 1024 * 1024;
        for (size_t k_begin = 0; k_begin < nk;) {
            size_t k_end = k_begin;
            double bytes = 0.0;
            while (k_end < nk) {
                const double b = 40.0 * plans[k_end].nray * (plans[k_end].ngate + 1);
                if (k_end > k_begin && bytes + b > budget) break;
                bytes += b;
                ++k_end;
            }
            if (!gaussian) {
                for (size_t k = k_begin; k < k_end; ++k)
                    plans[k].P.resize(static_cast<size_t>(5 * plans[k].nray * (plans[k].ngate + 1)));
                run(k_begin, k_end, [&](Plan& pl, int64_t q) { prefix_ray(pl, q); });
            }
            run(k_begin, k_end,
                [&](Plan& pl, int64_t i) { fit_ray(pl, i, gaussian, min_valid_fraction); });
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
