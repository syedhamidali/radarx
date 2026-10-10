// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Rain attenuation along a ray from the differential phase: the constrained
// ZPHI solution of Testud et al. (2000) and the linear PhiDP relation of
// Bringi et al. (1990).
//
// Every ray is independent. Its gates are (reflectivity in dBZ, differential
// phase in degrees, a flag that marks the rain gates); the rays form one pool
// of work that threads take in blocks from an atomic counter. The rain path of
// a ray runs from its first to its last flagged gate. Per gate the kernel
// returns the specific attenuation AH (dB/km, one way, NaN outside the rain
// gates) and the two-way path-integrated attenuation PIA (dB) at the gate
// centre; per ray the phase span of the path. The differential attenuation is
// the same fields times a constant ratio, applied by the caller.
//
// The equations, the references and the radarx choices are in the docstring of
// radarx/retrieve/attenuation.py. Every step follows the NumPy reference in
// that module, in the same order of operations.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

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
using BArray = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kLn10Over10 = 0.23025850929940457;  // ln(10) / 10
constexpr double kKappa = 2.0 * kLn10Over10;         // 0.46 of Testud et al. (2000)
constexpr double kMaxExponent = 300.0;               // guards exp() for huge PIA
constexpr int64_t kBlock = 32;                       // rays per scheduling block

template <class F>
void parallel_blocks(int64_t total, int n_threads, F&& body) {
    if (total <= 0) return;
    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    int nt = n_threads > 0 ? n_threads : static_cast<int>(hw);
    const int64_t nblocks = (total + kBlock - 1) / kBlock;
    nt = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(nt, nblocks)));
    std::atomic<int64_t> next{0};
    auto worker = [&]() {
        for (;;) {
            const int64_t g0 = next.fetch_add(kBlock);
            if (g0 >= total) break;
            const int64_t g1 = std::min(total, g0 + kBlock);
            body(g0, g1);
        }
    };
    std::vector<std::thread> pool;
    for (int i = 1; i < nt; ++i) pool.emplace_back(worker);
    worker();
    for (auto& th : pool) th.join();
}

// Constrained ZPHI: gate-integrated specific attenuation with the exact
// discrete form of Eq. (12) of Ryzhkov et al. (2014), so that the integral
// over the path is the constraint to rounding.
void zphi_ray(const double* z, const uint8_t* ok, int64_t n, int64_t i0, int64_t i1, double dr,
              double pia_total, double b, double* ah, double* pia, std::vector<double>& s) {
    const double x = std::min(0.5 * kKappa * b * pia_total, kMaxExponent);
    const double c = std::expm1(x);
    const double scale = kKappa * b * dr;
    double total = 0.0;
    for (int64_t i = n - 1; i >= i0; --i) {
        if (!ok[i]) continue;
        s[i] = scale * std::exp(b * kLn10Over10 * z[i]);
    }
    for (int64_t i = i1; i >= i0; --i) {
        if (ok[i]) total += s[i];
    }
    const double i_total = total;
    // tail = sum of s over the rain gates after gate i
    double tail = 0.0;
    for (int64_t i = i1; i >= i0; --i) {
        if (!ok[i]) continue;
        ah[i] = std::log1p(c * s[i] / (i_total + c * tail)) / scale;
        tail += s[i];
    }
    double before = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double a = (i >= i0 && i <= i1 && ok[i]) ? ah[i] : 0.0;
        pia[i] = 2.0 * (before + 0.5 * a * dr);
        before += a * dr;
    }
}

// Linear relation: PIA = alpha * (PhiDP - PhiDP at the first gate), made
// non-negative and non-decreasing along the path.
void linear_ray(const double* phi, const uint8_t* ok, int64_t n, int64_t i0, int64_t i1, double dr,
                double alpha, double max_pia, double* ah, double* pia) {
    std::fill(pia, pia + n, 0.0);
    double run = 0.0;
    int64_t p = i0;  // last rain gate at or before i
    int64_t q = i0;  // first rain gate at or after i
    for (int64_t i = i0; i <= i1; ++i) {
        if (ok[i]) p = i;
        if (q < i) {
            q = i;
            while (!ok[q]) ++q;  // terminates: i1 is a rain gate
        }
        double phif = phi[i];
        if (!ok[i]) phif = phi[p] + (phi[q] - phi[p]) * (static_cast<double>(i - p) / static_cast<double>(q - p));
        double raw = alpha * (phif - phi[i0]);
        if (raw < 0.0) raw = 0.0;
        if (raw > run) run = raw;
        pia[i] = run;
    }
    if (std::isfinite(max_pia)) {
        for (int64_t i = i0; i <= i1; ++i) pia[i] = std::min(pia[i], max_pia);
    }
    for (int64_t i = i1 + 1; i < n; ++i) pia[i] = pia[i1];
    for (int64_t i = i0; i <= i1; ++i) {
        if (!ok[i]) continue;
        const int64_t lo = std::max(i - 1, i0);
        const int64_t hi = std::min(i + 1, i1);
        ah[i] = (hi == lo) ? 0.0
                           : (pia[hi] - pia[lo]) / (2.0 * dr * static_cast<double>(hi - lo));
    }
}

void process_ray(const double* z, const double* phi, const uint8_t* valid, int64_t n, double dr,
                 int method, double alpha, double b, double min_dphi, double max_pia, double* ah,
                 double* pia, double* dphi_out, std::vector<double>& s, std::vector<uint8_t>& ok) {
    std::fill(ah, ah + n, kNaN);
    std::fill(pia, pia + n, 0.0);
    int64_t i0 = -1, i1 = -1;
    for (int64_t i = 0; i < n; ++i) {
        ok[i] = valid[i] != 0 && std::isfinite(z[i]) && std::isfinite(phi[i]);
        if (ok[i]) {
            if (i0 < 0) i0 = i;
            i1 = i;
        }
    }
    if (i0 < 0) {
        *dphi_out = kNaN;
        return;
    }
    const double dphi = phi[i1] - phi[i0];
    *dphi_out = dphi;
    if (!(dphi >= min_dphi)) return;
    if (method == 0) {
        double pia_total = alpha * dphi;
        if (std::isfinite(max_pia)) pia_total = std::min(pia_total, max_pia);
        zphi_ray(z, ok.data(), n, i0, i1, dr, pia_total, b, ah, pia, s);
    } else {
        linear_ray(phi, ok.data(), n, i0, i1, dr, alpha, max_pia, ah, pia);
    }
}

}  // namespace

// dbz, phidp: (nray, ngate); valid: (nray, ngate) flags of the rain gates.
// Returns (ah, pia, dphi): (nray, ngate), (nray, ngate), (nray,).
py::tuple correct_py(const DArray& dbz, const DArray& phidp, const BArray& valid, double dr,
                     int method, double alpha, double b, double min_dphi, double max_pia,
                     int n_threads) {
    if (dbz.ndim() != 2 || phidp.ndim() != 2 || valid.ndim() != 2)
        throw std::invalid_argument("dbz, phidp and valid must be 2-D (rays, gates)");
    const int64_t nr = dbz.shape(0);
    const int64_t ng = dbz.shape(1);
    if (phidp.shape(0) != nr || phidp.shape(1) != ng || valid.shape(0) != nr ||
        valid.shape(1) != ng)
        throw std::invalid_argument("dbz, phidp and valid must have the same shape");
    if (!(dr > 0.0)) throw std::invalid_argument("dr must be positive");
    if (method != 0 && method != 1) throw std::invalid_argument("method must be 0 or 1");

    py::array_t<double> ah({static_cast<py::ssize_t>(nr), static_cast<py::ssize_t>(ng)});
    py::array_t<double> pia({static_cast<py::ssize_t>(nr), static_cast<py::ssize_t>(ng)});
    py::array_t<double> dphi(static_cast<py::ssize_t>(nr));
    double* ahp = ah.mutable_data();
    double* piap = pia.mutable_data();
    double* dp = dphi.mutable_data();
    const double* zp = dbz.data();
    const double* pp = phidp.data();
    const uint8_t* vp = valid.data();
    {
        py::gil_scoped_release release;
        parallel_blocks(nr, n_threads, [&](int64_t r0, int64_t r1) {
            std::vector<double> s(ng);
            std::vector<uint8_t> ok(ng);
            for (int64_t r = r0; r < r1; ++r)
                process_ray(zp + r * ng, pp + r * ng, vp + r * ng, ng, dr, method, alpha, b,
                            min_dphi, max_pia, ahp + r * ng, piap + r * ng, dp + r, s, ok);
        });
    }
    return py::make_tuple(ah, pia, dphi);
}

PYBIND11_MODULE(_attenuation, m) {
    m.doc() = "ZPHI and linear PhiDP attenuation kernel";
    m.def("correct", &correct_py, py::arg("dbz"), py::arg("phidp"), py::arg("valid"),
          py::arg("dr"), py::arg("method"), py::arg("alpha"), py::arg("b"),
          py::arg("min_dphi"), py::arg("max_pia"), py::arg("n_threads") = 0);
}
