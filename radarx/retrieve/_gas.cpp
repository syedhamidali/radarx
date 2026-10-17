// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Attenuation of a millimetre-wave radar beam by the gases of the air and by
// cloud liquid water.
//
//  * specific(): the line-by-line specific attenuation of oxygen and water vapour
//    of Rec. ITU-R P.676-13, Annex 1, Eqs. (1)-(9): every point (frequency,
//    dry-air pressure, temperature, vapour pressure) is independent, the line
//    parameters (Tables 1 and 2) are passed in from Python.
//  * cloud_liquid(): the Rayleigh specific attenuation of cloud liquid water of
//    Rec. ITU-R P.840-9, Eqs. (2)-(10), times the liquid water content.
//  * integrate(): the two-way path-integrated attenuation along every ray from a
//    field of specific attenuation, by the trapezoid rule from the radar.
//
// The NumPy references in radarx/retrieve/gas_attenuation.py follow the same
// steps and order of operations. Threads take blocks of points (or rays) from an
// atomic counter and the GIL is released.

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

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

template <class F>
void parallel_blocks(int64_t total, int64_t block, int n_threads, F&& body) {
    if (total <= 0) return;
    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    int nt = n_threads > 0 ? n_threads : static_cast<int>(hw);
    const int64_t nblocks = (total + block - 1) / block;
    nt = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(nt, nblocks)));
    std::atomic<int64_t> next{0};
    auto worker = [&]() {
        for (;;) {
            const int64_t i0 = next.fetch_add(block);
            if (i0 >= total) break;
            body(i0, std::min(total, i0 + block));
        }
    };
    std::vector<std::thread> pool;
    for (int i = 1; i < nt; ++i) pool.emplace_back(worker);
    worker();
    for (auto& th : pool) th.join();
}

// line-shape factor, Eq. (5)
inline double line_shape(double f, double fi, double df, double delta) {
    const double a = fi - f, b = fi + f;
    return (f / fi) * ((df - delta * a) / (a * a + df * df) + (df - delta * b) / (b * b + df * df));
}

// oxygen: rows of Table 1 (f0, a1..a6), water vapour: rows of Table 2 (f0, b1..b6)
// p: dry-air pressure [hPa], e: water vapour partial pressure [hPa], t: K, f: GHz
void gas_point(double f, double p, double t, double e, const double* ox, int64_t nox, const double* wv,
               int64_t nwv, double* gamma_o, double* gamma_w) {
    if (!(f > 0.0) || !(t > 0.0) || !std::isfinite(p) || !std::isfinite(e) || !std::isfinite(f) ||
        !std::isfinite(t) || p < 0.0 || e < 0.0) {
        *gamma_o = kNaN;
        *gamma_w = kNaN;
        return;
    }
    const double th = 300.0 / t;
    double n_ox = 0.0;
    for (int64_t i = 0; i < nox; ++i) {
        const double* r = ox + 7 * i;
        const double fi = r[0], a1 = r[1], a2 = r[2], a3 = r[3], a4 = r[4], a5 = r[5], a6 = r[6];
        const double s = a1 * 1e-7 * p * th * th * th * std::exp(a2 * (1.0 - th));
        double df = a3 * 1e-4 * (p * std::pow(th, 0.8 - a4) + 1.1 * e * th);
        df = std::sqrt(df * df + 2.25e-6);
        const double delta = (a5 + a6 * th) * 1e-4 * (p + e) * std::pow(th, 0.8);
        n_ox += s * line_shape(f, fi, df, delta);
    }
    // dry continuum: Debye spectrum of oxygen and pressure-induced nitrogen, Eqs. (8), (9)
    const double d = 5.6e-4 * (p + e) * std::pow(th, 0.8);
    const double fd = f / d;
    n_ox += f * p * th * th *
            (6.14e-5 / (d * (1.0 + fd * fd)) + 1.4e-12 * p * std::pow(th, 1.5) / (1.0 + 1.9e-5 * std::pow(f, 1.5)));
    double n_wv = 0.0;
    for (int64_t i = 0; i < nwv; ++i) {
        const double* r = wv + 7 * i;
        const double fi = r[0], b1 = r[1], b2 = r[2], b3 = r[3], b4 = r[4], b5 = r[5], b6 = r[6];
        const double s = b1 * 1e-1 * e * std::pow(th, 3.5) * std::exp(b2 * (1.0 - th));
        double df = b3 * 1e-4 * (p * std::pow(th, b4) + b5 * e * std::pow(th, b6));
        df = 0.535 * df + std::sqrt(0.217 * df * df + 2.1316e-12 * fi * fi / th);
        n_wv += s * line_shape(f, fi, df, 0.0);
    }
    *gamma_o = 0.1820 * f * n_ox;
    *gamma_w = 0.1820 * f * n_wv;
}

// Rec. ITU-R P.840-9, Eqs. (2)-(10): [dB/km] per g/m3 of liquid water at f [GHz] and t [K]
double liquid_coefficient(double f, double t) {
    if (!(f > 0.0) || !(t > 0.0) || !std::isfinite(f) || !std::isfinite(t)) return kNaN;
    const double th = 300.0 / t;
    const double e0 = 77.66 + 103.3 * (th - 1.0);
    const double e1 = 0.0671 * e0;
    const double e2 = 3.52;
    const double fp = 20.20 - 146.0 * (th - 1.0) + 316.0 * (th - 1.0) * (th - 1.0);
    const double fs = 39.8 * fp;
    const double xp = f / fp, xs = f / fs;
    const double epp = f * (e0 - e1) / (fp * (1.0 + xp * xp)) + f * (e1 - e2) / (fs * (1.0 + xs * xs));
    const double ep = (e0 - e1) / (1.0 + xp * xp) + (e1 - e2) / (1.0 + xs * xs) + e2;
    const double eta = (2.0 + ep) / epp;
    return 0.819 * f / (epp * (1.0 + eta * eta));
}

// f, p, t, e: flat arrays of n points (f GHz, p dry-air hPa, t K, e hPa); ox: (nox, 7), wv: (nwv, 7)
py::tuple specific_py(const DArray& f, const DArray& p, const DArray& t, const DArray& e, const DArray& ox,
                      const DArray& wv, int n_threads) {
    if (f.ndim() != 1 || p.ndim() != 1 || t.ndim() != 1 || e.ndim() != 1)
        throw std::invalid_argument("f, p, t and e must be 1-D");
    const int64_t n = f.shape(0);
    if (p.shape(0) != n || t.shape(0) != n || e.shape(0) != n)
        throw std::invalid_argument("f, p, t and e must have the same length");
    if (ox.ndim() != 2 || ox.shape(1) != 7 || wv.ndim() != 2 || wv.shape(1) != 7)
        throw std::invalid_argument("the line tables must have 7 columns");
    const int64_t nox = ox.shape(0), nwv = wv.shape(0);
    DArray go(std::vector<py::ssize_t>{static_cast<py::ssize_t>(n)});
    DArray gw(std::vector<py::ssize_t>{static_cast<py::ssize_t>(n)});
    const double *pf = f.data(), *pp = p.data(), *pt = t.data(), *pe = e.data();
    const double *pox = ox.data(), *pwv = wv.data();
    double *po = go.mutable_data(), *pw = gw.mutable_data();
    {
        py::gil_scoped_release release;
        parallel_blocks(n, 1024, n_threads, [&](int64_t i0, int64_t i1) {
            for (int64_t i = i0; i < i1; ++i) gas_point(pf[i], pp[i], pt[i], pe[i], pox, nox, pwv, nwv, po + i, pw + i);
        });
    }
    return py::make_tuple(go, gw);
}

DArray cloud_liquid_py(const DArray& f, const DArray& t, const DArray& lwc, int n_threads) {
    if (f.ndim() != 1 || t.ndim() != 1 || lwc.ndim() != 1) throw std::invalid_argument("f, t and lwc must be 1-D");
    const int64_t n = f.shape(0);
    if (t.shape(0) != n || lwc.shape(0) != n) throw std::invalid_argument("f, t and lwc must have the same length");
    DArray out(std::vector<py::ssize_t>{static_cast<py::ssize_t>(n)});
    const double *pf = f.data(), *pt = t.data(), *pl = lwc.data();
    double* po = out.mutable_data();
    {
        py::gil_scoped_release release;
        parallel_blocks(n, 4096, n_threads, [&](int64_t i0, int64_t i1) {
            for (int64_t i = i0; i < i1; ++i) po[i] = liquid_coefficient(pf[i], pt[i]) * pl[i];
        });
    }
    return out;
}

// gamma: (rays, gates) specific attenuation [dB/km] on the gates; gamma0: (rays) the value at the
// radar; range_m: (gates) increasing. Returns the two-way attenuation [dB] at every gate: twice
// the trapezoid integral from range 0. A gate without a value carries the value of the gate
// before it (0 before the first gate that has one).
DArray integrate_py(const DArray& range_m, const DArray& gamma, const DArray& gamma0, int n_threads) {
    if (gamma.ndim() != 2) throw std::invalid_argument("gamma must be 2-D (rays, gates)");
    const int64_t nr = gamma.shape(0), ng = gamma.shape(1);
    if (range_m.ndim() != 1 || range_m.shape(0) != ng)
        throw std::invalid_argument("range_m must have one value per gate");
    if (gamma0.ndim() != 1 || gamma0.shape(0) != nr) throw std::invalid_argument("gamma0 must have one value per ray");
    DArray out(std::vector<py::ssize_t>{static_cast<py::ssize_t>(nr), static_cast<py::ssize_t>(ng)});
    const double *pr = range_m.data(), *pg = gamma.data(), *p0 = gamma0.data();
    double* po = out.mutable_data();
    {
        py::gil_scoped_release release;
        parallel_blocks(nr, 16, n_threads, [&](int64_t r0, int64_t r1) {
            for (int64_t r = r0; r < r1; ++r) {
                const double* g = pg + r * ng;
                double* o = po + r * ng;
                double prev_r = 0.0;
                double prev_g = std::isfinite(p0[r]) ? p0[r] : 0.0;
                double total = 0.0;  // one-way, dB
                for (int64_t k = 0; k < ng; ++k) {
                    const double gk = std::isfinite(g[k]) ? g[k] : prev_g;
                    total += 0.5 * (prev_g + gk) * (pr[k] - prev_r) * 1e-3;
                    prev_r = pr[k];
                    prev_g = gk;
                    o[k] = 2.0 * total;
                }
            }
        });
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(_gas, m) {
    m.doc() = "Gas and cloud liquid attenuation of millimetre-wave radar beams";
    m.def("specific", &specific_py, py::arg("f"), py::arg("p"), py::arg("t"), py::arg("e"), py::arg("oxygen"),
          py::arg("water_vapor"), py::arg("n_threads") = 0);
    m.def("cloud_liquid", &cloud_liquid_py, py::arg("f"), py::arg("t"), py::arg("lwc"), py::arg("n_threads") = 0);
    m.def("integrate", &integrate_py, py::arg("range_m"), py::arg("gamma"), py::arg("gamma0"),
          py::arg("n_threads") = 0);
}
