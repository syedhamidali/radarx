// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Polarimetric radar forward operator kernels.
//
// cells(): the radar variables of every model cell. Cells are independent, so
// threads take blocks of cells from an atomic counter. Per cell
//
// 1. rain: the gamma drop size distribution N(D) = N0 D^mu exp(-Lambda D)
//    follows from the rain mixing ratio and the air density (Lambda from the
//    third moment) with a fixed N0 or, where the number concentration is
//    given, from the number concentration and the shape mu. The moments over
//    the diameters of the T-matrix scattering table (trapezoid weights,
//    linear in the temperature of the table) give Z_H, Z_V, the copolar
//    covariance, K_DP and the Z_H-weighted fall speed.
// 2. ice classes (snow, hail, graupel): exponential distributions with a fixed
//    N0 and a bulk density, whose Rayleigh scattering by oblate spheroids with
//    a Gaussian canting angle is a constant times N0 Lambda^-7 (reflectivities
//    and covariance) or N0 Lambda^-4 (K_DP). The constants are computed by the
//    caller (radarx/retrieve/forward_operator.py documents the equations).
//
// sample(): fields of the model columns at the beam gates. Each gate has
// several neighbouring columns with weights; in each column the field is
// interpolated linearly in height between the model levels, and the columns
// are averaged with their weights.
//
// Every step follows the NumPy reference in radarx/retrieve/forward_operator.py
// in the same order of operations.

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
using FArray = py::array_t<float, py::array::forcecast>;
using IArray = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
// pi / 6 * 1000 kg m-3 * 1e-9 m3 mm-3: kg m-3 of a unit third moment (mm3 m-3)
constexpr double kMass = 3.14159265358979323846 / 6.0 * 1.0e-6;
constexpr double kPi = 3.14159265358979323846;
constexpr int64_t kBlock = 1024;  // cells (or gates) per scheduling block
constexpr int kOut = 6;           // Z_H, Z_V, Re, Im of the covariance, K_DP, Z_H v

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
            body(g0, std::min(total, g0 + kBlock));
        }
    };
    std::vector<std::thread> pool;
    for (int i = 1; i < nt; ++i) pool.emplace_back(worker);
    worker();
    for (auto& th : pool) th.join();
}

struct RainTable {
    const double* temps;  // water temperatures [degC]
    const double* diam;   // equal-volume diameters [mm]
    const double* wts;    // trapezoid weights [mm]
    const double* logd;   // log of the diameters
    const double* vfall;  // fall speeds [m s-1]
    const double* data;   // (nt, nd, 5): sigma_h, sigma_v, copol re, im, kdp
    int64_t nt = 0, nd = 0;
};

struct Ice {
    const double* q = nullptr;
    double n0, rho, zh, zv, cov, kdp, vt;
};

}  // namespace

// Radar variables of n cells. q_rain, n_rain, temp_c may be None; the arrays
// of the ice classes are in ice_q, their constants in ice_par (n_class, 7):
// N0 [m-4], bulk density [kg m-3], and the constants of Z_H, Z_V, the
// covariance and K_DP and the fall speed [m s-1]. Returns (6, n) float32:
// Z_H, Z_V, Re and Im of the covariance (mm6 m-3), K_DP (deg km-1) and
// Z_H times the fall speed.
py::array_t<float> cells(const py::object& q_rain, const py::object& n_rain,
                         const DArray& rho, const py::object& temp_c,
                         const DArray& temps, const DArray& diam, const DArray& wts,
                         const DArray& vfall, const DArray& table, double kfac,
                         double n0_rain, double mu_rain, double q_min,
                         double temp_default, bool nr_per_kg,
                         const std::vector<DArray>& ice_q, const DArray& ice_par,
                         int n_threads) {
    const int64_t n = rho.ndim() == 1 ? rho.shape(0) : -1;
    if (n < 0) throw std::invalid_argument("rho must be a 1-D array");
    DArray qr_keep, nr_keep, tc_keep;
    auto optional = [&](const py::object& obj, DArray& keep, const char* name) {
        if (obj.is_none()) return static_cast<const double*>(nullptr);
        keep = obj.cast<DArray>();
        if (keep.ndim() != 1 || keep.shape(0) != n)
            throw std::invalid_argument(std::string(name) + " must have the shape of rho");
        return static_cast<const double*>(keep.data());
    };
    const double* qr = optional(q_rain, qr_keep, "q_rain");
    const double* nr = optional(n_rain, nr_keep, "n_rain");
    const double* tc = optional(temp_c, tc_keep, "temp_c");

    RainTable t;
    t.nt = temps.shape(0);
    t.nd = diam.shape(0);
    if (t.nt < 1 || t.nd < 2 || wts.shape(0) != t.nd || vfall.shape(0) != t.nd ||
        table.ndim() != 3 || table.shape(0) != t.nt || table.shape(1) != t.nd ||
        table.shape(2) != 5)
        throw std::invalid_argument("inconsistent rain scattering table");
    std::vector<double> logd(t.nd);
    for (int64_t j = 0; j < t.nd; ++j) logd[j] = std::log(diam.data()[j]);
    t.temps = temps.data();
    t.diam = diam.data();
    t.wts = wts.data();
    t.vfall = vfall.data();
    t.data = table.data();
    t.logd = logd.data();

    const size_t nc = ice_q.size();
    if (ice_par.ndim() != 2 || static_cast<size_t>(ice_par.shape(0)) != nc ||
        ice_par.shape(1) != 7)
        throw std::invalid_argument("ice_par must have shape (n_class, 7)");
    std::vector<Ice> ice(nc);
    for (size_t c = 0; c < nc; ++c) {
        if (ice_q[c].ndim() != 1 || ice_q[c].shape(0) != n)
            throw std::invalid_argument("ice mixing ratios must have the shape of rho");
        const double* p = ice_par.data() + 7 * c;
        ice[c] = {ice_q[c].data(), p[0], p[1], p[2], p[3], p[4], p[5], p[6]};
    }
    const double mu = mu_rain;
    const double g_mu1 = std::tgamma(mu + 1.0), g_mu4 = std::tgamma(mu + 4.0);
    const double n0m = n0_rain * 1.0e-3;  // m-4 to m-3 mm-1
    const double* rho_a = rho.data();

    py::array_t<float> out(std::vector<py::ssize_t>{kOut, n});
    float* o = out.mutable_data();
    {
        py::gil_scoped_release release;
        parallel_blocks(n, n_threads, [&](int64_t c0, int64_t c1) {
            for (int64_t i = c0; i < c1; ++i) {
                double zh = 0.0, zv = 0.0, cre = 0.0, cim = 0.0, kdp = 0.0, zvt = 0.0;
                const double rh = rho_a[i];
                const bool air = std::isfinite(rh) && rh > 0.0;
                if (qr && air && qr[i] > q_min) {
                    const double m3 = qr[i] * rh / kMass;  // mm3 m-3
                    double lam, n0;
                    const double nt_cell = nr ? (nr_per_kg ? nr[i] * rh : nr[i]) : -1.0;
                    if (nt_cell > 0.0 && std::isfinite(nt_cell)) {
                        lam = std::pow(nt_cell * g_mu4 / (g_mu1 * m3), 1.0 / 3.0);
                        n0 = nt_cell * std::pow(lam, mu + 1.0) / g_mu1;
                    } else {
                        lam = std::pow(n0m * g_mu4 / m3, 1.0 / (mu + 4.0));
                        n0 = n0m;
                    }
                    // table temperature position (clipped to the table)
                    double temp = (tc && std::isfinite(tc[i])) ? tc[i] : temp_default;
                    temp = std::min(std::max(temp, t.temps[0]), t.temps[t.nt - 1]);
                    int64_t k = 0;
                    double wt = 0.0;
                    if (t.nt > 1) {
                        k = static_cast<int64_t>(
                                std::upper_bound(t.temps, t.temps + t.nt, temp) - t.temps) -
                            1;
                        k = std::min(std::max<int64_t>(k, 0), t.nt - 2);
                        wt = (temp - t.temps[k]) / (t.temps[k + 1] - t.temps[k]);
                    }
                    const double* d0 = t.data + k * t.nd * 5;
                    const double* d1 = t.data + std::min<int64_t>(k + 1, t.nt - 1) * t.nd * 5;
                    double sh = 0, sv = 0, sre = 0, sim = 0, sk = 0, shv = 0;
                    for (int64_t j = 0; j < t.nd; ++j) {
                        const double nd = n0 * std::exp(mu * t.logd[j] - lam * t.diam[j]) * t.wts[j];
                        auto at = [&](int c) {
                            return d0[j * 5 + c] + wt * (d1[j * 5 + c] - d0[j * 5 + c]);
                        };
                        const double a = nd * at(0);
                        sh += a;
                        shv += a * t.vfall[j];
                        sv += nd * at(1);
                        sre += nd * at(2);
                        sim += nd * at(3);
                        sk += nd * at(4);
                    }
                    zh += kfac * sh;
                    zv += kfac * sv;
                    // the table holds the covariance in the backscatter-alignment
                    // sign convention (negative for the correlated small drops)
                    cre -= kfac * sre;
                    cim -= kfac * sim;
                    kdp += sk;
                    zvt += kfac * shv;
                }
                for (size_t c = 0; c < nc; ++c) {
                    const Ice& e = ice[c];
                    const double q = e.q[i];
                    if (!(air && q > q_min)) continue;
                    // exponential DSD: rho_a q = pi rho_i N0 / Lambda^4 (SI)
                    const double lam = std::pow(kPi * e.rho * e.n0 / (rh * q), 0.25) * 1.0e-3;
                    const double n0 = e.n0 * 1.0e-3;
                    const double z7 = 720.0 * n0 * std::pow(lam, -7.0);
                    const double zhc = e.zh * z7;
                    zh += zhc;
                    zv += e.zv * z7;
                    cre += e.cov * z7;
                    kdp += e.kdp * 6.0 * n0 * std::pow(lam, -4.0);
                    zvt += e.vt * zhc;
                }
                o[0 * n + i] = static_cast<float>(zh);
                o[1 * n + i] = static_cast<float>(zv);
                o[2 * n + i] = static_cast<float>(cre);
                o[3 * n + i] = static_cast<float>(cim);
                o[4 * n + i] = static_cast<float>(kdp);
                o[5 * n + i] = static_cast<float>(zvt);
            }
        });
    }
    return out;
}

// Fields of the model columns at the gates. fields: nf arrays (any strides) of
// shape (ncol, nz); height (ncol, nz) in increasing order along the level
// axis; idx and weight (ng, k): the columns of each gate (-1 for none) and
// their weights; gate_h (ng) the heights. Returns (nf, ng) float32; NaN where
// no column holds the gate height or the field is NaN.
py::array_t<float> sample(const std::vector<FArray>& fields, const FArray& height,
                          const IArray& idx, const DArray& weight, const DArray& gate_h,
                          int n_threads) {
    if (height.ndim() != 2) throw std::invalid_argument("height must be 2-D");
    const int64_t ncol = height.shape(0), nz = height.shape(1);
    if (idx.ndim() != 2 || weight.ndim() != 2 || idx.shape(0) != weight.shape(0) ||
        idx.shape(1) != weight.shape(1) || gate_h.ndim() != 1 ||
        gate_h.shape(0) != idx.shape(0))
        throw std::invalid_argument("idx, weight and gate_h are inconsistent");
    const int64_t ng = idx.shape(0), k = idx.shape(1);
    const int64_t nf = static_cast<int64_t>(fields.size());
    struct View {
        const float* p;
        int64_t sc, sz;
    };
    std::vector<View> fv(nf);
    for (int64_t f = 0; f < nf; ++f) {
        if (fields[f].ndim() != 2 || fields[f].shape(0) != ncol || fields[f].shape(1) != nz)
            throw std::invalid_argument("every field must have the shape of height");
        fv[f] = {fields[f].data(), fields[f].strides(0) / static_cast<int64_t>(sizeof(float)),
                 fields[f].strides(1) / static_cast<int64_t>(sizeof(float))};
    }
    const View hv = {height.data(), height.strides(0) / static_cast<int64_t>(sizeof(float)),
                     height.strides(1) / static_cast<int64_t>(sizeof(float))};
    const int64_t* ix = idx.data();
    const double* wg = weight.data();
    const double* gh = gate_h.data();

    py::array_t<float> out(std::vector<py::ssize_t>{nf, ng});
    float* o = out.mutable_data();
    {
        py::gil_scoped_release release;
        parallel_blocks(ng, n_threads, [&](int64_t g0, int64_t g1) {
            std::vector<double> sum(nf), wsum(nf);
            for (int64_t g = g0; g < g1; ++g) {
                std::fill(sum.begin(), sum.end(), 0.0);
                std::fill(wsum.begin(), wsum.end(), 0.0);
                const double h = gh[g];
                for (int64_t j = 0; j < k; ++j) {
                    const int64_t col = ix[g * k + j];
                    const double w = wg[g * k + j];
                    if (col < 0 || col >= ncol || !(w > 0.0) || !std::isfinite(h)) continue;
                    auto level = [&](int64_t l) {
                        return static_cast<double>(hv.p[col * hv.sc + l * hv.sz]);
                    };
                    if (!(h >= level(0) && h <= level(nz - 1))) continue;
                    // last level at or below h (binary search)
                    int64_t lo = 0, hi = nz - 1;
                    while (hi - lo > 1) {
                        const int64_t mid = (lo + hi) / 2;
                        if (level(mid) <= h) lo = mid; else hi = mid;
                    }
                    const double hl = level(lo), hu = level(lo + 1 < nz ? lo + 1 : lo);
                    const double t = hu > hl ? (h - hl) / (hu - hl) : 0.0;
                    const int64_t up = lo + 1 < nz ? lo + 1 : lo;
                    for (int64_t f = 0; f < nf; ++f) {
                        const double a = fv[f].p[col * fv[f].sc + lo * fv[f].sz];
                        const double b = fv[f].p[col * fv[f].sc + up * fv[f].sz];
                        const double v = a + t * (b - a);
                        if (std::isfinite(v)) {
                            sum[f] += w * v;
                            wsum[f] += w;
                        }
                    }
                }
                for (int64_t f = 0; f < nf; ++f)
                    o[f * ng + g] = wsum[f] > 0.0 ? static_cast<float>(sum[f] / wsum[f])
                                                  : static_cast<float>(kNaN);
            }
        });
    }
    return out;
}

PYBIND11_MODULE(_forward_operator, m) {
    m.doc() = "Compiled polarimetric radar forward operator kernels for radarx.";
    m.def("cells", &cells, py::arg("q_rain"), py::arg("n_rain"), py::arg("rho"),
          py::arg("temp_c"), py::arg("temps"), py::arg("diam"), py::arg("weights"),
          py::arg("vfall"), py::arg("table"), py::arg("kfac"), py::arg("n0_rain"),
          py::arg("mu_rain"), py::arg("q_min"), py::arg("temp_default"),
          py::arg("nr_per_kg"), py::arg("ice_q"), py::arg("ice_par"),
          py::arg("n_threads") = 0);
    m.def("sample", &sample, py::arg("fields"), py::arg("height"), py::arg("idx"),
          py::arg("weight"), py::arg("gate_height"), py::arg("n_threads") = 0);
}
