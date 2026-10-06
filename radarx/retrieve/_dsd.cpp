// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Drop size distribution retrieval kernel.
//
// Every gate is independent, so the gates of all sweeps (or of a grid, a QVP)
// form one pool of work that threads take in blocks from an atomic counter.
// Per gate:
//
// 1. table lookup: the lookup table holds one family of gamma DSDs (one
//    mu-Lambda relation, or one mu of the normalized gamma DSD) against
//    increasing ZDR. ZDR selects the position by binary search and linear
//    interpolation (clipped to the table's range); the table gives log10 of
//    Z_H and K_DP per unit intercept, the shape mu, the slope Lambda and the
//    conversion from the unit intercept to N0.
// 2. the intercept follows from Z_H, or from K_DP (linear in the intercept)
//    where K_DP is given and at least kdp_min.
// 3. the moments of the (untruncated) gamma DSD give Dm, D0, Nw, the liquid
//    water content and the rain rate in closed form. Gates whose Nw is
//    outside [nw_min, nw_max] are inconsistent with rain and left empty.
//
// Every step follows the NumPy reference in radarx/retrieve/dsd.py, in the
// same order of operations.

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
using BArray = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kPi = 3.14159265358979323846;  // M_PI is not standard (MSVC)
constexpr int kNOut = 8;          // N0, Nw, D0, Dm, mu, Lambda, rain rate, LWC
constexpr int64_t kBlock = 2048;  // gates per scheduling block

// Lookup table of n entries against increasing ZDR.
struct Table {
    const double* zdr;    // ZDR [dB]
    const double* logz;   // log10 Z_H [mm6 m-3] per unit intercept
    const double* logk;   // log10 K_DP [deg km-1] per unit intercept
    const double* logn0;  // log10 N0 per unit intercept
    const double* mu;
    const double* lam;  // Lambda [mm-1]
    int64_t n = 0;
    double kdp_min = 0.0;
    double nw_min = 0.0, nw_max = 0.0;  // plausible range of Nw [m-3 mm-1]
};

struct Entry {
    double logz, logk, logn0, mu, lam;
};

// Position of x in the table (clipped) and the interpolated values.
inline Entry lookup(const Table& t, double x) {
    const int64_t n = t.n;
    const double* z = t.zdr;
    const double xc = std::min(std::max(x, z[0]), z[n - 1]);
    int64_t i = static_cast<int64_t>(std::upper_bound(z, z + n, xc) - z) - 1;
    i = std::min(std::max<int64_t>(i, 0), n - 2);
    const double w = (xc - z[i]) / (z[i + 1] - z[i]);
    auto at = [&](const double* a) { return a[i] + w * (a[i + 1] - a[i]); };
    return {at(t.logz), at(t.logk), at(t.logn0), at(t.mu), at(t.lam)};
}

// Closed-form moments of N(D) = N0 D^mu exp(-Lambda D) (D in mm).
inline void moments(const Table& t, double logn0, double mu, double lam, double* out,
                    int64_t g, int64_t n) {
    const double n0 = std::pow(10.0, logn0);
    const double m3 = n0 * std::tgamma(mu + 4.0) * std::pow(lam, -(mu + 4.0));
    const double dm = (mu + 4.0) / lam;
    const double nw = 256.0 / 6.0 * m3 / (dm * dm * dm * dm);
    if (!(nw >= t.nw_min && nw <= t.nw_max)) {
        for (int j = 0; j < kNOut; ++j) out[j * n + g] = kNaN;
        return;
    }
    out[0 * n + g] = n0;
    out[1 * n + g] = nw;
    out[2 * n + g] = (mu + 3.67) / lam;
    out[3 * n + g] = dm;
    out[4 * n + g] = mu;
    out[5 * n + g] = lam;
    // fall speed 9.65 - 10.3 exp(-0.6 D) m s-1 (Atlas et al. 1973)
    out[6 * n + g] =
        6.0e-4 * kPi * m3 * (9.65 - 10.3 * std::pow(lam / (lam + 0.6), mu + 4.0));
    out[7 * n + g] = kPi / 6.0 * 1.0e-3 * m3;
}

// One input array (a sweep, a grid, ...) in the pool of gates.
struct Block {
    const double* z = nullptr;
    const double* zdr = nullptr;
    const double* kdp = nullptr;
    const uint8_t* mask = nullptr;
    double* out = nullptr;
    int64_t n = 0, first = 0;
};

void gate(const Table& t, const Block& b, int64_t g) {
    const double z = b.z[g], zdr = b.zdr[g];
    if ((b.mask && !b.mask[g]) || !std::isfinite(z) || !std::isfinite(zdr)) {
        for (int j = 0; j < kNOut; ++j) b.out[j * b.n + g] = kNaN;
        return;
    }
    const Entry e = lookup(t, zdr);
    double logn0 = 0.1 * z - e.logz + e.logn0;
    if (b.kdp) {
        const double k = b.kdp[g];
        if (std::isfinite(k) && k >= t.kdp_min && k > 0.0)
            logn0 = std::log10(k) - e.logk + e.logn0;
    }
    moments(t, logn0, e.mu, e.lam, b.out, g, b.n);
}

template <class T>
const T* optional_data(const py::object& obj, int64_t n,
                       py::array_t<T, py::array::c_style | py::array::forcecast>& keep,
                       const char* name) {
    if (obj.is_none()) return nullptr;
    keep = obj.cast<py::array_t<T, py::array::c_style | py::array::forcecast>>();
    if (keep.ndim() != 1 || keep.shape(0) != n)
        throw std::invalid_argument(std::string(name) + " must have the shape of dbzh");
    return keep.data();
}

}  // namespace

// Retrieve gamma DSD parameters for the gates of several arrays in one call.
// z, zdr: 1-D arrays per input; kdp and mask: 1-D arrays or None. The lookup
// table is given as 1-D arrays against increasing ZDR. Returns one (8, n)
// array per input: N0, Nw, D0, Dm, mu, Lambda, rain rate, LWC; gates with Nw
// outside [nw_min, nw_max] are NaN.
py::list retrieve(const std::vector<DArray>& z, const std::vector<DArray>& zdr,
                  const std::vector<py::object>& kdp, const std::vector<py::object>& mask,
                  const DArray& zdr_tab, const DArray& logz_tab, const DArray& logk_tab,
                  const DArray& logn0_tab, const DArray& mu_tab, const DArray& lam_tab,
                  double kdp_min, double nw_min, double nw_max, int n_threads) {
    const size_t ns = z.size();
    if (zdr.size() != ns || kdp.size() != ns || mask.size() != ns)
        throw std::invalid_argument("need one entry per input in every list");
    Table t;
    t.n = zdr_tab.ndim() == 1 ? zdr_tab.shape(0) : 0;
    if (t.n < 2) throw std::invalid_argument("the lookup table needs at least two entries");
    for (const DArray* a : {&logz_tab, &logk_tab, &logn0_tab, &mu_tab, &lam_tab})
        if (a->ndim() != 1 || a->shape(0) != t.n)
            throw std::invalid_argument("lookup table arrays must have the same size");
    for (int64_t i = 1; i < t.n; ++i)
        if (!(zdr_tab.data()[i] > zdr_tab.data()[i - 1]))
            throw std::invalid_argument("table ZDR must be strictly increasing");
    t.zdr = zdr_tab.data();
    t.logz = logz_tab.data();
    t.logk = logk_tab.data();
    t.logn0 = logn0_tab.data();
    t.mu = mu_tab.data();
    t.lam = lam_tab.data();
    t.kdp_min = kdp_min;
    t.nw_min = nw_min;
    t.nw_max = nw_max;

    std::vector<DArray> kdp_keep(ns);
    std::vector<BArray> mask_keep(ns);
    std::vector<py::array_t<double>> outs;
    std::vector<Block> blocks(ns);
    int64_t total = 0;
    for (size_t i = 0; i < ns; ++i) {
        Block& b = blocks[i];
        if (z[i].ndim() != 1 || zdr[i].ndim() != 1 || zdr[i].shape(0) != z[i].shape(0))
            throw std::invalid_argument("dbzh and zdr must be 1-D arrays of the same size");
        b.n = z[i].shape(0);
        b.z = z[i].data();
        b.zdr = zdr[i].data();
        b.kdp = optional_data<double>(kdp[i], b.n, kdp_keep[i], "kdp");
        b.mask = optional_data<uint8_t>(mask[i], b.n, mask_keep[i], "mask");
        outs.emplace_back(std::vector<py::ssize_t>{kNOut, b.n});
        b.out = outs.back().mutable_data();
        b.first = total;
        total += b.n;
    }

    {
        py::gil_scoped_release release;
        if (total > 0) {
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
                    // input holding gate g0 (inputs are few: linear search)
                    size_t k = 0;
                    while (g0 >= blocks[k].first + blocks[k].n) ++k;
                    for (int64_t q = g0; q < g1; ++q) {
                        while (q >= blocks[k].first + blocks[k].n) ++k;
                        gate(t, blocks[k], q - blocks[k].first);
                    }
                }
            };
            std::vector<std::thread> pool;
            for (int i = 1; i < nt; ++i) pool.emplace_back(worker);
            worker();
            for (auto& th : pool) th.join();
        }
    }
    py::list result;
    for (auto& o : outs) result.append(o);
    return result;
}

PYBIND11_MODULE(_dsd, m) {
    m.doc() = "Compiled drop size distribution retrieval kernel for radarx.";
    m.def("retrieve", &retrieve, py::arg("dbzh"), py::arg("zdr"), py::arg("kdp"),
          py::arg("mask"), py::arg("zdr_table"), py::arg("logz_table"),
          py::arg("logk_table"), py::arg("logn0_table"), py::arg("mu_table"),
          py::arg("lambda_table"), py::arg("kdp_min") = 0.0, py::arg("nw_min") = 0.0,
          py::arg("nw_max") = std::numeric_limits<double>::infinity(),
          py::arg("n_threads") = 0);
}
