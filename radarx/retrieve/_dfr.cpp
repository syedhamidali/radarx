// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Look-up of the dual-frequency ratio retrievals: every gate of a field of
// dual-frequency ratios is mapped through a monotonic table (ratio -> size,
// ratio -> normalization, ...) by piecewise-linear interpolation. The tables are
// built in Python from the scattering calculations (radarx/retrieve/dual_frequency.py,
// which also has the NumPy twin of this function). A ratio below the first table
// entry by at most `tolerance` takes the first entry (noise around the Rayleigh
// limit); any other ratio outside the table, and a missing ratio, is NaN.
//
// Gates are independent; threads take blocks of gates from an atomic counter.

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
constexpr int64_t kBlock = 8192;

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
            const int64_t i0 = next.fetch_add(kBlock);
            if (i0 >= total) break;
            body(i0, std::min(total, i0 + kBlock));
        }
    };
    std::vector<std::thread> pool;
    for (int i = 1; i < nt; ++i) pool.emplace_back(worker);
    worker();
    for (auto& th : pool) th.join();
}

// x: (n) ratios; lut_x: (m) increasing; lut_y: (k, m). Returns (k, n).
DArray lookup_py(const DArray& x, const DArray& lut_x, const DArray& lut_y, double tolerance, int n_threads) {
    if (x.ndim() != 1) throw std::invalid_argument("x must be 1-D");
    if (lut_x.ndim() != 1 || lut_x.shape(0) < 2) throw std::invalid_argument("lut_x must be 1-D with at least 2 entries");
    if (lut_y.ndim() != 2 || lut_y.shape(1) != lut_x.shape(0))
        throw std::invalid_argument("lut_y must have shape (k, len(lut_x))");
    const int64_t n = x.shape(0), m = lut_x.shape(0), k = lut_y.shape(0);
    const double* px = x.data();
    const double* lx = lut_x.data();
    const double* ly = lut_y.data();
    for (int64_t i = 1; i < m; ++i)
        if (!(lx[i] > lx[i - 1])) throw std::invalid_argument("lut_x must increase");
    DArray out(std::vector<py::ssize_t>{static_cast<py::ssize_t>(k), static_cast<py::ssize_t>(n)});
    double* po = out.mutable_data();
    {
        py::gil_scoped_release release;
        parallel_blocks(n, n_threads, [&](int64_t i0, int64_t i1) {
            for (int64_t i = i0; i < i1; ++i) {
                const double v = px[i];
                if (!std::isfinite(v) || v > lx[m - 1] || v < lx[0] - tolerance) {
                    for (int64_t c = 0; c < k; ++c) po[c * n + i] = kNaN;
                    continue;
                }
                if (v <= lx[0]) {
                    for (int64_t c = 0; c < k; ++c) po[c * n + i] = ly[c * m];
                    continue;
                }
                const int64_t j = std::upper_bound(lx, lx + m, v) - lx;  // lx[j-1] <= v < lx[j]
                if (j >= m) {
                    for (int64_t c = 0; c < k; ++c) po[c * n + i] = ly[c * m + m - 1];
                    continue;
                }
                const double w = (v - lx[j - 1]) / (lx[j] - lx[j - 1]);
                for (int64_t c = 0; c < k; ++c)
                    po[c * n + i] = ly[c * m + j - 1] + w * (ly[c * m + j] - ly[c * m + j - 1]);
            }
        });
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(_dfr, m) {
    m.doc() = "Look-up of the dual-frequency ratio retrievals";
    m.def("lookup", &lookup_py, py::arg("x"), py::arg("lut_x"), py::arg("lut_y"), py::arg("tolerance") = 0.0,
          py::arg("n_threads") = 0);
}
