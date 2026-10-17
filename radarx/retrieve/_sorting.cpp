// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Enhancement regions of KDP and ZDR with the adaptive threshold of Loeffler and Kumjian
// (2018): the largest threshold, in steps of `step` from `floor`, for which at least
// `min_cells` eligible grid cells reach it, and the median coordinates (centroid) of those
// cells. Every frame is independent, so threads take frames from an atomic counter. The
// NumPy reference in radarx/retrieve/sorting.py follows the same steps.

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
void parallel_items(int64_t total, int n_threads, F&& body) {
    if (total <= 0) return;
    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    int nt = n_threads > 0 ? n_threads : static_cast<int>(hw);
    nt = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(nt, total)));
    std::atomic<int64_t> next{0};
    auto worker = [&]() {
        for (;;) {
            const int64_t i = next.fetch_add(1);
            if (i >= total) break;
            body(i);
        }
    };
    std::vector<std::thread> pool;
    for (int i = 1; i < nt; ++i) pool.emplace_back(worker);
    worker();
    for (auto& th : pool) th.join();
}

double median_of(std::vector<double>& v) {
    const size_t n = v.size(), h = n / 2;
    std::nth_element(v.begin(), v.begin() + h, v.end());
    const double upper = v[h];
    if (n % 2) return upper;
    const double lower = *std::max_element(v.begin(), v.begin() + h);
    return 0.5 * (lower + upper);
}

// values: (nt, n), NaN where a cell is not eligible; x, y: (n).
// Returns the threshold, the median x and y of the cells at or above it and their number,
// each (nt); NaN (and 0 cells) where fewer than min_cells cells are eligible or the
// threshold would be below `floor`.
py::tuple adaptive_py(const DArray& values, const DArray& x, const DArray& y, int64_t min_cells,
                      double step, double floor_value, int n_threads) {
    if (values.ndim() != 2) throw std::invalid_argument("values must be 2-D (time, cells)");
    const int64_t nt = values.shape(0), n = values.shape(1);
    if (x.size() != n || y.size() != n) throw std::invalid_argument("x and y need one value per cell");
    if (!(step > 0.0)) throw std::invalid_argument("step must be positive");
    if (min_cells < 1) throw std::invalid_argument("min_cells must be at least 1");
    const std::vector<py::ssize_t> shape{static_cast<py::ssize_t>(nt)};
    DArray thr(shape), cx(shape), cy(shape), cnt(shape);
    const double* pv = values.data();
    const double* px = x.data();
    const double* py_ = y.data();
    double* pt = thr.mutable_data();
    double* pcx = cx.mutable_data();
    double* pcy = cy.mutable_data();
    double* pc = cnt.mutable_data();
    {
        py::gil_scoped_release release;
        parallel_items(nt, n_threads, [&](int64_t t) {
            const double* v = pv + t * n;
            pt[t] = pcx[t] = pcy[t] = kNaN;
            pc[t] = 0.0;
            std::vector<double> finite;
            finite.reserve(n);
            for (int64_t i = 0; i < n; ++i)
                if (std::isfinite(v[i])) finite.push_back(v[i]);
            if (static_cast<int64_t>(finite.size()) < min_cells) return;
            // the min_cells-th largest value
            std::nth_element(finite.begin(), finite.begin() + (min_cells - 1), finite.end(),
                             [](double a, double b) { return a > b; });
            const double kth = finite[min_cells - 1];
            const double level = std::floor(kth / step) * step;
            if (level < floor_value) return;
            std::vector<double> sx, sy;
            for (int64_t i = 0; i < n; ++i)
                if (std::isfinite(v[i]) && v[i] >= level) {
                    sx.push_back(px[i]);
                    sy.push_back(py_[i]);
                }
            pt[t] = level;
            pcx[t] = median_of(sx);
            pcy[t] = median_of(sy);
            pc[t] = static_cast<double>(sx.size());
        });
    }
    return py::make_tuple(thr, cx, cy, cnt);
}

}  // namespace

PYBIND11_MODULE(_sorting, m) {
    m.doc() = "Adaptive-threshold enhancement regions of Loeffler and Kumjian (2018)";
    m.def("adaptive", &adaptive_py, py::arg("values"), py::arg("x"), py::arg("y"),
          py::arg("min_cells"), py::arg("step"), py::arg("floor_value"), py::arg("n_threads") = 0);
}
