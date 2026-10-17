// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Depth of the ZDR column of every vertical column of a grid: the vertically
// continuous depth above the freezing level in which ZDR is at least a threshold
// (Snyder et al. 2015), and the KDP quantities of a slab above the freezing level
// (van Lier-Walqui et al. 2016): the integral, the thickness at or above a threshold and
// the height of the highest such level. Every column is independent, so threads take
// blocks of columns from an atomic counter. The algorithms and the choices that are
// radarx's own are described in the docstring of radarx/retrieve/columns.py; the NumPy
// references there follow the same steps.

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
constexpr int64_t kBlock = 2048;  // columns per scheduling block

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
            const int64_t c0 = next.fetch_add(kBlock);
            if (c0 >= total) break;
            body(c0, std::min(total, c0 + kBlock));
        }
    };
    std::vector<std::thread> pool;
    for (int i = 1; i < nt; ++i) pool.emplace_back(worker);
    worker();
    for (auto& th : pool) th.join();
}

// zdr, keep: (n, nz) with the levels of a column contiguous and increasing in height;
// z, thickness: (nz); level: (n), the freezing level of every column.
// A level counts when ZDR is finite, at least `threshold`, and keep is non-zero. The
// run of counting levels must start at the first level at or above the freezing level.
// The depth is the sum of the layer thicknesses of the run; it is NaN where the column
// has no finite ZDR at or above the freezing level (or no freezing level).
DArray depth_py(const DArray& zdr, const DArray& keep, const DArray& z, const DArray& thickness,
                const DArray& level, double threshold, int n_threads) {
    if (zdr.ndim() != 2 || keep.ndim() != 2) throw std::invalid_argument("zdr and keep must be 2-D");
    const int64_t n = zdr.shape(0), nz = zdr.shape(1);
    if (keep.shape(0) != n || keep.shape(1) != nz) throw std::invalid_argument("keep must have the shape of zdr");
    if (z.size() != nz || thickness.size() != nz) throw std::invalid_argument("z and thickness must have one value per level");
    if (level.size() != n) throw std::invalid_argument("level must have one value per column");
    DArray out(std::vector<py::ssize_t>{static_cast<py::ssize_t>(n)});
    const double* pz = zdr.data();
    const double* pk = keep.data();
    const double* zz = z.data();
    const double* th = thickness.data();
    const double* lv = level.data();
    double* po = out.mutable_data();
    {
        py::gil_scoped_release release;
        parallel_blocks(n, n_threads, [&](int64_t c0, int64_t c1) {
            for (int64_t c = c0; c < c1; ++c) {
                const double f = lv[c];
                if (std::isnan(f)) {
                    po[c] = kNaN;
                    continue;
                }
                int64_t k = 0;
                while (k < nz && zz[k] < f) ++k;
                const double* v = pz + c * nz;
                const double* w = pk + c * nz;
                bool any = false, run = true;
                double depth = 0.0;
                for (; k < nz; ++k) {
                    const bool finite = std::isfinite(v[k]);
                    any = any || finite;
                    if (run) {
                        if (finite && v[k] >= threshold && w[k] != 0.0)
                            depth += th[k];
                        else
                            run = false;
                    }
                    if (!run && any) break;
                }
                po[c] = any ? depth : kNaN;
            }
        });
    }
    return out;
}

// The KDP of a slab from the first level at or above the freezing level to slab_depth above
// it (levels with z < level + slab_depth). Per column: the integral of the positive KDP of
// the counted levels (degree), the summed thickness of the levels with KDP >= threshold (m)
// and the height of the highest level at or above the freezing level, in the whole column,
// with KDP >= threshold (m). A level counts when KDP is finite and keep is non-zero. The
// integral and the thickness are NaN where no level of the slab counts; the top is NaN
// where no level has KDP >= threshold.
py::tuple kdp_slab_py(const DArray& kdp, const DArray& keep, const DArray& z,
                      const DArray& thickness, const DArray& level, double threshold,
                      double slab_depth, int n_threads) {
    if (kdp.ndim() != 2 || keep.ndim() != 2) throw std::invalid_argument("kdp and keep must be 2-D");
    const int64_t n = kdp.shape(0), nz = kdp.shape(1);
    if (keep.shape(0) != n || keep.shape(1) != nz) throw std::invalid_argument("keep must have the shape of kdp");
    if (z.size() != nz || thickness.size() != nz) throw std::invalid_argument("z and thickness must have one value per level");
    if (level.size() != n) throw std::invalid_argument("level must have one value per column");
    const std::vector<py::ssize_t> shape{static_cast<py::ssize_t>(n)};
    DArray integral(shape), depth(shape), top(shape);
    const double* pk = kdp.data();
    const double* pw = keep.data();
    const double* zz = z.data();
    const double* th = thickness.data();
    const double* lv = level.data();
    double* pi = integral.mutable_data();
    double* pd = depth.mutable_data();
    double* pt = top.mutable_data();
    {
        py::gil_scoped_release release;
        parallel_blocks(n, n_threads, [&](int64_t c0, int64_t c1) {
            for (int64_t c = c0; c < c1; ++c) {
                const double f = lv[c];
                pi[c] = pd[c] = pt[c] = kNaN;
                if (std::isnan(f)) continue;
                const double* v = pk + c * nz;
                const double* w = pw + c * nz;
                bool any = false;
                double sum = 0.0, thick = 0.0, highest = kNaN;
                for (int64_t k = 0; k < nz; ++k) {
                    if (zz[k] < f || !std::isfinite(v[k]) || w[k] == 0.0) continue;
                    const bool in_slab = zz[k] < f + slab_depth;
                    if (in_slab) {
                        any = true;
                        if (v[k] > 0.0) sum += v[k] * th[k] / 1000.0;
                        if (v[k] >= threshold) thick += th[k];
                    }
                    if (v[k] >= threshold) highest = zz[k];
                }
                if (any) {
                    pi[c] = sum;
                    pd[c] = thick;
                }
                pt[c] = highest;
            }
        });
    }
    return py::make_tuple(integral, depth, top);
}

}  // namespace

PYBIND11_MODULE(_zdr_column, m) {
    m.doc() = "Depth of the ZDR column and the KDP slab of every vertical column of a grid";
    m.def("kdp_slab", &kdp_slab_py, py::arg("kdp"), py::arg("keep"), py::arg("z"),
          py::arg("thickness"), py::arg("level"), py::arg("threshold"), py::arg("slab_depth"),
          py::arg("n_threads") = 0);
    m.def("depth", &depth_py, py::arg("zdr"), py::arg("keep"), py::arg("z"), py::arg("thickness"),
          py::arg("level"), py::arg("threshold"), py::arg("n_threads") = 0);
}
