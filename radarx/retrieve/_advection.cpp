// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Semi-Lagrangian advection kernel.
//
// data holds nk planes on an (ny, nx) grid. src_row and src_col hold ng sets
// of departure points on an (oy, ox) output grid, in fractional row/column
// indices of the data grid. Output cell (g, k, i, j) is plane k interpolated
// at departure point (g, i, j). All planes share the departure points, so the
// interpolation stencil of an output row is computed once and applied to
// every plane. One call handles all fields, levels and time steps; the work
// is split over (set, row) pairs handed out to threads by an atomic counter.
//
// order 1: bilinear interpolation from the four surrounding cells.
// order 3: cubic convolution (Keys 1981, a = -1/2) from the 4 x 4 surrounding
//          cells, clipped to the range of the four nearest valid cells so it
//          cannot create new extremes.
//
// Missing data (NaN, or a neighbour outside the grid) is handled with an
// advected validity mask: the bilinear weights of the valid neighbours are
// summed, and the cell is defined only if that sum reaches min_weight; the
// value is then the weighted mean of the valid neighbours. The cubic value is
// used only where every neighbour with a non-zero cubic weight is valid,
// otherwise the cell falls back to the masked bilinear value.

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

// Keys (1981) cubic convolution weights for fractional offset t in [0, 1).
inline void keys_weights(double t, double* w) {
    const double a = -0.5;
    const double t1 = 1.0 + t, t2 = t, t3 = 1.0 - t, t4 = 2.0 - t;
    w[0] = ((a * t1 - 5.0 * a) * t1 + 8.0 * a) * t1 - 4.0 * a;
    w[1] = ((a + 2.0) * t2 - (a + 3.0)) * t2 * t2 + 1.0;
    w[2] = ((a + 2.0) * t3 - (a + 3.0)) * t3 * t3 + 1.0;
    w[3] = ((a * t4 - 5.0 * a) * t4 + 8.0 * a) * t4 - 4.0 * a;
}

// Interpolation stencil of one output cell (shared by all planes).
struct Stencil {
    int64_t i0 = 0, j0 = 0;  // upper-left cell of the bilinear 2 x 2 block
    double fr = 0.0, fc = 0.0;
    double w[4] = {0, 0, 0, 0};                          // bilinear weights
    double wy[4] = {0, 0, 0, 0}, wx[4] = {0, 0, 0, 0};  // cubic weights
    bool valid = false;   // departure point is a finite number
    bool inside = false;  // the whole stencil lies inside the grid
};

// Value of plane p at one stencil (NaN if undefined). Bounds = false skips
// the bounds checks for stencils that lie entirely inside the grid.
template <bool Bounds, typename T>
inline double sample_impl(const T* p, int64_t ny, int64_t nx, const Stencil& s, int order,
                          double min_weight) {
    auto value_at = [&](int64_t i, int64_t j, double& v) -> bool {
        if (Bounds && (i < 0 || i >= ny || j < 0 || j >= nx)) return false;
        v = static_cast<double>(p[i * nx + j]);
        return !std::isnan(v);
    };
    const int64_t di[4] = {0, 0, 1, 1}, dj[4] = {0, 1, 0, 1};
    double num = 0.0, den = 0.0;
    double lo = std::numeric_limits<double>::infinity(), hi = -lo;
    for (int q = 0; q < 4; ++q) {
        double v;
        if (!value_at(s.i0 + di[q], s.j0 + dj[q], v)) continue;
        num += s.w[q] * v;
        den += s.w[q];
        lo = std::min(lo, v);
        hi = std::max(hi, v);
    }
    if (!(den > 0.0) || den < min_weight) return kNaN;
    const double linear = num / den;
    if (order != 3) return linear;
    double acc = 0.0;
    for (int a = 0; a < 4; ++a) {
        if (s.wy[a] == 0.0) continue;
        double row = 0.0;
        for (int b = 0; b < 4; ++b) {
            if (s.wx[b] == 0.0) continue;
            double v;
            if (!value_at(s.i0 - 1 + a, s.j0 - 1 + b, v)) return linear;
            row += s.wx[b] * v;
        }
        acc += s.wy[a] * row;
    }
    return std::min(hi, std::max(lo, acc));
}

template <typename T>
inline double sample(const T* p, int64_t ny, int64_t nx, const Stencil& s, int order,
                     double min_weight) {
    if (!s.valid) return kNaN;
    if (s.inside) return sample_impl<false>(p, ny, nx, s, order, min_weight);
    return sample_impl<true>(p, ny, nx, s, order, min_weight);
}

template <typename T>
py::array_t<T> advect(py::array_t<T, py::array::c_style | py::array::forcecast> data,
                      const DArray& src_row, const DArray& src_col, int order,
                      double min_weight, int n_threads) {
    if (data.ndim() != 3) throw std::invalid_argument("data must be 3-D (plane, y, x)");
    if (order != 1 && order != 3) throw std::invalid_argument("order must be 1 or 3");
    if (src_row.ndim() != 3 || src_col.ndim() != 3)
        throw std::invalid_argument("departure points must be 3-D (set, y, x)");
    for (int d = 0; d < 3; ++d)
        if (src_row.shape(d) != src_col.shape(d))
            throw std::invalid_argument("src_row and src_col differ in shape");
    const int64_t nk = data.shape(0), ny = data.shape(1), nx = data.shape(2);
    const int64_t ng = src_row.shape(0), oy = src_row.shape(1), ox = src_row.shape(2);

    py::array_t<T> result({ng, nk, oy, ox});
    T* out = result.mutable_data();
    const T* in = data.data();
    const double* pr = src_row.data();
    const double* pc = src_col.data();
    const int64_t plane = ny * nx, oplane = oy * ox;
    const int64_t nrows = ng * oy;  // independent work items

    {
        py::gil_scoped_release release;
        unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        int nt = n_threads > 0 ? n_threads : static_cast<int>(hw);
        nt = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(nt, nrows)));
        const int64_t block = std::max<int64_t>(1, std::min<int64_t>(16, nrows / (8 * nt)));
        std::atomic<int64_t> next{0};

        auto worker = [&]() {
            std::vector<Stencil> st(ox);  // one allocation per thread
            for (;;) {
                const int64_t w0 = next.fetch_add(block);
                if (w0 >= nrows) break;
                const int64_t w1 = std::min(nrows, w0 + block);
                for (int64_t w = w0; w < w1; ++w) {
                    const int64_t g = w / oy, i = w % oy;
                    const double* rr = pr + w * ox;
                    const double* cc = pc + w * ox;
                    for (int64_t j = 0; j < ox; ++j) {
                        Stencil& s = st[j];
                        const double r = rr[j], c = cc[j];
                        s.valid = std::isfinite(r) && std::isfinite(c) && r > -2.0 &&
                                  c > -2.0 && r < ny + 1.0 && c < nx + 1.0;
                        if (!s.valid) continue;
                        const double fi = std::floor(r), fj = std::floor(c);
                        s.i0 = static_cast<int64_t>(fi);
                        s.j0 = static_cast<int64_t>(fj);
                        s.fr = r - fi;
                        s.fc = c - fj;
                        s.w[0] = (1 - s.fr) * (1 - s.fc);
                        s.w[1] = (1 - s.fr) * s.fc;
                        s.w[2] = s.fr * (1 - s.fc);
                        s.w[3] = s.fr * s.fc;
                        const int64_t m = order == 3 ? 1 : 0;  // cubic reaches one more
                        s.inside = s.i0 - m >= 0 && s.i0 + 1 + m < ny && s.j0 - m >= 0 &&
                                   s.j0 + 1 + m < nx;
                        if (order == 3) {
                            keys_weights(s.fr, s.wy);
                            keys_weights(s.fc, s.wx);
                        }
                    }
                    for (int64_t k = 0; k < nk; ++k) {
                        const T* p = in + k * plane;
                        T* o = out + (g * nk + k) * oplane + i * ox;
                        for (int64_t j = 0; j < ox; ++j)
                            o[j] = static_cast<T>(sample(p, ny, nx, st[j], order, min_weight));
                    }
                }
            }
        };
        std::vector<std::thread> pool;
        for (int t = 1; t < nt; ++t) pool.emplace_back(worker);
        worker();
        for (auto& th : pool) th.join();
    }
    return result;
}

}  // namespace

PYBIND11_MODULE(_advection, m) {
    m.doc() = "Compiled semi-Lagrangian advection kernel for radarx.";
    m.def("advect", &advect<float>, py::arg("data").noconvert(), py::arg("src_row"),
          py::arg("src_col"), py::arg("order") = 1, py::arg("min_weight") = 0.5,
          py::arg("n_threads") = 0);
    m.def("advect", &advect<double>, py::arg("data"), py::arg("src_row"), py::arg("src_col"),
          py::arg("order") = 1, py::arg("min_weight") = 0.5, py::arg("n_threads") = 0);
}
