// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Kernels of radarx.retrieve.motion.optical_flow.
//
// lk_solve: the Lucas-Kanade least squares at every cell. From the derivatives
//   ix, iy (space) and it (time) of a stack of frame pairs, the products
//   ix*ix, ix*iy, iy*iy, ix*it, iy*it are summed over the pairs, smoothed with
//   a Gaussian window (separable taps, edges repeated) and the 2 x 2 normal
//   equations are solved in closed form at each cell. Returns the increment
//   (du, dv) and the smallest and largest eigenvalue of the structure tensor.
//   Threads take rows from an atomic counter in two passes (horizontal,
//   vertical).
// darts_normal: the normal equations of the least squares fit of the
//   optical-flow constraint ix*u + iy*v + it = 0 with u and v expanded in the
//   cosine basis cos(pi p (i + 1/2) / ny) cos(pi q (j + 1/2) / nx),
//   p, q = 0..order, over all cells of a stack of frame pairs. Threads sum
//   their own rows into private matrices that are added at the end.

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

template <typename Fn>
void parallel_items(int64_t n, int n_threads, Fn&& fn) {
    if (n <= 0) return;
    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    int nt = n_threads > 0 ? n_threads : static_cast<int>(hw);
    nt = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(nt, n)));
    const int64_t block = std::max<int64_t>(1, std::min<int64_t>(8, n / (8 * nt)));
    std::atomic<int64_t> next{0};
    auto worker = [&](int tid) {
        for (;;) {
            const int64_t w0 = next.fetch_add(block);
            if (w0 >= n) break;
            fn(tid, w0, std::min(n, w0 + block));
        }
    };
    std::vector<std::thread> pool;
    for (int t = 1; t < nt; ++t) pool.emplace_back(worker, t);
    worker(0);
    for (auto& th : pool) th.join();
}

inline int64_t clamp_index(int64_t i, int64_t n) { return i < 0 ? 0 : (i >= n ? n - 1 : i); }

void check_triple(const DArray& ix, const DArray& iy, const DArray& it) {
    if (ix.ndim() != 3) throw std::invalid_argument("derivatives must be 3-D (pair, y, x)");
    for (int k = 0; k < 3; ++k)
        if (iy.shape(k) != ix.shape(k) || it.shape(k) != ix.shape(k))
            throw std::invalid_argument("ix, iy and it differ in shape");
}

// Returns (4, ny, nx): du, dv, smallest and largest eigenvalue.
DArray lk_solve(const DArray& ix, const DArray& iy, const DArray& it, const DArray& taps,
                int n_threads) {
    check_triple(ix, iy, it);
    if (taps.ndim() != 1 || taps.size() % 2 != 1)
        throw std::invalid_argument("taps must be 1-D with an odd length");
    const int64_t n = ix.shape(0), ny = ix.shape(1), nx = ix.shape(2);
    const int64_t r = taps.size() / 2;
    const double* gx = ix.data();
    const double* gy = iy.data();
    const double* gt = it.data();
    const double* tp = taps.data();
    std::vector<double> tmp(static_cast<size_t>(5 * ny * nx));
    DArray out(std::vector<py::ssize_t>{4, static_cast<py::ssize_t>(ny), static_cast<py::ssize_t>(nx)});
    double* po = out.mutable_data();
    const int64_t plane = ny * nx;
    {
        py::gil_scoped_release release;
        // pass 1: products summed over the pairs, smoothed along the row
        parallel_items(ny, n_threads, [&](int, int64_t i0, int64_t i1) {
            std::vector<double> prod(static_cast<size_t>(5 * nx));
            for (int64_t i = i0; i < i1; ++i) {
                std::fill(prod.begin(), prod.end(), 0.0);
                for (int64_t k = 0; k < n; ++k) {
                    const int64_t base = k * plane + i * nx;
                    for (int64_t j = 0; j < nx; ++j) {
                        const double a = gx[base + j], b = gy[base + j], c = gt[base + j];
                        prod[j] += a * a;
                        prod[nx + j] += a * b;
                        prod[2 * nx + j] += b * b;
                        prod[3 * nx + j] += a * c;
                        prod[4 * nx + j] += b * c;
                    }
                }
                for (int q = 0; q < 5; ++q) {
                    const double* src = prod.data() + q * nx;
                    double* dst = tmp.data() + q * plane + i * nx;
                    for (int64_t j = 0; j < nx; ++j) {
                        double s = 0.0;
                        for (int64_t d = -r; d <= r; ++d)
                            s += tp[d + r] * src[clamp_index(j + d, nx)];
                        dst[j] = s;
                    }
                }
            }
        });
        // pass 2: smoothing along the column and the 2 x 2 solve
        parallel_items(ny, n_threads, [&](int, int64_t i0, int64_t i1) {
            std::vector<double> s(5);
            for (int64_t i = i0; i < i1; ++i) {
                for (int64_t j = 0; j < nx; ++j) {
                    for (int q = 0; q < 5; ++q) {
                        double v = 0.0;
                        for (int64_t d = -r; d <= r; ++d)
                            v += tp[d + r] * tmp[q * plane + clamp_index(i + d, ny) * nx + j];
                        s[q] = v;
                    }
                    const double sxx = s[0], sxy = s[1], syy = s[2], sxt = s[3], syt = s[4];
                    const double tr = 0.5 * (sxx + syy);
                    const double det = sxx * syy - sxy * sxy;
                    const double disc = std::sqrt(std::max(0.0, tr * tr - det));
                    const double lmin = tr - disc, lmax = tr + disc;
                    double du = 0.0, dv = 0.0;
                    if (det > 1e-12 * std::max(tr * tr, 1e-300) && det > 0.0) {
                        du = -(syy * sxt - sxy * syt) / det;
                        dv = -(sxx * syt - sxy * sxt) / det;
                    }
                    const int64_t o = i * nx + j;
                    po[o] = du;
                    po[plane + o] = dv;
                    po[2 * plane + o] = lmin;
                    po[3 * plane + o] = lmax;
                }
            }
        });
    }
    return out;
}

// Returns (2 M, 2 M + 1): the normal matrix and, in the last column, the right
// hand side, with M = (order + 1)^2 and the unknowns [cu_0..cu_M-1, cv_0..cv_M-1].
DArray darts_normal(const DArray& ix, const DArray& iy, const DArray& it, int order,
                    int n_threads) {
    check_triple(ix, iy, it);
    if (order < 0) throw std::invalid_argument("order must not be negative");
    const int64_t n = ix.shape(0), ny = ix.shape(1), nx = ix.shape(2);
    const int64_t P = order + 1, M = P * P, D = 2 * M;
    const double pi = 3.14159265358979323846;
    std::vector<double> cy(static_cast<size_t>(P * ny)), cx(static_cast<size_t>(P * nx));
    for (int64_t p = 0; p < P; ++p) {
        for (int64_t i = 0; i < ny; ++i) cy[p * ny + i] = std::cos(pi * p * (i + 0.5) / ny);
        for (int64_t j = 0; j < nx; ++j) cx[p * nx + j] = std::cos(pi * p * (j + 0.5) / nx);
    }
    const double* gx = ix.data();
    const double* gy = iy.data();
    const double* gt = it.data();
    const int64_t plane = ny * nx;
    const int nt_max = n_threads > 0 ? n_threads
                                      : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    std::vector<std::vector<double>> acc(static_cast<size_t>(nt_max),
                                         std::vector<double>(static_cast<size_t>(D * (D + 1)), 0.0));
    {
        py::gil_scoped_release release;
        parallel_items(ny, n_threads, [&](int tid, int64_t i0, int64_t i1) {
            std::vector<double>& a = acc[static_cast<size_t>(tid)];
            std::vector<double> phi(static_cast<size_t>(M)), row(static_cast<size_t>(D));
            for (int64_t i = i0; i < i1; ++i) {
                for (int64_t j = 0; j < nx; ++j) {
                    for (int64_t p = 0; p < P; ++p)
                        for (int64_t q = 0; q < P; ++q)
                            phi[p * P + q] = cy[p * ny + i] * cx[q * nx + j];
                    for (int64_t k = 0; k < n; ++k) {
                        const int64_t o = k * plane + i * nx + j;
                        const double a1 = gx[o], b1 = gy[o], c1 = gt[o];
                        if (a1 == 0.0 && b1 == 0.0) continue;
                        for (int64_t m = 0; m < M; ++m) {
                            row[m] = phi[m] * a1;
                            row[M + m] = phi[m] * b1;
                        }
                        for (int64_t r1 = 0; r1 < D; ++r1) {
                            const double v = row[r1];
                            double* arow = a.data() + r1 * (D + 1);
                            for (int64_t r2 = r1; r2 < D; ++r2) arow[r2] += v * row[r2];
                            arow[D] -= v * c1;
                        }
                    }
                }
            }
        });
    }
    DArray out(std::vector<py::ssize_t>{static_cast<py::ssize_t>(D), static_cast<py::ssize_t>(D + 1)});
    double* po = out.mutable_data();
    std::fill(po, po + D * (D + 1), 0.0);
    for (const auto& a : acc)
        for (int64_t r1 = 0; r1 < D; ++r1)
            for (int64_t r2 = r1; r2 <= D; ++r2) po[r1 * (D + 1) + r2] += a[r1 * (D + 1) + r2];
    for (int64_t r1 = 0; r1 < D; ++r1)
        for (int64_t r2 = 0; r2 < r1; ++r2) po[r1 * (D + 1) + r2] = po[r2 * (D + 1) + r1];
    return out;
}

}  // namespace

PYBIND11_MODULE(_optical_flow, m) {
    m.doc() = "Compiled optical-flow kernels for radarx.";
    m.def("lk_solve", &lk_solve, py::arg("ix"), py::arg("iy"), py::arg("it"), py::arg("taps"),
          py::arg("n_threads") = 0);
    m.def("darts_normal", &darts_normal, py::arg("ix"), py::arg("iy"), py::arg("it"),
          py::arg("order"), py::arg("n_threads") = 0);
}
