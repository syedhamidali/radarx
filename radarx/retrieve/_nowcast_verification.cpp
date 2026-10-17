// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Verification kernels of radarx.retrieve.nowcast_verification.
//
// fss_sums: the sums of the fractions skill score (Roberts and Lean 2008) of
//   many forecast/observation pairs. The fraction of cells above a threshold
//   in the square neighbourhood of every cell comes from a summed-area table,
//   so the work does not depend on the size of the neighbourhood. Threads take
//   pairs (2-D slices) from an atomic counter.
// crps_ensemble: the continuous ranked probability score of the ensemble of
//   every point. With the members sorted, the sum of the absolute differences
//   of all pairs is a weighted sum of the sorted values, so the cost is
//   O(m log m) for m members. Threads take blocks of points.

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

template <typename Fn>
void parallel_items(int64_t n, int64_t block, int n_threads, Fn&& fn) {
    if (n <= 0) return;
    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    int nt = n_threads > 0 ? n_threads : static_cast<int>(hw);
    nt = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(nt, (n + block - 1) / block)));
    std::atomic<int64_t> next{0};
    auto worker = [&]() {
        for (;;) {
            const int64_t w0 = next.fetch_add(block);
            if (w0 >= n) break;
            fn(w0, std::min(n, w0 + block));
        }
    };
    std::vector<std::thread> pool;
    for (int t = 1; t < nt; ++t) pool.emplace_back(worker);
    worker();
    for (auto& th : pool) th.join();
}

// Summed-area table of a (ny, nx) array into (ny + 1, nx + 1).
inline void integral(const std::vector<double>& a, int64_t ny, int64_t nx,
                     std::vector<double>& s) {
    const int64_t w = nx + 1;
    std::fill(s.begin(), s.begin() + w, 0.0);
    for (int64_t i = 0; i < ny; ++i) {
        double row = 0.0;
        s[(i + 1) * w] = 0.0;
        for (int64_t j = 0; j < nx; ++j) {
            row += a[i * nx + j];
            s[(i + 1) * w + j + 1] = s[i * w + j + 1] + row;
        }
    }
}

inline double box(const std::vector<double>& s, int64_t w, int64_t i0, int64_t i1, int64_t j0,
                  int64_t j1) {
    return s[i1 * w + j1] - s[i0 * w + j1] - s[i1 * w + j0] + s[i0 * w + j0];
}

// fc, ob: (n, ny, nx). Returns (n, 4): sum of squared differences of the
// fractions, sum of the squared forecast fractions, sum of the squared
// observed fractions and the number of cells counted. A cell counts if both
// values are finite. edge 0: the fraction is taken over the cells of the
// neighbourhood that are inside the grid and finite in both fields; edge 1:
// over the whole neighbourhood, so cells outside the grid and missing cells
// count as no event.
DArray fss_sums(const DArray& fc, const DArray& ob, double threshold, int64_t hy, int64_t hx,
                int edge, int n_threads) {
    if (fc.ndim() != 3 || ob.ndim() != 3) throw std::invalid_argument("fields must be 3-D");
    for (int k = 0; k < 3; ++k)
        if (fc.shape(k) != ob.shape(k)) throw std::invalid_argument("fields differ in shape");
    if (hy < 0 || hx < 0) throw std::invalid_argument("half widths must not be negative");
    if (edge != 0 && edge != 1) throw std::invalid_argument("edge must be 0 or 1");
    const int64_t n = fc.shape(0), ny = fc.shape(1), nx = fc.shape(2);
    DArray out(std::vector<py::ssize_t>{static_cast<py::ssize_t>(n), 4});
    const double* pf = fc.data();
    const double* po = ob.data();
    double* res = out.mutable_data();
    {
        py::gil_scoped_release release;
        parallel_items(n, 1, n_threads, [&](int64_t k0, int64_t k1) {
            std::vector<double> ef(ny * nx), eo(ny * nx), ev(ny * nx);
            const int64_t w = nx + 1;
            std::vector<double> sf((ny + 1) * w), so((ny + 1) * w), sv((ny + 1) * w);
            for (int64_t k = k0; k < k1; ++k) {
                const double* f = pf + k * ny * nx;
                const double* o = po + k * ny * nx;
                for (int64_t q = 0; q < ny * nx; ++q) {
                    const bool ok = std::isfinite(f[q]) && std::isfinite(o[q]);
                    ev[q] = ok ? 1.0 : 0.0;
                    ef[q] = ok && f[q] >= threshold ? 1.0 : 0.0;
                    eo[q] = ok && o[q] >= threshold ? 1.0 : 0.0;
                }
                integral(ef, ny, nx, sf);
                integral(eo, ny, nx, so);
                integral(ev, ny, nx, sv);
                double sse = 0.0, sf2 = 0.0, so2 = 0.0, count = 0.0;
                for (int64_t i = 0; i < ny; ++i) {
                    const int64_t i0 = std::max<int64_t>(0, i - hy);
                    const int64_t i1 = std::min<int64_t>(ny, i + hy + 1);
                    for (int64_t j = 0; j < nx; ++j) {
                        if (ev[i * nx + j] == 0.0) continue;
                        const int64_t j0 = std::max<int64_t>(0, j - hx);
                        const int64_t j1 = std::min<int64_t>(nx, j + hx + 1);
                        const double area = edge == 0 ? box(sv, w, i0, i1, j0, j1)
                                                      : (2.0 * hy + 1.0) * (2.0 * hx + 1.0);
                        const double pfr = box(sf, w, i0, i1, j0, j1) / area;
                        const double por = box(so, w, i0, i1, j0, j1) / area;
                        sse += (pfr - por) * (pfr - por);
                        sf2 += pfr * pfr;
                        so2 += por * por;
                        count += 1.0;
                    }
                }
                res[4 * k] = sse;
                res[4 * k + 1] = sf2;
                res[4 * k + 2] = so2;
                res[4 * k + 3] = count;
            }
        });
    }
    return out;
}

// ens: (n, m) members of n points; obs: (n). The CRPS of each point:
//   (1/m) sum |x_i - y| - 1/(2 m (m - d)) sum_{i,j} |x_i - x_j|
// with d = 0 (the CRPS of the empirical distribution of the members) or d = 1
// (the fair CRPS, Ferro 2014). NaN members are left out, m is the number of
// the others; NaN where the observation is NaN or no member is finite (the
// fair score also needs two members).
DArray crps_ensemble(const DArray& ens, const DArray& obs, int fair, int n_threads) {
    if (ens.ndim() != 2) throw std::invalid_argument("ens must be 2-D (point, member)");
    const int64_t n = ens.shape(0), m = ens.shape(1);
    if (obs.size() != n) throw std::invalid_argument("obs needs one value per point");
    DArray out(std::vector<py::ssize_t>{static_cast<py::ssize_t>(n)});
    const double* pe = ens.data();
    const double* po = obs.data();
    double* res = out.mutable_data();
    {
        py::gil_scoped_release release;
        parallel_items(n, 4096, n_threads, [&](int64_t p0, int64_t p1) {
            std::vector<double> x(static_cast<size_t>(m));
            for (int64_t p = p0; p < p1; ++p) {
                const double y = po[p];
                int64_t c = 0;
                for (int64_t i = 0; i < m; ++i) {
                    const double v = pe[p * m + i];
                    if (!std::isnan(v)) x[c++] = v;
                }
                if (std::isnan(y) || c == 0 || (fair && c < 2)) {
                    res[p] = kNaN;
                    continue;
                }
                std::sort(x.begin(), x.begin() + c);
                double abs_err = 0.0, spread = 0.0;
                for (int64_t i = 0; i < c; ++i) {
                    abs_err += std::fabs(x[i] - y);
                    spread += (2.0 * (i + 1) - c - 1.0) * x[i];
                }
                // sum over all ordered pairs |x_i - x_j| is 2 * spread
                const double denom = fair ? static_cast<double>(c) * (c - 1.0)
                                          : static_cast<double>(c) * c;
                res[p] = abs_err / c - spread / denom;
            }
        });
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(_nowcast_verification, m) {
    m.doc() = "Compiled verification kernels for radarx nowcasts.";
    m.def("fss_sums", &fss_sums, py::arg("fc"), py::arg("ob"), py::arg("threshold"),
          py::arg("hy"), py::arg("hx"), py::arg("edge") = 0, py::arg("n_threads") = 0);
    m.def("crps_ensemble", &crps_ensemble, py::arg("ens"), py::arg("obs"), py::arg("fair") = 0,
          py::arg("n_threads") = 0);
}
