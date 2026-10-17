// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Volume matching of a spaceborne radar and a ground radar. Two reductions that
// every match volume needs:
//
//   match_gates  the (weighted) mean, spread and count of ground radar gates inside
//                a disc around a point, for all sweeps in one call: each sweep is
//                sorted into a uniform grid of buckets once, a query reads the
//                buckets that its disc touches;
//   match_bins   the same for the range bins of a satellite ray between two heights.
//
// A query is independent of the others, so threads take blocks of queries from an
// atomic counter. The algorithm and the choices that are radarx's own are described
// in the docstring of radarx/retrieve/satellite_match.py; the NumPy reference in
// _satellite_match_numpy.py computes the same sums.
//
// Per variable the kernel accumulates the weights w, the sums of w*t(v), w*(v-k) and
// w*(v-k)^2 (k: the first value) over the finite values v, where t(v) = v for mode 0 (arithmetic mean) and
// t(v) = 10^(v/10) for mode 1 (mean of a quantity in dB in linear units). The mean
// is sum(w t)/sum(w) (mode 1 is converted back by the caller), the spread is the
// weighted standard deviation of v itself.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <vector>

namespace py = pybind11;
using DArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using IArray = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kLn2 = 0.69314718055994530942;
constexpr int64_t kBlock = 256;  // queries per scheduling block
constexpr int64_t kMaxCells = 4000000;

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
            const int64_t c0 = next.fetch_add(block);
            if (c0 >= total) break;
            body(c0, std::min(total, c0 + block));
        }
    };
    std::vector<std::thread> pool;
    for (int i = 1; i < nt; ++i) pool.emplace_back(worker);
    worker();
    for (auto& th : pool) th.join();
}

struct Sums {
    double k0 = 0.0;  // shift of the second moment, the first value: no cancellation
    bool has = false;
    double w = 0.0, wt = 0.0, ws = 0.0, wss = 0.0;
    int64_t n = 0;
    inline void add(double v, double weight, int mode) {
        if (!has) {
            k0 = v;
            has = true;
        }
        const double t = mode == 1 ? std::pow(10.0, 0.1 * v) : v;
        const double d = v - k0;
        w += weight;
        wt += weight * t;
        ws += weight * d;
        wss += weight * d * d;
        ++n;
    }
};

struct Buckets {
    double x0 = 0.0, y0 = 0.0, cs = 1.0;
    int64_t nx = 1, ny = 1;
    std::vector<int64_t> start;
    std::vector<int64_t> order;
};

Buckets build(const double* x, const double* y, int64_t a, int64_t b, double cs) {
    Buckets g;
    double xmin = std::numeric_limits<double>::infinity(), xmax = -xmin;
    double ymin = xmin, ymax = -xmin;
    for (int64_t i = a; i < b; ++i) {
        if (!std::isfinite(x[i]) || !std::isfinite(y[i])) continue;
        xmin = std::min(xmin, x[i]);
        xmax = std::max(xmax, x[i]);
        ymin = std::min(ymin, y[i]);
        ymax = std::max(ymax, y[i]);
    }
    if (xmin > xmax) {  // no finite gate
        g.start.assign(2, 0);
        return g;
    }
    while ((std::floor((xmax - xmin) / cs) + 1.0) * (std::floor((ymax - ymin) / cs) + 1.0) >
           static_cast<double>(kMaxCells))
        cs *= 2.0;
    g.x0 = xmin;
    g.y0 = ymin;
    g.cs = cs;
    g.nx = static_cast<int64_t>(std::floor((xmax - xmin) / cs)) + 1;
    g.ny = static_cast<int64_t>(std::floor((ymax - ymin) / cs)) + 1;
    const int64_t ncell = g.nx * g.ny;
    std::vector<int64_t> cell(b - a, -1);
    g.start.assign(ncell + 1, 0);
    for (int64_t i = a; i < b; ++i) {
        if (!std::isfinite(x[i]) || !std::isfinite(y[i])) continue;
        const int64_t cx = std::min<int64_t>(g.nx - 1, static_cast<int64_t>((x[i] - g.x0) / cs));
        const int64_t cy = std::min<int64_t>(g.ny - 1, static_cast<int64_t>((y[i] - g.y0) / cs));
        cell[i - a] = cy * g.nx + cx;
        ++g.start[cell[i - a] + 1];
    }
    for (int64_t c = 0; c < ncell; ++c) g.start[c + 1] += g.start[c];
    g.order.resize(g.start[ncell]);
    std::vector<int64_t> fill(g.start.begin(), g.start.end() - 1);
    for (int64_t i = a; i < b; ++i)
        if (cell[i - a] >= 0) g.order[fill[cell[i - a]]++] = i;
    return g;
}

using Out = std::tuple<DArray, DArray, DArray, IArray>;

Out make_out(int64_t nv, int64_t nq) {
    std::vector<py::ssize_t> s2{static_cast<py::ssize_t>(nv), static_cast<py::ssize_t>(nq)};
    return Out(DArray(s2), DArray(s2), DArray(s2), IArray(std::vector<py::ssize_t>{static_cast<py::ssize_t>(nq)}));
}

void finish(const std::vector<Sums>& acc, int64_t nv, int64_t q, int64_t nq, double* mean, double* sd,
            double* count) {
    for (int64_t v = 0; v < nv; ++v) {
        const Sums& s = acc[v];
        const int64_t k = v * nq + q;
        count[k] = static_cast<double>(s.n);
        if (s.n == 0 || !(s.w > 0.0)) {
            mean[k] = kNaN;
            sd[k] = kNaN;
        } else {
            mean[k] = s.wt / s.w;
            const double m = s.ws / s.w;
            sd[k] = std::sqrt(std::max(0.0, s.wss / s.w - m * m));
        }
    }
}

// Gates of all sweeps: gx, gy (n); offsets (nsweep + 1) delimit the sweeps; values
// (nv, n); modes (nv); queries (qx, qy, qs) with qs the sweep of a query.
Out match_gates(const DArray& gx, const DArray& gy, const IArray& offsets, const DArray& values,
                const IArray& modes, const DArray& qx, const DArray& qy, const IArray& qs,
                double radius, bool gaussian, int n_threads) {
    if (gx.ndim() != 1 || gy.ndim() != 1 || gx.size() != gy.size())
        throw std::invalid_argument("gx and gy must be 1-D with the same length");
    if (values.ndim() != 2 || values.shape(1) != gx.size())
        throw std::invalid_argument("values must have the shape (variables, gates)");
    const int64_t n = gx.size(), nv = values.shape(0), nq = qx.size();
    const int64_t ns = offsets.size() - 1;
    if (ns < 0 || modes.size() != nv) throw std::invalid_argument("offsets or modes have the wrong length");
    if (qy.size() != nq || qs.size() != nq) throw std::invalid_argument("qx, qy and qs must have one value per query");
    if (!(radius > 0.0)) throw std::invalid_argument("radius must be positive");
    const int64_t* off = offsets.data();
    if (ns >= 0 && (off[0] != 0 || off[ns] != n)) throw std::invalid_argument("offsets must run from 0 to the number of gates");
    for (int64_t s = 0; s < ns; ++s)
        if (off[s + 1] < off[s]) throw std::invalid_argument("offsets must not decrease");
    for (int64_t q = 0; q < nq; ++q)
        if (qs.data()[q] < 0 || qs.data()[q] >= ns) throw std::invalid_argument("sweep index out of range");
    Out out = make_out(nv, nq);
    double* mean = std::get<0>(out).mutable_data();
    double* sd = std::get<1>(out).mutable_data();
    double* cnt = std::get<2>(out).mutable_data();
    int64_t* total = std::get<3>(out).mutable_data();
    const double *px = gx.data(), *py_ = gy.data(), *pv = values.data(), *pqx = qx.data(), *pqy = qy.data();
    const int64_t *md = modes.data(), *pqs = qs.data();
    {
        py::gil_scoped_release release;
        std::vector<Buckets> grids(ns);
        parallel_blocks(ns, 1, n_threads, [&](int64_t s0, int64_t s1) {
            for (int64_t s = s0; s < s1; ++s) grids[s] = build(px, py_, off[s], off[s + 1], radius);
        });
        const double r2 = radius * radius;
        parallel_blocks(nq, kBlock, n_threads, [&](int64_t q0, int64_t q1) {
            std::vector<Sums> acc(nv);
            for (int64_t q = q0; q < q1; ++q) {
                std::fill(acc.begin(), acc.end(), Sums());
                int64_t seen = 0;
                const Buckets& g = grids[pqs[q]];
                const double x = pqx[q], y = pqy[q];
                if (g.order.empty() || !std::isfinite(x) || !std::isfinite(y)) {
                    total[q] = 0;
                    finish(acc, nv, q, nq, mean, sd, cnt);
                    continue;
                }
                const int64_t ix0 = static_cast<int64_t>(std::floor((x - radius - g.x0) / g.cs));
                const int64_t ix1 = static_cast<int64_t>(std::floor((x + radius - g.x0) / g.cs));
                const int64_t iy0 = static_cast<int64_t>(std::floor((y - radius - g.y0) / g.cs));
                const int64_t iy1 = static_cast<int64_t>(std::floor((y + radius - g.y0) / g.cs));
                for (int64_t cy = std::max<int64_t>(iy0, 0); cy <= std::min<int64_t>(iy1, g.ny - 1); ++cy)
                    for (int64_t cx = std::max<int64_t>(ix0, 0); cx <= std::min<int64_t>(ix1, g.nx - 1); ++cx) {
                        const int64_t c = cy * g.nx + cx;
                        for (int64_t k = g.start[c]; k < g.start[c + 1]; ++k) {
                            const int64_t i = g.order[k];
                            const double dx = px[i] - x, dy = py_[i] - y;
                            const double d2 = dx * dx + dy * dy;
                            if (d2 > r2) continue;
                            ++seen;
                            const double w = gaussian ? std::exp(-kLn2 * d2 / r2) : 1.0;
                            for (int64_t v = 0; v < nv; ++v) {
                                const double val = pv[v * n + i];
                                if (std::isfinite(val)) acc[v].add(val, w, static_cast<int>(md[v]));
                            }
                        }
                    }
                total[q] = seen;
                finish(acc, nv, q, nq, mean, sd, cnt);
            }
        });
    }
    return out;
}

// Range bins of the rays: z (nray, nbin) heights, values (nv, nray, nbin); queries
// (qr, zlo, zhi, zc) select the bins of ray qr with zlo <= z <= zhi; with gaussian
// the weight is exp(-ln2 ((z - zc) / h)^2), h = (zhi - zlo) / 2.
Out match_bins(const DArray& z, const DArray& values, const IArray& modes, const IArray& qr,
               const DArray& zlo, const DArray& zhi, const DArray& zc, bool gaussian, int n_threads) {
    if (z.ndim() != 2) throw std::invalid_argument("z must have the shape (rays, bins)");
    if (values.ndim() != 3 || values.shape(1) != z.shape(0) || values.shape(2) != z.shape(1))
        throw std::invalid_argument("values must have the shape (variables, rays, bins)");
    const int64_t nray = z.shape(0), nbin = z.shape(1), nv = values.shape(0), nq = qr.size();
    if (modes.size() != nv) throw std::invalid_argument("modes must have one value per variable");
    if (zlo.size() != nq || zhi.size() != nq || zc.size() != nq)
        throw std::invalid_argument("zlo, zhi and zc must have one value per query");
    for (int64_t q = 0; q < nq; ++q)
        if (qr.data()[q] < 0 || qr.data()[q] >= nray) throw std::invalid_argument("ray index out of range");
    Out out = make_out(nv, nq);
    double* mean = std::get<0>(out).mutable_data();
    double* sd = std::get<1>(out).mutable_data();
    double* cnt = std::get<2>(out).mutable_data();
    int64_t* total = std::get<3>(out).mutable_data();
    const double *pz = z.data(), *pv = values.data(), *plo = zlo.data(), *phi = zhi.data(), *pc = zc.data();
    const int64_t *md = modes.data(), *pr = qr.data();
    {
        py::gil_scoped_release release;
        parallel_blocks(nq, kBlock, n_threads, [&](int64_t q0, int64_t q1) {
            std::vector<Sums> acc(nv);
            for (int64_t q = q0; q < q1; ++q) {
                std::fill(acc.begin(), acc.end(), Sums());
                int64_t seen = 0;
                const int64_t r = pr[q];
                const double h = 0.5 * (phi[q] - plo[q]);
                for (int64_t k = 0; k < nbin; ++k) {
                    const double h_k = pz[r * nbin + k];
                    if (!(h_k >= plo[q] && h_k <= phi[q])) continue;
                    ++seen;
                    double w = 1.0;
                    if (gaussian) {
                        const double u = h > 0.0 ? (h_k - pc[q]) / h : 0.0;
                        w = std::exp(-kLn2 * u * u);
                    }
                    for (int64_t v = 0; v < nv; ++v) {
                        const double val = pv[(v * nray + r) * nbin + k];
                        if (std::isfinite(val)) acc[v].add(val, w, static_cast<int>(md[v]));
                    }
                }
                total[q] = seen;
                finish(acc, nv, q, nq, mean, sd, cnt);
            }
        });
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(_satellite_match, m) {
    m.doc() = "Volume matching of a spaceborne radar and a ground radar";
    m.def("match_gates", &match_gates, py::arg("gx"), py::arg("gy"), py::arg("offsets"), py::arg("values"),
          py::arg("modes"), py::arg("qx"), py::arg("qy"), py::arg("qs"), py::arg("radius"),
          py::arg("gaussian") = false, py::arg("n_threads") = 0);
    m.def("match_bins", &match_bins, py::arg("z"), py::arg("values"), py::arg("modes"), py::arg("qr"),
          py::arg("zlo"), py::arg("zhi"), py::arg("zc"), py::arg("gaussian") = false,
          py::arg("n_threads") = 0);
}
