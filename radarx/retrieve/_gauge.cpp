// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Radar-gauge merging: the empirical variogram of the gauges, ordinary
// kriging and kriging with an external drift (the radar field) on any set of
// target points, the Brandes (1975) spatial adjustment and the mean field
// bias.
//
// Kriging in the variogram form of Wackernagel (2003, ch. 11 and 22): for the
// gauges i = 1..m near the target 0
//
//     sum_j w_j g(h_ij) + mu + lambda f_i = g(h_i0)      (i = 1..m)
//     sum_j w_j = 1,   sum_j w_j f_j = f_0               (drift only)
//
// with the variogram g (g(0) = 0), the Lagrange multipliers mu and lambda and
// the drift f (the radar value). The estimate is sum w_i z_i and the kriging
// variance sum w_i g(h_i0) + mu + lambda f_0 (the dot product of the solution
// and the right-hand side). The system is solved by LU decomposition with
// partial pivoting. With all gauges as neighbours it is factorized once per
// group of values that share it; with a limited number of neighbours every
// target solves a small system. Targets are independent: threads take blocks
// of them from an atomic counter. Every step follows the NumPy reference in
// radarx/retrieve/gauge.py.
//
// Variogram models (code: shape), with the practical range a:
//   0 linear         g = c0 + s h                  (slope s in `psill`)
//   1 spherical      g = c0 + c (1.5 h/a - 0.5 (h/a)^3), c0 + c for h >= a
//   2 exponential    g = c0 + c (1 - exp(-3 h / a))
//   3 gaussian       g = c0 + c (1 - exp(-3 h^2 / a^2))
// and g(0) = 0 for every model.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace py = pybind11;
using DArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using IArray = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr int64_t kBlock = 256;  // targets per scheduling block

int thread_count(int64_t blocks, int n_threads) {
    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    int nt = n_threads > 0 ? n_threads : static_cast<int>(hw);
    return static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(nt, blocks)));
}

// body(worker, begin, end) for blocks of [0, total) taken from an atomic counter.
template <class F>
void parallel_blocks(int64_t total, int64_t block, int n_threads, F&& body) {
    if (total <= 0) return;
    const int64_t nblocks = (total + block - 1) / block;
    const int nt = thread_count(nblocks, n_threads);
    std::atomic<int64_t> next{0};
    auto worker = [&](int id) {
        for (;;) {
            const int64_t b0 = next.fetch_add(block);
            if (b0 >= total) break;
            body(id, b0, std::min(total, b0 + block));
        }
    };
    std::vector<std::thread> pool;
    for (int i = 1; i < nt; ++i) pool.emplace_back(worker, i);
    worker(0);
    for (auto& th : pool) th.join();
}

struct Vario {
    int model;
    double nug, psill, range;
};

inline double semivariance(const Vario& v, double h) {
    if (h <= 0.0) return 0.0;
    switch (v.model) {
        case 0:
            return v.nug + v.psill * h;
        case 1: {
            if (h >= v.range) return v.nug + v.psill;
            const double x = h / v.range;
            return v.nug + v.psill * (1.5 * x - 0.5 * x * x * x);
        }
        case 2:
            return v.nug + v.psill * (1.0 - std::exp(-3.0 * h / v.range));
        default: {
            const double x = h / v.range;
            return v.nug + v.psill * (1.0 - std::exp(-3.0 * x * x));
        }
    }
}

// In-place LU decomposition with partial pivoting (row-major n x n). False if
// a pivot is negligible against the largest entry.
bool lu_factor(std::vector<double>& a, int n, std::vector<int>& piv) {
    double scale = 0.0;
    for (double x : a) scale = std::max(scale, std::fabs(x));
    if (scale == 0.0) return false;
    const double tol = 1e-12 * scale;
    piv.assign(n, 0);
    for (int k = 0; k < n; ++k) {
        int p = k;
        double best = std::fabs(a[static_cast<size_t>(k) * n + k]);
        for (int i = k + 1; i < n; ++i) {
            const double v = std::fabs(a[static_cast<size_t>(i) * n + k]);
            if (v > best) {
                best = v;
                p = i;
            }
        }
        if (!(best > tol)) return false;
        piv[k] = p;
        if (p != k)
            for (int j = 0; j < n; ++j)
                std::swap(a[static_cast<size_t>(k) * n + j], a[static_cast<size_t>(p) * n + j]);
        const double d = a[static_cast<size_t>(k) * n + k];
        for (int i = k + 1; i < n; ++i) {
            double* row = &a[static_cast<size_t>(i) * n];
            const double f = row[k] / d;
            row[k] = f;
            if (f != 0.0) {
                const double* krow = &a[static_cast<size_t>(k) * n];
                for (int j = k + 1; j < n; ++j) row[j] -= f * krow[j];
            }
        }
    }
    return true;
}

// Solve with the factors of lu_factor, b is overwritten by the solution.
void lu_solve(const std::vector<double>& a, const std::vector<int>& piv, int n, double* b) {
    for (int k = 0; k < n; ++k)
        if (piv[k] != k) std::swap(b[k], b[piv[k]]);
    for (int k = 0; k < n; ++k) {
        const double bk = b[k];
        if (bk != 0.0)
            for (int i = k + 1; i < n; ++i) b[i] -= a[static_cast<size_t>(i) * n + k] * bk;
    }
    for (int i = n - 1; i >= 0; --i) {
        double s = b[i];
        const double* row = &a[static_cast<size_t>(i) * n];
        for (int j = i + 1; j < n; ++j) s -= row[j] * b[j];
        b[i] = s / row[i];
    }
}

// Kriging matrix of the gauges `idx` (m of them), with a drift column if f != nullptr
// (f has one value per entry of idx).
void build_matrix(const std::vector<int>& idx, const double* xg, const double* yg,
                  const std::vector<double>* f, const Vario& v, std::vector<double>& a, int& n) {
    const int m = static_cast<int>(idx.size());
    n = m + 1 + (f != nullptr ? 1 : 0);
    a.assign(static_cast<size_t>(n) * n, 0.0);
    for (int i = 0; i < m; ++i) {
        for (int j = i + 1; j < m; ++j) {
            const double dx = xg[idx[i]] - xg[idx[j]], dy = yg[idx[i]] - yg[idx[j]];
            const double g = semivariance(v, std::sqrt(dx * dx + dy * dy));
            a[static_cast<size_t>(i) * n + j] = g;
            a[static_cast<size_t>(j) * n + i] = g;
        }
        a[static_cast<size_t>(i) * n + m] = 1.0;
        a[static_cast<size_t>(m) * n + i] = 1.0;
        if (f != nullptr) {
            a[static_cast<size_t>(i) * n + m + 1] = (*f)[i];
            a[static_cast<size_t>(m + 1) * n + i] = (*f)[i];
        }
    }
}

void build_rhs(const std::vector<int>& idx, const double* xg, const double* yg, double x0,
               double y0, double f0, bool drift, const Vario& v, std::vector<double>& b) {
    const int m = static_cast<int>(idx.size());
    b.assign(m + 1 + (drift ? 1 : 0), 0.0);
    for (int i = 0; i < m; ++i) {
        const double dx = xg[idx[i]] - x0, dy = yg[idx[i]] - y0;
        b[i] = semivariance(v, std::sqrt(dx * dx + dy * dy));
    }
    b[m] = 1.0;
    if (drift) b[m + 1] = f0;
}

// True if the drift is constant to a relative tolerance among the gauges `idx`: the
// drift system is then singular.
inline bool drift_degenerate(const double* f, const std::vector<int>& idx) {
    double lo = std::numeric_limits<double>::infinity(), hi = -lo, top = 0.0;
    for (int i : idx) {
        lo = std::min(lo, f[i]);
        hi = std::max(hi, f[i]);
        top = std::max(top, std::fabs(f[i]));
    }
    return (hi - lo) <= 1e-9 * top;
}

// Drift of the gauges `idx` shifted and scaled to zero mean and unit variance. The kriging
// weights do not depend on this affine change; it keeps the system well conditioned.
// Returns the mean and the standard deviation used for the drift at the target.
void normalise_drift(const double* f, const std::vector<int>& idx, std::vector<double>& fn,
                     double& mean, double& sd) {
    const size_t m = idx.size();
    mean = 0.0;
    for (int i : idx) mean += f[i];
    mean /= static_cast<double>(m);
    double ss = 0.0;
    for (int i : idx) ss += (f[i] - mean) * (f[i] - mean);
    sd = std::sqrt(ss / static_cast<double>(m));
    fn.resize(m);
    for (size_t k = 0; k < m; ++k) fn[k] = (f[idx[k]] - mean) / sd;
}

// The m gauges nearest to (x0, y0), ties broken by the lower index.
void nearest(const double* xg, const double* yg, int64_t ng, double x0, double y0, int m,
             std::vector<std::pair<double, int>>& scratch, std::vector<int>& idx) {
    scratch.resize(ng);
    for (int64_t i = 0; i < ng; ++i) {
        const double dx = xg[i] - x0, dy = yg[i] - y0;
        scratch[i] = {dx * dx + dy * dy, static_cast<int>(i)};
    }
    std::partial_sort(scratch.begin(), scratch.begin() + m, scratch.end());
    idx.resize(m);
    for (int i = 0; i < m; ++i) idx[i] = scratch[i].second;
}

}  // namespace

// Empirical (Matheron) semivariance of nv value sets at the same ng points, in
// n_lags equal bins of width max_lag / n_lags. Returns the mean distance of
// the pairs of every bin, the semivariance of every set and the pair counts.
py::tuple variogram_py(const DArray& x, const DArray& y, const DArray& vals, int64_t n_lags,
                       double max_lag, int n_threads) {
    if (vals.ndim() != 2) throw std::invalid_argument("values must have 2 dimensions");
    const int64_t nv = vals.shape(0), ng = vals.shape(1);
    if (x.size() != ng || y.size() != ng)
        throw std::invalid_argument("x and y must have one value per gauge");
    if (n_lags < 1 || !(max_lag > 0.0))
        throw std::invalid_argument("n_lags must be positive and max_lag greater than 0");
    const double* px = x.data();
    const double* py_ = y.data();
    const double* pv = vals.data();
    const double width = max_lag / static_cast<double>(n_lags);
    // Rows of gauges go to threads in blocks; every block keeps its own sums and the blocks
    // are added in order, so the result does not depend on the number of threads.
    constexpr int64_t kRows = 16;
    const int64_t nblocks = (ng + kRows - 1) / kRows;
    std::vector<std::vector<double>> bd(nblocks), bg(nblocks);
    std::vector<std::vector<int64_t>> bc(nblocks);
    {
        py::gil_scoped_release release;
        parallel_blocks(nblocks, 1, n_threads, [&](int, int64_t b0, int64_t b1) {
            for (int64_t blk = b0; blk < b1; ++blk) {
                std::vector<double>& ds = bd[blk];
                std::vector<double>& gs = bg[blk];
                std::vector<int64_t>& cs = bc[blk];
                ds.assign(n_lags, 0.0);
                gs.assign(static_cast<size_t>(nv) * n_lags, 0.0);
                cs.assign(n_lags, 0);
                const int64_t i0 = blk * kRows, i1 = std::min(ng, i0 + kRows);
                for (int64_t i = i0; i < i1; ++i) {
                    for (int64_t j = i + 1; j < ng; ++j) {
                        const double dx = px[i] - px[j], dy = py_[i] - py_[j];
                        const double h = std::sqrt(dx * dx + dy * dy);
                        if (!(h < max_lag)) continue;
                        const int64_t b =
                            std::min<int64_t>(static_cast<int64_t>(h / width), n_lags - 1);
                        ds[b] += h;
                        cs[b] += 1;
                        for (int64_t t = 0; t < nv; ++t) {
                            const double d = pv[t * ng + i] - pv[t * ng + j];
                            gs[static_cast<size_t>(t) * n_lags + b] += d * d;
                        }
                    }
                }
            }
        });
    }
    std::vector<double> dist_sum(n_lags, 0.0), gam_sum(static_cast<size_t>(nv) * n_lags, 0.0);
    std::vector<int64_t> count(n_lags, 0);
    for (int64_t blk = 0; blk < nblocks; ++blk) {
        for (int64_t b = 0; b < n_lags; ++b) {
            dist_sum[b] += bd[blk][b];
            count[b] += bc[blk][b];
        }
        for (size_t k = 0; k < gam_sum.size(); ++k) gam_sum[k] += bg[blk][k];
    }
    py::array_t<double> lag(n_lags), gamma({nv, n_lags});
    py::array_t<int64_t> cnt(n_lags);
    double* pl = lag.mutable_data();
    double* pg = gamma.mutable_data();
    int64_t* pc = cnt.mutable_data();
    for (int64_t b = 0; b < n_lags; ++b) {
        pc[b] = count[b];
        pl[b] = count[b] > 0 ? dist_sum[b] / static_cast<double>(count[b]) : (b + 0.5) * width;
        for (int64_t t = 0; t < nv; ++t)
            pg[t * n_lags + b] = count[b] > 0 ? gam_sum[static_cast<size_t>(t) * n_lags + b] /
                                                    (2.0 * static_cast<double>(count[b]))
                                              : kNaN;
    }
    return py::make_tuple(lag, gamma, cnt);
}

// Kriging of nv value sets at the gauges (xg, yg) to the targets (xt, yt).
// `drift_g` (nv, ng) and `drift_t` (nv, npix) switch on the external drift
// (size 0: ordinary kriging). Value sets with the same `group` share the
// variogram parameters and, without drift, the kriging weights.
// With n_neighbours > 0 and fewer than ng, every target uses its nearest gauges.
// A singular drift system falls back to ordinary kriging; the number of such
// targets is returned for every value set. A target whose drift is NaN gets NaN.
py::tuple krige_py(const DArray& xg, const DArray& yg, const DArray& vals, const DArray& drift_g,
                   const DArray& xt, const DArray& yt, const DArray& drift_t, int model,
                   const DArray& params, const IArray& group, int64_t n_neighbours,
                   int n_threads) {
    if (vals.ndim() != 2) throw std::invalid_argument("values must have 2 dimensions");
    if (model < 0 || model > 3) throw std::invalid_argument("model must be 0, 1, 2 or 3");
    const int64_t nv = vals.shape(0), ng = vals.shape(1), npix = xt.size();
    if (xg.size() != ng || yg.size() != ng)
        throw std::invalid_argument("xg and yg must have one value per gauge");
    if (yt.size() != npix) throw std::invalid_argument("xt and yt must have the same size");
    if (params.ndim() != 2 || params.shape(0) != nv || params.shape(1) != 3)
        throw std::invalid_argument("params must have shape (n_values, 3)");
    if (group.size() != nv) throw std::invalid_argument("group must have one entry per value set");
    const bool drift = drift_g.size() > 0;
    if (drift && (drift_g.size() != nv * ng || drift_t.size() != nv * npix))
        throw std::invalid_argument("drift_g must be (n_values, ng) and drift_t (n_values, npix)");
    if (ng < 1) throw std::invalid_argument("at least one gauge is needed");

    std::map<int32_t, std::vector<int64_t>> groups;
    for (int64_t t = 0; t < nv; ++t) groups[group.data()[t]].push_back(t);
    if (drift)
        for (auto& kv : groups)
            if (kv.second.size() != 1)
                throw std::invalid_argument("with a drift every value set needs its own group");

    py::array_t<double> est({nv, npix}), var({nv, npix});
    py::array_t<int64_t> fallback(nv);
    double* pe = est.mutable_data();
    double* pvar = var.mutable_data();
    int64_t* pf = fallback.mutable_data();
    for (int64_t t = 0; t < nv; ++t) pf[t] = 0;
    const double* pxg = xg.data();
    const double* pyg = yg.data();
    const double* pxt = xt.data();
    const double* pyt = yt.data();
    const double* pval = vals.data();
    const bool local = n_neighbours > 0 && n_neighbours < ng;
    const int m_local = local ? static_cast<int>(n_neighbours) : static_cast<int>(ng);
    std::vector<int> all(ng);
    std::iota(all.begin(), all.end(), 0);

    {
        py::gil_scoped_release release;
        for (auto& kv : groups) {
            const std::vector<int64_t>& members = kv.second;
            const int64_t first = members[0];
            Vario vg{model, params.data()[first * 3], params.data()[first * 3 + 1],
                     params.data()[first * 3 + 2]};
            // The solution of the system does not change if the variogram is divided by a
            // constant (the multipliers and the variance scale with it): divide by its
            // largest value between the gauges to keep the matrix well conditioned.
            double gscale = 0.0;
            for (int64_t i = 0; i < ng; ++i)
                for (int64_t j = i + 1; j < ng; ++j) {
                    const double dx = pxg[i] - pxg[j], dy = pyg[i] - pyg[j];
                    gscale = std::max(gscale, semivariance(vg, std::sqrt(dx * dx + dy * dy)));
                }
            if (!(gscale > 0.0))
                throw std::runtime_error("the kriging system of the gauges is singular");
            vg.nug /= gscale;
            vg.psill /= gscale;
            const double* fg = drift ? drift_g.data() + first * ng : nullptr;
            const double* ft = drift ? drift_t.data() + first * npix : nullptr;

            // global neighbourhood: factorize once
            std::vector<double> A;
            std::vector<int> piv;
            int n = 0;
            bool global_ok = false;
            bool global_fallback = false;
            std::vector<double> A_ok;
            std::vector<int> piv_ok;
            int n_ok = 0;
            std::vector<double> fn_all;
            double f_mean = 0.0, f_sd = 1.0;
            if (!local) {
                if (drift && drift_degenerate(fg, all)) {
                    global_ok = false;
                } else {
                    if (drift) normalise_drift(fg, all, fn_all, f_mean, f_sd);
                    build_matrix(all, pxg, pyg, drift ? &fn_all : nullptr, vg, A, n);
                    global_ok = lu_factor(A, n, piv);
                }
                if (drift && !global_ok) {
                    global_fallback = true;
                    build_matrix(all, pxg, pyg, nullptr, vg, A_ok, n_ok);
                    if (!lu_factor(A_ok, n_ok, piv_ok))
                        throw std::runtime_error("the kriging system of the gauges is singular");
                } else if (!global_ok) {
                    throw std::runtime_error("the kriging system of the gauges is singular");
                }
            }
            std::atomic<bool> singular{false};
            std::vector<int64_t> local_fallback(static_cast<size_t>(thread_count(
                                                    (npix + kBlock - 1) / kBlock, n_threads)),
                                                0);
            parallel_blocks(npix, kBlock, n_threads, [&](int id, int64_t p0, int64_t p1) {
                std::vector<std::pair<double, int>> scratch;
                std::vector<int> idx;
                std::vector<double> B, M, fn_loc;
                std::vector<int> pv_;
                for (int64_t p = p0; p < p1 && !singular.load(); ++p) {
                    const double x0 = pxt[p], y0 = pyt[p];
                    double f0 = 0.0, fmean_p = f_mean, fsd_p = f_sd;
                    if (drift) {
                        f0 = ft[p];
                        if (std::isnan(f0)) {
                            for (int64_t t : members) {
                                pe[t * npix + p] = kNaN;
                                pvar[t * npix + p] = kNaN;
                            }
                            continue;
                        }
                    }
                    bool use_drift = drift;
                    const std::vector<double>* LUa = &A;
                    const std::vector<int>* LUp = &piv;
                    int ns = n;
                    const std::vector<int>* sel = &all;
                    if (!local) {
                        if (global_fallback) {
                            use_drift = false;
                            LUa = &A_ok;
                            LUp = &piv_ok;
                            ns = n_ok;
                            local_fallback[id] += 1;
                        }
                    } else {
                        nearest(pxg, pyg, ng, x0, y0, m_local, scratch, idx);
                        sel = &idx;
                        bool factored = false;
                        if (!drift || !drift_degenerate(fg, idx)) {
                            if (drift) normalise_drift(fg, idx, fn_loc, fmean_p, fsd_p);
                            build_matrix(idx, pxg, pyg, drift ? &fn_loc : nullptr, vg, M, ns);
                            factored = lu_factor(M, ns, pv_);
                        }
                        if (!factored) {
                            if (!drift) {
                                singular = true;
                                break;
                            }
                            use_drift = false;
                            local_fallback[id] += 1;
                            build_matrix(idx, pxg, pyg, nullptr, vg, M, ns);
                            if (!lu_factor(M, ns, pv_)) {
                                singular = true;
                                break;
                            }
                        }
                        LUa = &M;
                        LUp = &pv_;
                    }
                    build_rhs(*sel, pxg, pyg, x0, y0, (f0 - fmean_p) / fsd_p, use_drift, vg, B);
                    std::vector<double> rhs = B;
                    lu_solve(*LUa, *LUp, ns, rhs.data());
                    double s2 = 0.0;
                    for (size_t k = 0; k < B.size(); ++k) s2 += rhs[k] * B[k];
                    s2 = std::max(s2, 0.0) * gscale;
                    const int m = static_cast<int>(sel->size());
                    for (int64_t t : members) {
                        double z = 0.0;
                        for (int i = 0; i < m; ++i) z += rhs[i] * pval[t * ng + (*sel)[i]];
                        pe[t * npix + p] = z;
                        pvar[t * npix + p] = s2;
                    }
                }
            });
            if (singular.load())
                throw std::runtime_error("the kriging system of the gauges is singular");
            if (drift)
                for (int64_t c : local_fallback) pf[first] += c;
        }
    }
    return py::make_tuple(est, var, fallback);
}

// Brandes (1975) spatial adjustment: the factor at the targets is the mean of the
// factors at the gauges (nv sets; NaN excludes a gauge) with the weights
// exp(-d^2 / k). The weights are shifted by the smallest squared distance, which
// leaves the weighted mean unchanged and avoids underflow far from the gauges.
py::array_t<double> brandes_py(const DArray& xg, const DArray& yg, const DArray& ratio,
                               const DArray& xt, const DArray& yt, double k, int n_threads) {
    if (ratio.ndim() != 2) throw std::invalid_argument("ratio must have 2 dimensions");
    const int64_t nv = ratio.shape(0), ng = ratio.shape(1), npix = xt.size();
    if (xg.size() != ng || yg.size() != ng)
        throw std::invalid_argument("xg and yg must have one value per gauge");
    if (yt.size() != npix) throw std::invalid_argument("xt and yt must have the same size");
    if (!(k > 0.0)) throw std::invalid_argument("k must be positive");
    py::array_t<double> out({nv, npix});
    double* po = out.mutable_data();
    const double *pxg = xg.data(), *pyg = yg.data(), *pr = ratio.data();
    const double *pxt = xt.data(), *pyt = yt.data();
    {
        py::gil_scoped_release release;
        parallel_blocks(npix, kBlock, n_threads, [&](int, int64_t p0, int64_t p1) {
            std::vector<double> d2(ng);
            for (int64_t p = p0; p < p1; ++p) {
                for (int64_t i = 0; i < ng; ++i) {
                    const double dx = pxg[i] - pxt[p], dy = pyg[i] - pyt[p];
                    d2[i] = dx * dx + dy * dy;
                }
                for (int64_t t = 0; t < nv; ++t) {
                    double dmin = std::numeric_limits<double>::infinity();
                    for (int64_t i = 0; i < ng; ++i)
                        if (!std::isnan(pr[t * ng + i])) dmin = std::min(dmin, d2[i]);
                    if (!std::isfinite(dmin)) {
                        po[t * npix + p] = kNaN;
                        continue;
                    }
                    double sw = 0.0, sr = 0.0;
                    for (int64_t i = 0; i < ng; ++i) {
                        const double r = pr[t * ng + i];
                        if (std::isnan(r)) continue;
                        const double wi = std::exp(-(d2[i] - dmin) / k);
                        sw += wi;
                        sr += wi * r;
                    }
                    po[t * npix + p] = sr / sw;
                }
            }
        });
    }
    return out;
}

// Mean field bias sum(G) / sum(R) of the pairs where both values are finite and
// above min_value. Returns the factor (NaN without pairs) and the number of pairs.
py::tuple mean_field_bias_py(const DArray& radar, const DArray& gauge, double min_value,
                             int n_threads) {
    if (radar.ndim() != 2 || gauge.ndim() != 2 || radar.shape(0) != gauge.shape(0) ||
        radar.shape(1) != gauge.shape(1))
        throw std::invalid_argument("radar and gauge must be 2-D arrays of the same shape");
    const int64_t nv = radar.shape(0), ng = radar.shape(1);
    py::array_t<double> factor(nv);
    py::array_t<int64_t> pairs(nv);
    double* pf = factor.mutable_data();
    int64_t* pn = pairs.mutable_data();
    const double *pr = radar.data(), *pg = gauge.data();
    {
        py::gil_scoped_release release;
        parallel_blocks(nv, 1, n_threads, [&](int, int64_t t0, int64_t t1) {
            for (int64_t t = t0; t < t1; ++t) {
                double sr = 0.0, sg = 0.0;
                int64_t n = 0;
                for (int64_t i = 0; i < ng; ++i) {
                    const double r = pr[t * ng + i], g = pg[t * ng + i];
                    if (r > min_value && g > min_value) {  // false for NaN
                        sr += r;
                        sg += g;
                        ++n;
                    }
                }
                pn[t] = n;
                pf[t] = n > 0 ? sg / sr : kNaN;
            }
        });
    }
    return py::make_tuple(factor, pairs);
}

PYBIND11_MODULE(_gauge, m) {
    m.doc() = "Variogram, kriging, Brandes adjustment and mean field bias for radar-gauge merging";
    m.def("variogram", &variogram_py, py::arg("x"), py::arg("y"), py::arg("values"),
          py::arg("n_lags"), py::arg("max_lag"), py::arg("n_threads") = 0);
    m.def("krige", &krige_py, py::arg("xg"), py::arg("yg"), py::arg("values"),
          py::arg("drift_g"), py::arg("xt"), py::arg("yt"), py::arg("drift_t"),
          py::arg("model"), py::arg("params"), py::arg("group"), py::arg("n_neighbours"),
          py::arg("n_threads") = 0);
    m.def("brandes", &brandes_py, py::arg("xg"), py::arg("yg"), py::arg("ratio"),
          py::arg("xt"), py::arg("yt"), py::arg("k"), py::arg("n_threads") = 0);
    m.def("mean_field_bias", &mean_field_bias_py, py::arg("radar"), py::arg("gauge"),
          py::arg("min_value"), py::arg("n_threads") = 0);
}
