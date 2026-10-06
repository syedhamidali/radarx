// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Disdrometer kernel: per-spectrum work for radarx.retrieve.disdrometer.
//
// Every spectrum (time step, station) is independent, so all spectra of all
// stations form one pool of work that threads take in blocks from an atomic
// counter (dynamic scheduling, as in radarx/grid/_cone.cpp).
//
// - velocity_shift: the velocity correction of Raupach and Berne (2015):
//   per diameter class, the counts are split into sub-classes of width
//   `step`, shifted by a whole number of sub-classes so that the mean
//   velocity of the class matches the terminal fall speed, and regrouped
//   into the velocity classes (counts shifted outside them are lost).
// - number_concentration: N(D) = scale * sum_v counts * weight, where the
//   caller folds the quality-control mask and 1/V into `weight` and
//   1 / (area dD dt) and correction factors into `scale`.
// - fit_gamma: gamma DSD parameters from three moments M_i, M_j, M_k
//   (i < j < k) of measured spectra, either of the untruncated gamma DSD
//   (method of moments, bisection on mu) or of the gamma DSD truncated at
//   the largest observed diameter (optionally also the smallest; truncated
//   moments, damped Newton iteration on mu and log Lambda for the two log
//   moment ratios, from the untruncated fit).
//
// Every step follows the NumPy reference in radarx/retrieve/disdrometer.py,
// in the same order of operations.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

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

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kLnSqrt2Pi = 0.91893853320467274178;  // log(sqrt(2 pi))
constexpr int64_t kBlock = 64;                         // spectra per block
constexpr int kBisect = 64;                            // bisection steps
constexpr double kAccept = 1e-7;  // residual accepted when Newton stalls
// admissible log Lambda of the truncated fit (Lambda in 1e-3 .. 1e3 mm-1)
constexpr double kLogLamLo = -6.907755278982137, kLogLamHi = 6.907755278982137;

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
            const int64_t g1 = std::min(total, g0 + kBlock);
            for (int64_t g = g0; g < g1; ++g) body(g);
        }
    };
    std::vector<std::thread> pool;
    for (int i = 1; i < nt; ++i) pool.emplace_back(worker);
    worker();
    for (auto& th : pool) th.join();
}

// log Gamma(x) for x > 0 (Lanczos approximation, g = 7, 9 terms); written
// out because std::lgamma may write the global signgam (not thread safe).
inline double log_gamma(double x) {
    static const double c[9] = {0.99999999999980993,  676.5203681218851,
                                -1259.1392167224028,  771.32342877765313,
                                -176.61502916214059,  12.507343278686905,
                                -0.13857109526572012, 9.9843695780195716e-6,
                                1.5056327351493116e-7};
    if (x < 0.5) {
        // reflection: Gamma(x) Gamma(1 - x) = pi / sin(pi x), 0 < x < 0.5
        const double pi = 3.14159265358979323846;
        return std::log(pi / std::sin(pi * x)) - log_gamma(1.0 - x);
    }
    x -= 1.0;
    double a = c[0];
    const double t = x + 7.5;
    for (int i = 1; i < 9; ++i) a += c[i] / (x + i);
    return kLnSqrt2Pi + (x + 0.5) * std::log(t) - t + std::log(a);
}

// Regularized incomplete gamma functions P(a, x) and Q(a, x) = 1 - P(a, x),
// a > 0, x >= 0: power series of P for x < a + 1, else the continued
// fraction of Q (modified Lentz evaluation).
inline void gamma_pq(double a, double x, double* p, double* q) {
    if (!(x > 0.0)) {
        *p = 0.0;
        *q = 1.0;
        return;
    }
    if (std::isinf(x)) {
        *p = 1.0;
        *q = 0.0;
        return;
    }
    const double front = -x + a * std::log(x) - log_gamma(a);
    if (x < a + 1.0) {
        double term = 1.0 / a, sum = term;
        for (int n = 1; n < 1000; ++n) {
            term *= x / (a + n);
            sum += term;
            if (std::fabs(term) < std::fabs(sum) * 1e-16) break;
        }
        *p = std::min(1.0, sum * std::exp(front));
        *q = 1.0 - *p;
        return;
    }
    const double tiny = 1e-300;
    double b = x + 1.0 - a, c = 1.0 / tiny, d = 1.0 / b, h = d;
    for (int i = 1; i < 1000; ++i) {
        const double an = -i * (i - a);
        b += 2.0;
        d = an * d + b;
        if (std::fabs(d) < tiny) d = tiny;
        c = b + an / c;
        if (std::fabs(c) < tiny) c = tiny;
        d = 1.0 / d;
        const double del = d * c;
        h *= del;
        if (std::fabs(del - 1.0) < 1e-16) break;
    }
    *q = std::max(0.0, std::exp(front) * h);
    *p = 1.0 - *q;
}

// log of P(a, x1) - P(a, x0), 0 <= x0 < x1 (difference of Q where x0 is
// large, to avoid cancellation).
inline double log_gamma_window(double a, double x0, double x1) {
    double p0, q0, p1, q1;
    gamma_pq(a, x1, &p1, &q1);
    if (!(x0 > 0.0)) return std::log(p1);
    gamma_pq(a, x0, &p0, &q0);
    return std::log(x0 >= a + 1.0 ? q0 - q1 : p1 - p0);
}

struct Fit {
    int i, j, k;
    double mu_lo, mu_hi;
};

// log of the moment ratio M_j^(k-i) / (M_i^(k-j) M_k^(j-i)) of a gamma DSD
inline double ratio(const Fit& f, double mu) {
    return (f.k - f.i) * log_gamma(mu + f.j + 1) - (f.k - f.j) * log_gamma(mu + f.i + 1) -
           (f.j - f.i) * log_gamma(mu + f.k + 1);
}

// Untruncated gamma parameters (mu, Lambda [mm-1], ln N0) from three moments.
// Returns false where the moments are not those of a gamma DSD with mu
// inside (mu_lo, mu_hi).
inline bool solve(const Fit& f, double mi, double mj, double mk, double* mu, double* lam,
                  double* ln0) {
    if (!(mi > 0.0 && mj > 0.0 && mk > 0.0)) return false;
    const double target =
        (f.k - f.i) * std::log(mj) - (f.k - f.j) * std::log(mi) - (f.j - f.i) * std::log(mk);
    double lo = f.mu_lo, hi = f.mu_hi;
    if (!(target > ratio(f, lo) && target < ratio(f, hi))) return false;
    for (int it = 0; it < kBisect; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (ratio(f, mid) < target)
            lo = mid;
        else
            hi = mid;
    }
    const double m = 0.5 * (lo + hi);
    const double ll = (std::log(mi) + log_gamma(m + f.j + 1) - std::log(mj) -
                       log_gamma(m + f.i + 1)) /
                      (f.j - f.i);
    *mu = m;
    *lam = std::exp(ll);
    *ln0 = std::log(mi) + (m + f.i + 1) * ll - log_gamma(m + f.i + 1);
    return true;
}

// log M_n / N0 of the gamma DSD truncated to [d0, d1] at (mu, log Lambda)
inline double log_moment(int n, double mu, double ll, double d0, double d1) {
    const double a = mu + n + 1.0, lam = std::exp(ll);
    return log_gamma(a) - a * ll + log_gamma_window(a, lam * d0, lam * d1);
}

// Residuals of the log moment ratios ln(M_j/M_i), ln(M_k/M_j).
inline void residual(const Fit& f, double mu, double ll, double d0, double d1, double r1,
                     double r2, double* out) {
    const double li = log_moment(f.i, mu, ll, d0, d1), lj = log_moment(f.j, mu, ll, d0, d1),
                 lk = log_moment(f.k, mu, ll, d0, d1);
    out[0] = lj - li - r1;
    out[1] = lk - lj - r2;
}

// Truncated-moment fit by a damped Newton iteration on (mu, log Lambda)
// from the start (mu, lam). Returns the iterations, or -1 without a fit.
inline int solve_truncated(const Fit& f, double mi, double mj, double mk, double d0,
                           double d1, int max_iter, double tol, double* mu, double* lam,
                           double* ln0) {
    const double r1 = std::log(mj / mi), r2 = std::log(mk / mj);
    double m = *mu, ll = std::log(*lam), F[2], G[2], H[2];
    residual(f, m, ll, d0, d1, r1, r2, F);
    for (int it = 0; it <= max_iter; ++it) {
        if (!(std::isfinite(F[0]) && std::isfinite(F[1]))) return -1;
        if (std::max(std::fabs(F[0]), std::fabs(F[1])) < tol) {
            *mu = m;
            *lam = std::exp(ll);
            *ln0 = std::log(mi) - log_moment(f.i, m, ll, d0, d1);
            return it;
        }
        if (it == max_iter) break;
        const double h = 1e-6;
        residual(f, m + h, ll, d0, d1, r1, r2, G);
        residual(f, m, ll + h, d0, d1, r1, r2, H);
        const double a = (G[0] - F[0]) / h, b = (H[0] - F[0]) / h, c = (G[1] - F[1]) / h,
                     d = (H[1] - F[1]) / h, det = a * d - b * c;
        if (!(std::fabs(det) > 0.0) || !std::isfinite(det)) return -1;
        const double dm = -(d * F[0] - b * F[1]) / det;
        const double dl = -(-c * F[0] + a * F[1]) / det;
        const double norm = std::max(std::fabs(F[0]), std::fabs(F[1]));
        double t = 1.0, mn = m, ln_ = ll, N[2] = {F[0], F[1]};
        bool moved = false;
        for (int s = 0; s < 30; ++s, t *= 0.5) {
            mn = m + t * dm;
            ln_ = ll + t * dl;
            if (!(mn > f.mu_lo && mn < f.mu_hi && ln_ > kLogLamLo && ln_ < kLogLamHi)) continue;
            residual(f, mn, ln_, d0, d1, r1, r2, N);
            if (std::isfinite(N[0]) && std::isfinite(N[1]) &&
                std::max(std::fabs(N[0]), std::fabs(N[1])) < norm) {
                moved = true;
                break;
            }
        }
        if (!moved) {
            // the line search stalls at the precision of the moment ratios:
            // accept a residual below kAccept
            if (norm < kAccept) {
                *mu = m;
                *lam = std::exp(ll);
                *ln0 = std::log(mi) - log_moment(f.i, m, ll, d0, d1);
                return it + 1;
            }
            return -1;
        }
        m = mn;
        ll = ln_;
        F[0] = N[0];
        F[1] = N[1];
    }
    return -1;
}

void check(const DArray& a, std::vector<py::ssize_t> shape, const char* name) {
    if (a.ndim() != static_cast<py::ssize_t>(shape.size()))
        throw std::invalid_argument(std::string(name) + " has the wrong number of dimensions");
    for (size_t d = 0; d < shape.size(); ++d)
        if (a.shape(d) != shape[d])
            throw std::invalid_argument(std::string(name) + " has the wrong shape");
}

}  // namespace

// Velocity correction of Raupach and Berne (2015). counts (n, nv, nd),
// class edges v_lower, v_upper (nv), terminal fall speed vt (n, nd).
py::array_t<double> velocity_shift_py(const DArray& counts, const DArray& v_lower,
                                      const DArray& v_upper, const DArray& vt, double step,
                                      int n_threads) {
    if (counts.ndim() != 3) throw std::invalid_argument("counts must be (time, velocity, diameter)");
    const int64_t n = counts.shape(0), nv = counts.shape(1), nd = counts.shape(2);
    check(v_lower, {nv}, "v_lower");
    check(v_upper, {nv}, "v_upper");
    check(vt, {n, nd}, "vt");
    if (!(step > 0.0)) throw std::invalid_argument("step must be positive");
    const double *pc = counts.data(), *lo = v_lower.data(), *up = v_upper.data(),
                 *pvt = vt.data();
    std::vector<int64_t> nsub(nv);
    for (int64_t v = 0; v < nv; ++v) {
        nsub[v] = std::max<int64_t>(1, std::llround((up[v] - lo[v]) / step));
        if (v > 0 && !(lo[v] >= lo[v - 1]))
            throw std::invalid_argument("velocity classes must be increasing");
    }
    py::array_t<double> out(std::vector<py::ssize_t>{n, nv, nd});
    double* po = out.mutable_data();
    {
        py::gil_scoped_release release;
        parallel_blocks(n * nd, n_threads, [&](int64_t g) {
            const int64_t t = g / nd, i = g % nd;
            const double* c = pc + t * nv * nd + i;
            double* o = po + t * nv * nd + i;
            double total = 0.0, sum = 0.0;
            for (int64_t v = 0; v < nv; ++v) {
                o[v * nd] = 0.0;
                total += c[v * nd];
                sum += c[v * nd] * 0.5 * (lo[v] + up[v]);
            }
            const double target = pvt[t * nd + i];
            if (!(total > 0.0) || !std::isfinite(target)) {
                for (int64_t v = 0; v < nv; ++v) o[v * nd] = c[v * nd];
                return;
            }
            const double shift = std::round((target - sum / total) / step) * step;
            for (int64_t v = 0; v < nv; ++v) {
                if (c[v * nd] == 0.0) continue;
                const double part = c[v * nd] / static_cast<double>(nsub[v]);
                for (int64_t s = 0; s < nsub[v]; ++s) {
                    const double x = lo[v] + (s + 0.5) * step + shift;
                    if (x < lo[0] || x >= up[nv - 1]) continue;  // shifted out
                    const int64_t w =
                        static_cast<int64_t>(std::upper_bound(lo, lo + nv, x) - lo) - 1;
                    if (x < up[w]) o[w * nd] += part;
                }
            }
        });
    }
    return out;
}

// N(D) [m-3 mm-1] = scale[t, i] * sum_v counts[t, v, i] * weight[w, v, i]
// with w = t if weight has one entry per spectrum, else 0.
py::array_t<double> number_concentration_py(const DArray& counts, const DArray& weight,
                                            const DArray& scale, int n_threads) {
    if (counts.ndim() != 3) throw std::invalid_argument("counts must be (time, velocity, diameter)");
    const int64_t n = counts.shape(0), nv = counts.shape(1), nd = counts.shape(2);
    if (weight.ndim() != 3 || (weight.shape(0) != 1 && weight.shape(0) != n) ||
        weight.shape(1) != nv || weight.shape(2) != nd)
        throw std::invalid_argument("weight must be (1 or time, velocity, diameter)");
    check(scale, {n, nd}, "scale");
    const bool per_time = weight.shape(0) == n && n != 1;
    const double *pc = counts.data(), *pw = weight.data(), *ps = scale.data();
    py::array_t<double> out(std::vector<py::ssize_t>{n, nd});
    double* po = out.mutable_data();
    {
        py::gil_scoped_release release;
        parallel_blocks(n, n_threads, [&](int64_t t) {
            const double* c = pc + t * nv * nd;
            const double* w = pw + (per_time ? t * nv * nd : 0);
            double* o = po + t * nd;
            for (int64_t i = 0; i < nd; ++i) o[i] = 0.0;
            for (int64_t v = 0; v < nv; ++v)  // rows are contiguous
                for (int64_t i = 0; i < nd; ++i) o[i] += c[v * nd + i] * w[v * nd + i];
            for (int64_t i = 0; i < nd; ++i) o[i] *= ps[t * nd + i];
        });
    }
    return out;
}

// Gamma fit of spectra nd (n, m) on bins with centres d, widths dd and edges
// lower, upper (m). Returns (4, n): log10 N0, mu, Lambda, iterations (0 for
// the untruncated fit; NaN where no fit).
py::array_t<double> fit_gamma_py(const DArray& nd, const DArray& d, const DArray& dd,
                                 const DArray& lower, const DArray& upper, int i, int j, int k,
                                 bool truncated, bool lower_cut, double mu_lo, double mu_hi,
                                 int max_iter, double tol, int n_threads) {
    if (nd.ndim() != 2) throw std::invalid_argument("nd must be (spectrum, diameter)");
    const int64_t n = nd.shape(0), m = nd.shape(1);
    check(d, {m}, "d");
    check(dd, {m}, "dd");
    check(lower, {m}, "lower");
    check(upper, {m}, "upper");
    if (!(0 <= i && i < j && j < k)) throw std::invalid_argument("moments must be 0 <= i < j < k");
    if (!(mu_lo > -(i + 1.0) && mu_hi > mu_lo))
        throw std::invalid_argument("mu range must satisfy -(i + 1) < mu_lo < mu_hi");
    const Fit f{i, j, k, mu_lo, mu_hi};
    const double *pn = nd.data(), *pd = d.data(), *pdd = dd.data(), *pl = lower.data(),
                 *pu = upper.data();
    py::array_t<double> out(std::vector<py::ssize_t>{4, n});
    double* po = out.mutable_data();
    {
        py::gil_scoped_release release;
        parallel_blocks(n, n_threads, [&](int64_t t) {
            const double* s = pn + t * m;
            double mi = 0.0, mj = 0.0, mk = 0.0;
            int64_t first = -1, last = -1;
            for (int64_t b = 0; b < m; ++b) {
                const double w = s[b] * pdd[b];
                mi += w * std::pow(pd[b], i);
                mj += w * std::pow(pd[b], j);
                mk += w * std::pow(pd[b], k);
                if (s[b] > 0.0) {
                    if (first < 0) first = b;
                    last = b;
                }
            }
            double mu = kNaN, lam = kNaN, ln0 = kNaN, iters = kNaN;
            const bool ok = solve(f, mi, mj, mk, &mu, &lam, &ln0);
            if (ok) iters = 0.0;
            if (truncated && mi > 0.0 && mj > 0.0 && mk > 0.0) {
                if (!ok) {  // start from mu = 0 with Lambda from M_j / M_i
                    mu = 0.0;
                    lam = std::exp((log_gamma(j + 1.0) - log_gamma(i + 1.0) -
                                    std::log(mj / mi)) /
                                   (j - i));
                }
                const double d0 = lower_cut ? pl[first] : 0.0, d1 = pu[last];
                const int it =
                    solve_truncated(f, mi, mj, mk, d0, d1, max_iter, tol, &mu, &lam, &ln0);
                if (it < 0)
                    mu = lam = ln0 = iters = kNaN;
                else
                    iters = it;
            }
            po[t] = ln0 / std::log(10.0);
            po[n + t] = mu;
            po[2 * n + t] = lam;
            po[3 * n + t] = iters;
        });
    }
    return out;
}

PYBIND11_MODULE(_disdrometer, mod) {
    mod.doc() = "Compiled disdrometer kernel for radarx.";
    mod.def("velocity_shift", &velocity_shift_py, py::arg("counts"), py::arg("v_lower"),
            py::arg("v_upper"), py::arg("vt"), py::arg("step"), py::arg("n_threads") = 0);
    mod.def("number_concentration", &number_concentration_py, py::arg("counts"),
            py::arg("weight"), py::arg("scale"), py::arg("n_threads") = 0);
    mod.def("fit_gamma", &fit_gamma_py, py::arg("nd"), py::arg("d"), py::arg("dd"),
            py::arg("lower"), py::arg("upper"), py::arg("i"), py::arg("j"), py::arg("k"),
            py::arg("truncated"), py::arg("lower_cut"), py::arg("mu_lo"), py::arg("mu_hi"), py::arg("max_iter"),
            py::arg("tol"), py::arg("n_threads") = 0);
}
