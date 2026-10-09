// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Bayesian drop size distribution retrieval kernel.
//
// The state of a gate is x = (t, Dm, mu) with t = log10 Nw. Dm and mu live on
// a fixed grid of nodes; for every node the forward model is linear in Nw:
//
//   Z_H [dBZ] = 10 t + L_j,   Z_DR = zdr_j,   K_DP = 10^t k_j,   A_H = 10^t a_j,
//
// and the prior is p(Dm_j, mu_j) N(t; pm_j, ps_j^2). Per gate and node the
// posterior in t is integrated by the Laplace method around its mode (found
// by Gauss-Newton; exact in one step without K_DP and A_H, where the
// posterior in t is Gaussian), which gives the posterior weight of the node
// and a Gaussian N(t; m_j, s_j^2). The posterior is the mixture over nodes.
//
// 1. pass 1: the closed-form log evidence of every node from Z_H, Z_DR and
//    the prior; nodes more than `prune` nats below the best are skipped;
// 2. pass 2: Laplace integration of the remaining nodes with all inputs;
// 3. posterior mean, standard deviation, MAP and quantiles of t, Dm, mu and
//    of log10 R, log10 W (rain rate and water content are 10^t times a node
//    constant). Dm and mu marginals are piecewise constant over the grid
//    cells; the others are Gaussian mixtures, whose quantiles are found by
//    safeguarded Newton iterations on the mixture CDF, with the normal CDF
//    and density interpolated linearly in a table.
//
// Every gate is independent: the gates of all inputs form one pool of work
// that threads take in blocks from an atomic counter. Every step follows the
// NumPy reference in radarx/retrieve/dsd_bayes.py in the same order.

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
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kLn10 = 2.30258509299404568402;
constexpr double kLog2Pi = 1.83787706640934548356;
constexpr int64_t kBlock = 64;  // gates per scheduling block
constexpr int kMaxNewton = 8;   // Gauss-Newton iterations in t
constexpr int kMaxQuantile = 40;
constexpr double kWeightMin = 1e-9;  // nodes ignored in the quantiles
constexpr int kNFixed = 16;          // outputs before the quantiles
// normal CDF and density table on [-kZMax, kZMax]
constexpr double kZMax = 8.5;
constexpr double kZRes = 256.0;  // points per unit
constexpr int64_t kNTab = static_cast<int64_t>(2 * kZMax * kZRes) + 1;

// Forward model and prior on the nodes (field-major, n nodes each).
enum Field { L = 0, ZDR, KDP, AH, LOGR, LOGW, DM, MU, LOGP, PMEAN, PSD, NFIELD };

struct Grid {
    const double* f[NFIELD];
    int64_t n = 0;  // nodes = n_dm * n_mu, node = i_dm * n_mu + i_mu
    int64_t n_dm = 0, n_mu = 0;
    double dm0 = 0, ddm = 0, mu0 = 0, dmu = 0;  // uniform axes
    std::vector<double> ivt, hlvt;  // pass 1: 1 / (zh2 + 100 ps^2), its half log
    std::vector<double> cdf, pdf;   // normal tables
};

struct Errors {
    double zh2, zdr2, kdp_abs, kdp_rel, ah_abs, ah_rel, prune;
};

struct Block {
    const double* z = nullptr;
    const double* zdr = nullptr;
    const double* kdp = nullptr;
    const double* ah = nullptr;
    const uint8_t* mask = nullptr;
    double* out = nullptr;
    int64_t n = 0, first = 0;
};

// per-thread scratch, allocated once
struct Work {
    std::vector<double> c, m, s, lw, marg_dm, marg_mu;
    std::vector<double> qw, qc, qis;  // compact mixture for the quantiles
    std::vector<int64_t> idx;
};

// normal CDF and density at z by linear interpolation in the tables
inline void normal(const Grid& g, double z, double& cdf, double& pdf) {
    if (z <= -kZMax) {
        cdf = 0.0;
        pdf = 0.0;
        return;
    }
    if (z >= kZMax) {
        cdf = 1.0;
        pdf = 0.0;
        return;
    }
    const double u = (z + kZMax) * kZRes;
    int64_t i = static_cast<int64_t>(u);
    if (i > kNTab - 2) i = kNTab - 2;
    const double f = u - static_cast<double>(i);
    cdf = g.cdf[i] + f * (g.cdf[i + 1] - g.cdf[i]);
    pdf = g.pdf[i] + f * (g.pdf[i + 1] - g.pdf[i]);
}

// Quantile q of the Gaussian mixture sum_k w_k N(m_k + off_k, s_k^2) within
// the bracket [lo, hi], starting at x.
double mixture_quantile(const Grid& g, const Work& wk, double q, double x, double lo,
                        double hi) {
    const size_t nk = wk.qw.size();
    const double* w = wk.qw.data();
    const double* c = wk.qc.data();
    const double* is = wk.qis.data();
    x = std::min(std::max(x, lo), hi);
    for (int it = 0; it < kMaxQuantile; ++it) {
        double cdf = 0.0, pdf = 0.0;
        for (size_t j = 0; j < nk; ++j) {
            double cj, pj;
            normal(g, (x - c[j]) * is[j], cj, pj);
            cdf += w[j] * cj;
            pdf += w[j] * pj * is[j];
        }
        const double r = cdf - q;
        if (std::fabs(r) < 1e-7) break;
        if (r > 0.0)
            hi = x;
        else
            lo = x;
        double xn = pdf > 0.0 ? x - r / pdf : 0.5 * (lo + hi);
        if (!(xn > lo && xn < hi)) xn = 0.5 * (lo + hi);
        const double dx = std::fabs(xn - x);
        x = xn;
        if (dx < 1e-7) break;
    }
    return x;
}

// Quantile q of a piecewise-constant density on a uniform axis.
double cell_quantile(const std::vector<double>& marg, int64_t n, double x0, double dx,
                     double q) {
    double acc = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double next = acc + marg[i];
        if (next >= q && marg[i] > 0.0) {
            const double frac = (q - acc) / marg[i];
            return x0 + (static_cast<double>(i) - 0.5 + frac) * dx;
        }
        acc = next;
    }
    return x0 + (static_cast<double>(n - 1) + 0.5) * dx;
}

void gate(const Grid& g, const Errors& e, const std::vector<double>& qs,
          const std::vector<double>& zq, const Block& b, int64_t k, Work& wk) {
    const int64_t nq = static_cast<int64_t>(qs.size());
    const int64_t nout = kNFixed + 5 * nq;
    double* out = b.out;
    const int64_t n = b.n;
    auto put = [&](int64_t i, double v) { out[i * n + k] = v; };
    auto empty = [&]() {
        for (int64_t i = 0; i < nout; ++i) put(i, kNaN);
    };
    const double z = b.z[k], zdr = b.zdr[k];
    if ((b.mask && !b.mask[k]) || !std::isfinite(z) || !std::isfinite(zdr)) return empty();
    double kobs = 0.0, aobs = 0.0, vk = 0.0, va = 0.0;
    bool use_k = false, use_a = false;
    if (b.kdp && std::isfinite(b.kdp[k])) {
        kobs = b.kdp[k];
        vk = e.kdp_abs * e.kdp_abs + (e.kdp_rel * kobs) * (e.kdp_rel * kobs);
        use_k = vk > 0.0;
    }
    if (b.ah && std::isfinite(b.ah[k])) {
        aobs = b.ah[k];
        va = e.ah_abs * e.ah_abs + (e.ah_rel * aobs) * (e.ah_rel * aobs);
        use_a = va > 0.0;
    }
    const double* L = g.f[Field::L];
    const double* ZD = g.f[Field::ZDR];
    const double* KK = g.f[Field::KDP];
    const double* AA = g.f[Field::AH];
    const double* LP = g.f[Field::LOGP];
    const double* PM = g.f[Field::PMEAN];
    const double* PS = g.f[Field::PSD];
    const double* LR = g.f[Field::LOGR];
    const double* LW = g.f[Field::LOGW];
    const double* DMv = g.f[Field::DM];
    const double* MUv = g.f[Field::MU];
    const double izdr2 = 1.0 / e.zdr2;

    // pass 1: closed-form log evidence from Z_H, Z_DR and the prior
    double cmax = -kInf;
    for (int64_t j = 0; j < g.n; ++j) {
        const double dz = zdr - ZD[j];
        const double rz = z - 10.0 * PM[j] - L[j];
        const double c =
            LP[j] - 0.5 * dz * dz * izdr2 - 0.5 * rz * rz * g.ivt[j] - g.hlvt[j];
        wk.c[j] = c;  // -inf for impossible nodes
        cmax = std::max(cmax, c);
    }
    if (!std::isfinite(cmax)) return empty();
    // pass 2: Laplace integration in t of the retained nodes
    wk.idx.clear();
    double lwmax = -kInf, best = -kInf;
    int64_t jbest = -1;
    const double cmin = cmax - e.prune;
    // offset of the mode from the Gaussian part at the previous node (warm
    // start along mu)
    double toff = 0.0;
    bool prev = false;
    for (int64_t j = 0; j < g.n; ++j) {
        if (j % g.n_mu == 0) prev = false;
        if (!(wk.c[j] >= cmin)) {
            prev = false;
            continue;
        }
        const double ip = 1.0 / (PS[j] * PS[j]);
        // Gaussian part (Z_H and prior): precision and mode
        const double h0 = 100.0 / e.zh2 + ip;
        const double t0 = (10.0 * (z - L[j]) / e.zh2 + PM[j] * ip) / h0;
        double t = t0;
        double h = h0;
        if (use_k || use_a) {
            if (prev) t = t0 + toff;
            for (int it = 0; it < kMaxNewton; ++it) {
                const double p10 = std::exp(kLn10 * t);
                double grad = -10.0 * (z - 10.0 * t - L[j]) / e.zh2 + (t - PM[j]) * ip;
                double hh = h0;
                if (use_k) {
                    const double jk = p10 * KK[j] * kLn10;
                    grad += (p10 * KK[j] - kobs) * jk / vk;
                    hh += jk * jk / vk;
                }
                if (use_a) {
                    const double ja = p10 * AA[j] * kLn10;
                    grad += (p10 * AA[j] - aobs) * ja / va;
                    hh += ja * ja / va;
                }
                const double step = std::min(std::max(grad / hh, -1.0), 1.0);
                t -= step;
                if (std::fabs(step) < 1e-8) break;
            }
            toff = t - t0;
            prev = true;
            // Hessian at the mode (with the second-order residual term, at
            // least a tenth of the Gauss-Newton one)
            const double p10 = std::exp(kLn10 * t);
            double hg = h0, hx = h0;
            if (use_k) {
                const double jk = p10 * KK[j] * kLn10;
                hg += jk * jk / vk;
                hx += (jk * jk + (p10 * KK[j] - kobs) * jk * kLn10) / vk;
            }
            if (use_a) {
                const double ja = p10 * AA[j] * kLn10;
                hg += ja * ja / va;
                hx += (ja * ja + (p10 * AA[j] - aobs) * ja * kLn10) / va;
            }
            h = std::max(hx, 0.1 * hg);
        }
        // negative log posterior at the mode (without the node prior)
        const double rz = z - 10.0 * t - L[j];
        const double dz = zdr - ZD[j];
        double f = 0.5 * rz * rz / e.zh2 + 0.5 * (t - PM[j]) * (t - PM[j]) * ip +
                   0.5 * dz * dz * izdr2 + std::log(PS[j]);
        if (use_k || use_a) {
            const double p10 = std::exp(kLn10 * t);
            if (use_k) {
                const double r = p10 * KK[j] - kobs;
                f += 0.5 * r * r / vk;
            }
            if (use_a) {
                const double r = p10 * AA[j] - aobs;
                f += 0.5 * r * r / va;
            }
        }
        const double peak = LP[j] - f;
        const double lw = peak + 0.5 * (kLog2Pi - std::log(h));
        wk.m[j] = t;
        wk.s[j] = 1.0 / std::sqrt(h);
        wk.lw[j] = lw;
        wk.idx.push_back(j);
        if (lw > lwmax) lwmax = lw;
        if (peak > best) {
            best = peak;
            jbest = j;
        }
    }
    // normalize
    double sum = 0.0;
    for (int64_t j : wk.idx) {
        wk.lw[j] = std::exp(wk.lw[j] - lwmax);
        sum += wk.lw[j];
    }
    // log evidence with the normalizations of the likelihoods and of the
    // prior in t (its log ps_j is in f)
    double lognorm = -0.5 * std::log(e.zh2) - 0.5 * std::log(e.zdr2) - kLog2Pi;
    if (use_k) lognorm -= 0.5 * (kLog2Pi + std::log(vk));
    if (use_a) lognorm -= 0.5 * (kLog2Pi + std::log(va));
    const double logev = lwmax + std::log(sum) + lognorm - 0.5 * kLog2Pi;
    std::fill(wk.marg_dm.begin(), wk.marg_dm.end(), 0.0);
    std::fill(wk.marg_mu.begin(), wk.marg_mu.end(), 0.0);
    double st = 0, st2 = 0, sd_ = 0, sd2 = 0, smu = 0, smu2 = 0, sr = 0, sr2 = 0, sw = 0,
           sw2 = 0, slr = 0, slr2 = 0, slw = 0, slw2 = 0;
    double tlo = kInf, thi = -kInf, rlo = kInf, rhi = -kInf, wlo = kInf, whi = -kInf;
    std::vector<int64_t>& use = wk.idx;
    size_t keep = 0;
    for (size_t i = 0; i < use.size(); ++i) {
        const int64_t j = use[i];
        const double w = wk.lw[j] / sum;
        wk.lw[j] = w;
        const double m = wk.m[j], s = wk.s[j], s2 = s * s;
        st += w * m;
        st2 += w * (m * m + s2);
        sd_ += w * DMv[j];
        sd2 += w * DMv[j] * DMv[j];
        smu += w * MUv[j];
        smu2 += w * MUv[j] * MUv[j];
        const double v = 0.5 * (s * kLn10) * (s * kLn10);
        const double ev = std::exp(v), ev4 = std::exp(4.0 * v);
        const double lr = m + LR[j], lwc = m + LW[j];
        const double r = std::exp(kLn10 * lr);
        const double ww = std::exp(kLn10 * lwc);
        sr += w * r * ev;
        sr2 += w * r * r * ev4;
        sw += w * ww * ev;
        sw2 += w * ww * ww * ev4;
        slr += w * lr;
        slr2 += w * (lr * lr + s2);
        slw += w * lwc;
        slw2 += w * (lwc * lwc + s2);
        wk.marg_dm[j / g.n_mu] += w;
        wk.marg_mu[j % g.n_mu] += w;
        if (w > kWeightMin) {
            use[keep++] = j;
            tlo = std::min(tlo, m - kZMax * s);
            thi = std::max(thi, m + kZMax * s);
            rlo = std::min(rlo, lr - kZMax * s);
            rhi = std::max(rhi, lr + kZMax * s);
            wlo = std::min(wlo, lwc - kZMax * s);
            whi = std::max(whi, lwc + kZMax * s);
        }
    }
    use.resize(keep);
    auto sdev = [](double m1, double m2) { return std::sqrt(std::max(m2 - m1 * m1, 0.0)); };
    put(0, st);
    put(1, sdev(st, st2));
    put(2, sd_);
    put(3, std::sqrt(std::max(sd2 - sd_ * sd_, 0.0) + g.ddm * g.ddm / 12.0));
    put(4, smu);
    put(5, std::sqrt(std::max(smu2 - smu * smu, 0.0) + g.dmu * g.dmu / 12.0));
    put(6, sr);
    put(7, sdev(sr, sr2));
    put(8, sw);
    put(9, sdev(sw, sw2));
    // MAP and its misfit
    const double tb = wk.m[jbest];
    put(10, tb);
    put(11, DMv[jbest]);
    put(12, MUv[jbest]);
    {
        const double pb = std::pow(10.0, tb);
        const double rz = z - 10.0 * tb - L[jbest];
        const double dz = zdr - ZD[jbest];
        double chi2 = rz * rz / e.zh2 + dz * dz * izdr2;
        if (use_k) {
            const double r = pb * KK[jbest] - kobs;
            chi2 += r * r / vk;
        }
        if (use_a) {
            const double r = pb * AA[jbest] - aobs;
            chi2 += r * r / va;
        }
        put(13, chi2);
    }
    put(14, logev);
    put(15, 2.0 + (use_k ? 1.0 : 0.0) + (use_a ? 1.0 : 0.0));
    // quantiles, starting from the normal approximation of each mixture
    const double sdt = sdev(st, st2), sdr = sdev(slr, slr2), sdw = sdev(slw, slw2);
    for (int64_t i = 0; i < nq; ++i) {
        put(kNFixed + nq + i, cell_quantile(wk.marg_dm, g.n_dm, g.dm0, g.ddm, qs[i]));
        put(kNFixed + 2 * nq + i, cell_quantile(wk.marg_mu, g.n_mu, g.mu0, g.dmu, qs[i]));
    }
    if (nq == 0) return;
    // compact mixtures of t, log10 R and log10 W
    wk.qw.resize(keep);
    wk.qc.resize(keep);
    wk.qis.resize(keep);
    for (size_t i = 0; i < keep; ++i) {
        wk.qw[i] = wk.lw[use[i]];
        wk.qis[i] = 1.0 / wk.s[use[i]];
    }
    const double* offs[3] = {nullptr, LR, LW};
    const double starts[3] = {st, slr, slw}, sds[3] = {sdt, sdr, sdw};
    const double los[3] = {tlo, rlo, wlo}, his[3] = {thi, rhi, whi};
    const int64_t slot[3] = {0, 3, 4};
    for (int v = 0; v < 3; ++v) {
        for (size_t i = 0; i < keep; ++i)
            wk.qc[i] = wk.m[use[i]] + (offs[v] ? offs[v][use[i]] : 0.0);
        for (int64_t i = 0; i < nq; ++i) {
            double x = mixture_quantile(g, wk, qs[i], starts[v] + zq[i] * sds[v], los[v], his[v]);
            if (v > 0) x = std::pow(10.0, x);
            put(kNFixed + slot[v] * nq + i, x);
        }
    }
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

// Posterior of (log10 Nw, Dm, mu) for the gates of several inputs in one call.
// grid: (11, n_dm * n_mu) array of the fields in `Field` order (LOGP is -inf
// for impossible nodes); axes: the first value and step of the Dm and mu
// axes; errors: (zh variance, zdr variance, kdp abs, kdp rel, ah abs, ah rel,
// prune); quantiles and the standard normal quantiles zq of the same levels.
// Returns one (16 + 5 * n_quantiles, n) array per input.
py::list retrieve(const std::vector<DArray>& z, const std::vector<DArray>& zdr,
                  const std::vector<py::object>& kdp, const std::vector<py::object>& ah,
                  const std::vector<py::object>& mask, const DArray& grid, int64_t n_dm,
                  int64_t n_mu, double dm0, double ddm, double mu0, double dmu,
                  const std::vector<double>& errors, const std::vector<double>& quantiles,
                  const std::vector<double>& zq, int n_threads) {
    const size_t ns = z.size();
    if (zdr.size() != ns || kdp.size() != ns || ah.size() != ns || mask.size() != ns)
        throw std::invalid_argument("need one entry per input in every list");
    if (n_dm < 1 || n_mu < 1 || grid.ndim() != 2 || grid.shape(0) != NFIELD ||
        grid.shape(1) != n_dm * n_mu)
        throw std::invalid_argument("grid must have shape (11, n_dm * n_mu)");
    if (errors.size() != 7) throw std::invalid_argument("errors needs seven values");
    if (!(errors[0] > 0.0 && errors[1] > 0.0))
        throw std::invalid_argument("the Z_H and Z_DR error variances must be positive");
    if (zq.size() != quantiles.size())
        throw std::invalid_argument("zq must have one value per quantile");
    for (double q : quantiles)
        if (!(q > 0.0 && q < 1.0)) throw std::invalid_argument("quantiles must be in (0, 1)");
    Grid g;
    g.n = n_dm * n_mu;
    g.n_dm = n_dm;
    g.n_mu = n_mu;
    g.dm0 = dm0;
    g.ddm = ddm;
    g.mu0 = mu0;
    g.dmu = dmu;
    for (int i = 0; i < NFIELD; ++i) g.f[i] = grid.data() + i * g.n;
    Errors e{errors[0], errors[1], errors[2], errors[3], errors[4], errors[5], errors[6]};
    g.ivt.resize(g.n);
    g.hlvt.resize(g.n);
    for (int64_t j = 0; j < g.n; ++j) {
        const double ps = g.f[Field::PSD][j];
        if (std::isfinite(g.f[Field::LOGP][j]) && !(ps > 0.0))
            throw std::invalid_argument("prior standard deviations must be positive");
        const double vt = e.zh2 + 100.0 * ps * ps;
        g.ivt[j] = 1.0 / vt;
        g.hlvt[j] = 0.5 * std::log(vt);
    }
    g.cdf.resize(kNTab);
    g.pdf.resize(kNTab);
    for (int64_t i = 0; i < kNTab; ++i) {
        const double x = -kZMax + static_cast<double>(i) / kZRes;
        g.cdf[i] = 0.5 * std::erfc(-x / std::sqrt(2.0));
        g.pdf[i] = std::exp(-0.5 * x * x - 0.5 * kLog2Pi);
    }
    const int64_t nout = kNFixed + 5 * static_cast<int64_t>(quantiles.size());

    std::vector<DArray> kdp_keep(ns), ah_keep(ns);
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
        b.ah = optional_data<double>(ah[i], b.n, ah_keep[i], "ah");
        b.mask = optional_data<uint8_t>(mask[i], b.n, mask_keep[i], "mask");
        outs.emplace_back(std::vector<py::ssize_t>{nout, b.n});
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
                Work wk;
                wk.c.resize(g.n);
                wk.m.resize(g.n);
                wk.s.resize(g.n);
                wk.lw.resize(g.n);
                wk.marg_dm.resize(g.n_dm);
                wk.marg_mu.resize(g.n_mu);
                wk.idx.reserve(g.n);
                for (;;) {
                    const int64_t g0 = next.fetch_add(kBlock);
                    if (g0 >= total) break;
                    const int64_t g1 = std::min(total, g0 + kBlock);
                    size_t k = 0;
                    while (g0 >= blocks[k].first + blocks[k].n) ++k;
                    for (int64_t q = g0; q < g1; ++q) {
                        while (q >= blocks[k].first + blocks[k].n) ++k;
                        gate(g, e, quantiles, zq, blocks[k], q - blocks[k].first, wk);
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

PYBIND11_MODULE(_dsd_bayes, m) {
    m.doc() = "Compiled Bayesian drop size distribution retrieval kernel for radarx.";
    m.def("retrieve", &retrieve, py::arg("dbzh"), py::arg("zdr"), py::arg("kdp"),
          py::arg("ah"), py::arg("mask"), py::arg("grid"), py::arg("n_dm"), py::arg("n_mu"),
          py::arg("dm0"), py::arg("ddm"), py::arg("mu0"), py::arg("dmu"), py::arg("errors"),
          py::arg("quantiles"), py::arg("zq"), py::arg("n_threads") = 0);
}
