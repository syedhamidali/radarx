// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Parcel diagnostics of soundings and ERA5 columns for radarx.io.parcel:
// CAPE, CIN, lifted index, LCL, LFC and EL of a surface-based, mixed-layer or
// most-unstable parcel.
//
// The thermodynamics are those of Bolton (1980, Mon. Wea. Rev. 108, 1046-1053,
// doi 10.1175/1520-0493(1980)108<1046:TCOEPT>2.0.CO;2): saturation vapour
// pressure eq. (10), potential temperature eq. (6), temperature at the lifting
// condensation level eq. (15) and the equivalent potential temperature eq.
// (43), whose conservation defines the pseudo-adiabat above the LCL (the
// equation forms are as given by implementations that cite the paper, not
// checked against its text). Below the LCL the parcel follows the dry
// adiabat of eq. (6) with its mixing ratio. The buoyancy is
// g (Tv_parcel - Tv_environment) / Tv_environment (virtual temperature
// T (1 + r / eps) / (1 + r), no condensate loading), integrated in height
// with linear interpolation between levels and exact zero crossings.
//
// Every column is independent: the columns are split over threads that take
// blocks from an atomic counter. The arithmetic is that of
// radarx/io/_parcel_numpy.py, step for step.

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
constexpr double kRd = 287.04749;
constexpr double kRv = 461.52311;
constexpr double kEps = kRd / kRv;
constexpr double kG0 = 9.80665;
constexpr double kT0 = 273.15;
constexpr double kPref = 1.0e5;
constexpr double kLiPressure = 50000.0;
constexpr int kBisect = 80;
constexpr int kRows = 12;

template <class F>
void parallel_for(int64_t n, int64_t block, int n_threads, F&& fn) {
    if (n <= 0) return;
    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    int nt = n_threads > 0 ? n_threads : static_cast<int>(hw);
    const int64_t nblocks = (n + block - 1) / block;
    nt = static_cast<int>(std::min<int64_t>(nt, nblocks));
    std::atomic<int64_t> next{0};
    auto worker = [&]() {
        for (;;) {
            int64_t b0 = next.fetch_add(block);
            if (b0 >= n) break;
            fn(b0, std::min(n, b0 + block));
        }
    };
    std::vector<std::thread> pool;
    for (int t = 1; t < nt; ++t) pool.emplace_back(worker);
    worker();
    for (auto& th : pool) th.join();
}

inline double esat(double t) {
    const double tc = t - kT0;
    return 611.2 * std::exp(17.67 * tc / (tc + 243.5));
}
inline double dewpoint(double e) {
    const double l = std::log(e / 611.2);
    return kT0 + 243.5 * l / (17.67 - l);
}
inline double mixing_ratio(double e, double p) { return kEps * e / (p - e); }
inline double kappa_of(double r) { return 0.2854 * (1.0 - 0.28 * r); }
inline double lcl_temperature(double t, double td) {
    if (td >= t) return t;
    return 1.0 / (1.0 / (td - 56.0) + std::log(t / td) / 800.0) + 56.0;
}
inline double theta_e_exponent(double r, double tl) {
    return std::exp(r * (1.0 + 0.81 * r) * (3376.0 / tl - 2.54));
}
inline double theta_e(double t, double td, double p) {
    const double r = mixing_ratio(esat(td), p);
    return t * std::pow(kPref / p, kappa_of(r)) * theta_e_exponent(r, lcl_temperature(t, td));
}
inline double virt(double t, double r, bool use) {
    return use ? t * (1.0 + r / kEps) / (1.0 + r) : t;
}

// Linear in ln p between the two levels that bracket pt, NaN outside.
double interp_lnp(const std::vector<double>& p, const std::vector<double>& f, double pt) {
    const size_t n = p.size();
    for (size_t k = 0; k + 1 < n; ++k) {
        if (p[k] >= pt && pt >= p[k + 1]) {
            const double w = std::log(pt / p[k]) / std::log(p[k + 1] / p[k]);
            return f[k] + w * (f[k + 1] - f[k]);
        }
    }
    return kNaN;
}

struct Start {
    double p, t, td;
    size_t index;
};

Start parcel_start(int kind, double depth, const std::vector<double>& p,
                   const std::vector<double>& t, const std::vector<double>& td) {
    if (kind == 0) return {p[0], t[0], td[0], 0};
    const double top = p[0] - depth;
    if (kind == 2) {
        double best = -1.0;
        size_t arg = 0;
        for (size_t k = 0; k < p.size(); ++k) {
            if (p[k] < top) break;
            const double th = theta_e(t[k], td[k], p[k]);
            if (th > best) {
                best = th;
                arg = k;
            }
        }
        return {p[arg], t[arg], td[arg], arg};
    }
    std::vector<double> pp, th, rr;
    for (size_t k = 0; k < p.size(); ++k) {
        if (p[k] < top) break;
        const double r = mixing_ratio(esat(td[k]), p[k]);
        pp.push_back(p[k]);
        rr.push_back(r);
        th.push_back(t[k] * std::pow(kPref / p[k], kappa_of(r)));
    }
    if (pp.back() > top && pp.size() < p.size()) {
        const size_t k = pp.size();
        const double w = std::log(top / p[k - 1]) / std::log(p[k] / p[k - 1]);
        const double r = mixing_ratio(esat(td[k - 1]), p[k - 1]);
        const double r2 = mixing_ratio(esat(td[k]), p[k]);
        const double t_top = t[k - 1] + w * (t[k] - t[k - 1]);
        const double r_top = r + w * (r2 - r);
        pp.push_back(top);
        rr.push_back(r_top);
        th.push_back(t_top * std::pow(kPref / top, kappa_of(r_top)));
    }
    if (pp.size() < 2) return {p[0], t[0], td[0], 0};
    double num_t = 0.0, num_r = 0.0;
    for (size_t k = 0; k + 1 < pp.size(); ++k) {
        const double dp = pp[k] - pp[k + 1];
        num_t += 0.5 * (th[k] + th[k + 1]) * dp;
        num_r += 0.5 * (rr[k] + rr[k + 1]) * dp;
    }
    const double span = pp.front() - pp.back();
    const double th_m = num_t / span, r_m = num_r / span;
    const double p0 = p[0];
    const double t0 = th_m * std::pow(p0 / kPref, kappa_of(r_m));
    const double e0 = r_m * p0 / (kEps + r_m);
    return {p0, t0, std::min(dewpoint(e0), t0), 0};
}

struct Parcel {
    double p0, t0, r0, kappa0, theta0, tl, pl, theta_e0;
    Parcel(double p0_, double t0_, double td0) : p0(p0_), t0(t0_) {
        r0 = mixing_ratio(esat(td0), p0);
        kappa0 = kappa_of(r0);
        theta0 = t0 * std::pow(kPref / p0, kappa0);
        tl = lcl_temperature(t0, td0);
        pl = p0 * std::pow(tl / t0, 1.0 / kappa0);
        theta_e0 = theta0 * theta_e_exponent(r0, tl);
    }
    // temperature and mixing ratio at pressure p
    void at(double p, double& t, double& r) const {
        if (p >= pl) {
            t = theta0 * std::pow(p / kPref, kappa0);
            r = r0;
            return;
        }
        auto g = [&](double tt) {
            const double rs = mixing_ratio(esat(tt), p);
            return tt * std::pow(kPref / p, kappa_of(rs)) * theta_e_exponent(rs, tt) - theta_e0;
        };
        double lo = 100.0, hi = tl + 1.0;
        for (int it = 0; it < kBisect; ++it) {
            const double mid = 0.5 * (lo + hi);
            if (g(mid) > 0.0) hi = mid; else lo = mid;
        }
        t = 0.5 * (lo + hi);
        r = mixing_ratio(esat(t), p);
    }
};

// Integral of max(b, 0) dz, b linear between the nodes.
double positive_area(const std::vector<double>& z, const std::vector<double>& b) {
    double total = 0.0;
    for (size_t k = 0; k + 1 < z.size(); ++k) {
        const double a = b[k], c = b[k + 1], dz = z[k + 1] - z[k];
        if (a >= 0.0 && c >= 0.0) total += 0.5 * (a + c) * dz;
        else if (a > 0.0 && 0.0 > c) total += 0.5 * a * dz * a / (a - c);
        else if (c > 0.0 && 0.0 > a) total += 0.5 * c * dz * c / (c - a);
    }
    return total;
}

double negative_area(const std::vector<double>& z, std::vector<double> b) {
    for (auto& v : b) v = -v;
    return -positive_area(z, b);
}

void column(const double* zc, const double* pc, const double* tc, const double* dc, int64_t nlev,
            int kind, double depth, bool use_virtual, double* out) {
    for (int k = 0; k < kRows; ++k) out[k] = kNaN;
    std::vector<double> z, p, t, td;
    for (int64_t k = 0; k < nlev; ++k) {
        const double zk = zc[k], pk = pc[k], tk = tc[k], dk = dc[k];
        if (!(std::isfinite(zk) && std::isfinite(pk) && std::isfinite(tk))) continue;
        if (!std::isfinite(dk) || pk <= 0.0 || tk <= 0.0 || dk <= 0.0) continue;
        if (!p.empty() && !(pk < p.back() && zk > z.back())) continue;
        z.push_back(zk);
        p.push_back(pk);
        t.push_back(tk);
        td.push_back(std::min(dk, tk));
    }
    const size_t n = p.size();
    if (n < 3) return;

    const Start st = parcel_start(kind, depth, p, t, td);
    const double p0 = st.p, t0 = st.t, td0 = st.td;
    const size_t i0 = st.index;
    const Parcel parcel(p0, t0, td0);
    out[9] = p0;
    out[10] = t0;
    out[11] = td0;
    std::vector<double> r_env(n);
    for (size_t k = 0; k < n; ++k) r_env[k] = mixing_ratio(esat(td[k]), p[k]);

    auto buoyancy = [&](double tp, double rp, double te, double re) {
        const double tvp = virt(tp, rp, use_virtual), tve = virt(te, re, use_virtual);
        return kG0 * (tvp - tve) / tve;
    };

    if (p.back() <= kLiPressure && kLiPressure <= p0) {
        double tp, rp;
        parcel.at(kLiPressure, tp, rp);
        out[2] = interp_lnp(p, t, kLiPressure) - tp;
    }

    std::vector<double> nz, np, nb;
    nz.push_back(z[i0]);
    np.push_back(p0);
    nb.push_back(buoyancy(t0, parcel.r0, t[i0], r_env[i0]));
    long lcl_node = -1;
    bool have_lcl = parcel.pl < p0 && parcel.pl >= p.back();
    double lcl_z = kNaN, lcl_p = kNaN;
    if (parcel.pl >= p0) {
        have_lcl = true;
        lcl_node = 0;
        lcl_z = z[i0];
        lcl_p = p0;
    }
    for (size_t k = i0 + 1; k < n; ++k) {
        if (have_lcl && lcl_node < 0 && p[k] <= parcel.pl) {
            const double pl = parcel.pl;
            lcl_p = pl;
            lcl_z = interp_lnp(p, z, pl);
            const double te = interp_lnp(p, t, pl);
            const double re = interp_lnp(p, r_env, pl);
            if (p[k] < pl) {
                nz.push_back(lcl_z);
                np.push_back(pl);
                nb.push_back(buoyancy(parcel.tl, parcel.r0, te, re));
                lcl_node = static_cast<long>(nz.size()) - 1;
            }
        }
        double tp, rp;
        parcel.at(p[k], tp, rp);
        nz.push_back(z[k]);
        np.push_back(p[k]);
        nb.push_back(buoyancy(tp, rp, t[k], r_env[k]));
        if (lcl_node < 0 && have_lcl && p[k] == parcel.pl) lcl_node = static_cast<long>(nz.size()) - 1;
    }
    out[3] = lcl_p;
    out[4] = lcl_z;
    if (lcl_node < 0) {
        out[0] = 0.0;
        return;
    }

    const long m = static_cast<long>(nz.size());
    auto p_at = [&](double zc_, long j) {
        const double w = (zc_ - nz[j]) / (nz[j + 1] - nz[j]);
        return np[j] * std::pow(np[j + 1] / np[j], w);
    };

    double lfc_z = kNaN, lfc_p = kNaN, lfc_b = 0.0;
    long lfc_seg = -1;
    if (nb[lcl_node] > 0.0) {
        lfc_z = nz[lcl_node];
        lfc_p = np[lcl_node];
        lfc_b = nb[lcl_node];
        lfc_seg = lcl_node;
    } else {
        for (long j = lcl_node; j + 1 < m; ++j) {
            if (nb[j] <= 0.0 && 0.0 < nb[j + 1]) {
                const double zx = nz[j] + (nz[j + 1] - nz[j]) * (0.0 - nb[j]) / (nb[j + 1] - nb[j]);
                lfc_z = zx;
                lfc_p = p_at(zx, j);
                lfc_seg = j;
                break;
            }
        }
    }
    if (lfc_seg < 0) {
        out[0] = 0.0;
        return;
    }
    out[5] = lfc_p;
    out[6] = lfc_z;

    double el_z = nz[m - 1], el_p = np[m - 1], el_b = nb[m - 1];
    for (long j = m - 2; j >= lfc_seg; --j) {
        if (nb[j] > 0.0 && 0.0 >= nb[j + 1]) {
            const double zx = nz[j] + (nz[j + 1] - nz[j]) * (0.0 - nb[j]) / (nb[j + 1] - nb[j]);
            el_z = zx;
            el_p = p_at(zx, j);
            el_b = 0.0;
            break;
        }
    }
    if (nb[m - 1] > 0.0) {
        el_z = nz[m - 1];
        el_p = np[m - 1];
        el_b = nb[m - 1];
    }
    out[7] = el_p;
    out[8] = el_z;

    std::vector<double> cz, cb;
    cz.push_back(lfc_z);
    cb.push_back(lfc_b);
    for (long k = 0; k < m; ++k) {
        if (lfc_z < nz[k] && nz[k] < el_z) {
            cz.push_back(nz[k]);
            cb.push_back(nb[k]);
        }
    }
    cz.push_back(el_z);
    cb.push_back(el_b);
    out[0] = positive_area(cz, cb);

    std::vector<double> iz, ib;
    for (long k = 0; k < m; ++k) {
        if (nz[k] < lfc_z) {
            iz.push_back(nz[k]);
            ib.push_back(nb[k]);
        }
    }
    iz.push_back(lfc_z);
    ib.push_back(lfc_b);
    out[1] = negative_area(iz, ib);
}

}  // namespace

// z, p, t, td: (ncol, nlev) heights [m], pressures [Pa], temperatures and dew
// points [K], ascending in height. kind: 0 surface-based, 1 mixed layer,
// 2 most unstable; depth [Pa] of the mixed and the most-unstable layers.
// Returns (12, ncol): see ROWS in radarx/io/_parcel_numpy.py.
py::array_t<double> parcel(const DArray& z, const DArray& p, const DArray& t, const DArray& td,
                           int kind, double depth, bool use_virtual, int n_threads) {
    if (z.ndim() != 2 || p.ndim() != 2 || t.ndim() != 2 || td.ndim() != 2)
        throw std::invalid_argument("z, p, t and td must be (ncol, nlev)");
    const int64_t ncol = z.shape(0), nlev = z.shape(1);
    for (const auto* a : {&p, &t, &td})
        if (a->shape(0) != ncol || a->shape(1) != nlev)
            throw std::invalid_argument("z, p, t and td must have the same shape");
    if (kind < 0 || kind > 2) throw std::invalid_argument("kind must be 0, 1 or 2");
    if (!(depth > 0.0)) throw std::invalid_argument("depth must be positive");
    py::array_t<double> result({static_cast<int64_t>(kRows), ncol});
    double* res = result.mutable_data();
    const double *pz = z.data(), *pp = p.data(), *pt = t.data(), *pd = td.data();
    {
        py::gil_scoped_release release;
        parallel_for(ncol, 16, n_threads, [&](int64_t c0, int64_t c1) {
            double row[kRows];
            for (int64_t c = c0; c < c1; ++c) {
                column(pz + c * nlev, pp + c * nlev, pt + c * nlev, pd + c * nlev, nlev, kind,
                       depth, use_virtual, row);
                for (int k = 0; k < kRows; ++k) res[k * ncol + c] = row[k];
            }
        });
    }
    return result;
}

PYBIND11_MODULE(_parcel, m) {
    m.doc() = "Compiled parcel diagnostics (CAPE, CIN, LCL, LFC, EL, lifted index) for radarx.";
    m.def("parcel", &parcel, py::arg("z"), py::arg("p"), py::arg("t"), py::arg("td"),
          py::arg("kind"), py::arg("depth"), py::arg("virtual"), py::arg("n_threads") = 0);
}
