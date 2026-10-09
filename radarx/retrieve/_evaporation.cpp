// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Rain evaporation kernel.
//
// rates(): every cell (gate, QVP level, grid point) is independent, so all
// cells form one pool of work that threads take in blocks from an atomic
// counter. Per cell, the ventilated diffusional evaporation of a gamma drop
// size distribution N(D) = N0 D^mu exp(-Lambda D) is integrated in closed form
// (gamma-function moments) and gives the evaporation rate, the cooling rate,
// the tendency of the reflectivity factor and the saturation deficit.
//
// integrate(): every cell is marched in time through a sequence of DSDs
// (sub-steps of at most max_step seconds), updating temperature and specific
// humidity; cells are independent and shared out the same way.
//
// Sources (details, references and radarx choices in the docstring of
// radarx/retrieve/evaporation.py): the single-drop law of Rogers and Yau
// (1989) as written by Kumjian and Ryzhkov (2010), Eq. 2 and Appendix
// Eqs. A1-A10, with the ventilation coefficient 0.78 + 0.308 N_Sc^(1/3)
// N_Re^(1/2) (Pruppacher and Klett 1997; Li and Srivastava 2001, Eq. 2) taken
// equal for heat and vapour; D_v with the reference pressure of 1000 hPa as in
// Eq. A7 (the fit of Pruppacher and Klett uses 1013.25 hPa, a 0.3-0.8 %
// effect on the rates); e_s of Buck (1981), Eq. 8, without the enhancement
// factor; the fall speed is a radarx fit to Atlas et al. (1973) times the
// (rho0 / rho)^0.4 correction attributed to Foote and du Toit (1969); the
// gamma-DSD integral follows the approach of Milbrandt and Yau (2005, Part II,
// Eq. 8). The saturation limit of the time step is a radarx choice.
//
// Every step follows the NumPy reference in radarx/retrieve/evaporation.py,
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
constexpr double kPi = 3.14159265358979323846;  // M_PI is not standard (MSVC)
constexpr double kRd = 287.04;
constexpr double kRv = 461.5;
constexpr double kEps = kRd / kRv;
constexpr double kCpd = 1005.7;
constexpr double kCpv = 1870.0;
constexpr double kRhoW = 1000.0;
constexpr double kT0 = 273.15;
constexpr int64_t kBlock = 1024;  // cells per scheduling block

struct Params {
    double a, b, f;    // fall speed a D^b exp(-f D), D in mm
    double av, bv;     // ventilation coefficients
    double rho0;       // density of the sea-level fall speeds
};

// log Gamma(x) for x > 0 (Lanczos, g = 7, n = 9); thread-safe unlike
// std::lgamma, which may write the global signgam.
inline double log_gamma(double x) {
    static const double c[9] = {0.99999999999980993,  676.5203681218851,
                                -1259.1392167224028,  771.32342877765313,
                                -176.61502916214059,  12.507343278686905,
                                -0.13857109526572012, 9.9843695780195716e-6,
                                1.5056327351493116e-7};
    if (x < 0.5) {
        // reflection: Gamma(x) Gamma(1 - x) = pi / sin(pi x), 0 < x < 0.5
        return std::log(kPi / std::sin(kPi * x)) - log_gamma(1.0 - x);
    }
    x -= 1.0;
    double s = c[0];
    for (int i = 1; i < 9; ++i) s += c[i] / (x + i);
    const double t = x + 7.5;
    return 0.5 * std::log(2.0 * kPi) + (x + 0.5) * std::log(t) - t + std::log(s);
}

struct Air {
    double es, ssat, rho, lv, dv, nu, fkd, cp, qs;
};

inline Air air(double t, double p, double qv) {
    Air r;
    const double tc = t - kT0;
    r.es = 611.21 * std::exp(17.502 * tc / (240.97 + tc));  // Buck (1981)
    const double e = qv * p / (kEps + (1.0 - kEps) * qv);
    r.rho = p / (kRd * t * (1.0 + (1.0 / kEps - 1.0) * qv));
    r.lv = 2.499e6 * std::pow(kT0 / t, 0.167 + 3.67e-4 * t);
    const double k = (0.441635 + 0.0071 * t) * 1.0e-2;
    r.dv = 2.11e-5 * std::pow(t / kT0, 1.94) * (1.0e5 / p);
    r.nu = (0.379565 + 0.0049 * t) * 1.0e-5 / r.rho;
    r.fkd = (r.lv / (kRv * t) - 1.0) * r.lv / (k * t) + kRv * t / (r.dv * r.es);
    r.cp = kCpd * (1.0 - qv) + kCpv * qv;
    r.qs = kEps * r.es / (p - (1.0 - kEps) * r.es);
    r.ssat = e / r.es - 1.0;
    return r;
}

// The DSD part of I_k = int D^k f_v(D) N(D) dD = av * A_k + bterm * B_k, in
// closed form; it does not depend on the air, so it is computed once per DSD.
// Gamma(x + 3) = Gamma(x) x (x + 1) (x + 2) gives the k = 4 terms from k = 1.
struct Dsd {
    double a1, b1, a4, b4, z;
    int state;  // 1: rain, 0: no rain (N0 = 0), -1: missing
};

inline Dsd dsd_terms(const Params& q, double n0, double mu, double lam) {
    Dsd d{0.0, 0.0, 0.0, 0.0, 0.0, -1};
    if (!(std::isfinite(n0) && std::isfinite(mu) && std::isfinite(lam))) return d;
    if (n0 == 0.0) {
        d.state = 0;
        return d;
    }
    if (!(n0 > 0.0 && lam > 0.0)) return d;
    d.state = 1;
    const double logn0 = std::log(n0);
    const double loglam = std::log(lam);
    const double loglam2 = std::log(lam + 0.5 * q.f);
    const double x1 = mu + 2.0;
    const double x2 = x1 + 0.5 * (q.b + 1.0);
    d.a1 = std::exp(logn0 + log_gamma(x1) - x1 * loglam);
    d.b1 = std::exp(logn0 + log_gamma(x2) - x2 * loglam2);
    const double l = 1.0 / lam, l2 = 1.0 / (lam + 0.5 * q.f);
    d.a4 = d.a1 * (x1 * l) * ((x1 + 1.0) * l) * ((x1 + 2.0) * l);
    d.b4 = d.b1 * (x2 * l2) * ((x2 + 1.0) * l2) * ((x2 + 2.0) * l2);
    // Z = N0 Gamma(mu + 7) / Lambda^(mu + 7) = A_1 (mu + 2) ... (mu + 6) / Lambda^5
    d.z = d.a4 * ((x1 + 3.0) * l) * ((x1 + 4.0) * l);
    return d;
}

inline double bterm_of(const Params& q, const Air& s) {
    const double corr = std::pow(q.rho0 / s.rho, 0.4);
    return q.bv * std::cbrt(s.nu / s.dv) * std::sqrt(1.0e-3 * corr * q.a / s.nu);
}

// Evaporation rate (kg kg-1 s-1) only, for the time integration.
inline double evap_rate(const Params& q, const Dsd& d, const Air& s) {
    if (d.state <= 0) return d.state == 0 && std::isfinite(s.ssat) ? 0.0 : kNaN;
    const double i1 = q.av * d.a1 + bterm_of(q, s) * d.b1;
    return -2.0e-3 * kPi * s.ssat / s.fkd * i1 / s.rho;
}

struct Rates {
    double evap, cool, dbz, ssat;
};

inline Rates rates(const Params& q, const Dsd& d, const Air& s) {
    Rates r{kNaN, kNaN, kNaN, s.ssat};
    if (d.state <= 0) {
        if (d.state == 0 && std::isfinite(s.ssat)) r.evap = r.cool = r.dbz = 0.0;
        return r;
    }
    const double bterm = bterm_of(q, s);
    const double i1 = q.av * d.a1 + bterm * d.b1;
    const double i4 = q.av * d.a4 + bterm * d.b4;
    r.evap = -2.0e-3 * kPi * s.ssat / s.fkd * i1 / s.rho;
    r.cool = s.lv * r.evap / s.cp;
    const double zrate = 2.4e7 * s.ssat / (kRhoW * s.fkd) * i4;
    r.dbz = 3600.0 * 10.0 / std::log(10.0) * zrate / d.z;
    return r;
}

// Limit a vapour increment to the saturation adjustment gap.
inline double limit(double dq, double gap) {
    if (dq > 0.0) dq = std::min(dq, std::max(gap, 0.0));
    if (dq < 0.0) dq = std::max(dq, std::min(gap, 0.0));
    return dq;
}

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

void check_1d(const DArray& a, int64_t n, const char* name) {
    if (a.ndim() != 1 || a.shape(0) != n)
        throw std::invalid_argument(std::string(name) + " must be 1-D of the size of n0");
}

}  // namespace

// Evaporation rate (kg kg-1 s-1), cooling rate (K s-1), dBZ tendency (dB h-1)
// and saturation deficit for 1-D arrays of cells. Returns a (4, n) array.
py::array_t<double> rates_py(const DArray& n0, const DArray& mu, const DArray& lam,
                             const DArray& t, const DArray& p, const DArray& qv,
                             double a, double b, double f, double av, double bv,
                             double rho0, int n_threads) {
    if (n0.ndim() != 1) throw std::invalid_argument("n0 must be 1-D");
    const int64_t n = n0.shape(0);
    check_1d(mu, n, "mu");
    check_1d(lam, n, "lam");
    check_1d(t, n, "temperature");
    check_1d(p, n, "pressure");
    check_1d(qv, n, "specific_humidity");
    const Params q{a, b, f, av, bv, rho0};
    py::array_t<double> out(std::vector<py::ssize_t>{4, n});
    double* o = out.mutable_data();
    const double *pn0 = n0.data(), *pmu = mu.data(), *plam = lam.data(),
                 *pt = t.data(), *pp = p.data(), *pq = qv.data();
    {
        py::gil_scoped_release release;
        parallel_blocks(n, n_threads, [&](int64_t g) {
            const Rates r =
                rates(q, dsd_terms(q, pn0[g], pmu[g], plam[g]), air(pt[g], pp[g], pq[g]));
            o[g] = r.evap;
            o[n + g] = r.cool;
            o[2 * n + g] = r.dbz;
            o[3 * n + g] = r.ssat;
        });
    }
    return out;
}

// March temperature and specific humidity of every cell through nt DSDs
// ((nt, n) arrays) separated by dt (nt - 1 intervals, s). Returns
// (temperature, specific humidity, evaporation rate, cooling rate), each (nt, n),
// the state and rates at each time before its interval is integrated.
py::tuple integrate_py(const DArray& n0, const DArray& mu, const DArray& lam,
                       const DArray& t0, const DArray& qv0, const DArray& p,
                       const DArray& dt, double max_step, double a, double b, double f,
                       double av, double bv, double rho0, int n_threads) {
    if (n0.ndim() != 2) throw std::invalid_argument("n0 must be (time, cell)");
    const int64_t nt = n0.shape(0), n = n0.shape(1);
    for (const DArray* x : {&mu, &lam})
        if (x->ndim() != 2 || x->shape(0) != nt || x->shape(1) != n)
            throw std::invalid_argument("mu and lam must have the shape of n0");
    check_1d(t0, n, "temperature");
    check_1d(qv0, n, "specific_humidity");
    check_1d(p, n, "pressure");
    if (dt.ndim() != 1 || dt.shape(0) != std::max<int64_t>(nt - 1, 0))
        throw std::invalid_argument("dt must have one interval less than the times");
    if (!(max_step > 0.0)) throw std::invalid_argument("max_step must be positive");
    const Params q{a, b, f, av, bv, rho0};
    std::vector<int64_t> nsub(std::max<int64_t>(nt - 1, 0));
    for (int64_t i = 0; i + 1 < nt; ++i)
        nsub[i] = std::max<int64_t>(1, static_cast<int64_t>(std::ceil(dt.data()[i] / max_step)));
    std::vector<py::ssize_t> shape{nt, n};
    py::array_t<double> tt(shape), qq(shape), ee(shape), cc(shape);
    double *ot = tt.mutable_data(), *oq = qq.mutable_data(), *oe = ee.mutable_data(),
           *oc = cc.mutable_data();
    const double *pn0 = n0.data(), *pmu = mu.data(), *plam = lam.data(),
                 *pt0 = t0.data(), *pq0 = qv0.data(), *pp = p.data(), *pdt = dt.data();
    {
        py::gil_scoped_release release;
        parallel_blocks(n, n_threads, [&](int64_t g) {
            double t = pt0[g], qv = pq0[g];
            const double pres = pp[g];
            for (int64_t i = 0; i < nt; ++i) {
                const int64_t k = i * n + g;
                ot[k] = t;
                oq[k] = qv;
                const Dsd d = dsd_terms(q, pn0[k], pmu[k], plam[k]);
                const Rates r = rates(q, d, air(t, pres, qv));
                oe[k] = r.evap;
                oc[k] = r.cool;
                if (i == nt - 1 || d.state != 1) continue;  // no rain: no change
                const double h = pdt[i] / static_cast<double>(nsub[i]);
                for (int64_t s = 0; s < nsub[i]; ++s) {
                    // Heun (trapezoidal predictor-corrector) step, limited at saturation
                    const Air st = air(t, pres, qv);
                    const double gap =
                        (st.qs - qv) / (1.0 + st.lv * st.lv * st.qs / (st.cp * kRv * t * t));
                    double e1 = evap_rate(q, d, st);
                    if (!std::isfinite(e1)) e1 = 0.0;
                    double dq = limit(e1 * h, gap);
                    const double t1 = t - st.lv * dq / st.cp;
                    double e2 = evap_rate(q, d, air(t1, pres, qv + dq));
                    if (!std::isfinite(e2)) e2 = 0.0;
                    dq = limit(0.5 * (e1 + e2) * h, gap);
                    qv = qv + dq;
                    t = t - st.lv * dq / st.cp;
                }
            }
        });
    }
    return py::make_tuple(tt, qq, ee, cc);
}

PYBIND11_MODULE(_evaporation, m) {
    m.doc() = "Compiled rain evaporation kernel for radarx.";
    m.def("rates", &rates_py, py::arg("n0"), py::arg("mu"), py::arg("lam"),
          py::arg("temperature"), py::arg("pressure"), py::arg("specific_humidity"),
          py::arg("a"), py::arg("b"), py::arg("f"), py::arg("av"), py::arg("bv"),
          py::arg("rho0"), py::arg("n_threads") = 0);
    m.def("integrate", &integrate_py, py::arg("n0"), py::arg("mu"), py::arg("lam"),
          py::arg("temperature"), py::arg("specific_humidity"), py::arg("pressure"),
          py::arg("dt"), py::arg("max_step"), py::arg("a"), py::arg("b"), py::arg("f"),
          py::arg("av"), py::arg("bv"), py::arg("rho0"), py::arg("n_threads") = 0);
}
