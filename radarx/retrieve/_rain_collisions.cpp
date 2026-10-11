// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Collisions, coalescence and breakup of raindrops in a rain shaft
// (radarx/retrieve/rain_collisions.py).
//
// The drop size distribution is a set of pivot sizes D_k with number concentrations N_k. A
// collision of a drop of pivot i with one of pivot j (i > j) happens at the rate K_ij N_i N_j with
// the geometric sweep-out kernel K_ij = pi/4 (D_i + D_j)^2 |V_i - V_j|. Its outcome follows Low and
// List (1982): with the coalescence efficiency E (Part I, Eqs. 4.5-4.6) the pair coalesces into one
// drop, and otherwise it breaks up as a filament, a sheet or a disk with the probabilities of Part
// II (Eqs. 4.40-4.46) and the mean number of fragments of Eqs. (3.3)-(3.9); the small fragments
// follow the Gaussian and log-normal distributions of Part II (Eqs. 4.12-4.15, 4.27-4.32,
// 4.37-4.39). The large remnant is not taken from the Gaussians of the paper: its size follows from
// the conservation of mass (as in McFarquhar 2004, Section 2b, after Brown 1997). Every drop that is
// produced is put on the two pivots around its mass so that the number of drops and the mass are
// conserved ("fixed pivot"), so every event conserves the mass of the two drops to rounding.
//
// The state of a shaft is the flux F_k = V_k N_k, V_k = V0_k s(z) with the air density factor s;
// dF_k/dz = s sum_{i>j} K_ij N_i N_j G_ijk, G the net change of the pivot k per event. Columns are
// independent and run on all cores; the pairs of the tensor are built in parallel as well. Every
// step follows radarx/retrieve/_rain_collisions_numpy.py in the same order.

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

constexpr double kPi = 3.14159265358979323846;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kSigma = 7.28e-2;
constexpr double kRhoW = 1000.0;
constexpr double kD0 = 0.01;
constexpr double kCke0 = 0.893e-6;
constexpr double kW0 = 0.86;
constexpr double kEtMax = 5.0e-6;
constexpr double kXMin = 0.1;
constexpr double kSigmaMax = 8.0;

template <class F>
void parallel_for(int64_t n, int n_threads, F&& f) {
    int nt = n_threads > 0 ? n_threads : static_cast<int>(std::thread::hardware_concurrency());
    nt = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(nt, n)));
    if (nt == 1) {
        for (int64_t i = 0; i < n; ++i) f(i);
        return;
    }
    std::atomic<int64_t> next{0};
    std::vector<std::thread> pool;
    for (int t = 0; t < nt; ++t)
        pool.emplace_back([&] {
            for (int64_t i = next++; i < n; i = next++) f(i);
        });
    for (auto& t : pool) t.join();
}

// ---------------------------------------------------------------------------------------------
// Low and List (1982); diameters in cm, energies in J

struct Energies {
    double cke, st, sc, dsig;
};

Energies energies(double dl, double ds, double vl, double vs) {
    const double dlm = dl * 1e-2, dsm = ds * 1e-2;
    const double dv = vl - vs;
    Energies e;
    e.cke = (kRhoW * kPi / 12.0) * dlm * dlm * dlm * dsm * dsm * dsm / (dlm * dlm * dlm + dsm * dsm * dsm) * dv * dv;
    e.st = kPi * kSigma * (dlm * dlm + dsm * dsm);
    e.sc = kPi * kSigma * std::pow(dlm * dlm * dlm + dsm * dsm * dsm, 2.0 / 3.0);
    e.dsig = e.st - e.sc;
    return e;
}

double coalescence_efficiency(double dl, double ds, const Energies& e) {
    const double et = e.cke + e.dsig;
    if (et >= kEtMax) return 0.0;
    return 0.778 * std::pow(1.0 + ds / dl, -2.0) * std::exp(-2.61e6 * kSigma * et * et / e.sc);
}

void breakup_fractions(double cke, double st, double& rf, double& rs, double& rd) {
    rf = cke < kCke0 ? 1.0 : 1.11e-4 * std::pow(cke, -0.654);
    const double w2 = cke / st;
    rs = w2 >= kW0 ? 0.685 * (1.0 - std::exp(-1.63 * (w2 - kW0))) : 0.0;
    rd = rf + rs <= 1.0 ? 1.0 - (rf + rs) : 0.0;
    const double total = rf + rs + rd;
    rf /= total;
    rs /= total;
    rd /= total;
}

void filament_numbers(double dl, double ds, double& f1, double& f2) {
    f1 = (-2.25e4 * (dl - 0.403) * (dl - 0.403) - 37.9) * std::pow(ds, 2.5) +
         9.67 * (dl + 0.170) * (dl + 0.170) + 4.95;
    f2 = 1.02e4 * std::pow(ds, 2.83) + 2.0;
}

void mean_numbers(double dl, double ds, double cke, double st, double& ff, double& fs, double& fd) {
    double f1, f2;
    filament_numbers(dl, ds, f1, f2);
    ff = std::max(2.0, std::min(f1, f2));
    fs = std::max(2.0, 5.0 * std::erf((st - 2.53e-6) / 1.85e-6) + 6.0);
    fd = cke > 0.0 ? std::max(2.0, 297.5 + 23.7 * std::log(cke)) : 2.0;
}

double crossing(double dl) {
    double lo = 1e-5, hi = 2.0;
    for (int it = 0; it < 60; ++it) {
        const double mid = 0.5 * (lo + hi);
        double f1, f2;
        filament_numbers(dl, mid, f1, f2);
        if (f1 > f2) lo = mid; else hi = mid;
    }
    return 0.5 * (lo + hi);
}

double lognormal_area(double mode, double peak, double sig) {
    const double h = peak * mode * std::exp(0.5 * sig * sig);
    const double mu = std::log(mode) + sig * sig;
    return h * sig * std::sqrt(kPi / 2.0) *
           (1.0 - std::erf((std::log(kD0) - mu) / (std::sqrt(2.0) * sig)));
}

double lognormal_sigma(double mode, double peak, double number) {
    if (!(number > 0.0 && peak > 0.0 && mode > 0.0)) return kNaN;
    double lo = 1e-3, hi = kSigmaMax;
    if (!(lognormal_area(mode, peak, lo) < number && number < lognormal_area(mode, peak, hi))) return kNaN;
    for (int it = 0; it < 80; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (lognormal_area(mode, peak, mid) < number) lo = mid; else hi = mid;
    }
    return 0.5 * (lo + hi);
}

void gauss_moments(double mu, double sig, double a, double b, double h, double& num, double& x3) {
    num = x3 = 0.0;
    if (!(b > a)) return;
    const double s2 = std::sqrt(2.0) * sig;
    const double za = (a - mu) / s2, zb = (b - mu) / s2;
    const double ea = std::exp(-za * za), eb = std::exp(-zb * zb);
    const double i0 = std::sqrt(kPi / 2.0) * sig * (std::erf(zb) - std::erf(za));
    const double i1 = mu * i0 - sig * sig * (eb - ea);
    const double i2 = mu * i1 + sig * sig * (i0 - (b * eb - a * ea));
    const double i3 = mu * i2 + sig * sig * (2.0 * i1 - (b * b * eb - a * a * ea));
    num = h * i0;
    x3 = h * i3;
}

void lognormal_moments(double mu, double sig, double a, double b, double h, double& num, double& x3) {
    num = x3 = 0.0;
    if (!(b > a)) return;
    const double s2 = std::sqrt(2.0) * sig;
    const double la = std::log(a), lb = std::log(b);
    num = h * sig * std::sqrt(kPi / 2.0) * (std::erf((lb - mu) / s2) - std::erf((la - mu) / s2));
    const double shift = mu + 3.0 * sig * sig;
    x3 = h * std::exp(3.0 * mu + 4.5 * sig * sig) * sig * std::sqrt(kPi / 2.0) *
         (std::erf((lb - shift) / s2) - std::erf((la - shift) / s2));
}

struct Comp {
    bool gauss;
    double mu, sig, h, lo, hi;
};

struct TypeParts {
    std::vector<Comp> parts;
    bool ok;
};

void make_components(double dl, double ds, double cke, TypeParts out[3]) {
    const double d_coal = std::cbrt(dl * dl * dl + ds * ds * ds);
    const double st = kPi * kSigma * ((dl * 1e-2) * (dl * 1e-2) + (ds * 1e-2) * (ds * 1e-2));
    double ff, fs, fd;
    mean_numbers(dl, ds, cke, st, ff, fs, fd);
    // filament
    const double hf2 = 4.18 * std::pow(ds, -1.17);
    out[0].parts.push_back({true, ds, 1.0 / (std::sqrt(2.0 * kPi) * hf2), hf2, 0.0, d_coal});
    const double ds0 = crossing(dl);
    const double mode = 0.241 * ds + 0.0129;
    const double p1 = 1.68e5 * std::pow(ds, 2.33);
    const double p2 = (43.4 * (dl + 1.81) * (dl + 1.81) - 159.0) / ds -
                      3870.0 * (dl - 0.285) * (dl - 0.285) - 58.1;
    double peak;
    if (ds <= ds0) {
        peak = p1;
    } else if (ds >= 1.2 * ds0) {
        peak = p2;
    } else {
        const double alpha = (ds - ds0) / (0.2 * ds0);
        peak = (1.0 - alpha) * p1 + alpha * p2;
    }
    if (!(peak > 0.0)) peak = p1;
    out[0].ok = true;
    if (ff - 2.0 > 1e-9) {
        const double sig = lognormal_sigma(mode, peak, ff - 2.0);
        if (sig == sig)
            out[0].parts.push_back({false, std::log(mode) + sig * sig, sig,
                                    peak * mode * std::exp(0.5 * sig * sig), kD0, d_coal});
    }
    // sheet
    {
        const double mode_s = 0.254 * std::pow(ds, 0.413) * std::exp(3.53 * std::pow(ds, 2.51) * (dl - ds));
        const double peak_s = 0.23 * std::pow(ds, -3.93) * std::pow(dl, 14.2 * std::exp(-17.2 * ds));
        const double sig = lognormal_sigma(mode_s, peak_s, fs - 1.0);
        out[1].ok = sig == sig;
        if (out[1].ok)
            out[1].parts.push_back({false, std::log(mode_s) + sig * sig, sig,
                                    peak_s * mode_s * std::exp(0.5 * sig * sig), kD0, d_coal});
    }
    // disk
    {
        const double mode_d = std::exp(-17.4 * ds - 0.671 * (dl - ds)) * ds;
        const double peak_d = 0.0884 * std::pow(ds, -2.52) * std::pow(dl - ds, 0.007 * std::pow(ds, -2.54));
        const double sig = lognormal_sigma(mode_d, peak_d, fd - 1.0);
        out[2].ok = sig == sig;
        if (out[2].ok)
            out[2].parts.push_back({false, std::log(mode_d) + sig * sig, sig,
                                    peak_d * mode_d * std::exp(0.5 * sig * sig), kD0, d_coal});
    }
}

// ---------------------------------------------------------------------------------------------
// fixed pivots

void assign(double x, double number, const std::vector<double>& cube, double* out) {
    const size_t n = cube.size();
    if (x <= cube[0]) {
        out[0] += number * x / cube[0];
    } else if (x >= cube[n - 1]) {
        out[n - 1] += number * x / cube[n - 1];
    } else {
        const size_t k = std::upper_bound(cube.begin(), cube.end(), x) - cube.begin() - 1;
        const double t = (x - cube[k]) / (cube[k + 1] - cube[k]);
        out[k] += number * (1.0 - t);
        out[k + 1] += number * t;
    }
}

double bin_component(const Comp& c, const std::vector<double>& edges, const std::vector<double>& cube,
                     double* out) {
    double carried = 0.0;
    const size_t n = cube.size();
    for (size_t k = 0; k < n; ++k) {
        double a = k == 0 ? c.lo : std::max(c.lo, edges[k]);
        double b = k == n - 1 ? c.hi : std::min(c.hi, edges[k + 1]);
        a = std::max(a, c.lo);
        b = std::min(b, c.hi);
        double num, x3;
        if (c.gauss) gauss_moments(c.mu, c.sig, a, b, c.h, num, x3);
        else lognormal_moments(c.mu, c.sig, a, b, c.h, num, x3);
        if (num > 0.0 && x3 > 0.0) {
            assign(x3 / num, num, cube, out);
            carried += x3;
        }
    }
    return carried;
}

// Average outcome of a collision (coalescence efficiency; breakup and coalesced drop at the pivots).
double pair_outcome(double dl, double ds, double vl, double vs, const std::vector<double>& edges,
                    const std::vector<double>& cube, std::vector<double>& brk, std::vector<double>& coal) {
    const size_t n = cube.size();
    const Energies e = energies(dl, ds, vl, vs);
    const double eff = coalescence_efficiency(dl, ds, e);
    const double x0 = dl * dl * dl + ds * ds * ds;
    double w[3];
    breakup_fractions(e.cke, e.st, w[0], w[1], w[2]);
    TypeParts comps[3];
    make_components(dl, ds, e.cke, comps);
    std::vector<std::vector<double>> types(3);
    bool have[3] = {false, false, false};
    for (int t = 0; t < 3; ++t) {
        std::vector<double> out(n, 0.0);
        double carried = 0.0;
        for (const Comp& c : comps[t].parts) carried += bin_component(c, edges, cube, out.data());
        double remnant = x0 - carried;
        if (t != 0 && (!comps[t].ok || remnant < kXMin * x0)) {
            w[0] += w[t];
            w[t] = 0.0;
            continue;
        }
        assign(remnant, 1.0, cube, out.data());
        types[t] = std::move(out);
        have[t] = true;
    }
    std::fill(brk.begin(), brk.end(), 0.0);
    for (int t = 0; t < 3; ++t)
        if (have[t])
            for (size_t k = 0; k < n; ++k) brk[k] += w[t] * types[t][k];
    std::fill(coal.begin(), coal.end(), 0.0);
    assign(x0, 1.0, cube, coal.data());
    return eff;
}

// ---------------------------------------------------------------------------------------------

py::tuple build(const DArray& diam, const DArray& v0, int mode, int n_threads) {
    if (diam.ndim() != 1 || v0.ndim() != 1 || diam.size() != v0.size() || diam.size() < 2)
        throw std::invalid_argument("diam and v0 must be one-dimensional of the same length, at least 2");
    if (mode < 0 || mode > 2) throw std::invalid_argument("mode must be 0, 1 or 2");
    const int64_t n = diam.size();
    const double* d = diam.data();
    const double* v = v0.data();
    for (int64_t k = 0; k < n; ++k)
        if (!(d[k] > 0.0) || (k > 0 && !(d[k] > d[k - 1])))
            throw std::invalid_argument("diameters must be positive and ascending");
    py::array_t<double> kernel({n, n});
    py::array_t<double> gain({n, n, n});
    py::array_t<double> eff({n, n});
    double* pk = kernel.mutable_data();
    double* pg = gain.mutable_data();
    double* pe = eff.mutable_data();
    std::fill(pg, pg + n * n * n, 0.0);
    std::fill(pe, pe + n * n, 0.0);
    std::vector<double> dcm(n), cube(n);
    for (int64_t k = 0; k < n; ++k) {
        dcm[k] = d[k] * 0.1;
        cube[k] = dcm[k] * dcm[k] * dcm[k];
    }
    std::vector<double> edges(n + 1);
    for (int64_t k = 1; k < n; ++k) edges[k] = std::sqrt(dcm[k - 1] * dcm[k]);
    edges[0] = dcm[0] * dcm[0] / edges[1];
    edges[n] = dcm[n - 1] * dcm[n - 1] / edges[n - 1];
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < n; ++j) {
            const double dsum = (d[i] + d[j]) * 1e-3;
            pk[i * n + j] = 0.25 * kPi * dsum * dsum * std::fabs(v[i] - v[j]);
        }
    {
        py::gil_scoped_release release;
        parallel_for(n, n_threads, [&](int64_t i) {
            if (mode == 0 || i == 0) return;
            std::vector<double> brk(n), coal(n);
            for (int64_t j = 0; j < i; ++j) {
                if (!(pk[i * n + j] > 0.0)) continue;
                double e = 1.0;
                double* g = pg + (i * n + j) * n;
                if (mode == 1) {
                    std::fill(coal.begin(), coal.end(), 0.0);
                    assign(cube[i] + cube[j], 1.0, cube, coal.data());
                    for (int64_t k = 0; k < n; ++k) g[k] = coal[k];
                } else {
                    e = pair_outcome(dcm[i], dcm[j], v[i], v[j], edges, cube, brk, coal);
                    for (int64_t k = 0; k < n; ++k) g[k] = (1.0 - e) * brk[k] + e * coal[k];
                }
                g[i] -= 1.0;
                g[j] -= 1.0;
                pe[i * n + j] = e;
            }
        });
    }
    return py::make_tuple(kernel, gain, eff);
}

struct Shaft {
    int64_t n;
    const double* kernel;
    const double* gain;
    const double* v0;
    const double* zs;
    const double* ss;
    int64_t nz;
};

double factor_at(const Shaft& sh, double z) {
    if (z <= sh.zs[0]) return sh.ss[0];
    if (z >= sh.zs[sh.nz - 1]) return sh.ss[sh.nz - 1];
    const int64_t i = std::upper_bound(sh.zs, sh.zs + sh.nz, z) - sh.zs - 1;
    const double w = (z - sh.zs[i]) / (sh.zs[i + 1] - sh.zs[i]);
    return sh.ss[i] + w * (sh.ss[i + 1] - sh.ss[i]);
}

// d flux / dz into out for the flux f at the distance z
void deriv(const Shaft& sh, const double* f, double z, double* conc, double* out) {
    const int64_t n = sh.n;
    const double s = factor_at(sh, z);
    for (int64_t k = 0; k < n; ++k) {
        conc[k] = f[k] / (sh.v0[k] * s);
        out[k] = 0.0;
    }
    for (int64_t i = 1; i < n; ++i) {
        for (int64_t j = 0; j < i; ++j) {
            const double r = s * sh.kernel[i * n + j] * conc[i] * conc[j];
            if (r == 0.0) continue;
            const double* g = sh.gain + (i * n + j) * n;
            for (int64_t k = 0; k < n; ++k) out[k] += r * g[k];
        }
    }
}

void evolve_column(const Shaft& sh, const double* n0, const double* z_out, int64_t nout, double dz_max,
                   double cfl, double* result) {
    const int64_t n = sh.n;
    std::vector<double> flux(n), conc(n), k1(n), k2(n), k3(n), k4(n), tmp(n);
    const double s0 = factor_at(sh, 0.0);
    for (int64_t k = 0; k < n; ++k) flux[k] = n0[k] * sh.v0[k] * s0;
    double z = 0.0;
    for (int64_t q = 0; q < nout; ++q) {
        const double target = z_out[q];
        while (z < target) {
            const double s = factor_at(sh, z);
            double lam = 0.0;
            for (int64_t i = 0; i < n; ++i) {
                double sum = 0.0;
                for (int64_t j = 0; j < n; ++j) sum += sh.kernel[i * n + j] * (flux[j] / (sh.v0[j] * s));
                lam = std::max(lam, sum / sh.v0[i]);
            }
            double dz = std::min(dz_max, target - z);
            if (lam > 0.0) dz = std::min(dz, cfl / lam);
            deriv(sh, flux.data(), z, conc.data(), k1.data());
            for (int64_t k = 0; k < n; ++k) tmp[k] = flux[k] + 0.5 * dz * k1[k];
            deriv(sh, tmp.data(), z + 0.5 * dz, conc.data(), k2.data());
            for (int64_t k = 0; k < n; ++k) tmp[k] = flux[k] + 0.5 * dz * k2[k];
            deriv(sh, tmp.data(), z + 0.5 * dz, conc.data(), k3.data());
            for (int64_t k = 0; k < n; ++k) tmp[k] = flux[k] + dz * k3[k];
            deriv(sh, tmp.data(), z + dz, conc.data(), k4.data());
            for (int64_t k = 0; k < n; ++k)
                flux[k] = flux[k] + dz / 6.0 * (k1[k] + 2.0 * k2[k] + 2.0 * k3[k] + k4[k]);
            z = z + dz < target ? z + dz : target;
        }
        const double sz = factor_at(sh, z);
        for (int64_t k = 0; k < n; ++k) result[q * n + k] = flux[k] / (sh.v0[k] * sz);
    }
}

py::array_t<double> evolve(const DArray& kernel, const DArray& gain, const DArray& v0, const DArray& n0,
                           const DArray& z_out, double dz_max, double cfl, const DArray& zs,
                           const DArray& ss, int n_threads) {
    if (kernel.ndim() != 2 || gain.ndim() != 3 || v0.ndim() != 1 || n0.ndim() != 2 ||
        z_out.ndim() != 1 || zs.ndim() != 1 || ss.ndim() != 1)
        throw std::invalid_argument("bad array dimensions");
    const int64_t n = v0.size();
    if (kernel.shape(0) != n || kernel.shape(1) != n || gain.shape(0) != n || gain.shape(1) != n ||
        gain.shape(2) != n || n0.shape(1) != n)
        throw std::invalid_argument("the arrays must share the number of pivots");
    if (zs.size() != ss.size() || zs.size() < 1) throw std::invalid_argument("zs and ss must match");
    if (!(dz_max > 0.0) || !(cfl > 0.0)) throw std::invalid_argument("dz_max and cfl must be positive");
    for (int64_t k = 0; k < n; ++k)
        if (!(v0.data()[k] > 0.0)) throw std::invalid_argument("fall speeds must be positive");
    for (int64_t q = 0; q < z_out.size(); ++q)
        if (!(z_out.data()[q] >= 0.0) || (q > 0 && !(z_out.data()[q] >= z_out.data()[q - 1])))
            throw std::invalid_argument("z_out must be ascending and not negative");
    for (int64_t q = 1; q < zs.size(); ++q)
        if (!(zs.data()[q] > zs.data()[q - 1])) throw std::invalid_argument("zs must ascend");
    for (int64_t q = 0; q < ss.size(); ++q)
        if (!(ss.data()[q] > 0.0)) throw std::invalid_argument("ss must be positive");
    const int64_t ncol = n0.shape(0), nout = z_out.size();
    py::array_t<double> out({nout, ncol, n});
    double* po = out.mutable_data();
    const Shaft sh{n, kernel.data(), gain.data(), v0.data(), zs.data(), ss.data(), zs.size()};
    const double* pn = n0.data();
    const double* pz = z_out.data();
    {
        py::gil_scoped_release release;
        parallel_for(ncol, n_threads, [&](int64_t c) {
            std::vector<double> res(nout * n);
            evolve_column(sh, pn + c * n, pz, nout, dz_max, cfl, res.data());
            for (int64_t q = 0; q < nout; ++q)
                for (int64_t k = 0; k < n; ++k) po[(q * ncol + c) * n + k] = res[q * n + k];
        });
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(_rain_collisions, m) {
    m.doc() = "Raindrop collisions, coalescence and breakup in a rain shaft";
    m.def("build", &build, py::arg("diam_mm"), py::arg("v0"), py::arg("mode"), py::arg("n_threads") = 0);
    m.def("evolve", &evolve, py::arg("kernel"), py::arg("gain"), py::arg("v0"), py::arg("n0"),
          py::arg("z_out"), py::arg("dz_max"), py::arg("cfl"), py::arg("zs"), py::arg("ss"),
          py::arg("n_threads") = 0);
}
