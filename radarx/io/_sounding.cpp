// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Profile kernels for radarx.io.sounding.
//
// The Python layer (radarx/io/sounding.py) pulls contiguous float64 arrays
// out of xarray objects, calls these functions once for all columns, times
// and target points, and wraps the results back into xarray. Every function
// releases the GIL and splits its work over std::thread workers that take
// blocks from an atomic counter (dynamic scheduling). The NumPy reference
// implementation in radarx/io/_sounding_numpy.py gives the same results.
//
// Profiles are columns of levels with heights ascending; NaN marks a missing
// value. Interpolation is linear in height (linear in log for flagged
// variables such as pressure) between the nearest valid levels.

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

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kPi = 3.14159265358979323846;  // M_PI is not standard (MSVC)
constexpr double kDeg = kPi / 180.0;

// Physical constants (SI); keep in sync with radarx/io/_sounding_numpy.py.
constexpr double kRd = 287.04749;    // gas constant of dry air [J kg-1 K-1]
constexpr double kRv = 461.52311;    // gas constant of water vapour [J kg-1 K-1]
constexpr double kEps = kRd / kRv;   // ratio of the gas constants
constexpr double kCpd = 1005.7;      // specific heat of dry air [J kg-1 K-1]
constexpr double kCpv = 1875.0;      // specific heat of water vapour [J kg-1 K-1]
constexpr double kT0 = 273.15;       // 0 degC [K]
constexpr double kG0 = 9.80665;      // standard gravity [m s-2]
constexpr double kRe = 6371008.8;    // mean Earth radius [m]

// Run fn(begin, end) over [0, n) in blocks taken from an atomic counter.
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

// Largest i with a[i] <= x (a ascending, n >= 1), -1 if x < a[0].
inline int64_t bisect_right(const double* a, int64_t n, double x) {
    if (x < a[0]) return -1;
    int64_t lo = 0, hi = n;  // invariant: a[lo] <= x, a[hi] > x (hi == n: none)
    while (hi - lo > 1) {
        int64_t mid = (lo + hi) >> 1;
        if (a[mid] <= x) lo = mid; else hi = mid;
    }
    return lo;
}

// Saturation vapour pressure over liquid water [Pa], Bolton (1980) eq. (10).
inline double esat(double t) {
    const double tc = t - kT0;
    return 611.2 * std::exp(17.67 * tc / (tc + 243.5));
}

// Dew point [K] for vapour pressure e [Pa]: eq. (10) inverted.
inline double dewpoint(double e) {
    if (!(e > 0.0)) return kNaN;
    const double l = std::log(e / 611.2);
    return kT0 + 243.5 * l / (17.67 - l);
}

// Latent heat of vaporization [J kg-1], Bolton (1980) eq. (2).
inline double latent_heat(double t) { return (2.501 - 0.00237 * (t - kT0)) * 1e6; }

inline double mixing_ratio(double e, double p) { return kEps * e / (p - e); }

// Isobaric wet-bulb temperature [K]: the root of
// (cpd + r cpv) (T - Tw) = L(Tw) (rs(Tw) - r), bracketed by [Td, T].
double wet_bulb(double p, double t, double td) {
    if (std::isnan(p) || std::isnan(t) || std::isnan(td)) return kNaN;
    if (td > t) td = t;
    const double e = esat(td);
    const double r = mixing_ratio(e, p);
    const double cp = kCpd + r * kCpv;
    auto f = [&](double tw) {
        const double es = esat(tw);
        return cp * (t - tw) - latent_heat(tw) * (mixing_ratio(es, p) - r);
    };
    double lo = td, hi = t;  // f(lo) >= 0 >= f(hi), f decreasing
    double tw = 0.5 * (lo + hi);
    for (int it = 0; it < 60; ++it) {
        const double fv = f(tw);
        if (fv > 0) lo = tw; else hi = tw;
        if (hi - lo < 1e-7) break;
        const double h = 1e-4;
        const double df = (f(tw + h) - f(tw - h)) / (2 * h);
        double next = df != 0.0 ? tw - fv / df : 0.5 * (lo + hi);
        if (!(next > lo && next < hi)) next = 0.5 * (lo + hi);  // keep the bracket
        if (std::fabs(next - tw) < 1e-7) {
            tw = next;
            break;
        }
        tw = next;
    }
    return tw;
}

// ---------------------------------------------------------------------------
// Elementwise thermodynamics on flat, already broadcast arrays.
// op: "esat"(T), "dewpoint"(e), "vapor_pressure"(q, p), "specific_humidity"(e, p),
//     "wet_bulb"(p, T, Td), "height"(geopotential), "density"(p, T, q)
py::array_t<double> thermo(const std::string& op, const std::vector<DArray>& inputs,
                           int n_threads) {
    if (inputs.empty()) throw std::invalid_argument("no inputs");
    const int64_t n = inputs[0].size();
    for (const auto& a : inputs)
        if (a.size() != n) throw std::invalid_argument("inputs must have the same size");
    std::vector<const double*> in;
    for (const auto& a : inputs) in.push_back(a.data());
    auto need = [&](size_t k) {
        if (inputs.size() != k) throw std::invalid_argument(op + " needs " + std::to_string(k) + " inputs");
    };
    int code = -1;
    if (op == "esat") { need(1); code = 0; }
    else if (op == "dewpoint") { need(1); code = 1; }
    else if (op == "vapor_pressure") { need(2); code = 2; }
    else if (op == "specific_humidity") { need(2); code = 3; }
    else if (op == "wet_bulb") { need(3); code = 4; }
    else if (op == "height") { need(1); code = 5; }
    else if (op == "density") { need(3); code = 6; }
    else throw std::invalid_argument("unknown op " + op);

    py::array_t<double> result(n);
    double* out = result.mutable_data();
    {
        py::gil_scoped_release release;
        parallel_for(n, code == 4 ? 1024 : 65536, n_threads, [&](int64_t i0, int64_t i1) {
            for (int64_t i = i0; i < i1; ++i) {
                double v;
                switch (code) {
                    case 0: v = esat(in[0][i]); break;
                    case 1: v = dewpoint(in[0][i]); break;
                    case 2: {  // e from q and p
                        const double q = in[0][i];
                        v = q * in[1][i] / (kEps + (1.0 - kEps) * q);
                        break;
                    }
                    case 3: {  // q from e and p
                        const double e = in[0][i];
                        v = kEps * e / (in[1][i] - (1.0 - kEps) * e);
                        break;
                    }
                    case 4: v = wet_bulb(in[0][i], in[1][i], in[2][i]); break;
                    case 5: {  // geometric height from geopotential
                        const double h = in[0][i] / kG0;
                        v = kRe * h / (kRe - h);
                        break;
                    }
                    default: {  // density of moist air
                        const double tv = in[1][i] * (1.0 + (1.0 / kEps - 1.0) * in[2][i]);
                        v = in[0][i] / (kRd * tv);
                        break;
                    }
                }
                out[i] = v;
            }
        });
    }
    return result;
}

// Nearest valid level (finite value and height) at or below index a, or -1.
inline int64_t valid_below(const double* zc, const double* vc, int64_t a) {
    while (a >= 0 && (std::isnan(vc[a]) || std::isnan(zc[a]))) --a;
    return a;
}

// Nearest valid level at or above index b, or n.
inline int64_t valid_above(const double* zc, const double* vc, int64_t b, int64_t n) {
    while (b < n && (std::isnan(vc[b]) || std::isnan(zc[b]))) ++b;
    return b;
}

// Linear (or log-linear) interpolation between levels a and b at height zt.
inline double interp_pair(const double* zc, const double* vc, int64_t a, int64_t b, double zt,
                          bool lg) {
    const double w = (zt - zc[a]) / (zc[b] - zc[a]);
    if (lg) return std::exp(std::log(vc[a]) + w * (std::log(vc[b]) - std::log(vc[a])));
    return vc[a] + w * (vc[b] - vc[a]);
}

// Value of one profile column at height zt; lo is bisect_right(zc, nlev, zt).
inline double interp_at(const double* zc, const double* vc, int64_t nlev, int64_t lo, double zt,
                        bool lg, bool extrapolate) {
    const int64_t a = valid_below(zc, vc, std::min(lo, nlev - 1));
    const int64_t b = valid_above(zc, vc, lo + 1, nlev);
    if (a >= 0 && zc[a] == zt) return vc[a];
    if (a >= 0 && b < nlev) return interp_pair(zc, vc, a, b, zt, lg);
    if (!extrapolate) return kNaN;
    if (a >= 0) return vc[a];
    return b < nlev ? vc[b] : kNaN;
}

// ---------------------------------------------------------------------------
// Vertical interpolation.
// z: (nprof, nlev) ascending heights; values: (nvar, nprof, nlev);
// target: (nrow, nt) with nprof == 1 or nprof == nrow; log_var: (nvar).
// Returns (nvar, nrow, nt).
py::array_t<double> interp_vertical(const DArray& z, const DArray& values, const DArray& target,
                                    const py::array_t<bool, py::array::c_style | py::array::forcecast>& log_var,
                                    bool extrapolate, int n_threads) {
    if (z.ndim() != 2 || values.ndim() != 3 || target.ndim() != 2)
        throw std::invalid_argument("expected z (nprof, nlev), values (nvar, nprof, nlev), target (nrow, nt)");
    const int64_t nprof = z.shape(0), nlev = z.shape(1);
    const int64_t nvar = values.shape(0);
    const int64_t nrow = target.shape(0), nt = target.shape(1);
    if (values.shape(1) != nprof || values.shape(2) != nlev)
        throw std::invalid_argument("values must be (nvar, nprof, nlev)");
    if (nprof != 1 && nprof != nrow)
        throw std::invalid_argument("need one profile or one profile per target row");
    if (log_var.size() != nvar) throw std::invalid_argument("log_var must have nvar entries");
    if (nlev < 1) throw std::invalid_argument("empty profile");

    py::array_t<double> result({nvar, nrow, nt});
    double* out = result.mutable_data();
    const double* pz = z.data();
    const double* pv = values.data();
    const double* pt = target.data();
    std::vector<char> logv(nvar);
    for (int64_t k = 0; k < nvar; ++k) logv[k] = log_var.data()[k];

    {
        py::gil_scoped_release release;
        parallel_for(nrow * nt, 4096, n_threads, [&](int64_t i0, int64_t i1) {
            for (int64_t i = i0; i < i1; ++i) {
                const int64_t row = i / nt;
                const int64_t prof = nprof == 1 ? 0 : row;
                const double* zc = pz + prof * nlev;
                const double zt = pt[i];
                const int64_t lo = std::isnan(zt) ? -2 : bisect_right(zc, nlev, zt);
                for (int64_t k = 0; k < nvar; ++k) {
                    const double* vc = pv + (k * nprof + prof) * nlev;
                    const double res =
                        lo == -2 ? kNaN : interp_at(zc, vc, nlev, lo, zt, logv[k], extrapolate);
                    out[(k * nrow + row) * nt + (i % nt)] = res;
                }
            }
        });
    }
    return result;
}

// ---------------------------------------------------------------------------
// Height where values cross `level` going up from >= level to < level
// (e.g. temperature through an isotherm). z, v: (ncol, nlev); NaN levels are
// skipped. highest=true returns the top crossing (top of a warm layer).
py::array_t<double> level_crossing(const DArray& z, const DArray& v, double level, bool highest,
                                   int n_threads) {
    if (z.ndim() != 2 || v.ndim() != 2 || z.shape(0) != v.shape(0) || z.shape(1) != v.shape(1))
        throw std::invalid_argument("z and values must both be (ncol, nlev)");
    const int64_t ncol = z.shape(0), nlev = z.shape(1);
    py::array_t<double> result(ncol);
    double* out = result.mutable_data();
    const double* pz = z.data();
    const double* pv = v.data();
    {
        py::gil_scoped_release release;
        parallel_for(ncol, 1024, n_threads, [&](int64_t c0, int64_t c1) {
            for (int64_t c = c0; c < c1; ++c) {
                const double* zc = pz + c * nlev;
                const double* vc = pv + c * nlev;
                double res = kNaN;
                int64_t prev = -1;  // previous valid level
                for (int64_t l = 0; l < nlev; ++l) {
                    if (std::isnan(vc[l]) || std::isnan(zc[l])) continue;
                    if (prev >= 0 && vc[prev] >= level && vc[l] < level) {
                        const double w = (vc[prev] - level) / (vc[prev] - vc[l]);
                        res = zc[prev] + w * (zc[l] - zc[prev]);
                        if (!highest) break;
                    }
                    prev = l;
                }
                out[c] = res;
            }
        });
    }
    return result;
}

// ---------------------------------------------------------------------------
// Height-weighted layer mean over [bottom, top] of the piecewise-linear
// profile through the valid levels. NaN unless valid data span the layer.
// z: (ncol, nlev), values: (nvar, ncol, nlev). Returns (nvar, ncol).
py::array_t<double> layer_mean(const DArray& z, const DArray& values, double bottom, double top,
                               int n_threads) {
    if (z.ndim() != 2 || values.ndim() != 3 || values.shape(1) != z.shape(0) ||
        values.shape(2) != z.shape(1))
        throw std::invalid_argument("expected z (ncol, nlev) and values (nvar, ncol, nlev)");
    if (!(top > bottom)) throw std::invalid_argument("top must be above bottom");
    const int64_t ncol = z.shape(0), nlev = z.shape(1), nvar = values.shape(0);
    py::array_t<double> result({nvar, ncol});
    double* out = result.mutable_data();
    const double* pz = z.data();
    const double* pv = values.data();
    {
        py::gil_scoped_release release;
        parallel_for(nvar * ncol, 256, n_threads, [&](int64_t i0, int64_t i1) {
            for (int64_t i = i0; i < i1; ++i) {
                const int64_t k = i / ncol, c = i % ncol;
                const double* zc = pz + c * nlev;
                const double* vc = pv + (k * ncol + c) * nlev;
                double integral = 0.0, lowest = kNaN, highest = kNaN;
                int64_t prev = -1;
                for (int64_t l = 0; l < nlev; ++l) {
                    if (std::isnan(vc[l]) || std::isnan(zc[l])) continue;
                    if (std::isnan(lowest)) lowest = zc[l];
                    highest = zc[l];
                    if (prev >= 0) {
                        const double za = zc[prev], zb = zc[l];
                        const double a = std::max(za, bottom), b = std::min(zb, top);
                        if (b > a && zb > za) {
                            const double slope = (vc[l] - vc[prev]) / (zb - za);
                            const double va = vc[prev] + slope * (a - za);
                            const double vb = vc[prev] + slope * (b - za);
                            integral += 0.5 * (va + vb) * (b - a);
                        }
                    }
                    prev = l;
                }
                out[i] = (lowest <= bottom && highest >= top) ? integral / (top - bottom) : kNaN;
            }
        });
    }
    return result;
}

// ---------------------------------------------------------------------------
// Columns at arbitrary points from fields on a regular latitude/longitude grid,
// bilinear in the horizontal and linear in time.
// fields: (ntime, nvar, nlev, nlat, nlon); time_weights: (ntime);
// lat, lon: ascending axes; qlat, qlon: (npts). Returns (nvar, npts, nlev).
// Points outside the grid, or next to a missing value, get NaN.
py::array_t<double> bilinear_columns(const DArray& fields, const DArray& time_weights,
                                     const DArray& lat, const DArray& lon, const DArray& qlat,
                                     const DArray& qlon, int n_threads) {
    if (fields.ndim() != 5) throw std::invalid_argument("fields must be (ntime, nvar, nlev, nlat, nlon)");
    const int64_t ntime = fields.shape(0), nvar = fields.shape(1), nlev = fields.shape(2);
    const int64_t nlat = fields.shape(3), nlon = fields.shape(4);
    const int64_t npts = qlat.size();
    if (time_weights.size() != ntime || lat.size() != nlat || lon.size() != nlon ||
        qlon.size() != npts)
        throw std::invalid_argument("inconsistent shapes");
    if (nlat < 2 || nlon < 2) throw std::invalid_argument("need at least 2 x 2 grid points");
    py::array_t<double> result({nvar, npts, nlev});
    double* out = result.mutable_data();
    const double* pf = fields.data();
    const double* pw = time_weights.data();
    const double* pla = lat.data();
    const double* plo = lon.data();
    const double* pqa = qlat.data();
    const double* pqo = qlon.data();
    const int64_t plane = nlat * nlon;
    {
        py::gil_scoped_release release;
        parallel_for(npts, 64, n_threads, [&](int64_t p0, int64_t p1) {
            for (int64_t p = p0; p < p1; ++p) {
                const double y = pqa[p], x = pqo[p];
                const bool inside = y >= pla[0] && y <= pla[nlat - 1] && x >= plo[0] &&
                                    x <= plo[nlon - 1];
                int64_t j = 0, i = 0;
                double fy = 0.0, fx = 0.0;
                if (inside) {
                    j = std::min(bisect_right(pla, nlat, y), nlat - 2);
                    i = std::min(bisect_right(plo, nlon, x), nlon - 2);
                    fy = (y - pla[j]) / (pla[j + 1] - pla[j]);
                    fx = (x - plo[i]) / (plo[i + 1] - plo[i]);
                }
                const double w[4] = {(1 - fy) * (1 - fx), (1 - fy) * fx, fy * (1 - fx), fy * fx};
                const int64_t idx[4] = {j * nlon + i, j * nlon + i + 1, (j + 1) * nlon + i,
                                        (j + 1) * nlon + i + 1};
                for (int64_t k = 0; k < nvar; ++k) {
                    for (int64_t l = 0; l < nlev; ++l) {
                        double res = kNaN;
                        if (inside) {
                            res = 0.0;
                            for (int64_t t = 0; t < ntime && !std::isnan(res); ++t) {
                                if (pw[t] == 0.0) continue;
                                const double* f = pf + ((t * nvar + k) * nlev + l) * plane;
                                for (int q = 0; q < 4; ++q) res += pw[t] * w[q] * f[idx[q]];
                            }
                        }
                        out[(k * npts + p) * nlev + l] = res;
                    }
                }
            }
        });
    }
    return result;
}

// ---------------------------------------------------------------------------
// Rotate earth-relative winds to grid axes. angle: direction of true north
// measured clockwise from the grid +y axis [deg]. Returns (2, n): x and y
// components.
py::array_t<double> rotate_wind(const DArray& u, const DArray& v, const DArray& angle,
                                int n_threads) {
    const int64_t n = u.size();
    if (v.size() != n || angle.size() != n) throw std::invalid_argument("u, v, angle sizes differ");
    py::array_t<double> result({static_cast<int64_t>(2), n});
    double* out = result.mutable_data();
    const double* pu = u.data();
    const double* pv = v.data();
    const double* pa = angle.data();
    {
        py::gil_scoped_release release;
        parallel_for(n, 65536, n_threads, [&](int64_t i0, int64_t i1) {
            for (int64_t i = i0; i < i1; ++i) {
                const double s = std::sin(pa[i] * kDeg), c = std::cos(pa[i] * kDeg);
                out[i] = pu[i] * c + pv[i] * s;
                out[n + i] = -pu[i] * s + pv[i] * c;
            }
        });
    }
    return result;
}

}  // namespace

PYBIND11_MODULE(_sounding, m) {
    m.doc() = "Compiled profile kernels for radarx.io.sounding.";
    m.def("thermo", &thermo, py::arg("op"), py::arg("inputs"), py::arg("n_threads") = 0);
    m.def("interp_vertical", &interp_vertical, py::arg("z"), py::arg("values"), py::arg("target"),
          py::arg("log_var"), py::arg("extrapolate") = false, py::arg("n_threads") = 0);
    m.def("level_crossing", &level_crossing, py::arg("z"), py::arg("values"), py::arg("level"),
          py::arg("highest") = true, py::arg("n_threads") = 0);
    m.def("layer_mean", &layer_mean, py::arg("z"), py::arg("values"), py::arg("bottom"),
          py::arg("top"), py::arg("n_threads") = 0);
    m.def("bilinear_columns", &bilinear_columns, py::arg("fields"), py::arg("time_weights"),
          py::arg("lat"), py::arg("lon"), py::arg("qlat"), py::arg("qlon"),
          py::arg("n_threads") = 0);
    m.def("rotate_wind", &rotate_wind, py::arg("u"), py::arg("v"), py::arg("angle"),
          py::arg("n_threads") = 0);
}
