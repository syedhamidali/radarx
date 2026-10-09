// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Compiled kernel of radarx.retrieve.lagrangian and
// radarx.retrieve.diabatic_lagrangian: gridpoint air trajectories through
// time-dependent 3-D winds and the diabatic Lagrangian analysis (DLA) of
// Ziegler (2013a, b). The sources of the numbers are given, with equation, table
// and page pointers, in the documentation of the Python modules
// radarx/retrieve/lagrangian.py and diabatic_lagrangian.py and in
// _lagrangian_numpy.py; here Z13a = Ziegler (2013a, J. Atmos. Oceanic Technol.
// 30, 2248-2265, doi:10.1175/JTECH-D-12-00194.1), Z07 = Ziegler et al. (2007,
// Mon. Wea. Rev. 135, 2417-2442, doi:10.1175/MWR3396.1) and LFO83 = Lin, Farley
// and Orville (1983, J. Climate Appl. Meteor. 22, 1065-1092,
// doi:10.1175/1520-0450(1983)022<1065:BPOTSF>2.0.CO;2). The LFO83 rates are
// implemented from LFO83 itself, not from the "modified LFO" supplement of
// Gilmore et al. (2004a), which was not consulted. Numbers without a pointer
// are radarx choices. The NumPy reference implementation in
// radarx/retrieve/_lagrangian_numpy.py follows the same steps in the same
// order and is the test oracle of this file.
//
// Gridded inputs are float32 arrays of shape (nt, nz, ny, nx, nvar) with the
// variables of one grid node next to each other (one cache line per node).
// The winds hold (u, v, w, Z_H, valid, environment mask); "valid" is 1 where
// the wind is analysed and 0 where it is a background, "environment mask" is 1
// where a parcel may count as environmental (termination mode 2).
// Times are seconds relative to the analysis time. All work is spread over
// std::thread workers with an atomic chunk counter; the GIL is released.

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
using F32 = py::array_t<float, py::array::c_style | py::array::forcecast>;
using F64 = py::array_t<double, py::array::c_style | py::array::forcecast>;
using I8 = py::array_t<int8_t, py::array::c_style | py::array::forcecast>;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kPi = 3.14159265358979323846;
constexpr int64_t kChunk = 16;

// Trajectory flags (bit mask)
constexpr int kEnvDbz = 1;      // N > min_steps and Z_H < env_dbz (Ziegler 2013a, sect. 2a, i)
constexpr int kEnvW = 2;        // N > min_steps and w < env_w for env_w_steps steps (ii)
constexpr int kBoundary = 4;    // left the analysed domain through a lateral boundary (iii)
constexpr int kEndOfData = 8;   // reached the end of the wind time series
constexpr int kMaxSteps = 16;   // reached max_steps
constexpr int kMissing = 32;    // missing (NaN) wind at the parcel
// bit 64 (too little time in valid winds) is set by the Python layer of the DLA
constexpr int kBoundaryStorm = 128;  // left through a lateral boundary that is not environment
constexpr int kNW = 6;          // variables of the wind pack

// Process switches of the DLA (bit mask)
constexpr int kCond = 1;
constexpr int kRevp = 2;
constexpr int kRacw = 4;
constexpr int kGacw = 8;
constexpr int kGmlt = 16;
constexpr int kGsub = 32;
constexpr int kGfr = 64;
constexpr int kDamp = 128;
constexpr int kFlux = 256;
constexpr int kIce = 512;      // mixed-phase adjustment with cloud ice (Tao et al. 1989)

// DLA flag: initial state from in situ observations (Ziegler et al. 2007)
constexpr int kInsitu = 256;

// Thermodynamic constants
constexpr double kT0 = 273.15;
constexpr double kP0 = 1.0e5;
constexpr double kRd = 287.04;             // Ziegler (2013a), sect. 2a
constexpr double kKappa = 0.2854;          // Ziegler (2013a), sect. 2a
constexpr double kCp = kRd / kKappa;      // radarx choice (1005.8; LFO83 list 1005)
constexpr double kRv = 461.5;              // LFO83 appendix (R_w)
constexpr double kEps = kRd / kRv;
constexpr double kEs0 = 611.2;             // Bolton (1980) fit, e_s(0 degC), Pa; not re-checked against the paper

// LFO83 constants (appendix, pp. 1089-1092), SI
constexpr double kLvL = 2.5e6;     // L_v, J kg^-1
constexpr double kLfL = 3.336e5;   // L_f
constexpr double kLsL = 2.8336e6;  // L_s
constexpr double kCw = 4.187e3;    // C_w, J kg^-1 K^-1
constexpr double kAR = 841.99;            // a = 2115 cm^0.2 s^-1 in m^0.2 s^-1, eq. (7), p. 1069
constexpr double kBR = 0.8;               // b, eq. (7)
constexpr double kCD = 0.6;               // C_D, eq. (9), p. 1069
constexpr double kG = 9.805;              // g = 980.5 cm s^-2
constexpr double kApr = 0.66;             // A', Bigg freezing, eq. (45)
constexpr double kBpr = 100.0;            // B' (m^-3 s^-1), eq. (45)
constexpr double kRhoW = 1000.0;

// Tao, Simpson and McCumber (1989), eqs. (3a), (3b), p. 232: a = 17.2693882 and
// 21.8745584, b = 3.8 / P (P in mb)
constexpr double kTaoA1 = 17.2693882;
constexpr double kTaoA2 = 21.8745584;
constexpr double kTaoB = 3.8;
constexpr double kTHom = 233.15;  // homogeneous freezing at T <= -40 degC (LFO83 sect. 3f, p. 1077; Hsie et al. 1980, p. 956)

// ---------------------------------------------------------------------------
// Gridded field sampling: trilinear in space, linear in time, on a grid that
// moves with a constant storm motion (cx, cy) (Ziegler 2013a, sect. 2b;
// Ziegler 2013b, sect. 2c). Before the first and after the last analysis the
// nearest analysis is moved with the storm motion for ext_before / ext_after
// seconds (time morphing, steady state following the storm).
// ---------------------------------------------------------------------------
struct Grid {
    const double *x, *y, *z, *t;
    int64_t nx, ny, nz, nt;
    const float* data;
    int nvar;
    double cx, cy, ext_before, ext_after;
};

inline int64_t bracket(const double* c, int64_t n, double v) {
    const int64_t i = (std::upper_bound(c, c + n, v) - c) - 1;
    return std::min<int64_t>(std::max<int64_t>(i, 0), n - 2);
}

// Returns 0 or a flag (kBoundary, kEndOfData, kMissing when a wind is NaN and
// check_nan is set); out[nvar] holds the values.
inline int sample(const Grid& g, double xp, double yp, double zp, double tp, double* out,
                  bool check_nan) {
    int64_t lev[2] = {0, 0};
    double wt[2] = {1.0, 0.0};
    const double tol = 1.0e-6;
    if (tp < g.t[0]) {
        if (tp < g.t[0] - g.ext_before - tol) return kEndOfData;
        lev[0] = 0;
    } else if (tp > g.t[g.nt - 1]) {
        if (tp > g.t[g.nt - 1] + g.ext_after + tol) return kEndOfData;
        lev[0] = g.nt - 1;
    } else if (g.nt > 1) {
        const int64_t i = bracket(g.t, g.nt, tp);
        const double a = (tp - g.t[i]) / (g.t[i + 1] - g.t[i]);
        lev[0] = i;
        lev[1] = i + 1;
        wt[0] = 1.0 - a;
        wt[1] = a;
    }
    for (int v = 0; v < g.nvar; ++v) out[v] = 0.0;
    const int64_t k = bracket(g.z, g.nz, zp);
    const double fz = (zp - g.z[k]) / (g.z[k + 1] - g.z[k]);
    for (int l = 0; l < 2; ++l) {
        if (!(wt[l] > 0.0)) continue;
        const double dtl = tp - g.t[lev[l]];
        const double xs = xp - g.cx * dtl;
        const double ys = yp - g.cy * dtl;
        if (xs < g.x[0] || xs > g.x[g.nx - 1] || ys < g.y[0] || ys > g.y[g.ny - 1])
            return kBoundary;
        const int64_t i = bracket(g.x, g.nx, xs);
        const int64_t j = bracket(g.y, g.ny, ys);
        const double fx = (xs - g.x[i]) / (g.x[i + 1] - g.x[i]);
        const double fy = (ys - g.y[j]) / (g.y[j + 1] - g.y[j]);
        const double w000 = (1 - fz) * (1 - fy) * (1 - fx), w001 = (1 - fz) * (1 - fy) * fx;
        const double w010 = (1 - fz) * fy * (1 - fx), w011 = (1 - fz) * fy * fx;
        const double w100 = fz * (1 - fy) * (1 - fx), w101 = fz * (1 - fy) * fx;
        const double w110 = fz * fy * (1 - fx), w111 = fz * fy * fx;
        const int64_t sx = g.nvar, sy = g.nx * sx, sz = g.ny * sy;
        const float* b = g.data + (((lev[l] * g.nz + k) * g.ny + j) * g.nx + i) * g.nvar;
        for (int v = 0; v < g.nvar; ++v) {
            const double c = w000 * b[v] + w001 * b[sx + v] + w010 * b[sy + v] +
                             w011 * b[sy + sx + v] + w100 * b[sz + v] + w101 * b[sz + sx + v] +
                             w110 * b[sz + sy + v] + w111 * b[sz + sy + sx + v];
            out[v] += wt[l] * c;
        }
    }
    if (check_nan)
        for (int v = 0; v < 3; ++v)
            if (std::isnan(out[v])) return kMissing;
    return 0;
}

struct PathParams {
    double dt;
    int n_iter;
    int64_t max_steps;
    int dir;
    int mode;  // 0: no environment tests, 1: Ziegler (2013a), 2: outside precipitation
    int64_t min_steps;
    double env_dbz, env_w;
    int64_t env_w_steps, env_dbz_steps;
    double cold_pool_depth, z_sfc;
    // classification of lateral-boundary exits of backward trajectories with an
    // environment test: 0 every exit is environment (Ziegler 2013a, sect. 2a,
    // test iii), 1 only where the environment mask is set at the boundary point,
    // 2 only outside precipitation (Z_H < env_dbz) or where the mask is set,
    // 3 only through the sides in the bit mask `sides`
    int boundary;
    int sides;
};

// Sides of the analysed domain (bit mask) beyond which (x, y) lies.
constexpr int kWest = 1, kEast = 2, kSouth = 4, kNorth = 8;

inline bool outside(const Grid& g, double xp, double yp) {
    return xp < g.x[0] || xp > g.x[g.nx - 1] || yp < g.y[0] || yp > g.y[g.ny - 1];
}

inline double clampz(const Grid& g, double zp) {
    return std::min(std::max(zp, g.z[0]), g.z[g.nz - 1]);
}

inline int sides_of(const Grid& g, double xp, double yp) {
    int s = 0;
    if (xp < g.x[0]) s |= kWest;
    if (xp > g.x[g.nx - 1]) s |= kEast;
    if (yp < g.y[0]) s |= kSouth;
    if (yp > g.y[g.ny - 1]) s |= kNorth;
    return s;
}

// Sides through which a parcel at (xp, yp, tp) left the domain: of the fixed
// grid, else of the analysis that sample() found it outside of (in the frame
// moving with the storm).
int exit_sides(const Grid& g, double xp, double yp, double tp) {
    int s = sides_of(g, xp, yp);
    if (s) return s;
    int64_t lev[2] = {0, 0};
    double wt[2] = {1.0, 0.0};
    if (tp > g.t[g.nt - 1]) {
        lev[0] = g.nt - 1;
    } else if (tp >= g.t[0] && g.nt > 1) {
        const int64_t i = bracket(g.t, g.nt, tp);
        const double a = (tp - g.t[i]) / (g.t[i + 1] - g.t[i]);
        lev[0] = i;
        lev[1] = i + 1;
        wt[0] = 1.0 - a;
        wt[1] = a;
    }
    for (int l = 0; l < 2; ++l) {
        if (!(wt[l] > 0.0)) continue;
        const double dtl = tp - g.t[lev[l]];
        s = sides_of(g, xp - g.cx * dtl, yp - g.cy * dtl);
        if (s) return s;
    }
    return 0;
}

// Flag of a lateral-boundary exit; vlast holds the wind pack at the last
// point inside the domain (the boundary point).
inline int boundary_flag(const PathParams& p, int sides, const double* vlast) {
    if (p.mode == 0) return kBoundary;
    bool env = true;
    if (p.boundary == 1) env = vlast[5] >= 0.5;
    else if (p.boundary == 2) env = vlast[3] < p.env_dbz || vlast[5] >= 0.5;
    else if (p.boundary == 3) env = (sides & p.sides) != 0;
    return env ? kBoundary : kBoundaryStorm;
}

// One trajectory from (x0, y0, z0) at t = 0: predictor (Euler) plus n_iter
// trapezoidal corrector iterations per step: radarx's reading of the
// "first-order predictor corrector scheme as in Z07" with three iterations
// (Z13a sect. 2b, p. 2250; Z07 p. 2422). pos[4 * n] = (x, y, z, t), val[kNW * n] = the wind pack
// at each stored point. Returns the number of stored points.
int64_t build_path(const Grid& g, const PathParams& p, double x0, double y0, double z0,
                   double* pos, double* val, int& flags) {
    flags = 0;
    double xn = x0, yn = y0, zn = clampz(g, z0), tn = 0.0;
    double vn[kNW], v1[kNW];
    if (outside(g, xn, yn)) {
        flags = kBoundary;
        return 0;
    }
    int st = sample(g, xn, yn, zn, tn, vn, true);
    if (st) {
        flags = st;
        return 0;
    }
    pos[0] = xn;
    pos[1] = yn;
    pos[2] = zn;
    pos[3] = tn;
    for (int v = 0; v < kNW; ++v) val[v] = vn[v];
    int64_t npts = 1;
    int64_t wcount = 0, dcount = 0;
    const double h = p.dir * p.dt;
    for (int64_t n = 1; n <= p.max_steps; ++n) {
        const double t1 = tn + h;
        double xs = xn + h * vn[0], ys = yn + h * vn[1], zs = clampz(g, zn + h * vn[2]);
        if (outside(g, xs, ys)) {
            flags |= boundary_flag(p, sides_of(g, xs, ys), vn);
            break;
        }
        bool ok = true;
        for (int it = 0; it < p.n_iter; ++it) {
            st = sample(g, xs, ys, zs, t1, v1, true);
            if (st) {
                flags |= st == kBoundary ? boundary_flag(p, exit_sides(g, xs, ys, t1), vn) : st;
                ok = false;
                break;
            }
            xs = xn + 0.5 * h * (vn[0] + v1[0]);
            ys = yn + 0.5 * h * (vn[1] + v1[1]);
            zs = clampz(g, zn + 0.5 * h * (vn[2] + v1[2]));
            if (outside(g, xs, ys)) {
                flags |= boundary_flag(p, sides_of(g, xs, ys), vn);
                ok = false;
                break;
            }
        }
        if (!ok) break;
        st = sample(g, xs, ys, zs, t1, v1, true);
        if (st) {
            flags |= st == kBoundary ? boundary_flag(p, exit_sides(g, xs, ys, t1), vn) : st;
            break;
        }
        double* q = pos + 4 * npts;
        q[0] = xs;
        q[1] = ys;
        q[2] = zs;
        q[3] = t1;
        for (int v = 0; v < kNW; ++v) val[kNW * npts + v] = v1[v];
        ++npts;
        xn = xs;
        yn = ys;
        zn = zs;
        tn = t1;
        for (int v = 0; v < kNW; ++v) vn[v] = v1[v];
        if (p.mode == 1) {
            // Ziegler (2013a), sect. 2a, tests (i) and (ii)
            wcount = (v1[2] < p.env_w) ? wcount + 1 : 0;
            if (n > p.min_steps) {
                if (v1[3] < p.env_dbz) flags |= kEnvDbz;
                if (wcount >= p.env_w_steps) flags |= kEnvW;
                if (flags) break;
            }
        } else if (p.mode == 2) {
            // outside precipitation for env_dbz_steps steps, above the cold
            // pool or where the environment mask allows it
            dcount = (v1[3] < p.env_dbz) ? dcount + 1 : 0;
            if (n > p.min_steps && dcount >= p.env_dbz_steps &&
                (zs - p.z_sfc >= p.cold_pool_depth || v1[5] >= 0.5)) {
                flags |= kEnvDbz;
                break;
            }
        }
        if (n == p.max_steps) flags |= kMaxSteps;
    }
    return npts;
}

template <class F>
void parallel_for(int64_t total, int n_threads, F&& body) {
    if (total <= 0) return;
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    int nt = n_threads > 0 ? n_threads : static_cast<int>(hw);
    const int64_t nchunks = (total + kChunk - 1) / kChunk;
    nt = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(nt, nchunks)));
    std::atomic<int64_t> next{0};
    auto worker = [&](int tid) {
        for (;;) {
            const int64_t g0 = next.fetch_add(kChunk);
            if (g0 >= total) break;
            const int64_t g1 = std::min(total, g0 + kChunk);
            for (int64_t g = g0; g < g1; ++g) body(tid, g);
        }
    };
    std::vector<std::thread> pool;
    for (int i = 1; i < nt; ++i) pool.emplace_back(worker, i);
    worker(0);
    for (auto& th : pool) th.join();
}

int thread_count(int n_threads) {
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    return n_threads > 0 ? n_threads : static_cast<int>(hw);
}

Grid make_grid(const F64& x, const F64& y, const F64& z, const F64& t, const F32& data,
               double cx, double cy, double eb, double ea, int nvar) {
    Grid g;
    g.x = x.data();
    g.y = y.data();
    g.z = z.data();
    g.t = t.data();
    g.nx = x.shape(0);
    g.ny = y.shape(0);
    g.nz = z.shape(0);
    g.nt = t.shape(0);
    if (g.nx < 2 || g.ny < 2 || g.nz < 2 || g.nt < 1)
        throw std::invalid_argument("the grid needs at least 2 points along x, y and z");
    if (data.ndim() != 5 || data.shape(0) != g.nt || data.shape(1) != g.nz ||
        data.shape(2) != g.ny || data.shape(3) != g.nx || data.shape(4) != nvar)
        throw std::invalid_argument("gridded data must be (time, z, y, x, " +
                                    std::to_string(nvar) + ")");
    g.data = data.data();
    g.nvar = nvar;
    g.cx = cx;
    g.cy = cy;
    g.ext_before = eb;
    g.ext_after = ea;
    return g;
}

PathParams make_path_params(const F64& par) {
    if (par.ndim() != 1 || par.shape(0) != 14)
        throw std::invalid_argument("path parameters must have 14 values");
    const double* q = par.data();
    PathParams p;
    p.dt = q[0];
    p.n_iter = static_cast<int>(q[1]);
    p.max_steps = static_cast<int64_t>(q[2]);
    p.dir = q[3] < 0 ? -1 : 1;
    p.mode = static_cast<int>(q[4]);
    p.min_steps = static_cast<int64_t>(q[5]);
    p.env_dbz = q[6];
    p.env_w = q[7];
    p.env_w_steps = static_cast<int64_t>(q[8]);
    p.env_dbz_steps = static_cast<int64_t>(q[9]);
    p.cold_pool_depth = q[10];
    p.z_sfc = q[11];
    p.boundary = static_cast<int>(q[12]);
    p.sides = static_cast<int>(q[13]);
    if (!(p.dt > 0.0)) throw std::invalid_argument("dt must be positive");
    if (p.max_steps < 0) throw std::invalid_argument("max_steps must be >= 0");
    return p;
}

// ---------------------------------------------------------------------------
// Thermodynamics
// ---------------------------------------------------------------------------

// Saturation vapour pressure over water, Bolton (1980) eq. (10), Pa.
inline double es_water(double t) {
    const double tc = t - kT0;
    return kEs0 * std::exp(17.67 * tc / (tc + 243.5));
}

inline double qvs_water(double t, double p) {
    const double e = es_water(t);
    return kEps * e / std::max(p - e, 1.0);
}

// Saturation vapour pressure over ice from the Clausius-Clapeyron equation
// with the constant latent heat of sublimation of LFO83 from e_s(T0).
inline double es_ice(double t) {
    return kEs0 * std::exp(kLsL / kRv * (1.0 / kT0 - 1.0 / t));
}

inline double lv_bolton(double t) { return 2.501e6 - 2370.0 * (t - kT0); }

inline double exner(double p) { return std::pow(p / kP0, kKappa); }

// Air density of Ziegler (2013a, sect. 2a): p / (R_d T) with T = theta Pi.
inline double air_density(double theta, double p) {
    return kP0 * std::pow(p / kP0, 1.0 - kKappa) / (kRd * theta);
}

// Isobaric saturation adjustment (condensation of excess vapour, evaporation
// of cloud water until saturation or q_c = 0; Ziegler 2013a, sect. 2g, after
// Soong and Ogura 1973). Newton iterations on q_v - dq = q_vs(T + L dq / c_p).
// Returns the heating d(theta).
inline double adjust(double& th, double& qv, double& qc, double p) {
    const double pi = exner(p);
    const double t = th * pi;
    if (qv <= qvs_water(t, p) && qc <= 0.0) return 0.0;
    const double lv = lv_bolton(t);
    double dq = 0.0;
    for (int it = 0; it < 6; ++it) {
        const double tn = t + lv * dq / kCp;
        const double e = es_water(tn);
        const double qs = kEps * e / std::max(p - e, 1.0);
        const double tc = tn - kT0;
        const double dlnes = 17.67 * 243.5 / ((tc + 243.5) * (tc + 243.5));
        const double dqs = qs * p / std::max(p - e, 1.0) * dlnes;
        const double f = qv - dq - qs;
        const double fp = -1.0 - lv / kCp * dqs;
        dq -= f / fp;
    }
    if (dq < -qc) dq = -qc;
    const double dth = lv * dq / (kCp * pi);
    th += dth;
    qv -= dq;
    qc += dq;
    return dth;
}

// Ice-water saturation adjustment of Tao, Simpson and McCumber (1989).
// First the phase changes of cloud condensate that LFO83 (sect. 3f) and Hsie
// et al. (1980, sect. 3b5) prescribe: cloud ice melts instantaneously above
// 0 degC (P_IMLT) and cloud water freezes at or below -40 degC (P_IHOM), with
// the heating of Tao et al. eq. (4a) for dq_c = -dq_i. Then one
// non-iterative step: the saturation mixing ratio is the mass-weighted mix
// (1) of Teten's values over water and ice (3a, 3b), the excess vapour
// dq = r1 / (1 + r2 A3) (6a-6e, 7b) is split into cloud water and cloud ice
// in proportions CND and DEP linear in T between T00 and 0 degC (2b, 2c),
// evaporation limited by the available q_c and q_i, and theta changes by
// (4a). Without condensate the weights of (1) are CND and DEP (those of the
// condensate the step produces). Returns the heating of the adjustment; the
// heating of the phase changes is added to dfrz.
inline double adjust_ice(double& th, double& qv, double& qc, double& qi, double p, double t00,
                         double& dfrz) {
    const double pi = exner(p);
    double t = th * pi;
    const double melt = t > kT0 ? qi : 0.0;
    const double frz = t <= kTHom ? qc : 0.0;
    const double dth_f = (kLsL - kLvL) * (frz - melt) / (kCp * pi);
    qc = qc + melt - frz;
    qi = qi - melt + frz;
    th = th + dth_f;
    dfrz += dth_f;
    t = th * pi;
    const double cnd = std::min(std::max((t - t00) / (kT0 - t00), 0.0), 1.0);
    const double dep = 1.0 - cnd;
    const double b = kTaoB / (p / 100.0);
    const double qws = b * std::exp(kTaoA1 * (t - 273.16) / (t - 35.86));
    const double qis = b * std::exp(kTaoA2 * (t - 273.16) / (t - 7.66));
    const double cloud = qc + qi;
    const bool has = cloud > 0.0;
    const double wc = has ? qc / cloud : cnd;
    const double wi = has ? qi / cloud : dep;
    const double qvs = wc * qws + wi * qis;
    const double a1 = 237.3 * kTaoA1 * pi / ((t - 35.86) * (t - 35.86));
    const double a2 = 265.5 * kTaoA2 * pi / ((t - 7.66) * (t - 7.66));
    const double r1 = qv - qvs;                                // (6a)
    if (!(r1 > 0.0) && !has) return 0.0;
    const double r2 = a1 * wc * qws + a2 * wi * qis;           // (6b)
    const double a3 = (kLvL * cnd + kLsL * dep) / (kCp * pi);  // (6e)
    const double dq = r1 / (1.0 + r2 * a3);                    // (7b)
    const double dqc = std::max(dq * cnd, -qc);                // (2b)
    const double dqi = std::max(dq * dep, -qi);                // (2c)
    const double dth = (kLvL * dqc + kLsL * dqi) / (kCp * pi);  // (4a)
    th += dth;
    qv = qv - dqc - dqi;
    qc += dqc;
    qi += dqi;
    return dth;
}

// ---------------------------------------------------------------------------
// Microphysical rates of Lin, Farley and Orville (1983, LFO83), SI units
// (every rate is dimensionally homogeneous). Equation numbers of LFO83.
// Rates are signed source terms of the hydrometeor as in LFO83: P_REVP,
// P_GMLT, P_GSUB < 0 (loss of rain or graupel), P_RACW, P_GACW, P_GFR > 0.
// ---------------------------------------------------------------------------
struct Rates {
    double revp, racw, gacw, gacr, gmlt, gsub, gfr;
};

struct Props {
    double ka, psi, nu, sc;
};

// Thermal conductivity, vapour diffusivity and kinematic viscosity of air
// (as listed in the appendix of Kumjian and Ryzhkov 2010; not checked against
// that paper).
inline Props air_props(double t, double p, double rho) {
    Props r;
    r.ka = (0.441635 + 0.0071 * t) * 1.0e-2;
    r.psi = 2.11e-5 * std::pow(t / kT0, 1.94) * (1.0e5 / p);
    r.nu = (0.379565 + 0.0049 * t) * 1.0e-5 / rho;
    r.sc = r.nu / r.psi;
    return r;
}

Rates lfo_rates(double t, double p, double rho, double qv, double qc, double qr, double nr,
                double qg, double ng, double rhog, double rho0, int sw, double qi) {
    Rates r{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const bool rain = qr > 1.0e-12 && nr > 0.0;
    const bool graupel = qg > 1.0e-12 && ng > 0.0;
    if (!rain && !graupel) return r;
    const Props a = air_props(t, p, rho);
    const double tc = t - kT0;
    double lr = 0.0, n0r = 0.0, lg = 0.0, n0g = 0.0;
    if (rain) {
        lr = std::cbrt(kPi * kRhoW * nr / (rho * qr));  // eqs. (4), (6), p. 1068, with n0 = N lambda: lambda^3 = pi rho N / (rho_air q)
        n0r = nr * lr;
    }
    if (graupel) {
        lg = std::cbrt(kPi * rhog * ng / (rho * qg));
        n0g = ng * lg;
    }
    const double gfall = std::sqrt(4.0 * kG * rhog / (3.0 * kCD * rho));  // eq. (9)
    if (rain) {
        // (51) accretion of cloud water by rain
        if ((sw & kRacw) && qc > 0.0)
            r.racw = kPi * n0r * kAR * qc * std::tgamma(3.0 + kBR) /
                     (4.0 * std::pow(lr, 3.0 + kBR)) * std::sqrt(rho0 / rho);
        // (52) evaporation of rain in subsaturated air
        const double qs = qvs_water(t, p);
        const double s = qv / qs;
        if ((sw & kRevp) && s < 1.0) {
            const double vent = 0.78 / (lr * lr) +
                                0.31 * std::cbrt(a.sc) * std::tgamma(0.5 * (kBR + 5.0)) *
                                    std::sqrt(kAR) / std::sqrt(a.nu) *
                                    std::pow(rho0 / rho, 0.25) *
                                    std::pow(lr, -0.5 * (kBR + 5.0));
            const double den = kLvL * kLvL / (a.ka * kRv * t * t) + 1.0 / (rho * qs * a.psi);
            r.revp = 2.0 * kPi * (s - 1.0) * n0r * vent / rho / den;
        }
        // (45) Bigg freezing of rain
        if ((sw & kGfr) && t < kT0)
            r.gfr = 20.0 * kPi * kPi * kBpr * n0r * (kRhoW / rho) *
                    (std::exp(kApr * (kT0 - t)) - 1.0) * std::pow(lr, -7.0);
    }
    if (graupel) {
        // (40) accretion of cloud water by graupel
        if (((sw & kGacw) || (sw & kGmlt)) && qc > 0.0)
            r.gacw = kPi * n0g * qc * std::tgamma(3.5) / (4.0 * std::pow(lg, 3.5)) * gfall;
        // ventilation bracket of (46) and (47), with the air density inside
        // the fourth root as required by the fall speed (9)
        const double vent = 0.78 / (lg * lg) + 0.31 * std::cbrt(a.sc) * std::tgamma(2.75) *
                                                   std::sqrt(gfall) / std::sqrt(a.nu) *
                                                   std::pow(lg, -2.75);
        if ((sw & kGmlt) && t >= kT0) {
            // (42) accretion of rain by graupel, with the mass-weighted fall
            // speeds (11) and (13); used in the sensible-heat term of (47)
            if (rain) {
                const double ur = kAR * std::tgamma(4.0 + kBR) / (6.0 * std::pow(lr, kBR)) *
                                  std::sqrt(rho0 / rho);
                const double ug = std::tgamma(4.5) / (6.0 * std::sqrt(lg)) * gfall;
                r.gacr = kPi * kPi * n0g * n0r * std::fabs(ug - ur) * (kRhoW / rho) *
                         (5.0 / (std::pow(lr, 6.0) * lg) + 2.0 / (std::pow(lr, 5.0) * lg * lg) +
                          0.5 / (std::pow(lr, 4.0) * lg * lg * lg));
            }
            // (47) melting of graupel; Delta r_s = r_s0 - r
            const double drs = kEps * kEs0 / (p - kEs0) - qv;
            r.gmlt = -2.0 * kPi / (rho * kLfL) * (a.ka * tc - kLvL * a.psi * rho * drs) * n0g *
                         vent -
                     kCw * tc / kLfL * (r.gacw + r.gacr);
            if (r.gmlt > 0.0) r.gmlt = 0.0;
        }
        // (46) sublimation of graupel outside cloud (delta_1 of eq. 20:
        // l_CW + l_CI = 0) below 0 degC
        if ((sw & kGsub) && t < kT0 && qc + qi <= 0.0) {
            const double ei = es_ice(t);
            const double qsi = kEps * ei / std::max(p - ei, 1.0);
            const double si = qv / qsi;
            if (si < 1.0) {
                const double a2 = kLsL * kLsL / (a.ka * kRv * t * t);  // A'' (31)
                const double b2 = 1.0 / (rho * qsi * a.psi);           // B'' (31)
                r.gsub = 2.0 * kPi * (si - 1.0) / (rho * (a2 + b2)) * n0g * vent;
            }
        }
        if (!(sw & kGacw)) r.gacw = 0.0;
    }
    return r;
}

// Tendencies of theta, q_v, q_c from the rates, limited so that a time step
// dt does not overshoot (numerical safeguards): evaporation and sublimation
// at most to saturation and to the available rain or graupel, collection at
// most the cloud water, melting at most the graupel, freezing at most the rain.
struct Tend {
    double th, qv, qc;
    double revp, gmlt, gsub, frz;  // theta tendencies of each process
};

Tend tendencies(const Rates& r0, double th, double p, double qv, double qc, double qr,
                double qg, double dt) {
    Rates r = r0;
    const double pi = exner(p);
    const double t = th * pi;
    if (r.revp < 0.0) {
        const double qs = qvs_water(t, p);
        const double gap = std::max(qs - qv, 0.0) /
                           (1.0 + kLvL * kLvL * qs / (kCp * kRv * t * t));
        const double lim = std::min(gap, qr) / dt;
        if (-r.revp > lim) r.revp = -lim;
    }
    if (r.gsub < 0.0) {
        const double ei = es_ice(t);
        const double qsi = kEps * ei / std::max(p - ei, 1.0);
        const double gap = std::max(qsi - qv, 0.0) /
                           (1.0 + kLsL * kLsL * qsi / (kCp * kRv * t * t));
        const double lim = std::min(gap, qg) / dt;
        if (-r.gsub > lim) r.gsub = -lim;
    }
    const double col = r.racw + r.gacw;
    if (col * dt > qc && col > 0.0) {
        const double f = qc / (col * dt);
        r.racw *= f;
        r.gacw *= f;
    }
    if (-r.gmlt * dt > qg) r.gmlt = -qg / dt;
    if (r.gfr * dt > qr) r.gfr = qr / dt;
    Tend d;
    const double c = 1.0 / (kCp * pi);
    d.revp = c * kLvL * r.revp;
    d.gsub = c * kLsL * r.gsub;
    d.gmlt = c * kLfL * r.gmlt;
    d.frz = c * kLfL * (r.gfr + (t < kT0 ? r.gacw : 0.0));
    d.th = d.revp + d.gsub + d.gmlt + d.frz;
    d.qv = -r.revp - r.gsub;
    d.qc = -(r.racw + r.gacw);
    return d;
}

// Vertical profile on a uniform table (base state).
struct Table {
    double z0, dz;
    int64_t n;
    const double* v;  // (n, nv)
    int nv;
};

inline void profile(const Table& b, double zp, double* out) {
    double f = (zp - b.z0) / b.dz;
    if (f < 0.0) f = 0.0;
    if (f > static_cast<double>(b.n - 1)) f = static_cast<double>(b.n - 1);
    int64_t i = static_cast<int64_t>(f);
    if (i > b.n - 2) i = b.n - 2;
    const double a = f - static_cast<double>(i);
    for (int k = 0; k < b.nv; ++k)
        out[k] = (1.0 - a) * b.v[i * b.nv + k] + a * b.v[(i + 1) * b.nv + k];
}

// Base-state profile columns
enum { kBP = 0, kBTheta = 1, kBQv = 2, kBU = 3, kBV = 4, kBNv = 5 };

// Thermodynamic parameters (order shared with the Python layer)
enum {
    qDtSmall = 0, qCd, qBd, qW0, qLd1, qLd2, qLw1, qLw2, qCmin, qCmax, qQp0, qQ0, qQ1, qZbl,
    qBf, qZsfc, qRhoGsfc, qRhoG5, qRho0, qFluxTh, qFluxQv, qSwitch, qT00, qNpar
};

struct Thermo {
    Table base;
    const Grid* precip;    // nvar 4: q_r, N_r, q_g, N_g (or nullptr)
    const Grid* meso;      // nvar 2: theta, q_v (any nt) or nullptr
    const Grid* grad;      // nvar 4: dtheta/dx, dtheta/dy, dqv/dx, dqv/dy (nz = 2) or nullptr
    const double* q;
};

inline void base_at(const Thermo& th, double xp, double yp, double zp, double tp,
                    double& theta, double& qv) {
    if (th.meso) {
        double m[2];
        sample(*th.meso, xp, yp, zp, tp, m, false);
        theta = m[0];
        qv = m[1];
        return;
    }
    double b[kBNv];
    profile(th.base, zp, b);
    theta = b[kBTheta];
    qv = b[kBQv];
}

inline double graupel_density(const double* q, double zagl) {
    const double f = std::min(std::max(zagl / 5000.0, 0.0), 1.0);
    return q[qRhoGsfc] + (q[qRhoG5] - q[qRhoGsfc]) * f;
}

// Damping coefficient of Ziegler (2013a) eqs. (22)-(26), s^-1.
inline double damping_rate(const double* q, double w, double u, double v, double ub, double vb,
                           double qp, double zagl) {
    const double aw = std::fabs(w);
    double vel, ld;
    if (w > q[qW0]) {
        vel = aw;
        ld = q[qLd1] + (aw - q[qW0]) * q[qLw1];
    } else if (w < -q[qW0]) {
        vel = aw;
        ld = q[qLd2] + (aw - q[qW0]) * q[qLw2];
    } else {
        const double f = std::min(std::max(qp / q[qQp0], 0.0), 1.0);
        const double c0 = (1.0 - f) * q[qCmin] + f * q[qCmax];
        vel = std::sqrt((u - ub) * (u - ub) + (v - vb) * (v - vb));
        ld = q[qCd] / c0;
    }
    return q[qCd] * vel / (ld * std::exp(q[qBd] * zagl / 1000.0));
}

constexpr int kNBudget = 7;  // theta budget: cond, revp, gmlt, gsub, frz, damp, flux

// In situ observations (Ziegler et al. 2007): (nobs, 6) x, y, z, t, theta,
// q_v and the options window, radius, z tolerance, kappa_s, tau_i, tau_L.
struct Obs {
    const double* v;
    int64_t n;
    double window, radius, ztol, kap, tau_i, tau_l;
};

// Candidates are the observations within the data window (at or before the
// analysis time) whose position is within radius (horizontal) and z
// tolerance of the trajectory at the observation time (linear between stored
// points). Each has the weight of Ziegler et al. (2007) eq. (1) with t_i =
// t_o and t_L = |t_o|. Returns the index of the stored point nearest the time
// of the candidate with the largest weight (-1 without a candidate), the
// weight sum and the weighted theta and q_v.
int64_t insitu_match(const Obs& ob, const double* pos, int64_t npts, double dt, double& wsum,
                     double& th0, double& qv0) {
    const double h = -dt;
    double sth = 0.0, sqv = 0.0, wbest = 0.0;
    int64_t kbest = -1;
    wsum = 0.0;
    for (int64_t j = 0; j < ob.n; ++j) {
        const double* o = ob.v + 6 * j;
        const double to = o[3];
        if (!(to <= 0.0 && -to <= ob.window)) continue;
        const double f = to / h;
        const double fk = std::floor(f);
        const int64_t k0 = static_cast<int64_t>(fk);
        const double a = f - fk;
        const int64_t need = k0 + (a > 0.0 ? 1 : 0);
        if (need > npts - 1) continue;
        const double* p0 = pos + 4 * k0;
        double pt[3];
        if (a > 0.0) {
            const double* p1 = p0 + 4;
            for (int v = 0; v < 3; ++v) pt[v] = (1.0 - a) * p0[v] + a * p1[v];
        } else {
            for (int v = 0; v < 3; ++v) pt[v] = p0[v];
        }
        const double dx = pt[0] - o[0];
        const double dy = pt[1] - o[1];
        const double r2 = dx * dx + dy * dy;
        if (!(r2 <= ob.radius * ob.radius) || !(std::fabs(pt[2] - o[2]) <= ob.ztol)) continue;
        const double w = std::exp(-r2 / ob.kap - to * to / ob.tau_i - to * to / ob.tau_l);
        wsum += w;
        sth += w * o[4];
        sqv += w * o[5];
        if (w > wbest) {
            wbest = w;
            kbest = static_cast<int64_t>(std::floor(f + 0.5));
        }
    }
    if (kbest < 0 || !(wsum > 0.0)) return -1;
    th0 = sth / wsum;
    qv0 = sqv / wsum;
    return kbest;
}

// Saturation adjustment of the DLA: water only (Ziegler 2013a) or, with
// kIce, the ice-water adjustment of Tao et al. (1989).
inline void condense(const double* q, int sw, double& theta, double& qv, double& qc, double& qi,
                     double p, double* bud) {
    if (sw & kIce)
        bud[0] += adjust_ice(theta, qv, qc, qi, p, q[qT00], bud[4]);
    else
        bud[0] += adjust(theta, qv, qc, p);
}

// Forward integration of theta, q_v, q_c (and q_i) along a stored backward
// path from point kstart to point 0. The initial theta, q_v are those of the
// environment at the start point, or th_init, qv_init (in situ) if use_init.
// out: theta, q_v, q_c, q_i; bud: theta budget.
void integrate(const Thermo& th, const double* pos, const double* val, int64_t kstart,
               bool surface, double dt, bool use_init, double th_init, double qv_init,
               double* out, double* bud) {
    const double* q = th.q;
    const int sw = static_cast<int>(q[qSwitch]);
    const double zsfc = q[qZsfc];
    for (int k = 0; k < kNBudget; ++k) bud[k] = 0.0;
    const double* p0 = pos + 4 * kstart;
    double theta, qv, qc = 0.0, qi = 0.0;
    base_at(th, p0[0], p0[1], p0[2], p0[3], theta, qv);
    if (use_init) {
        theta = th_init;
        qv = qv_init;
    }
    double b[kBNv];
    profile(th.base, p0[2], b);
    double pa = b[kBP];
    if (sw & kCond) condense(q, sw, theta, qv, qc, qi, pa, bud);
    const int64_t ns = std::max<int64_t>(1, static_cast<int64_t>(std::ceil(dt / q[qDtSmall] - 1e-9)));
    const double h = dt / static_cast<double>(ns);
    const double qthr = (surface ? q[qQ0] : q[qQ1]);
    for (int64_t m = kstart; m >= 1; --m) {
        const double* A = pos + 4 * m;
        const double* B = pos + 4 * (m - 1);
        const double* VA = val + kNW * m;
        const double zagl = A[2] - zsfc;
        profile(th.base, B[2], b);
        const double pb = b[kBP];
        double pr[4] = {0.0, 0.0, 0.0, 0.0};
        if (th.precip) {
            if (sample(*th.precip, A[0], A[1], A[2], A[3], pr, false)) {
                pr[0] = pr[1] = pr[2] = pr[3] = 0.0;
            }
            for (int k = 0; k < 4; ++k)
                if (!(pr[k] > 0.0)) pr[k] = 0.0;
        }
        const double qr = pr[0], nr = pr[1], qg = pr[2], ng = pr[3];
        // microphysical rates at the start of the step, held over the step
        const double t = theta * exner(pa);
        const double rho = air_density(theta, pa);
        Tend d{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        if (sw & (kRevp | kRacw | kGacw | kGmlt | kGsub | kGfr)) {
            const Rates r = lfo_rates(t, pa, rho, qv, qc, qr, nr, qg, ng,
                                      graupel_density(q, zagl), q[qRho0], sw, qi);
            d = tendencies(r, theta, pa, qv, qc, qr, qg, dt);
        }
        // surface flux, Ziegler (2013a) eq. (27)
        double fth = 0.0, fqv = 0.0;
        if ((sw & kFlux) && zagl <= q[qZbl] && qc + qi + qr + qg <= q[qQ1]) {
            double gth = 0.0, gqv = 0.0;
            if (th.grad) {
                double gr[4];
                if (!sample(*th.grad, A[0], A[1], th.grad->z[0], A[3], gr, false)) {
                    gth = VA[0] * gr[0] + VA[1] * gr[1];
                    gqv = VA[0] * gr[2] + VA[1] * gr[3];
                }
            }
            const double e = std::exp(-q[qBf] * zagl / 1000.0);
            fth = e * (gth + q[qFluxTh]);
            fqv = e * (gqv + q[qFluxQv]);
        }
        // sub-steps: pressure change with the displacement, sources, and
        // saturation adjustment every dt_small (Ziegler 2013a, sect. 2g)
        for (int64_t s = 1; s <= ns; ++s) {
            const double ps = pa + (pb - pa) * static_cast<double>(s) / static_cast<double>(ns);
            theta += h * (d.th + fth);
            qv += h * (d.qv + fqv);
            qc += h * d.qc;
            if (qv < 0.0) qv = 0.0;
            if (qc < 0.0) qc = 0.0;
            if (sw & kCond) condense(q, sw, theta, qv, qc, qi, ps, bud);
        }
        bud[1] += dt * d.revp;
        bud[2] += dt * d.gmlt;
        bud[3] += dt * d.gsub;
        bud[4] += dt * d.frz;
        bud[6] += dt * fth;
        // Lagrangian damping, eqs. (22)-(26), integrated exactly over the step
        if ((sw & kDamp) && qr + qg >= qthr) {
            double pa_b[kBNv];
            profile(th.base, A[2], pa_b);
            const double kd = damping_rate(q, VA[2], VA[0], VA[1], pa_b[kBU], pa_b[kBV], qr + qg,
                                           zagl);
            const double f = std::exp(-kd * dt);
            double thb, qvb;
            base_at(th, B[0], B[1], B[2], B[3], thb, qvb);
            const double th_new = thb + (theta - thb) * f;
            bud[5] += th_new - theta;
            theta = th_new;
            qv = qvb + (qv - qvb) * f;
            qc = qc * f;
            qi = qi * f;
        }
        pa = pb;
    }
    out[0] = theta;
    out[1] = qv;
    out[2] = qc;
    out[3] = qi;
}

}  // namespace

// Trajectories from n start points (x, y, z) at t = 0. Returns positions
// (n, max_steps + 1, 4: x, y, z, t), values (n, max_steps + 1, 6: u, v, w,
// Z_H, valid, environment mask), the number of points and the flags.
py::tuple trajectories_py(const F32& fields, const F64& x, const F64& y, const F64& z,
                          const F64& t, const F64& starts, const F64& par, double cx, double cy,
                          double ext_before, double ext_after, int n_threads) {
    const Grid g = make_grid(x, y, z, t, fields, cx, cy, ext_before, ext_after, kNW);
    const PathParams p = make_path_params(par);
    if (starts.ndim() != 2 || starts.shape(1) != 3)
        throw std::invalid_argument("starts must be (n, 3)");
    const int64_t n = starts.shape(0), m = p.max_steps + 1;
    py::array_t<double> pos(std::vector<py::ssize_t>{n, m, 4});
    py::array_t<double> val(std::vector<py::ssize_t>{n, m, kNW});
    py::array_t<int64_t> npts(n);
    py::array_t<int32_t> flags(n);
    double *pp = pos.mutable_data(), *pv = val.mutable_data();
    int64_t* pn = npts.mutable_data();
    int32_t* pf = flags.mutable_data();
    const double* ps = starts.data();
    {
        py::gil_scoped_release release;
        parallel_for(n, n_threads, [&](int, int64_t i) {
            double* a = pp + i * m * 4;
            double* b = pv + i * m * kNW;
            int f = 0;
            const int64_t k = build_path(g, p, ps[3 * i], ps[3 * i + 1], ps[3 * i + 2], a, b, f);
            for (int64_t s = k * 4; s < m * 4; ++s) a[s] = kNaN;
            for (int64_t s = k * kNW; s < m * kNW; ++s) b[s] = kNaN;
            pn[i] = k;
            pf[i] = f;
        });
    }
    return py::make_tuple(pos, val, npts, flags);
}

// The diabatic Lagrangian analysis at n start points. Returns (n, 4) theta,
// q_v, q_c, q_i, (n, 7) theta budget, (n, 5) origin (x, y, z, t) and fraction
// of the trajectory points with valid winds, the number of points, the flags
// and (n, 2) in situ weight sum and start time (NaN without observation).
py::tuple dla_py(const F32& fields, const F64& x, const F64& y, const F64& z, const F64& t,
                 const F64& starts, const I8& surface, const F64& par, double cx, double cy,
                 double ext_before, double ext_after, const F64& base, double base_z0,
                 double base_dz, const F32& precip, bool has_precip, const F32& meso,
                 const F64& meso_t, bool has_meso, const F32& grad, bool has_grad,
                 const F64& thermo, const F64& obs, const F64& obs_par, bool has_obs,
                 int n_threads) {
    const Grid g = make_grid(x, y, z, t, fields, cx, cy, ext_before, ext_after, kNW);
    const PathParams p = make_path_params(par);
    if (starts.ndim() != 2 || starts.shape(1) != 3)
        throw std::invalid_argument("starts must be (n, 3)");
    const int64_t n = starts.shape(0);
    if (surface.ndim() != 1 || surface.shape(0) != n)
        throw std::invalid_argument("surface must be (n,)");
    if (base.ndim() != 2 || base.shape(1) != kBNv || base.shape(0) < 2)
        throw std::invalid_argument("base must be (n, 5)");
    if (thermo.ndim() != 1 || thermo.shape(0) != qNpar)
        throw std::invalid_argument("thermo must have " + std::to_string(qNpar) + " values");
    Obs ob{nullptr, 0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    if (has_obs) {
        if (obs.ndim() != 2 || obs.shape(1) != 6) throw std::invalid_argument("obs must be (n, 6)");
        if (obs_par.ndim() != 1 || obs_par.shape(0) != 6)
            throw std::invalid_argument("obs_par must have 6 values");
        const double* op = obs_par.data();
        ob = Obs{obs.data(), obs.shape(0), op[0], op[1], op[2], op[3], op[4], op[5]};
    }
    Grid gp, gm, gg;
    // the mesoscale analysis holds its first and last times beyond its span
    const double hold = 1.0e30;
    if (has_precip) gp = make_grid(x, y, z, t, precip, cx, cy, ext_before, ext_after, 4);
    if (has_meso) gm = make_grid(x, y, z, meso_t, meso, 0.0, 0.0, hold, hold, 2);
    F64 zz(2);
    zz.mutable_data()[0] = 0.0;
    zz.mutable_data()[1] = 1.0;
    if (has_grad) gg = make_grid(x, y, zz, meso_t, grad, 0.0, 0.0, hold, hold, 4);
    Thermo th;
    th.base = Table{base_z0, base_dz, base.shape(0), base.data(), kBNv};
    th.precip = has_precip ? &gp : nullptr;
    th.meso = has_meso ? &gm : nullptr;
    th.grad = has_grad ? &gg : nullptr;
    th.q = thermo.data();
    const int nthr = thread_count(n_threads);
    const int64_t m = p.max_steps + 1;
    std::vector<std::vector<double>> bpos(nthr, std::vector<double>(4 * m));
    std::vector<std::vector<double>> bval(nthr, std::vector<double>(kNW * m));
    py::array_t<double> out(std::vector<py::ssize_t>{n, 4});
    py::array_t<double> init(std::vector<py::ssize_t>{n, 2});
    py::array_t<double> bud(std::vector<py::ssize_t>{n, kNBudget});
    py::array_t<double> org(std::vector<py::ssize_t>{n, 5});
    py::array_t<int64_t> npts(n);
    py::array_t<int32_t> flags(n);
    double *po = out.mutable_data(), *pb = bud.mutable_data(), *pg = org.mutable_data();
    double* pin = init.mutable_data();
    int64_t* pn = npts.mutable_data();
    int32_t* pf = flags.mutable_data();
    const double* ps = starts.data();
    const int8_t* psf = surface.data();
    {
        py::gil_scoped_release release;
        parallel_for(n, nthr, [&](int tid, int64_t i) {
            double* a = bpos[tid].data();
            double* b = bval[tid].data();
            int f = 0;
            const int64_t k = build_path(g, p, ps[3 * i], ps[3 * i + 1], ps[3 * i + 2], a, b, f);
            pn[i] = k;
            pf[i] = f;
            for (int v = 0; v < 4; ++v) pg[5 * i + v] = k > 0 ? a[4 * (k - 1) + v] : kNaN;
            int64_t nv = 0;
            for (int64_t s = 0; s < k; ++s) nv += b[kNW * s + 4] >= 0.5 ? 1 : 0;
            pg[5 * i + 4] = k > 0 ? static_cast<double>(nv) / static_cast<double>(k) : kNaN;
            const bool env = (f & (kEnvDbz | kEnvW | kBoundary)) != 0;
            int64_t kstart = k - 1;
            double wsum = 0.0, th0 = 0.0, qv0 = 0.0;
            int64_t kin = -1;
            if (ob.n > 0 && k > 0) kin = insitu_match(ob, a, k, p.dt, wsum, th0, qv0);
            pin[2 * i] = kNaN;
            pin[2 * i + 1] = kNaN;
            if (kin >= 0) {
                kstart = kin;
                pf[i] = f | kInsitu;
                pin[2 * i] = wsum;
                pin[2 * i + 1] = a[4 * kin + 3];
            }
            if (k < 1 || (!env && kin < 0)) {
                for (int v = 0; v < 4; ++v) po[4 * i + v] = kNaN;
                for (int v = 0; v < kNBudget; ++v) pb[kNBudget * i + v] = kNaN;
                return;
            }
            integrate(th, a, b, kstart, psf[i] != 0, p.dt, kin >= 0, th0, qv0, po + 4 * i,
                      pb + kNBudget * i);
        });
    }
    return py::make_tuple(out, bud, org, npts, flags, init);
}

// LFO83 rates and limited theta/q_v/q_c tendencies for 1-D arrays of states.
// Returns (n, 7) rates (revp, racw, gacw, gacr, gmlt, gsub, gfr) and (n, 3)
// tendencies (theta, q_v, q_c) for a step dt.
py::tuple rates_py(const F64& theta, const F64& p, const F64& qv, const F64& qc, const F64& qr,
                   const F64& nr, const F64& qg, const F64& ng, const F64& rhog, double rho0,
                   int switches, double dt, int n_threads) {
    const int64_t n = theta.size();
    for (const F64* a : {&p, &qv, &qc, &qr, &nr, &qg, &ng, &rhog})
        if (a->size() != n) throw std::invalid_argument("all states must have the same size");
    py::array_t<double> rr(std::vector<py::ssize_t>{n, 7});
    py::array_t<double> tt(std::vector<py::ssize_t>{n, 3});
    double *o = rr.mutable_data(), *o2 = tt.mutable_data();
    const double *a0 = theta.data(), *a1 = p.data(), *a2 = qv.data(), *a3 = qc.data(),
                 *a4 = qr.data(), *a5 = nr.data(), *a6 = qg.data(), *a7 = ng.data(),
                 *a8 = rhog.data();
    {
        py::gil_scoped_release release;
        parallel_for(n, n_threads, [&](int, int64_t i) {
            const double tk = a0[i] * exner(a1[i]);
            const double rho = air_density(a0[i], a1[i]);
            const Rates r = lfo_rates(tk, a1[i], rho, a2[i], a3[i], a4[i], a5[i], a6[i], a7[i],
                                      a8[i], rho0, switches, 0.0);
            o[7 * i + 0] = r.revp;
            o[7 * i + 1] = r.racw;
            o[7 * i + 2] = r.gacw;
            o[7 * i + 3] = r.gacr;
            o[7 * i + 4] = r.gmlt;
            o[7 * i + 5] = r.gsub;
            o[7 * i + 6] = r.gfr;
            const Tend d = tendencies(r, a0[i], a1[i], a2[i], a3[i], a4[i], a6[i], dt);
            o2[3 * i + 0] = d.th;
            o2[3 * i + 1] = d.qv;
            o2[3 * i + 2] = d.qc;
        });
    }
    return py::make_tuple(rr, tt);
}

// Saturation adjustment of 1-D states (water only, or ice-water with ice).
// Returns (n, 4) theta, q_v, q_c, q_i and (n, 2) heating of the adjustment
// and of the freezing/melting of cloud condensate.
py::tuple adjust_py(const F64& theta, const F64& p, const F64& qv, const F64& qc, const F64& qi,
                    double t00, bool ice) {
    const int64_t n = theta.size();
    for (const F64* a : {&p, &qv, &qc, &qi})
        if (a->size() != n) throw std::invalid_argument("all states must have the same size");
    py::array_t<double> st(std::vector<py::ssize_t>{n, 4});
    py::array_t<double> ht(std::vector<py::ssize_t>{n, 2});
    double *o = st.mutable_data(), *o2 = ht.mutable_data();
    for (int64_t i = 0; i < n; ++i) {
        double th = theta.data()[i], v = qv.data()[i], c = qc.data()[i], x = qi.data()[i];
        double dfrz = 0.0, dth;
        if (ice)
            dth = adjust_ice(th, v, c, x, p.data()[i], t00, dfrz);
        else
            dth = adjust(th, v, c, p.data()[i]);
        o[4 * i] = th;
        o[4 * i + 1] = v;
        o[4 * i + 2] = c;
        o[4 * i + 3] = x;
        o2[2 * i] = dth;
        o2[2 * i + 1] = dfrz;
    }
    return py::make_tuple(st, ht);
}

PYBIND11_MODULE(_lagrangian, m) {
    m.doc() = "Compiled trajectory and diabatic Lagrangian analysis kernel for radarx.";
    m.def("trajectories", &trajectories_py, py::arg("fields"), py::arg("x"), py::arg("y"),
          py::arg("z"), py::arg("t"), py::arg("starts"), py::arg("par"), py::arg("cx"),
          py::arg("cy"), py::arg("ext_before"), py::arg("ext_after"), py::arg("n_threads") = 0);
    m.def("dla", &dla_py, py::arg("fields"), py::arg("x"), py::arg("y"), py::arg("z"),
          py::arg("t"), py::arg("starts"), py::arg("surface"), py::arg("par"), py::arg("cx"),
          py::arg("cy"), py::arg("ext_before"), py::arg("ext_after"), py::arg("base"),
          py::arg("base_z0"), py::arg("base_dz"), py::arg("precip"), py::arg("has_precip"),
          py::arg("meso"), py::arg("meso_t"), py::arg("has_meso"), py::arg("grad"),
          py::arg("has_grad"), py::arg("thermo"), py::arg("obs"), py::arg("obs_par"),
          py::arg("has_obs"), py::arg("n_threads") = 0);
    m.def("adjust", &adjust_py, py::arg("theta"), py::arg("pressure"), py::arg("qv"),
          py::arg("qc"), py::arg("qi"), py::arg("t00"), py::arg("ice"));
    m.def("rates", &rates_py, py::arg("theta"), py::arg("pressure"), py::arg("qv"),
          py::arg("qc"), py::arg("qr"), py::arg("nr"), py::arg("qg"), py::arg("ng"),
          py::arg("rho_g"), py::arg("rho0"), py::arg("switches"), py::arg("dt"),
          py::arg("n_threads") = 0);
}
