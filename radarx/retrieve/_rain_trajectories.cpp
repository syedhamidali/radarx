// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Raindrop trajectory kernel.
//
// integrate(): every drop (source point x size bin x ensemble member) is an
// independent initial value problem for the state (x, y, z, s = D^2, L):
//
//   dx/dt = u + u',  dy/dt = v + v',  dz/dt = w + w' - Vt(D, rho(z)),
//   ds/dt = 8e6 f_v (S - 1) / (rho_w (F_K + F_D)),   dL/dt = -div,
//
// integrated forward or backward in time with RK4 (or Heun) steps of fixed
// length until the drop crosses the stop height, shrinks below the minimum
// diameter or exceeds the maximum time. The step that ends the path is
// finished with the cubic Hermite interpolant of the step, so the landing
// point is as accurate as the step itself. All drops form one pool of work
// that threads take in small blocks from an atomic counter.
//
// Every operation follows the NumPy reference in
// radarx/retrieve/rain_trajectories.py in the same order.

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
constexpr double kRhoW = 1000.0;
constexpr double kDFloor = 1.0e-3;  // mm, smallest diameter in the right-hand side
constexpr int kBisect = 60;
constexpr int kNewton = 3;
constexpr int64_t kBlock = 8;  // drops per scheduling block
constexpr int kNout = 9;

enum Status { kAloft = 0, kLanded = 1, kEvaporated = 2, kInvalid = 3 };

struct Ctx {
    // wind grid (nt may be 0: no gridded wind) and background profile
    int nt = 0, nz = 0, ny = 0, nx = 0;
    const double *wt = nullptr, *wz = nullptr, *wy = nullptr, *wx = nullptr;
    const double* uvw = nullptr;
    int nb = 0;
    const double *bz = nullptr, *bu = nullptr, *bv = nullptr, *bw = nullptr;
    // thermodynamic profile
    int ne = 0;
    // optional stop surface z(y, x) (terrain, or the cone of a sweep)
    int nsy = 0, nsx = 0;
    const double *sy = nullptr, *sx = nullptr, *sz = nullptr;
    const double *ez = nullptr, *erho = nullptr, *enu = nullptr, *efkd = nullptr,
                 *essat = nullptr, *ecs = nullptr;
    // parameters
    double dir = 1.0, dt = 1.0, max_time = 0.0, smin = 0.0, rho0 = 1.204;
    double vav = 0.78, vbv = 0.308, cx = 0.0, cy = 0.0;
    double sigma_h = 0.0, sigma_w = 0.0, timescale = 1.0;
    int scheme = 4, fall_kind = 0, dens = 1, evap = 0, ratio = 0, wdiv = 0, turb = 0;
    int stride = 0, nrec = 0, nfc = 0;
    const double* fc = nullptr;
    uint64_t seed = 0;
};

// -------------------------------------------------------------------------
// interpolation
// -------------------------------------------------------------------------

// Index and weight of q on a sorted axis, clamped at the ends (the same as
// np.searchsorted(side="right") - 1, clipped); `in` tells whether q is within
// the axis. `hint` is the index found last time on this axis: consecutive
// points of a trajectory are in the same cell, which saves the search.
inline void locate(const double* ax, int n, double q, int& i, double& w, bool& in,
                   int& hint) {
    if (n <= 1) {
        i = 0;
        w = 0.0;
        in = true;
        return;
    }
    if (hint >= 0 && hint <= n - 2 && ax[hint] <= q && q < ax[hint + 1]) {
        i = hint;
    } else {
        const double* p = std::upper_bound(ax, ax + n, q);
        const int k = static_cast<int>(p - ax) - 1;
        i = std::min(std::max(k, 0), n - 2);
        hint = i;
    }
    const double ww = (q - ax[i]) / (ax[i + 1] - ax[i]);
    w = std::min(std::max(ww, 0.0), 1.0);
    in = (q >= ax[0]) && (q <= ax[n - 1]);
}

struct Hints {
    int t = -1, z = -1, b = -1, e = -1, sy = -1, sx = -1;
    int y[2] = {-1, -1}, x[2] = {-1, -1};
};

struct Wind {
    double u = 0, v = 0, w = 0, dudx = 0, dvdy = 0, dwdz = 0;
};

inline Wind background(const Ctx& c, Hints& h, double z) {
    Wind r;
    int i;
    double w;
    bool in;
    locate(c.bz, c.nb, z, i, w, in, h.b);
    const int j = std::min(i + 1, c.nb - 1);
    r.u = c.bu[i] + w * (c.bu[j] - c.bu[i]);
    r.v = c.bv[i] + w * (c.bv[j] - c.bv[i]);
    r.w = c.bw[i] + w * (c.bw[j] - c.bw[i]);
    if (c.wdiv && c.nb > 1 && in) r.dwdz = (c.bw[j] - c.bw[i]) / (c.bz[j] - c.bz[i]);
    return r;
}

// Wind at (t, x, y, z). Each analysis is a frozen pattern that moves with the
// storm motion from its own valid time: the wind at time t is read from the
// two analyses around t at the position moved back by the motion times the
// time since that analysis, trilinearly in space, and the two are blended
// linearly in time. The background profile is used where the drop is outside
// the grid (horizontally or above its top) or a neighbouring value is missing.
inline Wind sample_wind(const Ctx& c, Hints& h, double t, double x, double y, double z) {
    if (c.nt <= 0) return background(c, h, z);
    int it, iz;
    double wt, wz;
    bool in_t, in_z;
    locate(c.wt, c.nt, t, it, wt, in_t, h.t);
    locate(c.wz, c.nz, z, iz, wz, in_z, h.z);
    if (c.nz > 1 && z > c.wz[c.nz - 1]) return background(c, h, z);
    const int nslice = (wt != 0.0) ? 2 : 1;
    int iy[2], ix[2];
    double wy[2], wx[2];
    bool in_y[2], in_x[2];
    int slice[2];
    double sw[2];
    for (int s = 0; s < nslice; ++s) {
        slice[s] = std::min(it + s, c.nt - 1);
        sw[s] = (s == 0) ? 1.0 - wt : wt;
        const double dt = t - c.wt[slice[s]];
        locate(c.wy, c.ny, y - c.cy * dt, iy[s], wy[s], in_y[s], h.y[s]);
        locate(c.wx, c.nx, x - c.cx * dt, ix[s], wx[s], in_x[s], h.x[s]);
        if ((c.nx > 1 && !in_x[s]) || (c.ny > 1 && !in_y[s])) return background(c, h, z);
    }
    // 1 / cell width per axis (0 outside the axis or on a single point)
    double invz = 0.0;
    if (c.nz > 1 && in_z) invz = 1.0 / (c.wz[iz + 1] - c.wz[iz]);
    double acc[3] = {0.0, 0.0, 0.0};
    double dxu = 0.0, dyv = 0.0, dzw = 0.0;
    const int64_t nxy = static_cast<int64_t>(c.ny) * c.nx;
    for (int s = 0; s < nslice; ++s) {
        double invy = 0.0, invx = 0.0;
        if (c.ny > 1 && in_y[s]) invy = 1.0 / (c.wy[iy[s] + 1] - c.wy[iy[s]]);
        if (c.nx > 1 && in_x[s]) invx = 1.0 / (c.wx[ix[s] + 1] - c.wx[ix[s]]);
        for (int corner = 0; corner < 8; ++corner) {
            const int hz = (corner >> 2) & 1, hy = (corner >> 1) & 1, hx = corner & 1;
            const double cw = (hz ? wz : 1.0 - wz) * (hy ? wy[s] : 1.0 - wy[s]) *
                              (hx ? wx[s] : 1.0 - wx[s]);
            const double weight = cw * sw[s];
            // a corner with zero weight does not matter (unless derivatives are wanted)
            if (weight == 0.0 && !c.wdiv) continue;
            const int64_t flat =
                (static_cast<int64_t>(slice[s]) * c.nz +
                 std::min(iz + hz, c.nz - 1)) * nxy +
                static_cast<int64_t>(std::min(iy[s] + hy, c.ny - 1)) * c.nx +
                std::min(ix[s] + hx, c.nx - 1);
            const double* q = c.uvw + 3 * flat;
            if (!(std::isfinite(q[0]) && std::isfinite(q[1]) && std::isfinite(q[2])))
                return background(c, h, z);
            acc[0] += weight * q[0];
            acc[1] += weight * q[1];
            acc[2] += weight * q[2];
            if (c.wdiv) {
                // derivative along one axis: that axis' weight becomes +-1 / width
                const double wzc = hz ? wz : 1.0 - wz;
                const double wyc = hy ? wy[s] : 1.0 - wy[s];
                const double wxc = hx ? wx[s] : 1.0 - wx[s];
                const double dz = wyc * wxc * ((hz ? 1.0 : -1.0) * invz);
                const double dy = wzc * wxc * ((hy ? 1.0 : -1.0) * invy);
                const double dx = wzc * wyc * ((hx ? 1.0 : -1.0) * invx);
                dzw += (dz * sw[s]) * q[2];
                dyv += (dy * sw[s]) * q[1];
                dxu += (dx * sw[s]) * q[0];
            }
        }
    }
    Wind r;
    r.u = acc[0];
    r.v = acc[1];
    r.w = acc[2];
    r.dudx = dxu;
    r.dvdy = dyv;
    r.dwdz = dzw;
    return r;
}

// Environment at height z from the tables at the levels of the profile (linear
// in height): air density and its slope, and the properties for evaporation.
struct Env {
    double rho, drho, nu, fkd, ssat, cs;
};

inline Env environment(const Ctx& c, Hints& h, double z) {
    int i;
    double w;
    bool in;
    locate(c.ez, c.ne, z, i, w, in, h.e);
    const int j = std::min(i + 1, c.ne - 1);
    Env r;
    r.rho = c.erho[i] + w * (c.erho[j] - c.erho[i]);
    r.drho = (c.ne > 1 && in) ? (c.erho[j] - c.erho[i]) / (c.ez[j] - c.ez[i]) : 0.0;
    r.nu = r.fkd = r.ssat = r.cs = 0.0;
    if (c.evap) {
        r.nu = c.enu[i] + w * (c.enu[j] - c.enu[i]);
        r.fkd = c.efkd[i] + w * (c.efkd[j] - c.efkd[i]);
        r.ssat = c.essat[i] + w * (c.essat[j] - c.essat[i]);
        r.cs = c.ecs[i] + w * (c.ecs[j] - c.ecs[i]);
    }
    return r;
}

// Stop height and its horizontal gradient at (x, y): bilinear in the grid
// z(y, x) (clamped at its edges), else the constant of the drop.
struct Stop {
    double z, gx, gy;
};

inline Stop stop_at(const Ctx& c, Hints& h, double x, double y, double zconst) {
    Stop r{zconst, 0.0, 0.0};
    if (c.nsy <= 0) return r;
    int iy, ix;
    double wy, wx;
    bool in_y, in_x;
    locate(c.sy, c.nsy, y, iy, wy, in_y, h.sy);
    locate(c.sx, c.nsx, x, ix, wx, in_x, h.sx);
    const int jy = std::min(iy + 1, c.nsy - 1), jx = std::min(ix + 1, c.nsx - 1);
    const double z00 = c.sz[static_cast<int64_t>(iy) * c.nsx + ix];
    const double z01 = c.sz[static_cast<int64_t>(iy) * c.nsx + jx];
    const double z10 = c.sz[static_cast<int64_t>(jy) * c.nsx + ix];
    const double z11 = c.sz[static_cast<int64_t>(jy) * c.nsx + jx];
    r.z = (1.0 - wy) * ((1.0 - wx) * z00 + wx * z01) + wy * ((1.0 - wx) * z10 + wx * z11);
    if (c.nsx > 1 && in_x)
        r.gx = ((1.0 - wy) * (z01 - z00) + wy * (z11 - z10)) / (c.sx[ix + 1] - c.sx[ix]);
    if (c.nsy > 1 && in_y)
        r.gy = ((1.0 - wx) * (z10 - z00) + wx * (z11 - z01)) / (c.sy[iy + 1] - c.sy[iy]);
    return r;
}

// -------------------------------------------------------------------------
// drop physics
// -------------------------------------------------------------------------

// Sea-level terminal speed (at least zero) and its derivative with respect to D.
inline void v0_of(const Ctx& c, double d, double& v, double& dv) {
    if (c.fall_kind == 0) {
        const double e = std::exp(-0.6 * d);
        v = 9.65 - 10.3 * e;
        dv = 6.18 * e;
    } else if (c.fall_kind == 1) {
        v = c.fc[0] * std::pow(d, c.fc[1]) * std::exp(-c.fc[2] * d);
        dv = v * (c.fc[1] / d - c.fc[2]);
    } else {
        v = 0.0;
        dv = 0.0;
        for (int k = c.nfc - 1; k >= 0; --k) {
            dv = dv * d + v;
            v = v * d + c.fc[k];
        }
    }
    if (!(v > 0.0)) {
        v = 0.0;
        dv = 0.0;
    }
}

// -------------------------------------------------------------------------
// random numbers (counter based, independent of the number of threads)
// -------------------------------------------------------------------------

inline uint64_t mix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

inline double normal(uint64_t key, uint64_t ctr) {
    const uint64_t h1 = mix64(key + (2 * ctr) * 0xD1B54A32D192ED03ULL);
    const uint64_t h2 = mix64(key + (2 * ctr + 1) * 0xD1B54A32D192ED03ULL);
    const double u1 = (static_cast<double>(h1 >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    const double u2 = (static_cast<double>(h2 >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * kPi * u2);
}

// -------------------------------------------------------------------------
// right-hand side and integration
// -------------------------------------------------------------------------

struct State {
    double v[5];
};

// F = dir * d(state)/dt, the derivative with respect to the integration time.
inline void rhs(const Ctx& c, Hints& h, double t, const State& y, const double* turb,
                State& f) {
    const Wind wnd = sample_wind(c, h, t, y.v[0], y.v[1], y.v[2]);
    const Env ev = environment(c, h, y.v[2]);
    const double d = std::sqrt(std::max(y.v[3], kDFloor * kDFloor));
    double v0, dv0;
    v0_of(c, d, v0, dv0);
    const double corr = c.dens ? std::pow(c.rho0 / ev.rho, 0.4) : 1.0;
    const double vt = v0 * corr;
    f.v[0] = wnd.u + turb[0];
    f.v[1] = wnd.v + turb[1];
    f.v[2] = wnd.w + turb[2] - vt;
    f.v[3] = 0.0;
    f.v[4] = 0.0;
    double kc = 0.0, sdot = 0.0, arg = 0.0;
    const double dm = d * 1.0e-3;
    if (c.evap) {
        arg = vt * dm / ev.nu;
        const double fv = c.vav + c.vbv * ev.cs * std::sqrt(arg);
        kc = 8.0e6 * ev.ssat / (ev.fkd * kRhoW);
        sdot = kc * fv;
        f.v[3] = sdot;
    }
    if (c.ratio) {
        double div = 0.0;
        if (c.wdiv) div = wnd.dudx + wnd.dvdy + wnd.dwdz;
        // Vt is proportional to rho^-0.4: dVt/dz = -0.4 Vt (drho/dz) / rho
        if (c.dens) div = div + 0.4 * vt * ev.drho / ev.rho;
        if (c.evap) {
            // d(dD/dt)/dD with dD/dt = (ds/dt) / (2 D)
            double dfv = 0.0;
            if (arg > 0.0)
                dfv = c.vbv * ev.cs * (0.5 / std::sqrt(arg)) *
                      (dv0 * corr * dm + vt * 1.0e-3) / ev.nu;
            div = div + kc * dfv / (2.0 * d) - sdot / (2.0 * d * d);
        }
        f.v[4] = -div;
    }
    for (int k = 0; k < 5; ++k) f.v[k] = c.dir * f.v[k];
}

inline State axpy(const State& y, double a, const State& k) {
    State r;
    for (int i = 0; i < 5; ++i) r.v[i] = y.v[i] + a * k.v[i];
    return r;
}

inline State step(const Ctx& c, Hints& hs, double t, const State& y, const State& k1,
                  const double* turb, double h) {
    State k2, k3, k4, r;
    if (c.scheme == 2) {
        rhs(c, hs, t + c.dir * h, axpy(y, h, k1), turb, k2);
        for (int i = 0; i < 5; ++i) r.v[i] = y.v[i] + 0.5 * h * (k1.v[i] + k2.v[i]);
        return r;
    }
    rhs(c, hs, t + c.dir * 0.5 * h, axpy(y, 0.5 * h, k1), turb, k2);
    rhs(c, hs, t + c.dir * 0.5 * h, axpy(y, 0.5 * h, k2), turb, k3);
    rhs(c, hs, t + c.dir * h, axpy(y, h, k3), turb, k4);
    for (int i = 0; i < 5; ++i)
        r.v[i] = y.v[i] + h / 6.0 * (k1.v[i] + 2.0 * k2.v[i] + 2.0 * k3.v[i] + k4.v[i]);
    return r;
}

inline double hermite(double th, double a0, double fa0, double a1, double fa1) {
    // cubic Hermite interpolant on [0, 1]; fa are derivatives with respect to th
    const double t2 = th * th, t3 = t2 * th;
    return (2.0 * t3 - 3.0 * t2 + 1.0) * a0 + (t3 - 2.0 * t2 + th) * fa0 +
           (-2.0 * t3 + 3.0 * t2) * a1 + (t3 - t2) * fa1;
}

// First sign change of g(th) = scale * (hermite(th) - level(th)) on [0, 1] by
// bisection (g(0) > 0 >= g(1)); the level varies linearly from lev0 to lev1.
inline double hermite_root(double scale, double lev0, double lev1, double a0, double fa0,
                           double a1, double fa1) {
    double lo = 0.0, hi = 1.0;
    for (int it = 0; it < kBisect; ++it) {
        const double mid = 0.5 * (lo + hi);
        const double g = scale * (hermite(mid, a0, fa0, a1, fa1) - ((1.0 - mid) * lev0 + mid * lev1));
        if (g > 0.0)
            lo = mid;
        else
            hi = mid;
    }
    return 0.5 * (lo + hi);
}

void run_drop(const Ctx& c, int64_t item, double x0, double y0, double z0, double t0,
              double d0, double zstop, double* out, double* path) {
    for (int k = 0; k < kNout; ++k) out[k] = kNaN;
    out[6] = kInvalid;
    if (!(std::isfinite(x0) && std::isfinite(y0) && std::isfinite(z0) && std::isfinite(t0) &&
          std::isfinite(d0) && std::isfinite(zstop) && d0 > 0.0))
        return;
    State y;
    y.v[0] = x0;
    y.v[1] = y0;
    y.v[2] = z0;
    y.v[3] = d0 * d0;
    y.v[4] = 0.0;
    const double h = c.dt;
    const int64_t nsteps = static_cast<int64_t>(std::ceil(c.max_time / h));
    uint64_t key = 0;
    double turb[3] = {0.0, 0.0, 0.0};
    double ar = 0.0, ar_s = 0.0;
    if (c.turb) {
        key = mix64(c.seed + static_cast<uint64_t>(item + 1) * 0x9E3779B97F4A7C15ULL);
        ar = std::exp(-h / c.timescale);
        ar_s = std::sqrt(1.0 - ar * ar);
        turb[0] = c.sigma_h * normal(key, 0);
        turb[1] = c.sigma_h * normal(key, 1);
        turb[2] = c.sigma_w * normal(key, 2);
    }
    State f0;
    Hints hs;
    rhs(c, hs, t0, y, turb, f0);
    const double vz0 = -c.dir * f0.v[2];  // downward speed of the drop at the start
    int rec = 0;
    auto record = [&](double el, const State& s) {
        if (!path || rec >= c.nrec) return;
        double* r = path + 5 * static_cast<int64_t>(rec++);
        r[0] = el;
        r[1] = s.v[0];
        r[2] = s.v[1];
        r[3] = s.v[2];
        r[4] = std::sqrt(std::max(s.v[3], 0.0));
    };
    auto finish = [&](double elapsed, const State& s, int status, const State& fend) {
        out[0] = s.v[0];
        out[1] = s.v[1];
        out[2] = s.v[2];
        out[3] = elapsed;
        out[4] = std::sqrt(std::max(s.v[3], 0.0));
        out[5] = s.v[4];
        out[6] = status;
        out[7] = vz0;
        out[8] = -c.dir * fend.v[2];
        record(elapsed, s);
    };
    record(0.0, y);
    Stop sp = stop_at(c, hs, x0, y0, zstop);
    if (!std::isfinite(sp.z)) return;  // keep the NaN output, status invalid
    if (c.dir * (z0 - sp.z) <= 0.0) {
        finish(0.0, y, kLanded, f0);
        return;
    }
    if (y.v[3] <= c.smin) {
        finish(0.0, y, kEvaporated, f0);
        return;
    }
    State fk = f0;
    for (int64_t n = 0; n < nsteps; ++n) {
        const double t = t0 + c.dir * (static_cast<double>(n) * h);
        const State y1 = step(c, hs, t, y, fk, turb, h);
        const Stop sp1 = stop_at(c, hs, y1.v[0], y1.v[1], zstop);
        const bool hit_z = c.dir * (y1.v[2] - sp1.z) <= 0.0;
        const bool hit_s = y1.v[3] <= c.smin;
        if (hit_z || hit_s) {
            State f1;
            rhs(c, hs, t + c.dir * h, y1, turb, f1);
            double th = 2.0;
            int status = kLanded;
            if (hit_z)
                th = hermite_root(c.dir, sp.z, sp1.z, y.v[2], h * fk.v[2], y1.v[2], h * f1.v[2]);
            if (hit_s) {
                const double ts =
                    hermite_root(1.0, c.smin, c.smin, y.v[3], h * fk.v[3], y1.v[3], h * f1.v[3]);
                if (ts < th) {
                    th = ts;
                    status = kEvaporated;
                }
            }
            State e, fe;
            if (status == kLanded) {
                // The trial step overshoots the surface, where the environment is
                // extrapolated, so the Hermite root is only a first guess: Newton
                // iterations with real (shortened) steps land on the surface with
                // the full accuracy of the scheme.
                for (int it = 0; it < kNewton; ++it) {
                    e = step(c, hs, t, y, fk, turb, th * h);
                    rhs(c, hs, t + c.dir * (th * h), e, turb, fe);
                    const Stop se = stop_at(c, hs, e.v[0], e.v[1], zstop);
                    const double g = c.dir * (e.v[2] - se.z);
                    const double gp =
                        c.dir * h * (fe.v[2] - se.gx * fe.v[0] - se.gy * fe.v[1]);
                    if (!(gp != 0.0) || !std::isfinite(gp)) break;
                    th = std::min(std::max(th - g / gp, 0.0), 1.0);
                }
                e = step(c, hs, t, y, fk, turb, th * h);
                e.v[2] = stop_at(c, hs, e.v[0], e.v[1], zstop).z;
            } else {
                for (int k = 0; k < 5; ++k)
                    e.v[k] = hermite(th, y.v[k], h * fk.v[k], y1.v[k], h * f1.v[k]);
                e.v[3] = c.smin;
            }
            // downward speed at the end point: from the end-point right-hand side
            rhs(c, hs, t + c.dir * (th * h), e, turb, fe);
            finish((static_cast<double>(n) + th) * h, e, status, fe);
            return;
        }
        y = y1;
        sp = sp1;
        if (c.turb) {
            const uint64_t m = static_cast<uint64_t>(n + 1);
            turb[0] = ar * turb[0] + ar_s * c.sigma_h * normal(key, 3 * m);
            turb[1] = ar * turb[1] + ar_s * c.sigma_h * normal(key, 3 * m + 1);
            turb[2] = ar * turb[2] + ar_s * c.sigma_w * normal(key, 3 * m + 2);
        }
        rhs(c, hs, t + c.dir * h, y, turb, fk);
        if (c.stride > 0 && ((n + 1) % c.stride) == 0)
            record(static_cast<double>(n + 1) * h, y);
    }
    finish(static_cast<double>(nsteps) * h, y, kAloft, fk);
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
        throw std::invalid_argument(std::string(name) + " must be 1-D of the size of x0");
}

void check_axis(const DArray& a, int64_t n, const char* name) {
    if (a.ndim() != 1 || a.shape(0) != n)
        throw std::invalid_argument(std::string(name) + " does not match the wind grid");
    for (int64_t i = 1; i < n; ++i)
        if (!(a.data()[i] > a.data()[i - 1]))
            throw std::invalid_argument(std::string(name) + " must be increasing");
}

}  // namespace

// Integrate all drops. `par` and `ipar` hold the scalar parameters (see
// rain_trajectories.py), `fc` the fall-speed coefficients. Returns
// (out (n, 9), path (n, nrec, 5)).
py::tuple integrate_py(const DArray& x0, const DArray& y0, const DArray& z0,
                       const DArray& t0, const DArray& d0, const DArray& zstop,
                       const DArray& wt, const DArray& wz, const DArray& wy,
                       const DArray& wx, const DArray& uvw, const DArray& bz,
                       const DArray& bu, const DArray& bv, const DArray& bw,
                       const DArray& sy, const DArray& sx, const DArray& sz,
                       const DArray& ez, const DArray& erho, const DArray& enu,
                       const DArray& efkd, const DArray& essat, const DArray& ecs,
                       const DArray& par, const py::array_t<int64_t,
                       py::array::c_style | py::array::forcecast>& ipar, const DArray& fc,
                       uint64_t seed, int n_threads) {
    if (x0.ndim() != 1) throw std::invalid_argument("x0 must be 1-D");
    const int64_t n = x0.shape(0);
    check_1d(y0, n, "y0");
    check_1d(z0, n, "z0");
    check_1d(t0, n, "t0");
    check_1d(d0, n, "d0");
    check_1d(zstop, n, "zstop");
    if (par.ndim() != 1 || par.shape(0) < 16) throw std::invalid_argument("par too short");
    if (ipar.ndim() != 1 || ipar.shape(0) < 10) throw std::invalid_argument("ipar too short");
    Ctx c;
    const double* p = par.data();
    c.dir = p[0];
    c.dt = p[1];
    c.max_time = p[2];
    c.smin = p[3];
    c.rho0 = p[4];
    c.vav = p[5];
    c.vbv = p[6];
    c.cx = p[7];
    c.cy = p[8];
    c.sigma_h = p[10];
    c.sigma_w = p[11];
    c.timescale = p[12];
    const int64_t* q = ipar.data();
    c.scheme = static_cast<int>(q[0]);
    c.fall_kind = static_cast<int>(q[1]);
    c.dens = static_cast<int>(q[2]);
    c.evap = static_cast<int>(q[3]);
    c.ratio = static_cast<int>(q[4]);
    c.wdiv = static_cast<int>(q[5]);
    c.turb = static_cast<int>(q[6]);
    c.stride = static_cast<int>(q[7]);
    c.nrec = static_cast<int>(q[8]);
    c.seed = seed;
    c.nfc = static_cast<int>(fc.shape(0));
    c.fc = fc.data();
    if (!(c.dt > 0.0) || !(c.max_time >= 0.0)) throw std::invalid_argument("bad time step");
    if (c.scheme != 2 && c.scheme != 4) throw std::invalid_argument("scheme must be 2 or 4");
    if (c.fall_kind < 0 || c.fall_kind > 2) throw std::invalid_argument("bad fall_kind");
    if (c.fall_kind == 1 && c.nfc < 3) throw std::invalid_argument("power law needs a, b, f");
    if (c.fall_kind == 2 && c.nfc < 1) throw std::invalid_argument("polynomial needs coefficients");
    if (c.turb && !(c.timescale > 0.0)) throw std::invalid_argument("bad timescale");
    // wind grid
    const int64_t nt = wt.shape(0), nz = wz.shape(0), ny = wy.shape(0), nx = wx.shape(0);
    if (nt > 0) {
        if (nz < 1 || ny < 1 || nx < 1) throw std::invalid_argument("empty wind axis");
        check_axis(wt, nt, "wind time");
        check_axis(wz, nz, "wind z");
        check_axis(wy, ny, "wind y");
        check_axis(wx, nx, "wind x");
        if (uvw.size() != nt * nz * ny * nx * 3)
            throw std::invalid_argument("wind array does not match its axes");
        c.nt = static_cast<int>(nt);
        c.nz = static_cast<int>(nz);
        c.ny = static_cast<int>(ny);
        c.nx = static_cast<int>(nx);
        c.wt = wt.data();
        c.wz = wz.data();
        c.wy = wy.data();
        c.wx = wx.data();
        c.uvw = uvw.data();
    }
    if (sy.shape(0) > 0) {
        if (sx.shape(0) < 1 || sz.ndim() != 2 || sz.shape(0) != sy.shape(0) ||
            sz.shape(1) != sx.shape(0))
            throw std::invalid_argument("stop surface does not match its axes");
        check_axis(sy, sy.shape(0), "stop surface y");
        check_axis(sx, sx.shape(0), "stop surface x");
        c.nsy = static_cast<int>(sy.shape(0));
        c.nsx = static_cast<int>(sx.shape(0));
        c.sy = sy.data();
        c.sx = sx.data();
        c.sz = sz.data();
    }
    const int64_t nb = bz.shape(0), ne = ez.shape(0);
    if (nb < 1 || bu.shape(0) != nb || bv.shape(0) != nb || bw.shape(0) != nb)
        throw std::invalid_argument("background wind profile is empty or inconsistent");
    if (ne < 1 || erho.shape(0) != ne || enu.shape(0) != ne || efkd.shape(0) != ne ||
        essat.shape(0) != ne || ecs.shape(0) != ne)
        throw std::invalid_argument("thermodynamic profile is empty or inconsistent");
    check_axis(bz, nb, "wind profile height");
    check_axis(ez, ne, "thermodynamic profile height");
    c.nb = static_cast<int>(nb);
    c.bz = bz.data();
    c.bu = bu.data();
    c.bv = bv.data();
    c.bw = bw.data();
    c.ne = static_cast<int>(ne);
    c.ez = ez.data();
    c.erho = erho.data();
    c.enu = enu.data();
    c.efkd = efkd.data();
    c.essat = essat.data();
    c.ecs = ecs.data();

    py::array_t<double> out(std::vector<py::ssize_t>{n, kNout});
    const int64_t nrec = c.stride > 0 ? c.nrec : 0;
    py::array_t<double> path(std::vector<py::ssize_t>{n, nrec, 5});
    double* po = out.mutable_data();
    double* pp = path.mutable_data();
    if (nrec > 0)
        std::fill(pp, pp + n * nrec * 5, kNaN);
    const double *px = x0.data(), *py_ = y0.data(), *pz = z0.data(), *pt = t0.data(),
                 *pd = d0.data(), *ps = zstop.data();
    {
        py::gil_scoped_release release;
        parallel_blocks(n, n_threads, [&](int64_t g) {
            run_drop(c, g, px[g], py_[g], pz[g], pt[g], pd[g], ps[g], po + kNout * g,
                     nrec > 0 ? pp + 5 * nrec * g : nullptr);
        });
    }
    return py::make_tuple(out, path);
}

PYBIND11_MODULE(_rain_trajectories, m) {
    m.doc() = "Compiled raindrop trajectory kernel for radarx.";
    m.def("integrate", &integrate_py, py::arg("x0"), py::arg("y0"), py::arg("z0"),
          py::arg("t0"), py::arg("d0"), py::arg("zstop"), py::arg("wind_t"),
          py::arg("wind_z"), py::arg("wind_y"), py::arg("wind_x"), py::arg("uvw"),
          py::arg("bg_z"), py::arg("bg_u"), py::arg("bg_v"), py::arg("bg_w"),
          py::arg("stop_y"), py::arg("stop_x"), py::arg("stop_z"),
          py::arg("env_z"), py::arg("env_rho"), py::arg("env_nu"), py::arg("env_fkd"),
          py::arg("env_ssat"), py::arg("env_cs"),
          py::arg("par"), py::arg("ipar"), py::arg("fall_coeffs"), py::arg("seed") = 0,
          py::arg("n_threads") = 0);
}
