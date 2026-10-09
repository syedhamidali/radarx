// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Cold-pool, wind-profile and baroclinity kernel.
//
// thermo():    potential temperatures of every sample (elementwise).
// cold_pool(): column integral C^2 = 2 int (-B) dz up to the top of the cold
//              pool, per column.
// profile():   winds at the bottom and top of a layer and the storm-relative
//              helicity over it, per column.
// gradient():  horizontal derivatives of a field on (batch, y, x), per row.
// vad():       velocity-azimuth display wind fit, per range ring.
//
// Every element, column or row is independent, so each function forms one
// pool of work that threads take in blocks from an atomic counter, with the
// GIL released. Every step follows the NumPy reference in
// radarx/retrieve/_coldpool_numpy.py, in the same order of operations.

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
constexpr double kCpd = 1005.7;
constexpr double kKappa = kRd / kCpd;
constexpr double kT0 = 273.15;
constexpr double kP0 = 100000.0;

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
            const int64_t g0 = next.fetch_add(block);
            if (g0 >= total) break;
            const int64_t g1 = std::min(total, g0 + block);
            for (int64_t g = g0; g < g1; ++g) body(g);
        }
    };
    std::vector<std::thread> pool;
    for (int i = 1; i < nt; ++i) pool.emplace_back(worker);
    worker();
    for (auto& th : pool) th.join();
}

inline double esat(double t) {  // Bolton (1980) eq. (10), Pa
    const double tc = t - kT0;
    return 611.2 * std::exp(17.67 * tc / (tc + 243.5));
}

// --------------------------------------------------------------------------
// thermodynamics

DArray thermo_py(DArray t, DArray p, DArray td, int n_threads) {
    const int64_t n = t.size();
    if (p.size() != n || td.size() != n) throw std::invalid_argument("size mismatch");
    DArray out({static_cast<py::ssize_t>(4), static_cast<py::ssize_t>(n)});
    const double* T = t.data();
    const double* P = p.data();
    const double* TD = td.data();
    double* o = out.mutable_data();
    {
        py::gil_scoped_release release;
        parallel_blocks(n, 4096, n_threads, [&](int64_t i) {
            const double tk = T[i], pp = P[i];
            // dew point capped at the temperature; NaN if either is missing
            const double d = std::isnan(TD[i]) ? kNaN : (TD[i] < tk ? TD[i] : tk);
            const double th = tk * std::pow(kP0 / pp, kKappa);
            const double e = esat(d);
            const double r = kEps * e / (pp - e);
            const double thv = th * (1.0 + r / kEps) / (1.0 + r);
            // Bolton (1980) eqs. (15) and (43), r in g/kg
            const double tl = 1.0 / (1.0 / (d - 56.0) + std::log(tk / d) / 800.0) + 56.0;
            const double rg = 1000.0 * r;
            const double the = tk * std::pow(kP0 / pp, 0.2854 * (1.0 - 0.28e-3 * rg)) *
                               std::exp((3.376 / tl - 0.00254) * rg * (1.0 + 0.81e-3 * rg));
            o[i] = r;
            o[n + i] = th;
            o[2 * n + i] = thv;
            o[3 * n + i] = the;
        });
    }
    return out;
}

// --------------------------------------------------------------------------
// cold-pool intensity

// One column: z ascending is not required; levels are visited in the given
// order and must increase (callers sort). Returns C, depth, open-top flag.
inline void cold_pool_column(const double* z, const double* b, int64_t nz,
                             double bottom, double threshold, double top,
                             double* c_out, double* h_out, double* open_out) {
    *c_out = kNaN;
    *h_out = kNaN;
    *open_out = kNaN;
    int64_t k0 = -1;
    for (int64_t k = 0; k < nz; ++k) {
        if (std::isfinite(z[k]) && std::isfinite(b[k]) &&
            (!std::isfinite(bottom) || z[k] >= bottom)) {
            k0 = k;
            break;
        }
    }
    if (k0 < 0) return;
    const double z0 = z[k0];
    double zp = z0, bp = b[k0], integral = 0.0;
    if (std::isfinite(top)) {  // fixed depth: integrate -B from z0 to z0 + top
        const double zt = z0 + top;
        for (int64_t k = k0 + 1; k < nz; ++k) {
            if (!std::isfinite(z[k]) || !std::isfinite(b[k])) continue;
            if (z[k] >= zt) {
                const double bt = bp + (b[k] - bp) * (zt - zp) / (z[k] - zp);
                integral += 0.5 * (-bp - bt) * (zt - zp);
                *c_out = std::sqrt(std::max(2.0 * integral, 0.0));
                *h_out = top;
                *open_out = 0.0;
                return;
            }
            integral += 0.5 * (-bp - b[k]) * (z[k] - zp);
            zp = z[k];
            bp = b[k];
        }
        return;  // profile does not reach the top: NaN
    }
    if (bp >= threshold) {  // no cold air at the bottom
        *c_out = 0.0;
        *h_out = 0.0;
        *open_out = 0.0;
        return;
    }
    for (int64_t k = k0 + 1; k < nz; ++k) {
        if (!std::isfinite(z[k]) || !std::isfinite(b[k])) continue;
        if (b[k] >= threshold) {
            const double zc = zp + (threshold - bp) * (z[k] - zp) / (b[k] - bp);
            integral += 0.5 * (-bp - threshold) * (zc - zp);
            *c_out = std::sqrt(std::max(2.0 * integral, 0.0));
            *h_out = zc - z0;
            *open_out = 0.0;
            return;
        }
        integral += 0.5 * (-bp - b[k]) * (z[k] - zp);
        zp = z[k];
        bp = b[k];
    }
    *c_out = std::sqrt(std::max(2.0 * integral, 0.0));
    *h_out = zp - z0;
    *open_out = 1.0;
}

DArray cold_pool_py(DArray z, DArray b, DArray bottom, double threshold, DArray top,
                    int n_threads) {
    if (b.ndim() != 2 || z.ndim() != 2) throw std::invalid_argument("z and b must be 2-D");
    const int64_t ncol = b.shape(0), nz = b.shape(1);
    if (z.shape(0) != ncol || z.shape(1) != nz || bottom.size() != ncol ||
        top.size() != ncol)
        throw std::invalid_argument("shape mismatch");
    DArray out({static_cast<py::ssize_t>(3), static_cast<py::ssize_t>(ncol)});
    const double* Z = z.data();
    const double* B = b.data();
    const double* BOT = bottom.data();
    const double* TOP = top.data();
    double* o = out.mutable_data();
    {
        py::gil_scoped_release release;
        parallel_blocks(ncol, 64, n_threads, [&](int64_t c) {
            cold_pool_column(Z + c * nz, B + c * nz, nz, BOT[c], threshold, TOP[c], o + c,
                             o + ncol + c, o + 2 * ncol + c);
        });
    }
    return out;
}

// --------------------------------------------------------------------------
// wind profile: winds at the layer limits and storm-relative helicity

inline bool interp_at(const double* z, const double* u, const double* v, int64_t nz,
                      double h, double* uo, double* vo) {
    int64_t prev = -1;
    for (int64_t k = 0; k < nz; ++k) {
        if (!std::isfinite(z[k]) || !std::isfinite(u[k]) || !std::isfinite(v[k])) continue;
        if (z[k] == h) {
            *uo = u[k];
            *vo = v[k];
            return true;
        }
        if (z[k] > h) {
            if (prev < 0) return false;
            const double w = (h - z[prev]) / (z[k] - z[prev]);
            *uo = u[prev] + w * (u[k] - u[prev]);
            *vo = v[prev] + w * (v[k] - v[prev]);
            return true;
        }
        prev = k;
    }
    return false;
}

void profile_column(const double* z, const double* u, const double* v, int64_t nz,
                    double ground, double bottom, double top, double cu, double cv,
                    double* o, int64_t stride) {
    // o rows: u, v at bottom; u, v at top; storm-relative helicity;
    // layer-mean u, v (height-weighted, trapezoidal)
    for (int j = 0; j < 7; ++j) o[j * stride] = kNaN;
    double g = ground;
    if (!std::isfinite(g)) {
        for (int64_t k = 0; k < nz; ++k) {
            if (std::isfinite(z[k]) && std::isfinite(u[k]) && std::isfinite(v[k])) {
                g = z[k];
                break;
            }
        }
        if (!std::isfinite(g)) return;
    }
    const double zb = g + bottom, zt = g + top;
    double ub, vb, ut, vt;
    if (!interp_at(z, u, v, nz, zb, &ub, &vb)) return;
    if (!interp_at(z, u, v, nz, zt, &ut, &vt)) return;
    o[0] = ub;
    o[stride] = vb;
    o[2 * stride] = ut;
    o[3 * stride] = vt;
    // SRH = sum over layers of (u_{k+1}-cu)(v_k-cv) - (u_k-cu)(v_{k+1}-cv)
    double srh = 0.0, su = 0.0, sv = 0.0;
    double zp = zb, up = ub, vp = vb;
    for (int64_t k = 0; k < nz; ++k) {
        if (!std::isfinite(z[k]) || !std::isfinite(u[k]) || !std::isfinite(v[k])) continue;
        if (z[k] <= zb) continue;
        if (z[k] >= zt) break;
        srh += (u[k] - cu) * (vp - cv) - (up - cu) * (v[k] - cv);
        su += 0.5 * (up + u[k]) * (z[k] - zp);
        sv += 0.5 * (vp + v[k]) * (z[k] - zp);
        zp = z[k];
        up = u[k];
        vp = v[k];
    }
    srh += (ut - cu) * (vp - cv) - (up - cu) * (vt - cv);
    su += 0.5 * (up + ut) * (zt - zp);
    sv += 0.5 * (vp + vt) * (zt - zp);
    o[4 * stride] = (std::isfinite(cu) && std::isfinite(cv)) ? srh : kNaN;
    if (zt > zb) {
        o[5 * stride] = su / (zt - zb);
        o[6 * stride] = sv / (zt - zb);
    } else {
        o[5 * stride] = ub;
        o[6 * stride] = vb;
    }
}

DArray profile_py(DArray z, DArray u, DArray v, DArray ground, double bottom, double top,
                  DArray cu, DArray cv, int n_threads) {
    if (u.ndim() != 2) throw std::invalid_argument("u must be 2-D");
    const int64_t ncol = u.shape(0), nz = u.shape(1);
    if (z.size() != ncol * nz || v.size() != ncol * nz || ground.size() != ncol ||
        cu.size() != ncol || cv.size() != ncol)
        throw std::invalid_argument("shape mismatch");
    DArray out({static_cast<py::ssize_t>(7), static_cast<py::ssize_t>(ncol)});
    const double* Z = z.data();
    const double* U = u.data();
    const double* V = v.data();
    const double* G = ground.data();
    const double* CU = cu.data();
    const double* CV = cv.data();
    double* o = out.mutable_data();
    {
        py::gil_scoped_release release;
        parallel_blocks(ncol, 64, n_threads, [&](int64_t c) {
            profile_column(Z + c * nz, U + c * nz, V + c * nz, nz, G[c], bottom, top, CU[c],
                           CV[c], o + c, ncol);
        });
    }
    return out;
}

// --------------------------------------------------------------------------
// horizontal gradient on (batch, y, x)

// derivative of f at i along a line of n points with stride s and coordinate x:
// centred where both neighbours are valid, one-sided where one is, NaN else.
inline double deriv(const double* f, const double* x, int64_t i, int64_t n, int64_t s) {
    if (!std::isfinite(f[i * s])) return kNaN;
    const bool lo = i > 0 && std::isfinite(f[(i - 1) * s]);
    const bool hi = i + 1 < n && std::isfinite(f[(i + 1) * s]);
    if (lo && hi) return (f[(i + 1) * s] - f[(i - 1) * s]) / (x[i + 1] - x[i - 1]);
    if (hi) return (f[(i + 1) * s] - f[i * s]) / (x[i + 1] - x[i]);
    if (lo) return (f[i * s] - f[(i - 1) * s]) / (x[i] - x[i - 1]);
    return kNaN;
}

DArray gradient_py(DArray f, DArray x, DArray y, int n_threads) {
    if (f.ndim() != 3) throw std::invalid_argument("f must be 3-D (batch, y, x)");
    const int64_t nb = f.shape(0), ny = f.shape(1), nx = f.shape(2);
    if (x.size() != nx || y.size() != ny) throw std::invalid_argument("coordinate size");
    DArray out({static_cast<py::ssize_t>(2), static_cast<py::ssize_t>(nb),
                static_cast<py::ssize_t>(ny), static_cast<py::ssize_t>(nx)});
    const double* F = f.data();
    const double* X = x.data();
    const double* Y = y.data();
    double* o = out.mutable_data();
    const int64_t plane = ny * nx, total = nb * plane;
    {
        py::gil_scoped_release release;
        parallel_blocks(nb * ny, 8, n_threads, [&](int64_t row) {
            const int64_t b = row / ny, j = row % ny;
            const double* fp = F + b * plane;
            for (int64_t i = 0; i < nx; ++i) {
                const int64_t idx = b * plane + j * nx + i;
                o[idx] = deriv(fp + j * nx, X, i, nx, 1);
                o[total + idx] = deriv(fp + i, Y, j, ny, nx);
            }
        });
    }
    return out;
}

// --------------------------------------------------------------------------
// velocity-azimuth display: least-squares fit of
// vr = a0 + b1 sin(az) + b2 cos(az) on every range ring (Browning and Wexler
// 1968); u = b1 / cos(el), v = b2 / cos(el)

void vad_ring(const double* vr, const double* az, int64_t naz, double cos_el,
              int64_t min_gates, double min_spread, double* o, int64_t stride) {
    for (int j = 0; j < 5; ++j) o[j * stride] = kNaN;
    double n = 0, ss = 0, sc = 0, sv = 0;
    for (int64_t k = 0; k < naz; ++k) {
        if (!std::isfinite(vr[k]) || !std::isfinite(az[k])) continue;
        n += 1;
        ss += std::sin(az[k]);
        sc += std::cos(az[k]);
        sv += vr[k];
    }
    o[4 * stride] = n;
    if (n < static_cast<double>(min_gates) || n < 3) return;
    const double ms = ss / n, mc = sc / n, mv = sv / n;
    double css = 0, ccc = 0, csc = 0, csv = 0, ccv = 0;
    for (int64_t k = 0; k < naz; ++k) {
        if (!std::isfinite(vr[k]) || !std::isfinite(az[k])) continue;
        const double ds = std::sin(az[k]) - ms, dc = std::cos(az[k]) - mc;
        const double dv = vr[k] - mv;
        css += ds * ds;
        ccc += dc * dc;
        csc += ds * dc;
        csv += ds * dv;
        ccv += dc * dv;
    }
    const double det = css * ccc - csc * csc;
    if (!(det / (n * n) >= min_spread)) return;
    const double b1 = (ccc * csv - csc * ccv) / det;
    const double b2 = (css * ccv - csc * csv) / det;
    const double a0 = mv - b1 * ms - b2 * mc;
    double sse = 0;
    for (int64_t k = 0; k < naz; ++k) {
        if (!std::isfinite(vr[k]) || !std::isfinite(az[k])) continue;
        const double r = vr[k] - a0 - b1 * std::sin(az[k]) - b2 * std::cos(az[k]);
        sse += r * r;
    }
    o[0] = b1 / cos_el;
    o[stride] = b2 / cos_el;
    o[2 * stride] = a0;
    o[3 * stride] = std::sqrt(sse / n);
}

DArray vad_py(DArray vr, DArray az, DArray cos_el, int64_t min_gates, double min_spread,
              int n_threads) {
    if (vr.ndim() != 2 || az.ndim() != 2) throw std::invalid_argument("vr, az must be 2-D");
    const int64_t nring = vr.shape(0), naz = vr.shape(1);
    if (az.shape(0) != nring || az.shape(1) != naz || cos_el.size() != nring)
        throw std::invalid_argument("shape mismatch");
    DArray out({static_cast<py::ssize_t>(5), static_cast<py::ssize_t>(nring)});
    const double* V = vr.data();
    const double* A = az.data();
    const double* C = cos_el.data();
    double* o = out.mutable_data();
    {
        py::gil_scoped_release release;
        parallel_blocks(nring, 64, n_threads, [&](int64_t r) {
            vad_ring(V + r * naz, A + r * naz, naz, C[r], min_gates, min_spread, o + r,
                     nring);
        });
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(_coldpool, m) {
    m.doc() = "Cold-pool, wind-profile and baroclinity kernel (multithreaded).";
    m.def("thermo", &thermo_py, py::arg("t"), py::arg("p"), py::arg("td"),
          py::arg("n_threads") = 0);
    m.def("cold_pool", &cold_pool_py, py::arg("z"), py::arg("b"), py::arg("bottom"),
          py::arg("threshold"), py::arg("top"), py::arg("n_threads") = 0);
    m.def("profile", &profile_py, py::arg("z"), py::arg("u"), py::arg("v"),
          py::arg("ground"), py::arg("bottom"), py::arg("top"), py::arg("cu"),
          py::arg("cv"), py::arg("n_threads") = 0);
    m.def("gradient", &gradient_py, py::arg("f"), py::arg("x"), py::arg("y"),
          py::arg("n_threads") = 0);
    m.def("vad", &vad_py, py::arg("vr"), py::arg("az"), py::arg("cos_el"),
          py::arg("min_gates"), py::arg("min_spread"), py::arg("n_threads") = 0);
}
