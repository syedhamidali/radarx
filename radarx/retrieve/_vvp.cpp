// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Volume velocity processing (VVP): least squares fit of a linear wind to the radial
// velocities inside analysis volumes (Waldteufel and Corbin 1979; the form of Doviak and
// Zrnic 1993, Eqs. 9.11 to 9.18).
//
// The data are scattered gates: position (x, y, z) in m east, north and above the
// reference level, the cosine and sine of the local elevation angle of the beam (c, sn),
// the sine and cosine of the azimuth (sa, ca) and the radial velocity vr. An analysis
// volume is a cylinder of radius R and half depth D around (x0, y0, z0). With the
// unknowns K = (u0', ux, uz, v0', vy, vz, uy + vx, w0, wx, wy, wz) the radial velocity is
// vr = sum_k P_k K_k with the predictors
//
//     P0 = c sa            P1 = c sa (x - x0)     P2 = c sa (z - z0)
//     P3 = c ca            P4 = c ca (y - y0)     P5 = c ca (z - z0)
//     P6 = c (cos(az) (x - x0) + sin(az) (y - y0)) / 2
//     P7 = sn              P8 = sn (x - x0)       P9 = sn (y - y0)    P10 = sn (z - z0)
//
// (Eq. 9.13). A model uses a subset of the unknowns, given as a bit mask. The normal
// equations are formed with the offsets scaled by R and D (the conditioning of the
// unscaled problem is poor), solved by a Cholesky factorisation, and the covariance of the
// estimates is s^2 (P^T P)^-1 with s^2 = RSS / (n - m) (Eq. 9.18). The module
// radarx/retrieve/vvp.py has the references; its NumPy implementation does the same steps.
//
// The gates are sorted into a uniform grid once (counting sort), so a volume only visits
// the cells that it touches. The volumes are independent: they run on a pool of threads
// and the work is handed out in small blocks through an atomic counter.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <vector>

namespace py = pybind11;
using DArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr int kParams = 11;
constexpr double kPivotTolerance = 1.0e-10;

template <class F>
void parallel_blocks(int64_t total, int n_threads, int64_t block, F&& body) {
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
            body(g0, g1);
        }
    };
    std::vector<std::thread> pool;
    for (int i = 1; i < nt; ++i) pool.emplace_back(worker);
    worker();
    for (auto& th : pool) th.join();
}

struct Settings {
    double radius;      // radius of the analysis cylinder (m)
    double half_depth;  // half depth (m)
    int mask;           // bit k set: unknown k is fitted
    int min_gates;      // fewest gates in a volume
};

struct Fit {
    double coef[kParams];
    double err[kParams];
    double div_err = kNaN;
    double rms = kNaN;
    double n = 0.0;
};

class Gates {
public:
    Gates(const double* x, const double* y, const double* z, const double* vr, const double* c,
          const double* sn, const double* sa, const double* ca, int64_t n, double cell_xy,
          double cell_z)
        : x_(x), y_(y), z_(z), vr_(vr), c_(c), sn_(sn), sa_(sa), ca_(ca), cxy_(cell_xy),
          cz_(cell_z) {
        double xmax = -std::numeric_limits<double>::infinity();
        double ymax = xmax, zmax = xmax;
        xmin_ = ymin_ = zmin_ = std::numeric_limits<double>::infinity();
        for (int64_t i = 0; i < n; ++i) {
            xmin_ = std::min(xmin_, x[i]);
            xmax = std::max(xmax, x[i]);
            ymin_ = std::min(ymin_, y[i]);
            ymax = std::max(ymax, y[i]);
            zmin_ = std::min(zmin_, z[i]);
            zmax = std::max(zmax, z[i]);
        }
        if (n == 0) {
            nx_ = ny_ = nz_ = 1;
            xmin_ = ymin_ = zmin_ = 0.0;
        } else {
            nx_ = static_cast<int64_t>(std::floor((xmax - xmin_) / cxy_)) + 1;
            ny_ = static_cast<int64_t>(std::floor((ymax - ymin_) / cxy_)) + 1;
            nz_ = static_cast<int64_t>(std::floor((zmax - zmin_) / cz_)) + 1;
        }
        const int64_t ncell = nx_ * ny_ * nz_;
        if (ncell > 200000000) throw std::invalid_argument("radius and depth are too small for the extent of the data");
        start_.assign(ncell + 1, 0);
        for (int64_t i = 0; i < n; ++i) start_[cell_of(x[i], y[i], z[i]) + 1]++;
        for (int64_t k = 0; k < ncell; ++k) start_[k + 1] += start_[k];
        order_.resize(n);
        std::vector<int64_t> fill(start_.begin(), start_.end() - 1);
        for (int64_t i = 0; i < n; ++i) order_[fill[cell_of(x[i], y[i], z[i])]++] = i;
    }

    bool fit(double x0, double y0, double z0, const Settings& s, Fit& out) const {
        int idx[kParams];
        int m = 0;
        for (int k = 0; k < kParams; ++k)
            if (s.mask & (1 << k)) idx[m++] = k;
        if (m == 0) return false;
        const double L = s.radius, D = s.half_depth;
        double N[kParams][kParams];
        double b[kParams];
        for (int a = 0; a < m; ++a) {
            b[a] = 0.0;
            for (int q = 0; q < m; ++q) N[a][q] = 0.0;
        }
        double vv = 0.0;
        int64_t count = 0;
        int64_t ix0, ix1, iy0, iy1, iz0, iz1;
        span(x0 - L, x0 + L, xmin_, cxy_, nx_, ix0, ix1);
        span(y0 - L, y0 + L, ymin_, cxy_, ny_, iy0, iy1);
        span(z0 - D, z0 + D, zmin_, cz_, nz_, iz0, iz1);
        if (ix1 < ix0 || iy1 < iy0 || iz1 < iz0) return false;
        const double L2 = L * L;
        double P[kParams];
        double p[kParams];
        for (int64_t iz = iz0; iz <= iz1; ++iz)
            for (int64_t iy = iy0; iy <= iy1; ++iy)
                for (int64_t ix = ix0; ix <= ix1; ++ix) {
                    const int64_t cell = (iz * ny_ + iy) * nx_ + ix;
                    for (int64_t q = start_[cell]; q < start_[cell + 1]; ++q) {
                        const int64_t i = order_[q];
                        const double dx = x_[i] - x0, dy = y_[i] - y0, dz = z_[i] - z0;
                        if (dx * dx + dy * dy > L2 || std::fabs(dz) > D) continue;
                        const double ex = dx / L, ey = dy / L, ez = dz / D;
                        const double c = c_[i], sn = sn_[i], sa = sa_[i], ca = ca_[i];
                        P[0] = c * sa;
                        P[1] = c * sa * ex;
                        P[2] = c * sa * ez;
                        P[3] = c * ca;
                        P[4] = c * ca * ey;
                        P[5] = c * ca * ez;
                        P[6] = 0.5 * c * (ca * ex + sa * ey);
                        P[7] = sn;
                        P[8] = sn * ex;
                        P[9] = sn * ey;
                        P[10] = sn * ez;
                        for (int a = 0; a < m; ++a) p[a] = P[idx[a]];
                        const double v = vr_[i];
                        for (int a = 0; a < m; ++a) {
                            b[a] += p[a] * v;
                            for (int q2 = 0; q2 <= a; ++q2) N[a][q2] += p[a] * p[q2];
                        }
                        vv += v * v;
                        ++count;
                    }
                }
        if (count < s.min_gates || count <= m) return false;
        // Cholesky factorisation N = G G^T (lower triangle)
        double G[kParams][kParams];
        for (int a = 0; a < m; ++a) {
            for (int q = 0; q <= a; ++q) {
                double sum = N[a][q];
                for (int r = 0; r < q; ++r) sum -= G[a][r] * G[q][r];
                if (a == q) {
                    if (!(sum > kPivotTolerance * N[a][a])) return false;
                    G[a][a] = std::sqrt(sum);
                } else {
                    G[a][q] = sum / G[q][q];
                }
            }
        }
        // solve for the coefficients, then for the inverse (column by column)
        double K[kParams];
        solve(G, m, b, K);
        double inv[kParams][kParams];
        for (int col = 0; col < m; ++col) {
            double e[kParams], x[kParams];
            for (int a = 0; a < m; ++a) e[a] = a == col ? 1.0 : 0.0;
            solve(G, m, e, x);
            for (int a = 0; a < m; ++a) inv[a][col] = x[a];
        }
        double rss = vv;
        for (int a = 0; a < m; ++a) rss -= K[a] * b[a];
        rss = std::max(rss, 0.0);
        const double s2 = rss / static_cast<double>(count - m);
        // back to unscaled units: lateral terms (1, 4, 6, 8, 9) per L, vertical terms
        // (2, 5, 10) per D
        double scale[kParams];
        for (int k = 0; k < kParams; ++k) scale[k] = 1.0;
        scale[1] = scale[4] = scale[6] = scale[8] = scale[9] = 1.0 / L;
        scale[2] = scale[5] = scale[10] = 1.0 / D;
        int pos[kParams];
        for (int k = 0; k < kParams; ++k) pos[k] = -1;
        for (int a = 0; a < m; ++a) pos[idx[a]] = a;
        for (int k = 0; k < kParams; ++k) {
            if (pos[k] < 0) {
                out.coef[k] = kNaN;
                out.err[k] = kNaN;
            } else {
                out.coef[k] = K[pos[k]] * scale[k];
                out.err[k] = std::sqrt(s2 * inv[pos[k]][pos[k]]) * scale[k];
            }
        }
        out.div_err = kNaN;
        if (pos[1] >= 0 && pos[4] >= 0) {
            const double var = inv[pos[1]][pos[1]] + inv[pos[4]][pos[4]] +
                               2.0 * inv[pos[1]][pos[4]];
            out.div_err = std::sqrt(s2 * std::max(var, 0.0)) / L;
        }
        out.rms = std::sqrt(rss / static_cast<double>(count));
        out.n = static_cast<double>(count);
        return true;
    }

private:
    static void solve(const double G[kParams][kParams], int m, const double* rhs, double* sol) {
        double t[kParams];
        for (int a = 0; a < m; ++a) {
            double sum = rhs[a];
            for (int r = 0; r < a; ++r) sum -= G[a][r] * t[r];
            t[a] = sum / G[a][a];
        }
        for (int a = m - 1; a >= 0; --a) {
            double sum = t[a];
            for (int r = a + 1; r < m; ++r) sum -= G[r][a] * sol[r];
            sol[a] = sum / G[a][a];
        }
    }

    static void span(double lo, double hi, double origin, double cell, int64_t n, int64_t& i0,
                     int64_t& i1) {
        i0 = std::max<int64_t>(0, static_cast<int64_t>(std::floor((lo - origin) / cell)));
        i1 = std::min<int64_t>(n - 1, static_cast<int64_t>(std::floor((hi - origin) / cell)));
    }

    int64_t cell_of(double x, double y, double z) const {
        const int64_t ix = static_cast<int64_t>(std::floor((x - xmin_) / cxy_));
        const int64_t iy = static_cast<int64_t>(std::floor((y - ymin_) / cxy_));
        const int64_t iz = static_cast<int64_t>(std::floor((z - zmin_) / cz_));
        return (iz * ny_ + iy) * nx_ + ix;
    }

    const double *x_, *y_, *z_, *vr_, *c_, *sn_, *sa_, *ca_;
    double cxy_, cz_, xmin_, ymin_, zmin_;
    int64_t nx_, ny_, nz_;
    std::vector<int64_t> start_, order_;
};

// Fit of every analysis volume. Returns (coef (nv, 11), err (nv, 11), extra (nv, 3)) with
// the extra columns: standard error of the divergence ux + vy, the rms residual and the
// number of gates. NaN for a volume that cannot be fitted.
std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>> vvp_fit_py(
    const DArray& x, const DArray& y, const DArray& z, const DArray& vr, const DArray& c,
    const DArray& sn, const DArray& sa, const DArray& ca, const DArray& x0, const DArray& y0,
    const DArray& z0, double radius, double half_depth, int mask, int min_gates, int n_threads) {
    const int64_t n = x.shape(0);
    for (const DArray* a : {&x, &y, &z, &vr, &c, &sn, &sa, &ca})
        if (a->ndim() != 1 || a->shape(0) != n)
            throw std::invalid_argument("the gate arrays must be 1-D and of the same length");
    const int64_t nv = x0.shape(0);
    if (x0.ndim() != 1 || y0.ndim() != 1 || z0.ndim() != 1 || y0.shape(0) != nv ||
        z0.shape(0) != nv)
        throw std::invalid_argument("x0, y0 and z0 must be 1-D and of the same length");
    if (!(radius > 0.0) || !(half_depth > 0.0))
        throw std::invalid_argument("radius and half_depth must be positive");
    if (mask <= 0 || mask >= (1 << kParams))
        throw std::invalid_argument("mask must select between 1 and 11 unknowns");
    py::array_t<double> coef({static_cast<py::ssize_t>(nv), static_cast<py::ssize_t>(kParams)});
    py::array_t<double> err({static_cast<py::ssize_t>(nv), static_cast<py::ssize_t>(kParams)});
    py::array_t<double> extra({static_cast<py::ssize_t>(nv), static_cast<py::ssize_t>(3)});
    double* co = coef.mutable_data();
    double* er = err.mutable_data();
    double* ex = extra.mutable_data();
    {
        py::gil_scoped_release release;
        const Gates gates(x.data(), y.data(), z.data(), vr.data(), c.data(), sn.data(), sa.data(),
                          ca.data(), n, radius, half_depth);
        const Settings s{radius, half_depth, mask, min_gates};
        const double *xp = x0.data(), *yp = y0.data(), *zp = z0.data();
        parallel_blocks(nv, n_threads, 16, [&](int64_t t0, int64_t t1) {
            for (int64_t t = t0; t < t1; ++t) {
                Fit f;
                const bool ok = gates.fit(xp[t], yp[t], zp[t], s, f);
                for (int k = 0; k < kParams; ++k) {
                    co[t * kParams + k] = ok ? f.coef[k] : kNaN;
                    er[t * kParams + k] = ok ? f.err[k] : kNaN;
                }
                ex[t * 3 + 0] = ok ? f.div_err : kNaN;
                ex[t * 3 + 1] = ok ? f.rms : kNaN;
                ex[t * 3 + 2] = ok ? f.n : kNaN;
            }
        });
    }
    return {coef, err, extra};
}

}  // namespace

PYBIND11_MODULE(_vvp, m) {
    m.doc() = "Volume velocity processing: linear wind fits in analysis volumes";
    m.def("vvp_fit", &vvp_fit_py, py::arg("x"), py::arg("y"), py::arg("z"), py::arg("vr"),
          py::arg("c"), py::arg("sn"), py::arg("sa"), py::arg("ca"), py::arg("x0"), py::arg("y0"),
          py::arg("z0"), py::arg("radius"), py::arg("half_depth"), py::arg("mask"),
          py::arg("min_gates") = 30, py::arg("n_threads") = 0);
}
