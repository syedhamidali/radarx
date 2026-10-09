// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Multi-Doppler variational wind retrieval: cost function and exact gradient.
//
// The structure (radial-velocity misfit, anelastic mass continuity as a weak
// constraint, smoothness, optional vertical vorticity equation) follows the
// variational dual-Doppler analyses of Gao et al. (1999, Mon. Wea. Rev. 127,
// 2128-2142), Shapiro et al. (2009, J. Atmos. Oceanic Technol. 26, 2089-2106)
// and Potvin et al. (2012, J. Atmos. Oceanic Technol. 29, 32-49). The papers'
// equations were not checked term by term; the scalings (h, h^2/U), the
// background term, the second-difference smoothness and every weight are
// radarx's own formulation.
//
// The state is (u, v, w) on a regular (z, y, x) grid. The cost is
//
//   J = Jo + Jm + Js + Jb + Jv
//
//   Jo = sum_k sum_i wo_k (a_k u + b_k v + c_k w - y_k)^2       observations
//   Jm = Cm sum_i (h / rho * (Dx(rho u) + Dy(rho v) + Dz(rho w)))^2   mass
//   Js = sum_i sum_{f = u, v, w} Csx (Sx f)^2 + Csy (Sy f)^2 + Csz (Sz f)^2
//   Jb = sum_i wb_u (u - ub)^2 + wb_v (v - vb)^2 + wb_w (w - wb)^2
//   Jv = Cv sum_i wv (s R)^2                               vertical vorticity
//
// with D first derivatives (centred inside, one-sided at the edges, as
// numpy.gradient), S second differences in grid units (zero at the edges)
// and R the residual of the steady vertical vorticity equation in a frame
// moving with the storm (ut, vt). Weights wo_k already contain Co.
//
// Jo, Jm, Js and Jb and their gradients are evaluated in ONE pass over the
// grid. The gradient is gathered rather than scattered: for every cell the
// residuals of all neighbouring cells it influences are recomputed from the
// state, so threads never write to the same cell and no residual fields are
// stored. The nonlinear vorticity term needs derivatives of derivatives and
// runs in three further passes with scratch fields (only when Cv > 0).
//
// Threads take blocks of grid rows (z, y) from an atomic counter.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

namespace py = pybind11;
using DArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

namespace {

struct Axis {
    int64_t n;       // points along the axis
    int64_t stride;  // flat-index stride
    double d;        // spacing [m]
};

// Coefficient of f[i] in the first derivative at j (numpy.gradient, edge_order=1).
inline double d1coef(int64_t j, int64_t i, const Axis& a) {
    const int64_t n = a.n;
    if (j == 0) {
        if (i == 1) return 1.0 / a.d;
        if (i == 0) return -1.0 / a.d;
        return 0.0;
    }
    if (j == n - 1) {
        if (i == n - 1) return 1.0 / a.d;
        if (i == n - 2) return -1.0 / a.d;
        return 0.0;
    }
    if (i == j + 1) return 0.5 / a.d;
    if (i == j - 1) return -0.5 / a.d;
    return 0.0;
}

// First derivative of f at flat index idx, position p along the axis.
inline double d1(const double* f, int64_t idx, int64_t p, const Axis& a) {
    const int64_t s = a.stride;
    if (p == 0) return (f[idx + s] - f[idx]) / a.d;
    if (p == a.n - 1) return (f[idx] - f[idx - s]) / a.d;
    return (f[idx + s] - f[idx - s]) * (0.5 / a.d);
}

// Adjoint of d1: sum_j coef(j -> p) g[j].
inline double d1T(const double* g, int64_t idx, int64_t p, const Axis& a) {
    double out = 0.0;
    const int64_t lo = std::max<int64_t>(0, p - 1), hi = std::min<int64_t>(a.n - 1, p + 1);
    for (int64_t j = lo; j <= hi; ++j) {
        const double c = d1coef(j, p, a);
        if (c != 0.0) out += c * g[idx + (j - p) * a.stride];
    }
    return out;
}

// Second difference (grid units) at position p; zero at the edges.
inline double d2(const double* f, int64_t idx, int64_t p, const Axis& a) {
    if (p <= 0 || p >= a.n - 1) return 0.0;
    return f[idx - a.stride] - 2.0 * f[idx] + f[idx + a.stride];
}

// Rows of the grid (z, y) handed out in blocks to all threads.
template <class F>
void parallel_rows(int64_t nrow, int n_threads, F&& body) {
    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    int nt = n_threads > 0 ? n_threads : static_cast<int>(hw);
    nt = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(nt, nrow)));
    const int64_t block = std::max<int64_t>(1, std::min<int64_t>(16, nrow / (4 * nt) + 1));
    std::atomic<int64_t> next{0};
    auto worker = [&](int tid) {
        for (;;) {
            const int64_t r0 = next.fetch_add(block);
            if (r0 >= nrow) break;
            const int64_t r1 = std::min(nrow, r0 + block);
            for (int64_t r = r0; r < r1; ++r) body(tid, r);
        }
    };
    std::vector<std::thread> pool;
    for (int t = 1; t < nt; ++t) pool.emplace_back(worker, t);
    worker(0);
    for (auto& th : pool) th.join();
}

int thread_count(int n_threads) {
    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    return n_threads > 0 ? n_threads : static_cast<int>(hw);
}

void check(const DArray& a, int64_t n, const char* name) {
    if (a.size() != n) throw std::invalid_argument(std::string(name) + " has the wrong size");
}

}  // namespace

// Cost terms (obs, mass, smooth, background, vorticity) and the gradient
// d J / d (u, v, w) as a (3, nz, ny, nx) array.
py::tuple cost_gradient(const DArray& state, const DArray& coef, const DArray& target,
                        const DArray& weight, const DArray& rho, const DArray& bg,
                        const DArray& bg_weight, const DArray& vort_weight, double dx,
                        double dy, double dz, double cm, double csx, double csy,
                        double csz, double cv, double ut, double vt, double coriolis,
                        double h, double vort_scale, int n_threads) {
    if (state.ndim() != 4 || state.shape(0) != 3)
        throw std::invalid_argument("state must be (3, nz, ny, nx)");
    const int64_t nz = state.shape(1), ny = state.shape(2), nx = state.shape(3);
    if (nz < 2 || ny < 3 || nx < 3) throw std::invalid_argument("grid must be at least 2 x 3 x 3");
    const int64_t n = nz * ny * nx;
    if (target.ndim() != 4 || target.shape(1) != nz || target.shape(2) != ny || target.shape(3) != nx)
        throw std::invalid_argument("target must be (nradar, nz, ny, nx)");
    const int64_t nr = target.shape(0);
    check(coef, 3 * nr * n, "coef");
    check(weight, nr * n, "weight");
    check(rho, n, "rho");
    check(bg, 3 * n, "bg");
    check(bg_weight, 3 * n, "bg_weight");
    check(vort_weight, n, "vort_weight");

    const double* U = state.data();
    const double* V = U + n;
    const double* W = V + n;
    const double* C = coef.data();
    const double* Y = target.data();
    const double* WO = weight.data();
    const double* RHO = rho.data();
    const double* B = bg.data();
    const double* BW = bg_weight.data();
    const double* VW = vort_weight.data();

    py::array_t<double> grad({int64_t(3), nz, ny, nx});
    double* G = grad.mutable_data();
    double* GU = G;
    double* GV = G + n;
    double* GW = G + 2 * n;
    double terms[5] = {0, 0, 0, 0, 0};

    {
        py::gil_scoped_release release;
        const Axis ax{nx, 1, dx}, ay{ny, nx, dy}, az{nz, nx * ny, dz};
        const int nt = thread_count(n_threads);
        std::vector<double> partial(static_cast<size_t>(nt) * 5 * 8, 0.0);  // padded

        // scaled mass-continuity residual at flat index j = (iz, iy, ix)
        auto mass = [&](int64_t j, int64_t iz, int64_t iy, int64_t ix) {
            // derivatives of the mass fluxes rho u, rho v, rho w
            auto flux = [&](const double* f, int64_t jj, int64_t p, const Axis& a) {
                const int64_t s = a.stride;
                if (p == 0) return (RHO[jj + s] * f[jj + s] - RHO[jj] * f[jj]) / a.d;
                if (p == a.n - 1) return (RHO[jj] * f[jj] - RHO[jj - s] * f[jj - s]) / a.d;
                return (RHO[jj + s] * f[jj + s] - RHO[jj - s] * f[jj - s]) * (0.5 / a.d);
            };
            return h / RHO[j] * (flux(U, j, ix, ax) + flux(V, j, iy, ay) + flux(W, j, iz, az));
        };

        parallel_rows(nz * ny, n_threads, [&](int tid, int64_t row) {
            const int64_t iz = row / ny, iy = row % ny;
            double jo = 0, jm = 0, js = 0, jb = 0;
            for (int64_t ix = 0; ix < nx; ++ix) {
                const int64_t i = row * nx + ix;
                double gu = 0, gv = 0, gw = 0;
                // observations (pointwise)
                for (int64_t k = 0; k < nr; ++k) {
                    const int64_t o = k * n + i;
                    const double wk = WO[o];
                    if (wk == 0.0) continue;
                    const double a = C[(3 * k) * n + i], b = C[(3 * k + 1) * n + i],
                                 c = C[(3 * k + 2) * n + i];
                    const double r = a * U[i] + b * V[i] + c * W[i] - Y[o];
                    jo += wk * r * r;
                    const double t = 2.0 * wk * r;
                    gu += t * a;
                    gv += t * b;
                    gw += t * c;
                }
                // background (pointwise)
                for (int q = 0; q < 3; ++q) {
                    const double wq = BW[q * n + i];
                    if (wq == 0.0) continue;
                    const double r = state.data()[q * n + i] - B[q * n + i];
                    jb += wq * r * r;
                    const double t = 2.0 * wq * r;
                    if (q == 0) gu += t; else if (q == 1) gv += t; else gw += t;
                }
                // smoothness: own residuals for the cost, neighbours' for the gradient
                const double* F[3] = {U, V, W};
                double gs[3] = {0, 0, 0};
                for (int q = 0; q < 3; ++q) {
                    const double* f = F[q];
                    const double sx = d2(f, i, ix, ax), sy = d2(f, i, iy, ay), sz = d2(f, i, iz, az);
                    js += csx * sx * sx + csy * sy * sy + csz * sz * sz;
                    double g = 0.0;
                    // d (S f)_j / d f_i is 1 for j = i +- 1 and -2 for j = i (interior j)
                    if (csx != 0.0) {
                        double acc = -2.0 * sx;
                        if (ix - 1 >= 1) acc += d2(f, i - 1, ix - 1, ax);
                        if (ix + 1 <= nx - 2) acc += d2(f, i + 1, ix + 1, ax);
                        g += 2.0 * csx * acc;
                    }
                    if (csy != 0.0) {
                        double acc = -2.0 * sy;
                        if (iy - 1 >= 1) acc += d2(f, i - nx, iy - 1, ay);
                        if (iy + 1 <= ny - 2) acc += d2(f, i + nx, iy + 1, ay);
                        g += 2.0 * csy * acc;
                    }
                    if (csz != 0.0) {
                        const int64_t sz_ = nx * ny;
                        double acc = -2.0 * sz;
                        if (iz - 1 >= 1) acc += d2(f, i - sz_, iz - 1, az);
                        if (iz + 1 <= nz - 2) acc += d2(f, i + sz_, iz + 1, az);
                        g += 2.0 * csz * acc;
                    }
                    gs[q] = g;
                }
                gu += gs[0];
                gv += gs[1];
                gw += gs[2];
                // mass continuity
                if (cm != 0.0) {
                    const double m0 = mass(i, iz, iy, ix);
                    jm += cm * m0 * m0;
                    double au = 0, av = 0, aw = 0;
                    for (int64_t jx = std::max<int64_t>(0, ix - 1); jx <= std::min(nx - 1, ix + 1); ++jx) {
                        const int64_t j = i + (jx - ix);
                        const double mj = jx == ix ? m0 : mass(j, iz, iy, jx);
                        au += mj / RHO[j] * d1coef(jx, ix, ax);
                    }
                    for (int64_t jy = std::max<int64_t>(0, iy - 1); jy <= std::min(ny - 1, iy + 1); ++jy) {
                        const int64_t j = i + (jy - iy) * nx;
                        const double mj = jy == iy ? m0 : mass(j, iz, jy, ix);
                        av += mj / RHO[j] * d1coef(jy, iy, ay);
                    }
                    for (int64_t jz = std::max<int64_t>(0, iz - 1); jz <= std::min(nz - 1, iz + 1); ++jz) {
                        const int64_t j = i + (jz - iz) * nx * ny;
                        const double mj = jz == iz ? m0 : mass(j, jz, iy, ix);
                        aw += mj / RHO[j] * d1coef(jz, iz, az);
                    }
                    const double t = 2.0 * cm * h * RHO[i];
                    gu += t * au;
                    gv += t * av;
                    gw += t * aw;
                }
                GU[i] = gu;
                GV[i] = gv;
                GW[i] = gw;
            }
            double* p = &partial[static_cast<size_t>(tid) * 40];
            p[0] += jo;
            p[1] += jm;
            p[2] += js;
            p[3] += jb;
        });

        if (cv != 0.0) {
            // vertical vorticity equation, steady in a frame moving with (ut, vt):
            // R = (u-ut) zx + (v-vt) zy + w zz + (zeta+f)(ux+vy) + wx vz - wy uz
            std::vector<double> zeta(n), A1(n), A2(n), A3(n), Bz(n), C1(n), C2(n), C3(n),
                C4(n), Gp(n), Gz(n);
            parallel_rows(nz * ny, n_threads, [&](int, int64_t row) {
                const int64_t iy = row % ny;
                for (int64_t ix = 0; ix < nx; ++ix) {
                    const int64_t i = row * nx + ix;
                    zeta[i] = d1(V, i, ix, ax) - d1(U, i, iy, ay);
                }
            });
            const double s2 = vort_scale * vort_scale;
            parallel_rows(nz * ny, n_threads, [&](int tid, int64_t row) {
                const int64_t iz = row / ny, iy = row % ny;
                double jv = 0;
                for (int64_t ix = 0; ix < nx; ++ix) {
                    const int64_t i = row * nx + ix;
                    const double zx = d1(zeta.data(), i, ix, ax), zy = d1(zeta.data(), i, iy, ay),
                                 zz = d1(zeta.data(), i, iz, az);
                    const double div = d1(U, i, ix, ax) + d1(V, i, iy, ay);
                    const double uz = d1(U, i, iz, az), vz = d1(V, i, iz, az);
                    const double wx = d1(W, i, ix, ax), wy = d1(W, i, iy, ay);
                    const double r = (U[i] - ut) * zx + (V[i] - vt) * zy + W[i] * zz +
                                     (zeta[i] + coriolis) * div + wx * vz - wy * uz;
                    const double wv = cv * VW[i];
                    jv += wv * s2 * r * r;
                    const double g = 2.0 * wv * s2 * r;
                    A1[i] = g * (U[i] - ut);
                    A2[i] = g * (V[i] - vt);
                    A3[i] = g * W[i];
                    Bz[i] = g * (zeta[i] + coriolis);
                    C1[i] = g * vz;
                    C2[i] = g * wx;
                    C3[i] = g * uz;
                    C4[i] = g * wy;
                    Gp[i] = g * div;
                    GU[i] += g * zx;
                    GV[i] += g * zy;
                    GW[i] += g * zz;
                }
                partial[static_cast<size_t>(tid) * 40 + 4] += jv;
            });
            parallel_rows(nz * ny, n_threads, [&](int, int64_t row) {
                const int64_t iz = row / ny, iy = row % ny;
                for (int64_t ix = 0; ix < nx; ++ix) {
                    const int64_t i = row * nx + ix;
                    Gz[i] = d1T(A1.data(), i, ix, ax) + d1T(A2.data(), i, iy, ay) +
                            d1T(A3.data(), i, iz, az) + Gp[i];
                }
            });
            parallel_rows(nz * ny, n_threads, [&](int, int64_t row) {
                const int64_t iz = row / ny, iy = row % ny;
                for (int64_t ix = 0; ix < nx; ++ix) {
                    const int64_t i = row * nx + ix;
                    GU[i] += -d1T(Gz.data(), i, iy, ay) + d1T(Bz.data(), i, ix, ax) -
                             d1T(C4.data(), i, iz, az);
                    GV[i] += d1T(Gz.data(), i, ix, ax) + d1T(Bz.data(), i, iy, ay) +
                             d1T(C2.data(), i, iz, az);
                    GW[i] += d1T(C1.data(), i, ix, ax) - d1T(C3.data(), i, iy, ay);
                }
            });
        }
        for (int t = 0; t < nt; ++t)
            for (int q = 0; q < 5; ++q) terms[q] += partial[static_cast<size_t>(t) * 40 + q];
    }
    py::array_t<double> out(5);
    std::copy(terms, terms + 5, out.mutable_data());
    return py::make_tuple(out, grad);
}

PYBIND11_MODULE(_multidoppler, m) {
    m.doc() = "Compiled multi-Doppler cost function and gradient for radarx.";
    m.def("cost_gradient", &cost_gradient, py::arg("state"), py::arg("coef"),
          py::arg("target"), py::arg("weight"), py::arg("rho"), py::arg("bg"),
          py::arg("bg_weight"), py::arg("vort_weight"), py::arg("dx"), py::arg("dy"),
          py::arg("dz"), py::arg("cm"), py::arg("csx"), py::arg("csy"), py::arg("csz"),
          py::arg("cv"), py::arg("ut"), py::arg("vt"), py::arg("coriolis"), py::arg("h"),
          py::arg("vort_scale"), py::arg("n_threads") = 0);
}
