// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Ring fits of the ground-based velocity track display (GBVTD) and the simplex search
// for the circulation centre of a tropical cyclone from single-Doppler velocities.
//
// The data are scattered gates: position (x, y) in m east and north of the radar, the
// horizontal projection v of the Doppler velocity (velocity / cos(elevation)) and a
// weight w (the gate area, which is proportional to the ground range of the gate).
// Around a trial centre (cx, cy) and for a ring of radius R every gate within the
// half width hw of the ring takes part with the triangular weight w (1 - |d - R| / hw),
// d being its distance from the centre, so that the fit is a continuous function of the
// centre. The angle psi of a gate is the angle from the beam to the outward radial
// direction of the ring at the gate (atan2(y - cy, x - cx) - atan2(y, x), in [0, 2 pi)),
// and the weighted least squares fit is
//
//     v = sum_{n=0..L} (A_n cos(n psi) + B_n sin(n psi)),   L <= 3,
//
// with L reduced when the largest angular gap of the data is wide (Chen et al. 2013:
// gaps up to 30, 60, 120 degrees allow L = 3, 2, 1). The module radarx/retrieve/tc_winds.py
// has the equations and the references; every step follows its NumPy implementation in the
// same order of operations.
//
// The data are binned in a uniform grid (counting sort) once, so a ring only visits the
// cells that touch it. Independent simplex searches and rings run on a pool of threads,
// the work being handed out in small blocks through an atomic counter.

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
constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;
constexpr double kDeg = kPi / 180.0;
constexpr double kBad = 1.0e30;  // cost of a centre that cannot be evaluated
constexpr int kMaxOrder = 3;
constexpr int kCoef = 2 * kMaxOrder + 1;

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

struct FitSettings {
    double half_width;  // half width of the ring (m)
    int max_order;      // largest Fourier order requested (0..3)
    int min_points;     // fewest gates in a ring
    double max_gap;     // largest angular gap of the data (rad)
};

// Result of one ring: the coefficients (A0, A1, B1, A2, B2, A3, B3), and diagnostics.
struct RingFit {
    double coef[kCoef];
    double n_points = 0.0;
    double gap_deg = kNaN;
    double order = 0.0;
    double rms = kNaN;
};

class Gates {
public:
    Gates(const double* x, const double* y, const double* v, const double* w, int64_t n,
          double cell)
        : x_(x), y_(y), v_(v), w_(w), cell_(cell) {
        // the beam direction of every gate, psi = phi_center - azimuth needs atan2(y, x)
        azim_.resize(n);
        for (int64_t i = 0; i < n; ++i) azim_[i] = std::atan2(y[i], x[i]);
        xmin_ = ymin_ = std::numeric_limits<double>::infinity();
        double xmax = -xmin_, ymax = -ymin_;
        for (int64_t i = 0; i < n; ++i) {
            xmin_ = std::min(xmin_, x[i]);
            xmax = std::max(xmax, x[i]);
            ymin_ = std::min(ymin_, y[i]);
            ymax = std::max(ymax, y[i]);
        }
        if (n == 0) {
            nx_ = ny_ = 1;
            xmin_ = ymin_ = 0.0;
        } else {
            nx_ = static_cast<int64_t>(std::floor((xmax - xmin_) / cell_)) + 1;
            ny_ = static_cast<int64_t>(std::floor((ymax - ymin_) / cell_)) + 1;
        }
        start_.assign(nx_ * ny_ + 1, 0);
        for (int64_t i = 0; i < n; ++i) start_[cell_of(x[i], y[i]) + 1]++;
        for (int64_t c = 0; c < nx_ * ny_; ++c) start_[c + 1] += start_[c];
        order_.resize(n);
        std::vector<int64_t> fill(start_.begin(), start_.end() - 1);
        for (int64_t i = 0; i < n; ++i) order_[fill[cell_of(x[i], y[i])]++] = i;
    }

    // Fit the ring (cx, cy, R). Returns false if it cannot be fitted.
    bool fit(double cx, double cy, double R, const FitSettings& s, RingFit& out) const {
        const double hw = s.half_width;
        std::vector<double> psi, val, wt;
        const int64_t ix0 = clamp_i(static_cast<int64_t>(std::floor((cx - R - hw - xmin_) / cell_)), nx_);
        const int64_t ix1 = clamp_i(static_cast<int64_t>(std::floor((cx + R + hw - xmin_) / cell_)), nx_);
        const int64_t iy0 = clamp_i(static_cast<int64_t>(std::floor((cy - R - hw - ymin_) / cell_)), ny_);
        const int64_t iy1 = clamp_i(static_cast<int64_t>(std::floor((cy + R + hw - ymin_) / cell_)), ny_);
        for (int64_t iy = iy0; iy <= iy1; ++iy) {
            const double y0 = ymin_ + iy * cell_, y1 = y0 + cell_;
            const double dy_min = (cy < y0) ? y0 - cy : (cy > y1 ? cy - y1 : 0.0);
            const double dy_max = std::max(std::fabs(cy - y0), std::fabs(cy - y1));
            for (int64_t ix = ix0; ix <= ix1; ++ix) {
                const double x0 = xmin_ + ix * cell_, x1 = x0 + cell_;
                const double dx_min = (cx < x0) ? x0 - cx : (cx > x1 ? cx - x1 : 0.0);
                const double dx_max = std::max(std::fabs(cx - x0), std::fabs(cx - x1));
                if (std::hypot(dx_min, dy_min) > R + hw || std::hypot(dx_max, dy_max) < R - hw)
                    continue;
                const int64_t c = iy * nx_ + ix;
                for (int64_t k = start_[c]; k < start_[c + 1]; ++k) {
                    const int64_t i = order_[k];
                    const double d = std::hypot(x_[i] - cx, y_[i] - cy);
                    const double t = 1.0 - std::fabs(d - R) / hw;
                    if (!(t > 0.0) || !std::isfinite(v_[i])) continue;
                    double p = std::atan2(y_[i] - cy, x_[i] - cx) - azim_[i];
                    p -= kTwoPi * std::floor(p / kTwoPi);
                    psi.push_back(p);
                    val.push_back(v_[i]);
                    wt.push_back(w_[i] * t);
                }
            }
        }
        const int64_t m = static_cast<int64_t>(psi.size());
        if (m < s.min_points || m < 2) return false;
        // largest angular gap, including the wrap-around
        std::vector<double> sorted(psi);
        std::sort(sorted.begin(), sorted.end());
        double gap = sorted.front() + kTwoPi - sorted.back();
        for (int64_t i = 1; i < m; ++i) gap = std::max(gap, sorted[i] - sorted[i - 1]);
        int order = 0;
        if (gap <= 30.0 * kDeg) order = 3;
        else if (gap <= 60.0 * kDeg) order = 2;
        else if (gap <= 120.0 * kDeg) order = 1;
        order = std::min(order, s.max_order);
        if (gap > s.max_gap || order < 1) return false;
        const int nc = 2 * order + 1;  // fewer gates than columns would leave a gap > 120 degrees
        double ata[kCoef][kCoef] = {}, atb[kCoef] = {};
        double basis[kCoef];
        for (int64_t i = 0; i < m; ++i) {
            basis[0] = 1.0;
            for (int n = 1; n <= order; ++n) {
                basis[2 * n - 1] = std::cos(n * psi[i]);
                basis[2 * n] = std::sin(n * psi[i]);
            }
            for (int a = 0; a < nc; ++a) {
                const double wa = wt[i] * basis[a];
                atb[a] += wa * val[i];
                for (int b = 0; b <= a; ++b) ata[a][b] += wa * basis[b];
            }
        }
        for (int a = 0; a < nc; ++a)
            for (int b = a + 1; b < nc; ++b) ata[a][b] = ata[b][a];
        double sol[kCoef];
        if (!solve(ata, atb, sol, nc)) return false;
        for (int a = 0; a < kCoef; ++a) out.coef[a] = a < nc ? sol[a] : 0.0;
        double sse = 0.0, sw = 0.0;
        for (int64_t i = 0; i < m; ++i) {
            double fit_v = sol[0];
            for (int n = 1; n <= order; ++n)
                fit_v += sol[2 * n - 1] * std::cos(n * psi[i]) + sol[2 * n] * std::sin(n * psi[i]);
            const double r = val[i] - fit_v;
            sse += wt[i] * r * r;
            sw += wt[i];
        }
        out.n_points = static_cast<double>(m);
        out.gap_deg = gap / kDeg;
        out.order = order;
        out.rms = std::sqrt(sse / sw);
        return true;
    }

private:
    int64_t clamp_i(int64_t i, int64_t n) const { return std::max<int64_t>(0, std::min<int64_t>(i, n - 1)); }
    int64_t cell_of(double x, double y) const {
        const int64_t ix = clamp_i(static_cast<int64_t>(std::floor((x - xmin_) / cell_)), nx_);
        const int64_t iy = clamp_i(static_cast<int64_t>(std::floor((y - ymin_) / cell_)), ny_);
        return iy * nx_ + ix;
    }

    // Gaussian elimination with partial pivoting of the n by n system a x = b.
    static bool solve(double a[kCoef][kCoef], const double b[kCoef], double x[kCoef], int n) {
        double m[kCoef][kCoef + 1];
        double scale = 0.0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) m[i][j] = a[i][j];
            m[i][n] = b[i];
            scale = std::max(scale, std::fabs(a[i][i]));
        }
        for (int c = 0; c < n; ++c) {
            int piv = c;
            for (int r = c + 1; r < n; ++r)
                if (std::fabs(m[r][c]) > std::fabs(m[piv][c])) piv = r;
            if (!(std::fabs(m[piv][c]) > 1.0e-12 * scale)) return false;
            if (piv != c)
                for (int j = 0; j <= n; ++j) std::swap(m[piv][j], m[c][j]);
            for (int r = c + 1; r < n; ++r) {
                const double f = m[r][c] / m[c][c];
                for (int j = c; j <= n; ++j) m[r][j] -= f * m[c][j];
            }
        }
        for (int r = n - 1; r >= 0; --r) {
            double acc = m[r][n];
            for (int j = r + 1; j < n; ++j) acc -= m[r][j] * x[j];
            x[r] = acc / m[r][r];
        }
        return true;
    }

    const double *x_, *y_, *v_, *w_;
    double cell_;
    double xmin_, ymin_;
    int64_t nx_, ny_;
    std::vector<double> azim_;
    std::vector<int64_t> start_, order_;
};

// Axisymmetric tangential wind of a ring, VT0 = -B1 - B3 - VM sin(theta_T - theta_M) sin(alpha_max),
// with VM sin(theta_T - theta_M) = (cy um - cx vm) / RT and sin(alpha_max) = R / RT.
// The ring needs the radar outside it (R < RT).
inline double vt0_of(const RingFit& f, double cx, double cy, double R, double um, double vm) {
    const double rt = std::hypot(cx, cy);
    return -f.coef[2] - f.coef[6] - (cy * um - cx * vm) / rt * (R / rt);
}

struct Search {
    const Gates* gates;
    FitSettings fit;
    double um, vm;
    double anchor_x, anchor_y, limit;  // the centre stays within `limit` of the anchor

    // Cost of a centre for the ring R: minus the axisymmetric tangential wind.
    double cost(double cx, double cy, double R, int64_t& n_eval) const {
        ++n_eval;
        if (std::hypot(cx - anchor_x, cy - anchor_y) > limit) return kBad;
        const double rt = std::hypot(cx, cy);
        if (!(rt > R * 1.0001)) return kBad;
        RingFit f;
        if (!gates->fit(cx, cy, R, fit, f)) return kBad;
        return -vt0_of(f, cx, cy, R, um, vm);
    }
};

}  // namespace

// x, y, v, w: (n,) gate arrays. cx, cy: (nc,) centres, radii: (nr,). Returns the
// coefficients (nc, nr, 7) and the diagnostics (nc, nr, 4): number of gates, largest
// angular gap (deg), order of the fit and the weighted rms residual; NaN for a ring
// that cannot be fitted.
std::tuple<py::array_t<double>, py::array_t<double>> gbvtd_rings_py(
    const DArray& x, const DArray& y, const DArray& v, const DArray& w, const DArray& cx,
    const DArray& cy, const DArray& radii, double half_width, int max_order, int min_points,
    double max_gap_deg, int n_threads) {
    if (x.ndim() != 1 || y.ndim() != 1 || v.ndim() != 1 || w.ndim() != 1 ||
        x.shape(0) != y.shape(0) || x.shape(0) != v.shape(0) || x.shape(0) != w.shape(0))
        throw std::invalid_argument("x, y, v and w must be 1-D arrays of the same length");
    if (cx.ndim() != 1 || cy.ndim() != 1 || cx.shape(0) != cy.shape(0) || radii.ndim() != 1)
        throw std::invalid_argument("cx and cy must be 1-D of the same length, radii 1-D");
    if (!(half_width > 0.0)) throw std::invalid_argument("half_width must be positive");
    if (max_order < 0 || max_order > kMaxOrder)
        throw std::invalid_argument("max_order must be between 0 and 3");
    const int64_t nc = cx.shape(0), nr = radii.shape(0);
    py::array_t<double> coef({static_cast<py::ssize_t>(nc), static_cast<py::ssize_t>(nr),
                              static_cast<py::ssize_t>(kCoef)});
    py::array_t<double> diag({static_cast<py::ssize_t>(nc), static_cast<py::ssize_t>(nr),
                              static_cast<py::ssize_t>(4)});
    double* co = coef.mutable_data();
    double* di = diag.mutable_data();
    {
        py::gil_scoped_release release;
        const Gates gates(x.data(), y.data(), v.data(), w.data(), x.shape(0), 2.0 * half_width);
        const FitSettings fs{half_width, max_order, min_points, max_gap_deg * kDeg};
        const double* cxp = cx.data();
        const double* cyp = cy.data();
        const double* rp = radii.data();
        parallel_blocks(nc * nr, n_threads, 4, [&](int64_t t0, int64_t t1) {
            for (int64_t t = t0; t < t1; ++t) {
                const int64_t ic = t / nr, ir = t % nr;
                RingFit f;
                const bool ok = gates.fit(cxp[ic], cyp[ic], rp[ir], fs, f);
                for (int a = 0; a < kCoef; ++a) co[t * kCoef + a] = ok ? f.coef[a] : kNaN;
                di[t * 4 + 0] = ok ? f.n_points : kNaN;
                di[t * 4 + 1] = ok ? f.gap_deg : kNaN;
                di[t * 4 + 2] = ok ? f.order : kNaN;
                di[t * 4 + 3] = ok ? f.rms : kNaN;
            }
        });
    }
    return {coef, diag};
}

// Nelder-Mead search (reflection 1, expansion 2, contraction 0.5, shrink 0.5) of the
// centre that maximises the axisymmetric tangential wind of the ring of every radius,
// from every start. starts: (ng, 2). Returns (ng, nr, 5): the centre x and y, the
// tangential wind, the number of evaluations and 1 if the simplex converged.
py::array_t<double> center_search_py(const DArray& x, const DArray& y, const DArray& v,
                                     const DArray& w, const DArray& starts, const DArray& radii,
                                     double half_width, int max_order, int min_points,
                                     double max_gap_deg, double um, double vm, double step,
                                     double tol, int max_iter, double limit, double anchor_x,
                                     double anchor_y, int n_threads) {
    if (x.ndim() != 1 || y.ndim() != 1 || v.ndim() != 1 || w.ndim() != 1 ||
        x.shape(0) != y.shape(0) || x.shape(0) != v.shape(0) || x.shape(0) != w.shape(0))
        throw std::invalid_argument("x, y, v and w must be 1-D arrays of the same length");
    if (starts.ndim() != 2 || starts.shape(1) != 2 || radii.ndim() != 1)
        throw std::invalid_argument("starts must have the shape (n, 2), radii must be 1-D");
    if (!(half_width > 0.0) || !(step > 0.0) || !(tol > 0.0) || max_iter < 1)
        throw std::invalid_argument("half_width, step and tol must be positive, max_iter at least 1");
    if (max_order < 0 || max_order > kMaxOrder)
        throw std::invalid_argument("max_order must be between 0 and 3");
    const int64_t ng = starts.shape(0), nr = radii.shape(0);
    py::array_t<double> out({static_cast<py::ssize_t>(ng), static_cast<py::ssize_t>(nr),
                             static_cast<py::ssize_t>(5)});
    double* op = out.mutable_data();
    {
        py::gil_scoped_release release;
        const Gates gates(x.data(), y.data(), v.data(), w.data(), x.shape(0), 2.0 * half_width);
        const Search search{&gates, FitSettings{half_width, max_order, min_points, max_gap_deg * kDeg},
                            um, vm, anchor_x, anchor_y, limit};
        const double* sp = starts.data();
        const double* rp = radii.data();
        parallel_blocks(ng * nr, n_threads, 1, [&](int64_t t0, int64_t t1) {
            for (int64_t t = t0; t < t1; ++t) {
                const int64_t ig = t / nr, ir = t % nr;
                const double R = rp[ir];
                int64_t n_eval = 0;
                double vx[3] = {sp[ig * 2], sp[ig * 2] + step, sp[ig * 2]};
                double vy[3] = {sp[ig * 2 + 1], sp[ig * 2 + 1], sp[ig * 2 + 1] + step};
                double fv[3];
                for (int k = 0; k < 3; ++k) fv[k] = search.cost(vx[k], vy[k], R, n_eval);
                bool converged = false;
                for (int it = 0; it < max_iter; ++it) {
                    // stable ordering of the three vertices by cost
                    int o[3] = {0, 1, 2};
                    for (int a = 1; a < 3; ++a)
                        for (int b = a; b > 0 && fv[o[b]] < fv[o[b - 1]]; --b) std::swap(o[b], o[b - 1]);
                    const int ib = o[0], im = o[1], iw = o[2];
                    const double size = std::max(std::hypot(vx[im] - vx[ib], vy[im] - vy[ib]),
                                                 std::hypot(vx[iw] - vx[ib], vy[iw] - vy[ib]));
                    if (fv[ib] >= kBad) break;
                    if (size < tol) {
                        converged = true;
                        break;
                    }
                    const double mx = 0.5 * (vx[ib] + vx[im]), my = 0.5 * (vy[ib] + vy[im]);
                    const double rx = mx + (mx - vx[iw]), ry = my + (my - vy[iw]);
                    const double fr = search.cost(rx, ry, R, n_eval);
                    bool shrink = false;
                    if (fr >= fv[ib] && fr < fv[im]) {
                        vx[iw] = rx; vy[iw] = ry; fv[iw] = fr;
                    } else if (fr < fv[ib]) {
                        const double ex = mx + 2.0 * (mx - vx[iw]), ey = my + 2.0 * (my - vy[iw]);
                        const double fe = search.cost(ex, ey, R, n_eval);
                        if (fe < fr) { vx[iw] = ex; vy[iw] = ey; fv[iw] = fe; }
                        else { vx[iw] = rx; vy[iw] = ry; fv[iw] = fr; }
                    } else if (fr < fv[iw]) {
                        const double cxo = mx + 0.5 * (rx - mx), cyo = my + 0.5 * (ry - my);
                        const double fc = search.cost(cxo, cyo, R, n_eval);
                        if (fc <= fr) { vx[iw] = cxo; vy[iw] = cyo; fv[iw] = fc; }
                        else shrink = true;
                    } else {
                        const double cxi = mx + 0.5 * (vx[iw] - mx), cyi = my + 0.5 * (vy[iw] - my);
                        const double fc = search.cost(cxi, cyi, R, n_eval);
                        if (fc < fv[iw]) { vx[iw] = cxi; vy[iw] = cyi; fv[iw] = fc; }
                        else shrink = true;
                    }
                    if (shrink) {
                        for (int k : {im, iw}) {
                            vx[k] = vx[ib] + 0.5 * (vx[k] - vx[ib]);
                            vy[k] = vy[ib] + 0.5 * (vy[k] - vy[ib]);
                            fv[k] = search.cost(vx[k], vy[k], R, n_eval);
                        }
                    }
                }
                int best = 0;
                for (int k = 1; k < 3; ++k)
                    if (fv[k] < fv[best]) best = k;
                const bool ok = fv[best] < kBad;
                double* row = op + t * 5;
                row[0] = ok ? vx[best] : kNaN;
                row[1] = ok ? vy[best] : kNaN;
                row[2] = ok ? -fv[best] : kNaN;
                row[3] = static_cast<double>(n_eval);
                row[4] = (ok && converged) ? 1.0 : 0.0;
            }
        });
    }
    return out;
}

PYBIND11_MODULE(_tc_winds, m) {
    m.doc() = "GBVTD ring fits and the simplex centre search of tropical cyclones";
    m.def("gbvtd_rings", &gbvtd_rings_py, py::arg("x"), py::arg("y"), py::arg("v"), py::arg("w"),
          py::arg("cx"), py::arg("cy"), py::arg("radii"), py::arg("half_width"),
          py::arg("max_order") = 3, py::arg("min_points") = 12, py::arg("max_gap") = 120.0,
          py::arg("n_threads") = 0);
    m.def("center_search", &center_search_py, py::arg("x"), py::arg("y"), py::arg("v"),
          py::arg("w"), py::arg("starts"), py::arg("radii"), py::arg("half_width"),
          py::arg("max_order") = 2, py::arg("min_points") = 12, py::arg("max_gap") = 120.0,
          py::arg("um") = 0.0, py::arg("vm") = 0.0, py::arg("step") = 2000.0,
          py::arg("tol") = 50.0, py::arg("max_iter") = 200, py::arg("limit") = 1.0e30,
          py::arg("anchor_x") = 0.0, py::arg("anchor_y") = 0.0, py::arg("n_threads") = 0);
}
