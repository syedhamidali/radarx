// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// T-matrix scattering by homogeneous spheroids (raindrops), in C++.
//
// This is a translation of the double-precision FORTRAN code of M. I. Mishchenko
// (NASA GISS) for rotationally symmetric particles, in the version that
// J. Leinonen distributes with pytmatrix (ampld.lp.f, "CALCULATION OF THE AMPLITUDE
// AND PHASE MATRICES FOR A PARTICLE WITH AN AXIALLY SYMMETRIC SHAPE", MIT-compatible
// licence with the permission of M. I. Mishchenko). Only the spheroid (NP = -1) is
// translated. Method: the extended boundary condition (null-field) method of
// Waterman (1971) with the improvements of Mishchenko and Travis (1994) and
// Mishchenko et al. (1996); the amplitude matrix of a particle in a fixed orientation
// follows Mishchenko (2000). The names of the subroutines and the order of the
// operations of the FORTRAN code are kept so that the two can be compared line by
// line; arrays are indexed from 1, in column-major order, as in the original.
//
// Differences from the FORTRAN code:
//  - everything is re-entrant (no COMMON blocks), so that particles are computed in
//    parallel, one thread per particle;
//  - the T-matrix is stored in double precision (the original keeps it in REAL*4);
//  - failures to converge throw an exception instead of STOP;
//  - the orientation average, the phase matrix and the loop over particles are done
//    here, not in Python, because they dominate the run time.
//
// Python interface (radarx/fundamentals/tmatrix.py):
//   scatter_batch(radius, axis_ratio, rat, wavelength, m_re, m_im, ddelt, ndgs,
//                 geometry, orientation, n_threads)
//     radius, axis_ratio: (n,) equal-volume radius (for rat = 1) and ratio of the
//       horizontal to the rotational axis (> 1 oblate), in the unit of wavelength;
//     geometry: (g, 4) thet0, thet, phi0, phi in degrees;
//     orientation: (k, 3) Euler angles alpha, beta in degrees and a weight; the
//       amplitude and phase matrices are the weighted means over the k orientations.
//   Returns S (n, g, 2, 2) complex, Z (n, g, 4, 4) and nmax (n,).

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace py = pybind11;
using cd = std::complex<double>;
using DArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

namespace {

constexpr double kPi = 3.14159265358979323846264338327950288;
constexpr int NPN1 = 100;   // largest order of the multipole expansion
constexpr int NPNG1 = 500;  // largest number of Gauss points per hemisphere

// Column-major matrix indexed from 1, as the FORTRAN arrays.
struct Mat {
    int ld = 0;
    std::vector<double> a;
    Mat() = default;
    Mat(int rows, int cols) : ld(rows), a(static_cast<size_t>(rows) * cols, 0.0) {}
    double& operator()(int i, int j) { return a[static_cast<size_t>(j - 1) * ld + (i - 1)]; }
    double operator()(int i, int j) const { return a[static_cast<size_t>(j - 1) * ld + (i - 1)]; }
};

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("tmatrix: " + what); }

// ---------------------------------------------------------------------------
// Gauss-Legendre points and weights on (-1, 1) (GAUSS with IND1 = 0)
// z and w are indexed from 1 to n.
void gauss(int n, std::vector<double>& z, std::vector<double>& w) {
    const double a = 1.0, b = 2.0, c = 3.0;
    const int ind = n % 2;
    const int k = n / 2 + ind;
    const double f = static_cast<double>(n);
    double x = 0.0;
    for (int i = 1; i <= k; ++i) {
        const int m = n + 1 - i;
        if (i == 1) x = a - b / ((f + a) * f);
        if (i == 2) x = (z[n] - a) * 4.0 + z[n];
        if (i == 3) x = (z[n - 1] - z[n]) * 1.6 + z[n - 1];
        if (i > 3) x = (z[m + 1] - z[m + 2]) * c + z[m + 3];
        if (i == k && ind == 1) x = 0.0;
        int niter = 0;
        double check = 1e-16;
        double pa, pb, pc, dj;
        for (;;) {
            pb = 1.0;
            ++niter;
            if (niter > 100) check *= 10.0;
            pc = x;
            dj = a;
            for (int j = 2; j <= n; ++j) {
                dj += a;
                pa = pb;
                pb = pc;
                pc = x * pb + (x * pb - pa) * (dj - a) / dj;
            }
            pa = a / ((pb - x * pc) * f);
            pb = pa * pc * (a - x * x);
            x -= pb;
            if (std::fabs(pb) > check * std::fabs(x)) continue;
            break;
        }
        z[m] = x;
        w[m] = pa * pa * (a - x * x);
        w[m] = b * w[m];
        if (i == k && ind == 1) continue;
        z[i] = -z[m];
        w[i] = w[m];
    }
}

// ---------------------------------------------------------------------------
// Spherical Bessel functions (RJB, RYB, CJB). Orders 1..nmax, arrays from 1.
void rjb(double x, std::vector<double>& y, std::vector<double>& u, int nmax, int nnmax) {
    const int l = nmax + nnmax;
    std::vector<double> z(l + 2, 0.0);
    const double xx = 1.0 / x;
    z[l] = 1.0 / (static_cast<double>(2 * l + 1) * xx);
    for (int i = 1; i <= l - 1; ++i) {
        const int i1 = l - i;
        z[i1] = 1.0 / (static_cast<double>(2 * i1 + 1) * xx - z[i1 + 1]);
    }
    const double z0 = 1.0 / (xx - z[1]);
    const double y0 = z0 * std::cos(x) * xx;
    const double y1 = y0 * z[1];
    u[1] = y0 - y1 * xx;
    y[1] = y1;
    for (int i = 2; i <= nmax; ++i) {
        const double yi1 = y[i - 1];
        const double yi = yi1 * z[i];
        u[i] = yi1 - static_cast<double>(i) * yi * xx;
        y[i] = yi;
    }
}

void ryb(double x, std::vector<double>& y, std::vector<double>& v, int nmax) {
    const double c = std::cos(x), s = std::sin(x);
    const double x1 = 1.0 / x, x2 = x1 * x1, x3 = x2 * x1;
    const double y1 = -c * x2 - s * x1;
    y[1] = y1;
    y[2] = (-3.0 * x3 + x1) * c - 3.0 * x2 * s;
    for (int i = 2; i <= nmax - 1; ++i) y[i + 1] = static_cast<double>(2 * i + 1) * x1 * y[i] - y[i - 1];
    v[1] = -x1 * (c + y1);
    for (int i = 2; i <= nmax; ++i) v[i] = y[i - 1] - static_cast<double>(i) * x1 * y[i];
}

void cjb(double xr, double xi, std::vector<double>& yr, std::vector<double>& yi, std::vector<double>& ur,
         std::vector<double>& ui, int nmax, int nnmax) {
    const int l = nmax + nnmax;
    std::vector<double> czr(l + 2, 0.0), czi(l + 2, 0.0);
    const double xrxi = 1.0 / (xr * xr + xi * xi);
    const double cxxr = xr * xrxi, cxxi = -xi * xrxi;
    double qf = 1.0 / static_cast<double>(2 * l + 1);
    czr[l] = xr * qf;
    czi[l] = xi * qf;
    for (int i = 1; i <= l - 1; ++i) {
        const int i1 = l - i;
        qf = static_cast<double>(2 * i1 + 1);
        const double ar = qf * cxxr - czr[i1 + 1];
        const double ai = qf * cxxi - czi[i1 + 1];
        const double ari = 1.0 / (ar * ar + ai * ai);
        czr[i1] = ar * ari;
        czi[i1] = -ai * ari;
    }
    double ar = cxxr - czr[1], ai = cxxi - czi[1];
    double ari = 1.0 / (ar * ar + ai * ai);
    const double cz0r = ar * ari, cz0i = -ai * ari;
    const double cr = std::cos(xr) * std::cosh(xi);
    const double ci = -std::sin(xr) * std::sinh(xi);
    ar = cz0r * cr - cz0i * ci;
    ai = cz0i * cr + cz0r * ci;
    const double cy0r = ar * cxxr - ai * cxxi;
    const double cy0i = ai * cxxr + ar * cxxi;
    const double cy1r = cy0r * czr[1] - cy0i * czi[1];
    const double cy1i = cy0i * czr[1] + cy0r * czi[1];
    ar = cy1r * cxxr - cy1i * cxxi;
    ai = cy1i * cxxr + cy1r * cxxi;
    yr[1] = cy1r;
    yi[1] = cy1i;
    ur[1] = cy0r - ar;
    ui[1] = cy0i - ai;
    for (int i = 2; i <= nmax; ++i) {
        const double qi = static_cast<double>(i);
        const double cyi1r = yr[i - 1], cyi1i = yi[i - 1];
        const double cyir = cyi1r * czr[i] - cyi1i * czi[i];
        const double cyii = cyi1i * czr[i] + cyi1r * czi[i];
        ar = cyir * cxxr - cyii * cxxi;
        ai = cyii * cxxr + cyir * cxxi;
        yr[i] = cyir;
        yi[i] = cyii;
        ur[i] = cyi1r - qi * ar;
        ui[i] = cyi1i - qi * ai;
    }
}

// ---------------------------------------------------------------------------
// Wigner d-functions d_{0m}^n(arccos x): DV1 = d / sin, DV2 = dd/dtheta (VIG), and the
// variants of VIGAMPL (DV1 divided by sin(theta)) used for the amplitude matrix.
void vig(double x, int nmax, int m, std::vector<double>& dv1, std::vector<double>& dv2) {
    double a = 1.0;
    const double qs = std::sqrt(1.0 - x * x);
    const double qs1 = 1.0 / qs;
    for (int n = 1; n <= nmax; ++n) dv1[n] = dv2[n] = 0.0;
    if (m == 0) {
        double d1 = 1.0, d2 = x;
        for (int n = 1; n <= nmax; ++n) {
            const double qn = n, qn1 = n + 1, qn2 = 2 * n + 1;
            const double d3 = (qn2 * x * d2 - qn * d1) / qn1;
            const double der = qs1 * (qn1 * qn / qn2) * (-d1 + d3);
            dv1[n] = d2;
            dv2[n] = der;
            d1 = d2;
            d2 = d3;
        }
        return;
    }
    const double qmm = static_cast<double>(m) * m;
    for (int i = 1; i <= m; ++i) {
        const int i2 = i * 2;
        a = a * std::sqrt(static_cast<double>(i2 - 1) / static_cast<double>(i2)) * qs;
    }
    double d1 = 0.0, d2 = a;
    for (int n = m; n <= nmax; ++n) {
        const double qn = n, qn2 = 2 * n + 1, qn1 = n + 1;
        const double qnm = std::sqrt(qn * qn - qmm);
        const double qnm1 = std::sqrt(qn1 * qn1 - qmm);
        const double d3 = (qn2 * x * d2 - qnm * d1) / qnm1;
        const double der = qs1 * (-qn1 * qnm * d1 + qn * qnm1 * d3) / qn2;
        dv1[n] = d2;
        dv2[n] = der;
        d1 = d2;
        d2 = d3;
    }
}

void vigampl(double x, int nmax, int m, std::vector<double>& dv1, std::vector<double>& dv2) {
    for (int n = 1; n <= nmax; ++n) dv1[n] = dv2[n] = 0.0;
    const double dx = std::fabs(x);
    if (std::fabs(1.0 - dx) <= 1e-10) {
        if (m != 1) return;
        for (int n = 1; n <= nmax; ++n) {
            double dn = 0.5 * std::sqrt(static_cast<double>(n * (n + 1)));
            if (x < 0.0) dn = dn * ((n + 1) % 2 == 0 ? 1.0 : -1.0);
            dv1[n] = dn;
            if (x < 0.0) dn = -dn;
            dv2[n] = dn;
        }
        return;
    }
    double a = 1.0;
    const double qs = std::sqrt(1.0 - x * x);
    const double qs1 = 1.0 / qs;
    const double dsi = qs1;
    if (m == 0) {
        double d1 = 1.0, d2 = x;
        for (int n = 1; n <= nmax; ++n) {
            const double qn = n, qn1 = n + 1, qn2 = 2 * n + 1;
            const double d3 = (qn2 * x * d2 - qn * d1) / qn1;
            const double der = qs1 * (qn1 * qn / qn2) * (-d1 + d3);
            dv1[n] = d2 * dsi;
            dv2[n] = der;
            d1 = d2;
            d2 = d3;
        }
        return;
    }
    const double qmm = static_cast<double>(m) * m;
    for (int i = 1; i <= m; ++i) {
        const int i2 = i * 2;
        a = a * std::sqrt(static_cast<double>(i2 - 1) / static_cast<double>(i2)) * qs;
    }
    double d1 = 0.0, d2 = a;
    for (int n = m; n <= nmax; ++n) {
        const double qn = n, qn2 = 2 * n + 1, qn1 = n + 1;
        const double qnm = std::sqrt(qn * qn - qmm);
        const double qnm1 = std::sqrt(qn1 * qn1 - qmm);
        const double d3 = (qn2 * x * d2 - qnm * d1) / qnm1;
        const double der = qs1 * (-qn1 * qnm * d1 + qn * qnm1 * d3) / qn2;
        dv1[n] = d2 * dsi;
        dv2[n] = der;
        d1 = d2;
        d2 = d3;
    }
}

// ---------------------------------------------------------------------------
// Everything that depends on the quadrature (NGAUSS, NMAX) and the particle.
struct Grid {
    int ngauss = 0, nmax = 0, ng = 0;
    std::vector<double> x, w, s, ss, an, r, dr, ddr, drr, dri;
    Mat ann;  // nmax x nmax
    double ppi = 0, pir = 0, pii = 0;
    // Bessel functions at the Gauss points (ng x nmax)
    Mat j, y, jr, ji, dj, dy, djr, dji;
};

void const_(Grid& g) {  // CONST
    const int nmax = g.nmax, ngauss = g.ngauss, ng = 2 * ngauss;
    g.an.assign(nmax + 1, 0.0);
    g.ann = Mat(nmax, nmax);
    std::vector<double> dd(nmax + 1, 0.0);
    for (int n = 1; n <= nmax; ++n) {
        const int nn = n * (n + 1);
        g.an[n] = static_cast<double>(nn);
        const double d = std::sqrt(static_cast<double>(2 * n + 1) / static_cast<double>(nn));
        dd[n] = d;
        for (int n1 = 1; n1 <= n; ++n1) {
            const double ddd = d * dd[n1] * 0.5;
            g.ann(n, n1) = ddd;
            g.ann(n1, n) = ddd;
        }
    }
    // The Gauss-Legendre points depend only on their number: every iteration of the
    // convergence loops, and every particle of a batch with the same ndgs, asks for the
    // same few sets, so each thread keeps the ones it has computed.
    struct Nodes {
        std::vector<double> x, w;
    };
    static thread_local std::vector<Nodes> cache(2 * NPNG1 + 1);
    Nodes& nodes = cache[ng];
    if (nodes.x.empty()) {
        nodes.x.assign(ng + 4, 0.0);
        nodes.w.assign(ng + 4, 0.0);
        gauss(ng, nodes.x, nodes.w);
    }
    g.x = nodes.x;
    g.w = nodes.w;
    g.s.assign(ng + 2, 0.0);
    g.ss.assign(ng + 2, 0.0);
    for (int i = 1; i <= ngauss; ++i) {
        double y = g.x[i];
        y = 1.0 / (1.0 - y * y);
        g.ss[i] = y;
        g.ss[ng - i + 1] = y;
        y = std::sqrt(y);
        g.s[i] = y;
        g.s[ng - i + 1] = y;
    }
}

void bess(Grid& g, const std::vector<double>& z, const std::vector<double>& zr, const std::vector<double>& zi,
          int nnmax1, int nnmax2) {  // BESS
    const int ng = g.ng, nmax = g.nmax;
    g.j = Mat(ng, nmax);
    g.y = Mat(ng, nmax);
    g.jr = Mat(ng, nmax);
    g.ji = Mat(ng, nmax);
    g.dj = Mat(ng, nmax);
    g.dy = Mat(ng, nmax);
    g.djr = Mat(ng, nmax);
    g.dji = Mat(ng, nmax);
    std::vector<double> aj(nmax + 1), ay(nmax + 1), ajr(nmax + 1), aji(nmax + 1), adj(nmax + 1), ady(nmax + 1),
        adjr(nmax + 1), adji(nmax + 1);
    for (int i = 1; i <= ng; ++i) {
        rjb(z[i], aj, adj, nmax, nnmax1);
        ryb(z[i], ay, ady, nmax);
        cjb(zr[i], zi[i], ajr, aji, adjr, adji, nmax, nnmax2);
        for (int n = 1; n <= nmax; ++n) {
            g.j(i, n) = aj[n];
            g.y(i, n) = ay[n];
            g.jr(i, n) = ajr[n];
            g.ji(i, n) = aji[n];
            g.dj(i, n) = adj[n];
            g.dy(i, n) = ady[n];
            g.djr(i, n) = adjr[n];
            g.dji(i, n) = adji[n];
        }
    }
}

void vary(Grid& g, double lam, double mrr, double mri, double a, double eps) {  // VARY with RSP1 (spheroid)
    const int ngauss = g.ngauss, ng = 2 * ngauss, nmax = g.nmax;
    g.r.assign(ng + 2, 0.0);
    g.dr.assign(ng + 2, 0.0);
    {  // RSP1
        const double aa0 = a * std::cbrt(eps);
        const double aa = aa0 * aa0;
        const double ee = eps * eps;
        const double ee1 = ee - 1.0;
        for (int i = 1; i <= ngauss; ++i) {
            const double c = g.x[i];
            const double cc = c * c;
            const double ss = 1.0 - cc;
            const double s = std::sqrt(ss);
            const double rr = 1.0 / (ss + ee * cc);
            g.r[i] = aa * rr;
            g.r[ng - i + 1] = g.r[i];
            g.dr[i] = rr * c * s * ee1;
            g.dr[ng - i + 1] = -g.dr[i];
        }
    }
    const double pi = kPi * 2.0 / lam;
    g.ppi = pi * pi;
    g.pir = g.ppi * mrr;
    g.pii = g.ppi * mri;
    double v = 1.0 / (mrr * mrr + mri * mri);
    const double prr = mrr * v, pri = -mri * v;
    double ta = 0.0;
    g.ddr.assign(ng + 2, 0.0);
    g.drr.assign(ng + 2, 0.0);
    g.dri.assign(ng + 2, 0.0);
    std::vector<double> z(ng + 2), zr(ng + 2), zi(ng + 2);
    for (int i = 1; i <= ng; ++i) {
        const double vv0 = std::sqrt(g.r[i]);
        v = vv0 * pi;
        ta = std::max(ta, v);
        const double vv = 1.0 / v;
        g.ddr[i] = vv;
        g.drr[i] = prr * vv;
        g.dri[i] = pri * vv;
        z[i] = v;
        zr[i] = v * mrr;
        zi[i] = v * mri;
    }
    if (nmax > NPN1) fail("nmax exceeds the maximum order");
    double tb = ta * std::sqrt(mrr * mrr + mri * mri);
    tb = std::max(tb, static_cast<double>(nmax));
    const int nnmax1 = static_cast<int>(1.2 * std::sqrt(std::max(ta, static_cast<double>(nmax))) + 3.0);
    int nnmax2 = static_cast<int>(tb + 4.0 * std::pow(tb, 0.33333) + 1.2 * std::sqrt(tb));
    nnmax2 = nnmax2 - nmax + 5;
    bess(g, z, zr, zi, nnmax1, nnmax2);
}

// ---------------------------------------------------------------------------
// Matrix T = -RG(Q) Q^-1 by the inverse of Q (complex Gauss-Jordan elimination with partial
// pivoting, the role of ZGETRF/ZGETRI of the original); q, rgq, t are (2*nm x 2*nm) with
// separate real and imaginary parts, as QR, QI, RGQR, RGQI, TR1, TI1.
void tt(int nm, const Mat& qr, const Mat& qi, const Mat& rgqr, const Mat& rgqi, Mat& tr1, Mat& ti1) {
    const int n = 2 * nm;
    // LU factorization with partial pivoting (the algorithm of ZGETRF), then the inverse by
    // solving A X = I column by column
    std::vector<cd> a(static_cast<size_t>(n) * n);  // row-major, overwritten by L and U
    for (int i = 1; i <= n; ++i)
        for (int j = 1; j <= n; ++j) a[static_cast<size_t>(i - 1) * n + (j - 1)] = cd(qr(i, j), qi(i, j));
    std::vector<int> piv(n);
    for (int k = 0; k < n; ++k) {
        int p = k;
        double best = std::abs(a[static_cast<size_t>(k) * n + k]);
        for (int r = k + 1; r < n; ++r) {
            const double v = std::abs(a[static_cast<size_t>(r) * n + k]);
            if (v > best) {
                best = v;
                p = r;
            }
        }
        if (best == 0.0) fail("singular matrix in the inversion");
        piv[k] = p;
        if (p != k)
            for (int c = 0; c < n; ++c) std::swap(a[static_cast<size_t>(p) * n + c], a[static_cast<size_t>(k) * n + c]);
        const cd d = 1.0 / a[static_cast<size_t>(k) * n + k];
        for (int r = k + 1; r < n; ++r) {
            const cd f = a[static_cast<size_t>(r) * n + k] * d;
            a[static_cast<size_t>(r) * n + k] = f;
            if (f == cd(0.0, 0.0)) continue;
            for (int c = k + 1; c < n; ++c) a[static_cast<size_t>(r) * n + c] -= f * a[static_cast<size_t>(k) * n + c];
        }
    }
    std::vector<cd> inv(static_cast<size_t>(n) * n, cd(0.0, 0.0));  // row-major
    std::vector<cd> col(n);
    for (int j = 0; j < n; ++j) {
        std::fill(col.begin(), col.end(), cd(0.0, 0.0));
        col[j] = cd(1.0, 0.0);
        for (int k = 0; k < n; ++k)  // apply the row interchanges to the right-hand side
            if (piv[k] != k) std::swap(col[k], col[piv[k]]);
        for (int r = 1; r < n; ++r) {  // L y = b (unit lower triangle)
            cd sum = col[r];
            for (int c = 0; c < r; ++c) sum -= a[static_cast<size_t>(r) * n + c] * col[c];
            col[r] = sum;
        }
        for (int r = n - 1; r >= 0; --r) {  // U x = y
            cd sum = col[r];
            for (int c = r + 1; c < n; ++c) sum -= a[static_cast<size_t>(r) * n + c] * col[c];
            col[r] = sum / a[static_cast<size_t>(r) * n + r];
        }
        for (int r = 0; r < n; ++r) inv[static_cast<size_t>(r) * n + j] = col[r];
    }
    for (int i = 1; i <= n; ++i)
        for (int j = 1; j <= n; ++j) {
            double tr = 0.0, ti = 0.0;
            for (int k = 1; k <= n; ++k) {
                const double arr = rgqr(i, k), ari = rgqi(i, k);
                const cd zq = inv[static_cast<size_t>(k - 1) * n + (j - 1)];
                tr = tr - arr * zq.real() + ari * zq.imag();
                ti = ti - arr * zq.imag() - ari * zq.real();
            }
            tr1(i, j) = tr;
            ti1(i, j) = ti;
        }
}

// ---------------------------------------------------------------------------
// T-matrix blocks of the azimuthal order m = 0 (TMATR0) and m >= 1 (TMATR). The integrals of
// the Q and RG(Q) matrices over the surface of the particle are sums over the Gauss points;
// they are built from the products of the Bessel and Wigner functions of one point.

struct Blocks {  // the 16 matrices R11 ... IG22 of the FORTRAN code (real and imaginary, Q and RG(Q))
    Mat r11, r12, r21, r22, i11, i12, i21, i22, rg11, rg12, rg21, rg22, ig11, ig12, ig21, ig22;
    explicit Blocks(int n)
        : r11(n, n), r12(n, n), r21(n, n), r22(n, n), i11(n, n), i12(n, n), i21(n, n), i22(n, n), rg11(n, n),
          rg12(n, n), rg21(n, n), rg22(n, n), ig11(n, n), ig12(n, n), ig21(n, n), ig22(n, n) {}
};

struct Sums {  // the sums over the Gauss points of one element (n1, n2): a = Q, g = RG(Q); r, i = parts
    double ar11 = 0, ar12 = 0, ar21 = 0, ar22 = 0, ai11 = 0, ai12 = 0, ai21 = 0, ai22 = 0;
    double gr11 = 0, gr12 = 0, gr21 = 0, gr22 = 0, gi11 = 0, gi12 = 0, gi21 = 0, gi22 = 0;
};

struct Terms {  // B (Bessel + Neumann of the exterior field) and C (regular part), eight each
    double b1r, b1i, b2r, b2i, b3r, b3i, b4r, b4i, b5r, b5i, b6r, b6i, b7r, b7i, b8r, b8i;
    double c1r, c1i, c2r, c2i, c3r, c3i, c4r, c4i, c5r, c5i, c6r, c6i, c7r, c7i, c8r, c8i;
};

// Products of the spherical Bessel functions of the particle (orders n1 outside, n2 inside) at
// the Gauss point i, as the temporaries C1..C8 and B1..B8 of the FORTRAN code.
Terms bessel_terms(const Grid& g, int i, int n1, int n2) {
    Terms t;
    const double qj1 = g.j(i, n1), qy1 = g.y(i, n1);
    const double qjr2 = g.jr(i, n2), qji2 = g.ji(i, n2);
    const double qdjr2 = g.djr(i, n2), qdji2 = g.dji(i, n2);
    const double qdj1 = g.dj(i, n1), qdy1 = g.dy(i, n1);
    t.c1r = qjr2 * qj1;
    t.c1i = qji2 * qj1;
    t.b1r = t.c1r - qji2 * qy1;
    t.b1i = t.c1i + qjr2 * qy1;
    t.c2r = qjr2 * qdj1;
    t.c2i = qji2 * qdj1;
    t.b2r = t.c2r - qji2 * qdy1;
    t.b2i = t.c2i + qjr2 * qdy1;
    const double ddri = g.ddr[i];
    t.c3r = ddri * t.c1r;
    t.c3i = ddri * t.c1i;
    t.b3r = ddri * t.b1r;
    t.b3i = ddri * t.b1i;
    t.c4r = qdjr2 * qj1;
    t.c4i = qdji2 * qj1;
    t.b4r = t.c4r - qdji2 * qy1;
    t.b4i = t.c4i + qdjr2 * qy1;
    const double drri = g.drr[i], drii = g.dri[i];
    t.c5r = t.c1r * drri - t.c1i * drii;
    t.c5i = t.c1i * drri + t.c1r * drii;
    t.b5r = t.b1r * drri - t.b1i * drii;
    t.b5i = t.b1i * drri + t.b1r * drii;
    t.c6r = qdjr2 * qdj1;
    t.c6i = qdji2 * qdj1;
    t.b6r = t.c6r - qdji2 * qdy1;
    t.b6i = t.c6i + qdjr2 * qdy1;
    t.c7r = t.c4r * ddri;
    t.c7i = t.c4i * ddri;
    t.b7r = t.b4r * ddri;
    t.b7i = t.b4i * ddri;
    t.c8r = t.c2r * drri - t.c2i * drii;
    t.c8i = t.c2i * drri + t.c2r * drii;
    t.b8r = t.b2r * drri - t.b2i * drii;
    t.b8i = t.b2i * drri + t.b2r * drii;
    return t;
}

// Blocks 12 and 21 (label 150 of TMATR, the whole of TMATR0): f1 = RR A22 (or RR AA2), the two
// other weights are RR DR AN a12 and RR DR AN a21.
void add_off_diagonal(Sums& s, const Terms& t, double f1, double f2a, double f2b) {
    s.ar12 += f1 * t.b2r + f2a * t.b3r;
    s.ai12 += f1 * t.b2i + f2a * t.b3i;
    s.gr12 += f1 * t.c2r + f2a * t.c3r;
    s.gi12 += f1 * t.c2i + f2a * t.c3i;
    s.ar21 += f1 * t.b4r + f2b * t.b5r;
    s.ai21 += f1 * t.b4i + f2b * t.b5i;
    s.gr21 += f1 * t.c4r + f2b * t.c5r;
    s.gi21 += f1 * t.c4i + f2b * t.c5i;
}

void add_block_11(Sums& s, const Terms& t, double e1) {
    s.ar11 += e1 * t.b1r;
    s.ai11 += e1 * t.b1i;
    s.gr11 += e1 * t.c1r;
    s.gi11 += e1 * t.c1i;
}

void add_block_22(Sums& s, const Terms& t, double e1, double e2, double e3) {
    s.ar22 += e1 * t.b6r + e2 * t.b7r + e3 * t.b8r;
    s.ai22 += e1 * t.b6i + e2 * t.b7i + e3 * t.b8i;
    s.gr22 += e1 * t.c6r + e2 * t.c7r + e3 * t.c8r;
    s.gi22 += e1 * t.c6i + e2 * t.c7i + e3 * t.c8i;
}

void store(Blocks& b, int n1, int n2, const Sums& s, double an12) {
    b.r11(n1, n2) = s.ar11 * an12;
    b.r12(n1, n2) = s.ar12 * an12;
    b.r21(n1, n2) = s.ar21 * an12;
    b.r22(n1, n2) = s.ar22 * an12;
    b.i11(n1, n2) = s.ai11 * an12;
    b.i12(n1, n2) = s.ai12 * an12;
    b.i21(n1, n2) = s.ai21 * an12;
    b.i22(n1, n2) = s.ai22 * an12;
    b.rg11(n1, n2) = s.gr11 * an12;
    b.rg12(n1, n2) = s.gr12 * an12;
    b.rg21(n1, n2) = s.gr21 * an12;
    b.rg22(n1, n2) = s.gr22 * an12;
    b.ig11(n1, n2) = s.gi11 * an12;
    b.ig12(n1, n2) = s.gi12 * an12;
    b.ig21(n1, n2) = s.gi21 * an12;
    b.ig22(n1, n2) = s.gi22 * an12;
}

// Wigner functions of the Gauss points of order m: D1, D2 (ng x nmax), symmetric about the equator.
void wigner_table(const Grid& g, int m, const std::vector<double>& sig, Mat& d1, Mat& d2) {
    const int nmax = g.nmax, ngauss = g.ngauss;
    std::vector<double> dv1(nmax + 2), dv2(nmax + 2);
    for (int i = 1; i <= ngauss; ++i) {
        const int i1 = ngauss + i, i2 = ngauss - i + 1;
        vig(g.x[i1], nmax, m, dv1, dv2);
        for (int n = 1; n <= nmax; ++n) {
            const double s = sig[n];
            d1(i1, n) = dv1[n];
            d2(i1, n) = dv2[n];
            d1(i2, n) = dv1[n] * s;
            d2(i2, n) = -dv2[n] * s;
        }
    }
}

std::vector<double> parity_signs(int n) {
    std::vector<double> sig(n + 2);
    double si = 1.0;
    for (int k = 1; k <= n; ++k) {
        si = -si;
        sig[k] = si;
    }
    return sig;
}

// m = 0 (TMATR0): blocks 12 and 21 only.
void blocks_m0(const Grid& g, bool ncheck, Blocks& b) {
    const int nmax = g.nmax, ngauss = g.ngauss;
    const int ngss = ncheck ? ngauss : 2 * ngauss;
    const double factor = ncheck ? 2.0 : 1.0;
    const std::vector<double> sig = parity_signs(2 * nmax);
    Mat d1(2 * ngauss, nmax), d2(2 * ngauss, nmax);
    wigner_table(g, 0, sig, d1, d2);
    std::vector<double> rr(ngss + 2);
    for (int i = 1; i <= ngss; ++i) rr[i] = g.w[i] * g.r[i];
    for (int n1 = 1; n1 <= nmax; ++n1)
        for (int n2 = 1; n2 <= nmax; ++n2) {
            Sums s;
            if (!(ncheck && sig[n1 + n2] < 0.0)) {
                for (int i = 1; i <= ngss; ++i) {
                    const double a12 = d1(i, n1) * d2(i, n2), a21 = d2(i, n1) * d1(i, n2), a22 = d2(i, n1) * d2(i, n2);
                    const Terms t = bessel_terms(g, i, n1, n2);
                    const double f1 = rr[i] * a22;
                    add_off_diagonal(s, t, f1, rr[i] * g.dr[i] * g.an[n1] * a12, rr[i] * g.dr[i] * g.an[n2] * a21);
                }
            }
            store(b, n1, n2, s, g.ann(n1, n2) * factor);
        }
}

// m >= 1 (TMATR): all four blocks. For spheroids (ncheck) the Gauss points of one hemisphere and the
// parity of n1 + n2 decide which blocks are non-zero.
void blocks_m(const Grid& g, int m, bool ncheck, Blocks& b) {
    const int nmax = g.nmax, ngauss = g.ngauss;
    const double qm = static_cast<double>(m), qmm = qm * qm;
    const int ngss = ncheck ? ngauss : 2 * ngauss;
    const double factor = ncheck ? 2.0 : 1.0;
    const std::vector<double> sig = parity_signs(2 * nmax);
    Mat d1(2 * ngauss, nmax), d2(2 * ngauss, nmax);
    wigner_table(g, m, sig, d1, d2);
    std::vector<double> ds(ngss + 2), dss(ngss + 2), rr(ngss + 2);
    for (int i = 1; i <= ngss; ++i) {
        const double wr = g.w[i] * g.r[i];
        ds[i] = g.s[i] * qm * wr;
        dss[i] = g.ss[i] * qmm;
        rr[i] = wr;
    }
    for (int n1 = m; n1 <= nmax; ++n1)
        for (int n2 = m; n2 <= nmax; ++n2) {
            Sums s;
            const double an1 = g.an[n1], an2 = g.an[n2];
            const bool odd = ncheck && sig[n1 + n2] > 0.0;  // only the blocks 12 and 21 are non-zero
            for (int i = 1; i <= ngss; ++i) {
                const double a11 = d1(i, n1) * d1(i, n2), a12 = d1(i, n1) * d2(i, n2);
                const double a21 = d2(i, n1) * d1(i, n2), a22 = d2(i, n1) * d2(i, n2);
                const Terms t = bessel_terms(g, i, n1, n2);
                const double uri = g.dr[i], dsi = ds[i], rri = rr[i];
                const double e1 = dsi * (a12 + a21);
                const double e2 = dsi * uri * a11;
                if (!odd) add_block_11(s, t, e1);
                if (!ncheck || odd)
                    add_off_diagonal(s, t, rri * (a11 * dss[i] + a22), rri * uri * an1 * a12, rri * uri * an2 * a21);
                if (!odd) add_block_22(s, t, e1, e2 * an1, e2 * an2);
            }
            store(b, n1, n2, s, g.ann(n1, n2) * factor);
        }
}

// The matrices Q and RG(Q) of the null-field equations from the blocks (rows and columns of the
// blocks 11, 12, 21, 22 of order nm = nmax - m + 1), then T = -RG(Q) Q^-1.
void tmatr(const Grid& g, int m, bool ncheck, Mat& tr1, Mat& ti1) {
    const int nmax = g.nmax;
    const bool zero = (m == 0);
    const int mm1 = zero ? 1 : m;
    Blocks b(nmax);
    if (zero)
        blocks_m0(g, ncheck, b);
    else
        blocks_m(g, m, ncheck, b);
    const double tpir = g.pir, tpii = g.pii, tppi = g.ppi;
    const int nm = zero ? nmax : nmax - mm1 + 1;
    Mat qr(2 * nm, 2 * nm), qi(2 * nm, 2 * nm), rgqr(2 * nm, 2 * nm), rgqi(2 * nm, 2 * nm);
    for (int n1 = mm1; n1 <= nmax; ++n1) {
        const int k1 = n1 - mm1 + 1, kk1 = k1 + nm;
        for (int n2 = mm1; n2 <= nmax; ++n2) {
            const int k2 = n2 - mm1 + 1, kk2 = k2 + nm;
            const double tar11 = -b.r11(n1, n2), tai11 = -b.i11(n1, n2), tgr11 = -b.rg11(n1, n2), tgi11 = -b.ig11(n1, n2);
            const double tar12 = b.i12(n1, n2), tai12 = -b.r12(n1, n2), tgr12 = b.ig12(n1, n2), tgi12 = -b.rg12(n1, n2);
            const double tar21 = -b.i21(n1, n2), tai21 = b.r21(n1, n2), tgr21 = -b.ig21(n1, n2), tgi21 = b.rg21(n1, n2);
            const double tar22 = -b.r22(n1, n2), tai22 = -b.i22(n1, n2), tgr22 = -b.rg22(n1, n2), tgi22 = -b.ig22(n1, n2);
            qr(k1, k2) = tpir * tar21 - tpii * tai21 + tppi * tar12;
            qi(k1, k2) = tpir * tai21 + tpii * tar21 + tppi * tai12;
            rgqr(k1, k2) = tpir * tgr21 - tpii * tgi21 + tppi * tgr12;
            rgqi(k1, k2) = tpir * tgi21 + tpii * tgr21 + tppi * tgi12;
            qr(kk1, kk2) = tpir * tar12 - tpii * tai12 + tppi * tar21;
            qi(kk1, kk2) = tpir * tai12 + tpii * tar12 + tppi * tai21;
            rgqr(kk1, kk2) = tpir * tgr12 - tpii * tgi12 + tppi * tgr21;
            rgqi(kk1, kk2) = tpir * tgi12 + tpii * tgr12 + tppi * tgi21;
            if (zero) continue;  // the blocks 11 and 22 are zero for m = 0 (the matrices start as zeros)
            qr(k1, kk2) = tpir * tar11 - tpii * tai11 + tppi * tar22;
            qi(k1, kk2) = tpir * tai11 + tpii * tar11 + tppi * tai22;
            rgqr(k1, kk2) = tpir * tgr11 - tpii * tgi11 + tppi * tgr22;
            rgqi(k1, kk2) = tpir * tgi11 + tpii * tgr11 + tppi * tgi22;
            qr(kk1, k2) = tpir * tar22 - tpii * tai22 + tppi * tar11;
            qi(kk1, k2) = tpir * tai22 + tpii * tar22 + tppi * tai11;
            rgqr(kk1, k2) = tpir * tgr22 - tpii * tgi22 + tppi * tgr11;
            rgqi(kk1, k2) = tpir * tgi22 + tpii * tgr22 + tppi * tgi11;
        }
    }
    tr1 = Mat(2 * nm, 2 * nm);
    ti1 = Mat(2 * nm, 2 * nm);
    tt(nm, qr, qi, rgqr, rgqi, tr1, ti1);
}

// Surface-area ratio of an oblate or prolate spheroid (SAREA): equal-surface-area radius.
double sarea(double d) {
    if (d < 1.0) {
        const double e = std::sqrt(1.0 - d * d);
        double r = 0.5 * (std::pow(d, 2.0 / 3.0) + std::pow(d, -1.0 / 3.0) * std::asin(e) / e);
        r = std::sqrt(r);
        return 1.0 / r;
    }
    const double e = std::sqrt(1.0 - 1.0 / (d * d));
    double r = 0.25 * (2.0 * std::pow(d, 2.0 / 3.0) + std::pow(d, -4.0 / 3.0) * std::log((1.0 + e) / (1.0 - e)) / e);
    r = std::sqrt(r);
    return 1.0 / r;
}

// ---------------------------------------------------------------------------
// The T-matrix of one particle: blocks 11, 12, 21, 22 (real and imaginary parts) of every
// azimuthal order m = 0..nmax at the absolute multipole indices n, n' (m <= n, n' <= nmax).
struct Particle {
    int nmax = 0;
    double lam = 0;
    std::vector<double> t;  // [comp][m][n][n'] flattened, comp = 0..7 (r11,r12,r21,r22,i11,i12,i21,i22)
    std::vector<cd> cal;    // CAL(n, n') of AMPL, row-major (nmax+1)^2
    size_t idx(int comp, int m, int n, int nn) const {
        const size_t s = static_cast<size_t>(nmax) + 1;
        return ((static_cast<size_t>(comp) * s + m) * s + n) * s + nn;
    }
};

double quantities(const Mat& tr1, const Mat& ti1, int nmax, double* qext) {
    double qsca = 0.0, qe = 0.0;
    for (int n = 1; n <= nmax; ++n) {
        const int n1 = n + nmax;
        const double tr1nn = tr1(n, n), ti1nn = ti1(n, n), tr1nn1 = tr1(n1, n1), ti1nn1 = ti1(n1, n1);
        const double dn1 = static_cast<double>(2 * n + 1);
        qsca += dn1 * (tr1nn * tr1nn + ti1nn * ti1nn + tr1nn1 * tr1nn1 + ti1nn1 * ti1nn1);
        qe += (tr1nn + tr1nn1) * dn1;
    }
    *qext = qe;
    return qsca;
}

Particle calctmat(double axi, double rat, double lam, double mrr, double mri, double eps, double ddelt, int ndgs) {
    const bool ncheck = true;  // spheroids
    if (std::fabs(rat - 1.0) > 1e-8) rat = sarea(eps);
    const double a = rat * axi;
    const double xev = 2.0 * kPi * a / lam;
    const int ixxx = static_cast<int>(xev + 4.05 * std::pow(xev, 0.333333));
    const int inm1 = std::max(4, ixxx);
    if (inm1 >= NPN1) fail("the size parameter is too large (the order would exceed " + std::to_string(NPN1) + ")");
    double qext1 = 0.0, qsca1 = 0.0;
    int nmax = 0, ngauss = 0;
    Grid g;
    Mat tr1, ti1;
    bool converged = false;
    for (int nma = inm1; nma <= NPN1; ++nma) {
        nmax = nma;
        ngauss = nmax * ndgs;
        if (ngauss > NPNG1) fail("ngauss exceeds " + std::to_string(NPNG1));
        g.nmax = nmax;
        g.ngauss = ngauss;
        g.ng = 2 * ngauss;
        const_(g);
        vary(g, lam, mrr, mri, a, eps);
        tmatr(g, 0, ncheck, tr1, ti1);
        double qext;
        const double qsca = quantities(tr1, ti1, nmax, &qext);
        const double dsca = std::fabs((qsca1 - qsca) / qsca);
        const double dext = std::fabs((qext1 - qext) / qext);
        qext1 = qext;
        qsca1 = qsca;
        if (dsca <= ddelt && dext <= ddelt) {
            converged = true;
            break;
        }
        if (nma == NPN1) fail("no convergence over the order up to " + std::to_string(NPN1));
    }
    if (!converged) fail("no convergence over the order");
    if (ngauss != NPNG1) {
        for (int ngaus = ngauss + 1; ngaus <= NPNG1; ++ngaus) {
            ngauss = ngaus;
            g.ngauss = ngauss;
            g.ng = 2 * ngauss;
            const_(g);
            vary(g, lam, mrr, mri, a, eps);
            tmatr(g, 0, ncheck, tr1, ti1);
            double qext;
            const double qsca = quantities(tr1, ti1, nmax, &qext);
            const double dsca = std::fabs((qsca1 - qsca) / qsca);
            const double dext = std::fabs((qext1 - qext) / qext);
            qext1 = qext;
            qsca1 = qsca;
            if (dsca <= ddelt && dext <= ddelt) break;
        }
    }
    Particle p;
    p.nmax = nmax;
    p.lam = lam;
    p.t.assign(8 * (static_cast<size_t>(nmax) + 1) * (nmax + 1) * (nmax + 1), 0.0);
    for (int n2 = 1; n2 <= nmax; ++n2) {
        const int nn2 = n2 + nmax;
        for (int n1 = 1; n1 <= nmax; ++n1) {
            const int nn1 = n1 + nmax;
            p.t[p.idx(0, 0, n1, n2)] = tr1(n1, n2);
            p.t[p.idx(4, 0, n1, n2)] = ti1(n1, n2);
            p.t[p.idx(1, 0, n1, n2)] = tr1(n1, nn2);
            p.t[p.idx(5, 0, n1, n2)] = ti1(n1, nn2);
            p.t[p.idx(2, 0, n1, n2)] = tr1(nn1, n2);
            p.t[p.idx(6, 0, n1, n2)] = ti1(nn1, n2);
            p.t[p.idx(3, 0, n1, n2)] = tr1(nn1, nn2);
            p.t[p.idx(7, 0, n1, n2)] = ti1(nn1, nn2);
        }
    }
    for (int m = 1; m <= nmax; ++m) {
        tmatr(g, m, ncheck, tr1, ti1);
        const int nm = nmax - m + 1;
        for (int n2 = 1; n2 <= nm; ++n2) {
            const int nn2 = n2 + m - 1, n22 = n2 + nm;
            for (int n1 = 1; n1 <= nm; ++n1) {
                const int nn1 = n1 + m - 1, n11 = n1 + nm;
                p.t[p.idx(0, m, nn1, nn2)] = tr1(n1, n2);
                p.t[p.idx(4, m, nn1, nn2)] = ti1(n1, n2);
                p.t[p.idx(1, m, nn1, nn2)] = tr1(n1, n22);
                p.t[p.idx(5, m, nn1, nn2)] = ti1(n1, n22);
                p.t[p.idx(2, m, nn1, nn2)] = tr1(n11, n2);
                p.t[p.idx(6, m, nn1, nn2)] = ti1(n11, n2);
                p.t[p.idx(3, m, nn1, nn2)] = tr1(n11, n22);
                p.t[p.idx(7, m, nn1, nn2)] = ti1(n11, n22);
            }
        }
    }
    const size_t s = static_cast<size_t>(nmax) + 1;
    p.cal.assign(s * s, cd(0.0, 0.0));
    for (int nn = 1; nn <= nmax; ++nn)
        for (int n = 1; n <= nmax; ++n) {
            // CAL(N, NN) = i^(NN - N - 1) * sqrt((2N+1)(2NN+1) / (N NN (N+1)(NN+1)))
            const int e = ((nn - n - 1) % 4 + 4) % 4;
            const cd ci = (e == 0) ? cd(1, 0) : (e == 1) ? cd(0, 1) : (e == 2) ? cd(-1, 0) : cd(0, -1);
            double dnn = static_cast<double>((2 * n + 1) * (2 * nn + 1));
            dnn = dnn / static_cast<double>(static_cast<int64_t>(n) * nn * (n + 1) * (nn + 1));
            p.cal[static_cast<size_t>(n) * s + nn] = ci * std::sqrt(dnn);
        }
    return p;
}

// ---------------------------------------------------------------------------
// Amplitude matrix of a particle in a fixed orientation (AMPL). Angles in degrees.

struct Rotations {  // the rotation of the scattering plane (Eqs. 8-21 of Mishchenko 2000)
    double r[2][2], r1[2][2];  // R and R^-1
    double dcth0, dcth, ph;    // cosines of the incident and scattered polar angles in the particle frame, phi difference
};

void check_angles(double alpha, double beta, double tl, double tl1, double pl, double pl1) {
    if (alpha < 0.0 || alpha > 360.0 || beta < 0.0 || beta > 180.0 || tl < 0.0 || tl > 180.0 || tl1 < 0.0 ||
        tl1 > 180.0 || pl < 0.0 || pl > 360.0 || pl1 < 0.0 || pl1 > 360.0)
        fail("an angle is outside its allowed range");
}

// Direction in the particle frame (THETP, PHIP of Eqs. 8, 19, 20) of a direction in the laboratory frame.
void particle_frame(double thetl, double phil, double alph, double cb, double sb, double* ctp, double* thetp,
                    double* phip) {
    const double pin = kPi;
    const double ct = std::cos(thetl), st = std::sin(thetl);
    const double cp = std::cos(phil - alph), sp = std::sin(phil - alph);
    *ctp = ct * cb + st * sb * cp;
    *thetp = std::acos(*ctp);
    const double cpp = cb * st * cp - sb * ct;
    const double spp = st * sp;
    double phi = std::atan(spp / cpp);
    if (phi > 0.0 && sp < 0.0) phi += pin;
    if (phi < 0.0 && sp > 0.0) phi += pin;
    if (phi < 0.0) phi += 2.0 * pin;
    *phip = phi;
}

// AL (Eq. 14) or AP^-1 (Eq. 15) of a direction (theta, phi) are 3x2 and 2x3 matrices.
void direction_matrices(double theta, double phi, double al[3][2]) {
    const double ct = std::cos(theta), st = std::sin(theta), cp = std::cos(phi), sp = std::sin(phi);
    al[0][0] = ct * cp;
    al[0][1] = -sp;
    al[1][0] = ct * sp;
    al[1][1] = cp;
    al[2][0] = -st;
    al[2][1] = 0.0;
}

void polarization_rotation(const double b[3][3], const double al[3][2], double thetp, double phip, double r[2][2]) {
    const double ct = std::cos(thetp), st = std::sin(thetp), cp = std::cos(phip), sp = std::sin(phip);
    const double ap[2][3] = {{ct * cp, ct * sp, -st}, {-sp, cp, 0.0}};
    double c[3][2];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 2; ++j) {
            double x = 0.0;
            for (int k = 0; k < 3; ++k) x += b[i][k] * al[k][j];
            c[i][j] = x;
        }
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
            double x = 0.0;
            for (int k = 0; k < 3; ++k) x += ap[i][k] * c[k][j];
            r[i][j] = x;
        }
}

Rotations rotations(double tl, double tl1, double pl, double pl1, double alpha, double beta) {
    const double pin = kPi, pin2 = pin * 0.5, pi = pin / 180.0;
    const double alph = alpha * pi;
    double bet = beta * pi;
    double thetl = tl * pi, phil = pl * pi, thetl1 = tl1 * pi, phil1 = pl1 * pi;
    const double eps = 1e-7;  // directions exactly on the poles or the equator are moved off them
    if (thetl < pin2) thetl += eps;
    if (thetl > pin2) thetl -= eps;
    if (thetl1 < pin2) thetl1 += eps;
    if (thetl1 > pin2) thetl1 -= eps;
    if (phil < pin) phil += eps;
    if (phil > pin) phil -= eps;
    if (phil1 < pin) phil1 += eps;
    if (phil1 > pin) phil1 -= eps;
    if (bet <= pin2 && pin2 - bet <= eps) bet -= eps;
    if (bet > pin2 && bet - pin2 <= eps) bet += eps;
    const double cb = std::cos(bet), sb = std::sin(bet);
    double ctp, thetp, phip, ctp1, thetp1, phip1;
    particle_frame(thetl, phil, alph, cb, sb, &ctp, &thetp, &phip);
    particle_frame(thetl1, phil1, alph, cb, sb, &ctp1, &thetp1, &phip1);
    const double ca = std::cos(alph), sa = std::sin(alph);
    const double b[3][3] = {{ca * cb, sa * cb, -sb}, {-sa, ca, 0.0}, {ca * sb, sa * sb, cb}};
    double al[3][2], al1[3][2];
    direction_matrices(thetl, phil, al);
    direction_matrices(thetl1, phil1, al1);
    Rotations rot;
    polarization_rotation(b, al, thetp, phip, rot.r);
    polarization_rotation(b, al1, thetp1, phip1, rot.r1);
    const double d = 1.0 / (rot.r1[0][0] * rot.r1[1][1] - rot.r1[0][1] * rot.r1[1][0]);
    const double x0 = rot.r1[0][0];
    rot.r1[0][0] = rot.r1[1][1] * d;
    rot.r1[0][1] = -rot.r1[0][1] * d;
    rot.r1[1][0] = -rot.r1[1][0] * d;
    rot.r1[1][1] = x0 * d;
    rot.dcth0 = ctp;
    rot.dcth = ctp1;
    rot.ph = phip1 - phip;
    return rot;
}

void ampl(const Particle& p, double tl, double tl1, double pl, double pl1, double alpha, double beta, cd out[4]) {
    const int nmax = p.nmax;
    check_angles(alpha, beta, tl, tl1, pl, pl1);
    const Rotations rot = rotations(tl, tl1, pl, pl1, alpha, beta);
    const double dcth0 = rot.dcth0, dcth = rot.dcth, ph = rot.ph;
    const auto& r = rot.r;
    const auto& r1 = rot.r1;
    cd vv(0, 0), vh(0, 0), hv(0, 0), hh(0, 0);
    std::vector<double> dv1(nmax + 2), dv2(nmax + 2), dv01(nmax + 2), dv02(nmax + 2);
    for (int m = 0; m <= nmax; ++m) {
        const int nmin = std::max(m, 1);
        vigampl(dcth, nmax, m, dv1, dv2);
        vigampl(dcth0, nmax, m, dv01, dv02);
        const double fc = 2.0 * std::cos(m * ph), fs = 2.0 * std::sin(m * ph);
        for (int nn = nmin; nn <= nmax; ++nn) {
            const double dv1nn = m * dv01[nn];
            const double dv2nn = dv02[nn];
            for (int n = nmin; n <= nmax; ++n) {
                const double dv1n = m * dv1[n];
                const double dv2n = dv2[n];
                const cd ct11(p.t[p.idx(0, m, n, nn)], p.t[p.idx(4, m, n, nn)]);
                const cd ct22(p.t[p.idx(3, m, n, nn)], p.t[p.idx(7, m, n, nn)]);
                const cd cal = p.cal[static_cast<size_t>(n) * (nmax + 1) + nn];
                if (m == 0) {
                    const cd cn = cal * dv2n * dv2nn;
                    vv += cn * ct22;
                    hh += cn * ct11;
                } else {
                    const cd ct12(p.t[p.idx(1, m, n, nn)], p.t[p.idx(5, m, n, nn)]);
                    const cd ct21(p.t[p.idx(2, m, n, nn)], p.t[p.idx(6, m, n, nn)]);
                    const cd cn1 = cal * fc, cn2 = cal * fs;
                    const double d11 = dv1n * dv1nn, d12 = dv1n * dv2nn, d21 = dv2n * dv1nn, d22 = dv2n * dv2nn;
                    vv += (ct11 * d11 + ct21 * d21 + ct12 * d12 + ct22 * d22) * cn1;
                    vh += (ct11 * d12 + ct21 * d22 + ct12 * d11 + ct22 * d21) * cn2;
                    hv -= (ct11 * d21 + ct21 * d11 + ct12 * d22 + ct22 * d12) * cn2;
                    hh += (ct11 * d22 + ct21 * d12 + ct12 * d21 + ct22 * d11) * cn1;
                }
            }
        }
    }
    const double dk = 2.0 * kPi / p.lam;
    vv /= dk;
    vh /= dk;
    hv /= dk;
    hh /= dk;
    const cd cvv = vv * r[0][0] + vh * r[1][0];
    const cd cvh = vv * r[0][1] + vh * r[1][1];
    const cd chv = hv * r[0][0] + hh * r[1][0];
    const cd chh = hv * r[0][1] + hh * r[1][1];
    out[0] = r1[0][0] * cvv + r1[0][1] * chv;
    out[1] = r1[0][0] * cvh + r1[0][1] * chh;
    out[2] = r1[1][0] * cvv + r1[1][1] * chv;
    out[3] = r1[1][0] * cvh + r1[1][1] * chh;
}

// Phase matrix from the amplitude matrix (CALCAMPL, Eqs. 13-29 of Mishchenko 2000); s is
// S(1,1), S(1,2), S(2,1), S(2,2); z is 4x4 row-major.
void phase_matrix(const cd s[4], double z[16]) {
    const cd s11 = s[0], s12 = s[1], s21 = s[2], s22 = s[3];
    const cd I(0.0, 1.0);
    auto cj = [](cd v) { return std::conj(v); };
    z[0] = 0.5 * (std::norm(s11) + std::norm(s12) + std::norm(s21) + std::norm(s22));
    z[1] = 0.5 * (std::norm(s11) - std::norm(s12) + std::norm(s21) - std::norm(s22));
    z[2] = (-s11 * cj(s12) - s22 * cj(s21)).real();
    z[3] = (I * (s11 * cj(s12) - s22 * cj(s21))).real();
    z[4] = 0.5 * (std::norm(s11) + std::norm(s12) - std::norm(s21) - std::norm(s22));
    z[5] = 0.5 * (std::norm(s11) - std::norm(s12) - std::norm(s21) + std::norm(s22));
    z[6] = (-s11 * cj(s12) + s22 * cj(s21)).real();
    z[7] = (I * (s11 * cj(s12) + s22 * cj(s21))).real();
    z[8] = (-s11 * cj(s21) - s22 * cj(s12)).real();
    z[9] = (-s11 * cj(s21) + s22 * cj(s12)).real();
    z[10] = (s11 * cj(s22) + s12 * cj(s21)).real();
    z[11] = (-I * (s11 * cj(s22) + s21 * cj(s12))).real();
    z[12] = (I * (s21 * cj(s11) + s22 * cj(s12))).real();
    z[13] = (I * (s21 * cj(s11) - s22 * cj(s12))).real();
    z[14] = (-I * (s22 * cj(s11) - s12 * cj(s21))).real();
    z[15] = (s22 * cj(s11) - s12 * cj(s21)).real();
}

template <class F>
void parallel_for(int64_t total, int n_threads, F&& body) {
    if (total <= 0) return;
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    int nt = n_threads > 0 ? n_threads : static_cast<int>(hw);
    nt = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(nt, total)));
    std::atomic<int64_t> next{0};
    std::atomic<bool> failed{false};
    std::string message;
    std::atomic_flag lock = ATOMIC_FLAG_INIT;
    auto worker = [&]() {
        for (;;) {
            const int64_t i = next.fetch_add(1);
            if (i >= total || failed.load()) break;
            try {
                body(i);
            } catch (const std::exception& e) {
                while (lock.test_and_set()) {
                }
                if (!failed.exchange(true)) message = e.what();
                lock.clear();
            }
        }
    };
    std::vector<std::thread> pool;
    for (int t = 1; t < nt; ++t) pool.emplace_back(worker);
    worker();
    for (auto& th : pool) th.join();
    if (failed.load()) throw std::runtime_error(message);
}

}  // namespace

py::tuple scatter_batch(DArray radius, DArray axis_ratio, double rat, double wavelength, double m_re, double m_im,
                        double ddelt, int ndgs, DArray geometry, DArray orientation, int n_threads) {
    if (radius.ndim() != 1 || axis_ratio.ndim() != 1 || radius.shape(0) != axis_ratio.shape(0))
        throw std::invalid_argument("radius and axis_ratio must be 1-D arrays of the same length");
    if (geometry.ndim() != 2 || geometry.shape(1) != 4) throw std::invalid_argument("geometry must have shape (g, 4)");
    if (orientation.ndim() != 2 || orientation.shape(1) != 3)
        throw std::invalid_argument("orientation must have shape (k, 3)");
    const int64_t n = radius.shape(0), ng = geometry.shape(0), nk = orientation.shape(0);
    const double* rp = radius.data();
    const double* ep = axis_ratio.data();
    const double* gp = geometry.data();
    const double* op = orientation.data();
    double wsum = 0.0;
    for (int64_t k = 0; k < nk; ++k) wsum += op[3 * k + 2];
    if (!(wsum > 0.0)) throw std::invalid_argument("the orientation weights must sum to a positive number");

    using ssz = py::ssize_t;
    py::array_t<cd> s_out(std::vector<ssz>{static_cast<ssz>(n), static_cast<ssz>(ng), 2, 2});
    py::array_t<double> z_out(std::vector<ssz>{static_cast<ssz>(n), static_cast<ssz>(ng), 4, 4});
    py::array_t<int> nmax_out(static_cast<ssz>(n));
    cd* sp = s_out.mutable_data();
    double* zp = z_out.mutable_data();
    int* np_ = nmax_out.mutable_data();
    {
        py::gil_scoped_release release;
        parallel_for(n, n_threads, [&](int64_t i) {
            const Particle p = calctmat(rp[i], rat, wavelength, m_re, m_im, ep[i], ddelt, ndgs);
            np_[i] = p.nmax;
            for (int64_t gidx = 0; gidx < ng; ++gidx) {
                const double* ge = gp + 4 * gidx;
                cd s_acc[4] = {cd(0, 0), cd(0, 0), cd(0, 0), cd(0, 0)};
                double z_acc[16] = {0};
                for (int64_t k = 0; k < nk; ++k) {
                    cd s[4];
                    ampl(p, ge[0], ge[1], ge[2], ge[3], op[3 * k], op[3 * k + 1], s);
                    double z[16];
                    phase_matrix(s, z);
                    const double w = op[3 * k + 2] / wsum;
                    for (int q = 0; q < 4; ++q) s_acc[q] += w * s[q];
                    for (int q = 0; q < 16; ++q) z_acc[q] += w * z[q];
                }
                for (int q = 0; q < 4; ++q) sp[(i * ng + gidx) * 4 + q] = s_acc[q];
                for (int q = 0; q < 16; ++q) zp[(i * ng + gidx) * 16 + q] = z_acc[q];
            }
        });
    }
    return py::make_tuple(s_out, z_out, nmax_out);
}


py::tuple tmatrix(double radius, double axis_ratio, double rat, double wavelength, double m_re, double m_im, double ddelt,
                  int ndgs) {
    Particle p;
    {
        py::gil_scoped_release release;
        p = calctmat(radius, rat, wavelength, m_re, m_im, axis_ratio, ddelt, ndgs);
    }
    const py::ssize_t s = p.nmax + 1;
    py::array_t<double> t(std::vector<py::ssize_t>{8, s, s, s});
    std::copy(p.t.begin(), p.t.end(), t.mutable_data());
    return py::make_tuple(p.nmax, t);
}

PYBIND11_MODULE(_tmatrix, m) {
    m.doc() = "T-matrix scattering by spheroids (translation of the code of Mishchenko)";
    m.def("tmatrix", &tmatrix, py::arg("radius"), py::arg("axis_ratio"), py::arg("rat"), py::arg("wavelength"),
          py::arg("m_re"), py::arg("m_im"), py::arg("ddelt"), py::arg("ndgs"),
          "T-matrix blocks (r11, r12, r21, r22, i11, i12, i21, i22) indexed [component, m, n, n'] (for tests)");
    m.def("scatter_batch", &scatter_batch, py::arg("radius"), py::arg("axis_ratio"), py::arg("rat"), py::arg("wavelength"),
          py::arg("m_re"), py::arg("m_im"), py::arg("ddelt"), py::arg("ndgs"), py::arg("geometry"),
          py::arg("orientation"), py::arg("n_threads") = 0);
}
