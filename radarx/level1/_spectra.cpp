// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Level I kernels on the IQ time series of a gate: Doppler power spectra,
// the lag sums of the pulse pair estimators and the polynomial regression
// ground clutter filter.
//
// Every series (one gate of one ray) is independent, so the kernels split the
// series into blocks that the threads take from an atomic counter. The FFT is
// a radix-2 transform for lengths that are powers of two and the chirp-z
// transform of Bluestein (1970) for the others. The NumPy reference of the
// same functions is radarx/level1/_spectra_numpy.py; the formulas and the
// radarx choices are in the docstrings of radarx/level1/spectra.py and
// radarx/level1/clutter.py.

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
using cplx = std::complex<double>;
using CArray = py::array_t<cplx, py::array::c_style | py::array::forcecast>;
using DArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

namespace {

constexpr double kPi = 3.14159265358979323846;

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
            const int64_t i0 = next.fetch_add(block);
            if (i0 >= total) break;
            body(i0, std::min(total, i0 + block));
        }
    };
    std::vector<std::thread> pool;
    for (int i = 1; i < nt; ++i) pool.emplace_back(worker);
    worker();
    for (auto& th : pool) th.join();
}

// ---------------------------------------------------------------- FFT plan

struct FFTPlan {
    int n = 0;          // transform length
    int m = 0;          // length of the radix-2 transform (m == n if n is a power of two)
    bool pow2 = true;
    std::vector<int> rev;
    std::vector<cplx> tw;     // exp(-2 pi i k / m), k < m / 2
    std::vector<cplx> chirp;  // exp(-i pi k^2 / n), k < n (Bluestein)
    std::vector<cplx> bfft;   // FFT of the conjugate chirp filter (Bluestein)

    explicit FFTPlan(int n_) : n(n_) {
        pow2 = (n & (n - 1)) == 0;
        if (pow2) {
            m = n;
        } else {
            m = 1;
            while (m < 2 * n - 1) m <<= 1;
        }
        rev.assign(m, 0);
        int bits = 0;
        while ((1 << bits) < m) ++bits;
        for (int i = 0; i < m; ++i) {
            int r = 0;
            for (int b = 0; b < bits; ++b)
                if (i & (1 << b)) r |= 1 << (bits - 1 - b);
            rev[i] = r;
        }
        tw.resize(std::max(1, m / 2));
        for (int k = 0; k < m / 2; ++k) {
            const double a = -2.0 * kPi * k / m;
            tw[k] = cplx(std::cos(a), std::sin(a));
        }
        if (!pow2) {
            chirp.resize(n);
            for (int k = 0; k < n; ++k) {
                const int64_t k2 = (static_cast<int64_t>(k) * k) % (2 * static_cast<int64_t>(n));
                const double a = -kPi * static_cast<double>(k2) / n;
                chirp[k] = cplx(std::cos(a), std::sin(a));
            }
            bfft.assign(m, cplx(0.0, 0.0));
            bfft[0] = std::conj(chirp[0]);
            for (int k = 1; k < n; ++k) {
                bfft[k] = std::conj(chirp[k]);
                bfft[m - k] = std::conj(chirp[k]);
            }
            radix2(bfft.data());
        }
    }

    // in-place forward radix-2 transform of length m
    void radix2(cplx* a) const {
        for (int i = 0; i < m; ++i)
            if (i < rev[i]) std::swap(a[i], a[rev[i]]);
        for (int len = 2; len <= m; len <<= 1) {
            const int half = len >> 1;
            const int step = m / len;
            for (int i = 0; i < m; i += len) {
                for (int j = 0; j < half; ++j) {
                    const cplx w = tw[static_cast<size_t>(j) * step];
                    const cplx u = a[i + j];
                    const cplx v = a[i + j + half] * w;
                    a[i + j] = u + v;
                    a[i + j + half] = u - v;
                }
            }
        }
    }

    // Forward DFT of buf[0..n) in place; buf must have m elements (work space).
    void forward(cplx* buf) const {
        if (pow2) {
            radix2(buf);
            return;
        }
        for (int k = 0; k < n; ++k) buf[k] *= chirp[k];
        for (int k = n; k < m; ++k) buf[k] = cplx(0.0, 0.0);
        radix2(buf);
        for (int k = 0; k < m; ++k) buf[k] *= bfft[k];
        // inverse transform of the product: conj, forward, conj, scale
        for (int k = 0; k < m; ++k) buf[k] = std::conj(buf[k]);
        radix2(buf);
        const double s = 1.0 / m;
        for (int k = 0; k < n; ++k) buf[k] = std::conj(buf[k]) * s * chirp[k];
    }
};

void check_series(const CArray& x) {
    if (x.ndim() != 2) throw std::invalid_argument("x must be 2-D (series, pulses)");
}

}  // namespace

// Power spectra of the series, averaged over ``n_seg`` consecutive segments of
// L = M / n_seg pulses (the pulses after n_seg * L are not used). Window w of
// length L; P_k = |sum_n w_n x_n exp(-2 pi i k n / L)|^2 / (L sum w^2), so that
// the bins add up to the mean power. Bin j of the result is the velocity
// (j - L/2) dv: the FFT bin (L/2 - j) mod L (velocity = -wavelength f / 2).
py::array_t<double> power_spectra_py(const CArray& x, const DArray& window, int n_seg,
                                     int n_threads) {
    check_series(x);
    const int64_t nser = x.shape(0);
    const int64_t M = x.shape(1);
    if (n_seg < 1) throw std::invalid_argument("n_seg must be at least 1");
    const int L = static_cast<int>(M / n_seg);
    if (L < 2) throw std::invalid_argument("a segment needs at least 2 pulses");
    if (window.ndim() != 1 || window.shape(0) != L)
        throw std::invalid_argument("window must have one value per pulse of a segment");
    const double* w = window.data();
    double sw2 = 0.0;
    for (int i = 0; i < L; ++i) sw2 += w[i] * w[i];
    if (!(sw2 > 0.0)) throw std::invalid_argument("the window is zero");
    const double norm = 1.0 / (static_cast<double>(L) * sw2 * n_seg);
    FFTPlan plan(L);
    std::vector<int> src(L);
    for (int j = 0; j < L; ++j) src[j] = (((L / 2 - j) % L) + L) % L;

    py::array_t<double> out(std::vector<py::ssize_t>{static_cast<py::ssize_t>(nser), static_cast<py::ssize_t>(L)});
    double* o = out.mutable_data();
    const cplx* xp = x.data();
    {
        py::gil_scoped_release release;
        parallel_blocks(nser, n_threads, 8, [&](int64_t i0, int64_t i1) {
            std::vector<cplx> buf(plan.m);
            std::vector<double> acc(L);
            for (int64_t i = i0; i < i1; ++i) {
                std::fill(acc.begin(), acc.end(), 0.0);
                for (int s = 0; s < n_seg; ++s) {
                    const cplx* seg = xp + i * M + static_cast<int64_t>(s) * L;
                    for (int k = 0; k < L; ++k) buf[k] = seg[k] * w[k];
                    plan.forward(buf.data());
                    for (int k = 0; k < L; ++k) acc[k] += std::norm(buf[k]);
                }
                double* row = o + i * L;
                for (int j = 0; j < L; ++j) row[j] = acc[src[j]] * norm;
            }
        });
    }
    return out;
}

// Lag sums of the pulse pair estimators: the mean power (lag 0) and the
// autocorrelation estimates at lags 1 and 2, each averaged over the M - lag
// products conj(x[m]) x[m + lag]. Columns: power, Re R1, Im R1, Re R2, Im R2.
py::array_t<double> lag_sums_py(const CArray& x, int n_threads) {
    check_series(x);
    const int64_t nser = x.shape(0);
    const int64_t M = x.shape(1);
    if (M < 3) throw std::invalid_argument("at least 3 pulses are needed");
    py::array_t<double> out(std::vector<py::ssize_t>{static_cast<py::ssize_t>(nser), 5});
    double* o = out.mutable_data();
    const cplx* xp = x.data();
    {
        py::gil_scoped_release release;
        parallel_blocks(nser, n_threads, 64, [&](int64_t i0, int64_t i1) {
            for (int64_t i = i0; i < i1; ++i) {
                const cplx* s = xp + i * M;
                double p0 = 0.0;
                cplx r1(0.0, 0.0), r2(0.0, 0.0);
                for (int64_t m = 0; m < M; ++m) {
                    p0 += std::norm(s[m]);
                    if (m + 1 < M) r1 += std::conj(s[m]) * s[m + 1];
                    if (m + 2 < M) r2 += std::conj(s[m]) * s[m + 2];
                }
                r1 /= static_cast<double>(M - 1);
                r2 /= static_cast<double>(M - 2);
                double* row = o + i * 5;
                row[0] = p0 / static_cast<double>(M);
                row[1] = r1.real();
                row[2] = r1.imag();
                row[3] = r2.real();
                row[4] = r2.imag();
            }
        });
    }
    return out;
}

// Polynomial regression filter: y = x - Q (Q^T x) with the M x K matrix Q of
// orthonormal polynomials. Returns the filtered series and the mean power that
// was removed, ||Q^T x||^2 / M.
py::tuple regression_filter_py(const CArray& x, const DArray& basis, int n_threads) {
    check_series(x);
    const int64_t nser = x.shape(0);
    const int64_t M = x.shape(1);
    if (basis.ndim() != 2 || basis.shape(0) != M)
        throw std::invalid_argument("basis must have one row per pulse");
    const int64_t K = basis.shape(1);
    CArray y(std::vector<py::ssize_t>{static_cast<py::ssize_t>(nser), static_cast<py::ssize_t>(M)});
    py::array_t<double> removed(nser);
    cplx* yp = y.mutable_data();
    double* rp = removed.mutable_data();
    const cplx* xp = x.data();
    const double* q = basis.data();
    {
        py::gil_scoped_release release;
        parallel_blocks(nser, n_threads, 16, [&](int64_t i0, int64_t i1) {
            std::vector<cplx> c(K);
            for (int64_t i = i0; i < i1; ++i) {
                const cplx* s = xp + i * M;
                cplx* d = yp + i * M;
                for (int64_t k = 0; k < K; ++k) c[k] = cplx(0.0, 0.0);
                for (int64_t m = 0; m < M; ++m)
                    for (int64_t k = 0; k < K; ++k) c[k] += q[m * K + k] * s[m];
                double pr = 0.0;
                for (int64_t k = 0; k < K; ++k) pr += std::norm(c[k]);
                for (int64_t m = 0; m < M; ++m) {
                    cplx fit(0.0, 0.0);
                    for (int64_t k = 0; k < K; ++k) fit += q[m * K + k] * c[k];
                    d[m] = s[m] - fit;
                }
                rp[i] = pr / static_cast<double>(M);
            }
        });
    }
    return py::make_tuple(y, removed);
}

PYBIND11_MODULE(_spectra, m) {
    m.doc() = "Doppler spectra, pulse pair lag sums and regression clutter filter of IQ series";
    m.def("power_spectra", &power_spectra_py, py::arg("x"), py::arg("window"), py::arg("n_seg"),
          py::arg("n_threads") = 0);
    m.def("lag_sums", &lag_sums_py, py::arg("x"), py::arg("n_threads") = 0);
    m.def("regression_filter", &regression_filter_py, py::arg("x"), py::arg("basis"),
          py::arg("n_threads") = 0);
}
