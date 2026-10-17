// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Level I kernels on Doppler power spectra: the noise level of every
// spectrum (Hildebrand and Sekhon 1974), the moments of its main peak or of
// all bins above the noise, the notch filter for ground clutter with
// interpolation or a Gaussian fill, and the unfolding of the mean velocity
// along a ray.
//
// A spectrum is one row of K bins, the bin j at the velocity (j - K / 2) dv.
// Rows are independent and are taken by the threads from an atomic counter.
// The NumPy reference is radarx/level1/_moments_numpy.py; the formulas and
// the radarx choices are in the docstring of radarx/level1/spectra.py and
// radarx/level1/clutter.py.

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

inline int64_t wrap(int64_t i, int64_t K) {
    i %= K;
    return i < 0 ? i + K : i;
}

bool all_finite(const double* s, int64_t K) {
    for (int64_t k = 0; k < K; ++k)
        if (!std::isfinite(s[k])) return false;
    return true;
}

void check_spectra(const DArray& s) {
    if (s.ndim() != 2) throw std::invalid_argument("spectra must be 2-D (rows, bins)");
}

void check_rows(const DArray& a, int64_t n, const char* name) {
    if (a.ndim() != 1 || a.shape(0) != n)
        throw std::invalid_argument(std::string(name) + " must have one value per spectrum");
}

// Hildebrand and Sekhon (1974): the noise of a row is the mean of the
// smallest n values, n the largest count for which mean^2 >= n_avg variance.
double noise_of_row(const double* s, int64_t K, double n_avg, int64_t min_bins,
                    std::vector<double>& work) {
    if (!all_finite(s, K)) return kNaN;
    work.resize(3 * K);
    double* sorted = work.data();
    double* cs = sorted + K;
    double* cs2 = cs + K;
    std::copy(s, s + K, sorted);
    std::sort(sorted, sorted + K);
    double a = 0.0, b = 0.0;
    for (int64_t k = 0; k < K; ++k) {
        a += sorted[k];
        b += sorted[k] * sorted[k];
        cs[k] = a;
        cs2[k] = b;
    }
    for (int64_t n = K; n >= min_bins; --n) {
        const double mean = cs[n - 1] / static_cast<double>(n);
        const double var = cs2[n - 1] / static_cast<double>(n) - mean * mean;
        if (n == min_bins || mean * mean >= n_avg * var) return mean;
    }
    return kNaN;  // unreachable: min_bins >= 1
}

}  // namespace

py::array_t<double> noise_rows_py(const DArray& spectra, double n_avg, int min_bins,
                                  int n_threads) {
    check_spectra(spectra);
    const int64_t n = spectra.shape(0), K = spectra.shape(1);
    if (min_bins < 1 || min_bins > K) throw std::invalid_argument("min_bins must be in 1..bins");
    py::array_t<double> out(n);
    double* o = out.mutable_data();
    const double* sp = spectra.data();
    {
        py::gil_scoped_release release;
        parallel_blocks(n, n_threads, 32, [&](int64_t i0, int64_t i1) {
            std::vector<double> work;
            for (int64_t i = i0; i < i1; ++i)
                o[i] = noise_of_row(sp + i * K, K, n_avg, min_bins, work);
        });
    }
    return out;
}

// Moments of a spectrum. Columns: noise per bin, signal power (sum of S - N
// over the bins used), mean velocity (not wrapped to the Nyquist interval),
// width, skewness, kurtosis, number of bins. mode 0: the contiguous bins above
// the threshold around the strongest bin (circular); mode 1: every bin above
// the threshold, placed within K / 2 bins of the strongest.
py::array_t<double> moments_py(const DArray& spectra, const DArray& noise, double n_avg,
                               double n_sigma, int mode, double dv, double win_var,
                               int n_threads) {
    check_spectra(spectra);
    const int64_t n = spectra.shape(0), K = spectra.shape(1);
    check_rows(noise, n, "noise");
    if (mode != 0 && mode != 1) throw std::invalid_argument("mode must be 0 or 1");
    py::array_t<double> out(std::vector<py::ssize_t>{static_cast<py::ssize_t>(n), 7});
    double* o = out.mutable_data();
    const double* sp = spectra.data();
    const double* np_ = noise.data();
    const double fac = 1.0 + n_sigma / std::sqrt(n_avg);
    {
        py::gil_scoped_release release;
        parallel_blocks(n, n_threads, 64, [&](int64_t i0, int64_t i1) {
            std::vector<double> dj, wj;
            for (int64_t i = i0; i < i1; ++i) {
                const double* s = sp + i * K;
                double* row = o + i * 7;
                const double N = np_[i];
                for (int c = 0; c < 7; ++c) row[c] = kNaN;
                row[0] = N;
                if (!std::isfinite(N) || !all_finite(s, K)) continue;
                const double T = N * fac;
                int64_t k0 = 0;
                for (int64_t k = 1; k < K; ++k)
                    if (s[k] > s[k0]) k0 = k;
                row[6] = 0.0;
                if (!(s[k0] > T)) continue;
                dj.clear();
                wj.clear();
                if (mode == 0) {
                    int64_t nl = 0, nr = 0;
                    while (nl + nr + 1 < K && s[wrap(k0 - nl - 1, K)] > T) ++nl;
                    while (nl + nr + 1 < K && s[wrap(k0 + nr + 1, K)] > T) ++nr;
                    for (int64_t j = -nl; j <= nr; ++j) {
                        dj.push_back(static_cast<double>(j));
                        wj.push_back(s[wrap(k0 + j, K)] - N);
                    }
                } else {
                    for (int64_t k = 0; k < K; ++k) {
                        if (!(s[k] > T)) continue;
                        const int64_t d = wrap(k - k0 + K / 2, K) - K / 2;
                        dj.push_back(static_cast<double>(d));
                        wj.push_back(s[k] - N);
                    }
                }
                const size_t m = dj.size();
                double P = 0.0, a1 = 0.0;
                for (size_t j = 0; j < m; ++j) {
                    P += wj[j];
                    a1 += wj[j] * dj[j];
                }
                const double m1 = a1 / P;
                double m2 = 0.0, m3 = 0.0, m4 = 0.0;
                for (size_t j = 0; j < m; ++j) {
                    const double d = dj[j] - m1;
                    const double d2 = d * d;
                    m2 += wj[j] * d2;
                    m3 += wj[j] * d2 * d;
                    m4 += wj[j] * d2 * d2;
                }
                m2 /= P;
                m3 /= P;
                m4 /= P;
                row[1] = P;
                row[2] = (static_cast<double>(k0) - static_cast<double>(K / 2) + m1) * dv;
                row[3] = dv * std::sqrt(std::max(m2 - win_var, 0.0));
                if (m2 > 0.0) {
                    row[4] = m3 / (m2 * std::sqrt(m2));
                    row[5] = m4 / (m2 * m2);
                }
                row[6] = static_cast<double>(m);
            }
        });
    }
    return out;
}

namespace {

// Solve the 3 x 3 system a x = b by Cramer's rule; false if it is singular.
bool solve3(const double a[3][3], const double b[3], double x[3]) {
    auto det = [](const double m[3][3]) {
        return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
               m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
               m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    };
    const double d = det(a);
    double scale = 0.0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) scale = std::max(scale, std::fabs(a[i][j]));
    if (!(std::fabs(d) > 1e-12 * scale * scale * scale)) return false;
    for (int c = 0; c < 3; ++c) {
        double m[3][3];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) m[i][j] = (j == c) ? b[i] : a[i][j];
        x[c] = det(m) / d;
    }
    return true;
}

}  // namespace

// Notch filter of the bins within half_width of the zero velocity bin (K / 2).
// fill 0: the notch bins get the noise level; 1: linear interpolation between
// the bins next to the notch; 2: the Gaussian that fits the peak outside the
// notch (Guo 2011, weighted least squares of ln y on a parabola, weights y^2),
// or interpolation if the fit fails. Returns the spectra and, per row, the
// power that was removed, sum (S - fill) over the notch.
py::tuple notch_py(const DArray& spectra, const DArray& noise, double n_avg, double n_sigma,
                   int half_width, int fill, int n_threads) {
    check_spectra(spectra);
    const int64_t n = spectra.shape(0), K = spectra.shape(1);
    check_rows(noise, n, "noise");
    if (half_width < 0) throw std::invalid_argument("half_width must not be negative");
    if (fill < 0 || fill > 2) throw std::invalid_argument("fill must be 0, 1 or 2");
    py::array_t<double> out(std::vector<py::ssize_t>{static_cast<py::ssize_t>(n),
                                                      static_cast<py::ssize_t>(K)});
    py::array_t<double> removed(n);
    double* o = out.mutable_data();
    double* rm = removed.mutable_data();
    const double* sp = spectra.data();
    const double* np_ = noise.data();
    const double fac = 1.0 + n_sigma / std::sqrt(n_avg);
    const int64_t c = K / 2;
    const int64_t lo = std::max<int64_t>(0, c - half_width);
    const int64_t hi = std::min<int64_t>(K - 1, c + half_width);
    {
        py::gil_scoped_release release;
        parallel_blocks(n, n_threads, 64, [&](int64_t i0, int64_t i1) {
            for (int64_t i = i0; i < i1; ++i) {
                const double* s = sp + i * K;
                double* d = o + i * K;
                std::copy(s, s + K, d);
                rm[i] = kNaN;
                const double N = np_[i];
                if (!std::isfinite(N) || !all_finite(s, K)) continue;
                const double T = N * fac;
                bool filled = false;
                if (fill == 2) {
                    // the strongest bin outside the notch and its contiguous bins
                    int64_t k0 = -1;
                    for (int64_t k = 0; k < K; ++k)
                        if ((k < lo || k > hi) && (k0 < 0 || s[k] > s[k0])) k0 = k;
                    if (k0 >= 0 && s[k0] > T) {
                        auto outside = [&](int64_t k) { return k < lo || k > hi; };
                        int64_t nl = 0, nr = 0;
                        while (nl + nr + 1 < K && outside(wrap(k0 - nl - 1, K)) &&
                               s[wrap(k0 - nl - 1, K)] > T)
                            ++nl;
                        while (nl + nr + 1 < K && outside(wrap(k0 + nr + 1, K)) &&
                               s[wrap(k0 + nr + 1, K)] > T)
                            ++nr;
                        if (nl + nr + 1 >= 3) {
                            double A[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
                            double B[3] = {0, 0, 0};
                            for (int64_t j = -nl; j <= nr; ++j) {
                                const double y = s[wrap(k0 + j, K)] - N;
                                const double w = y * y;
                                const double ly = std::log(y);
                                const double x = static_cast<double>(j);
                                const double pw[5] = {1.0, x, x * x, x * x * x, x * x * x * x};
                                for (int p = 0; p < 3; ++p) {
                                    for (int q = 0; q < 3; ++q) A[p][q] += w * pw[p + q];
                                    B[p] += w * pw[p] * ly;
                                }
                            }
                            double coef[3];
                            if (solve3(A, B, coef) && coef[2] < 0.0) {
                                for (int64_t k = lo; k <= hi; ++k) {
                                    const double x =
                                        static_cast<double>(wrap(k - k0 + K / 2, K) - K / 2);
                                    d[k] = N + std::exp(coef[0] + coef[1] * x + coef[2] * x * x);
                                }
                                filled = true;
                            }
                        }
                    }
                }
                if (!filled) {
                    if (fill == 0) {
                        for (int64_t k = lo; k <= hi; ++k) d[k] = N;
                    } else {
                        const int64_t a = lo - 1, b = hi + 1;
                        for (int64_t k = lo; k <= hi; ++k) {
                            if (a < 0 && b >= K)
                                d[k] = N;
                            else if (a < 0)
                                d[k] = s[b];
                            else if (b >= K)
                                d[k] = s[a];
                            else
                                d[k] = s[a] + (s[b] - s[a]) * static_cast<double>(k - a) /
                                                  static_cast<double>(b - a);
                        }
                    }
                }
                double r = 0.0;
                for (int64_t k = lo; k <= hi; ++k) r += s[k] - d[k];
                rm[i] = r;
            }
        });
    }
    return py::make_tuple(out, removed);
}

// Unfold the mean velocities of the rays (rows) gate by gate. A gate with a
// finite reference is brought to the Nyquist interval around the reference;
// without one it is brought to the interval around the last unfolded gate of
// the ray, if that is at most max_gap gates away (otherwise it stays in the
// principal interval). max_gap is the number of missing gates that still link two gates. Returns the unfolded velocity and the number of
// intervals (2 va each) that were added.
py::tuple unfold_py(const DArray& vel, const DArray& ref, double va, int max_gap,
                    int n_threads) {
    if (vel.ndim() != 2) throw std::invalid_argument("vel must be 2-D (rays, gates)");
    const int64_t n = vel.shape(0), G = vel.shape(1);
    const bool has_ref = ref.size() > 0;
    if (has_ref && (ref.ndim() != 2 || ref.shape(0) != n || ref.shape(1) != G))
        throw std::invalid_argument("ref must have the shape of vel");
    if (!(va > 0.0)) throw std::invalid_argument("the Nyquist velocity must be positive");
    py::array_t<double> out(std::vector<py::ssize_t>{static_cast<py::ssize_t>(n),
                                                      static_cast<py::ssize_t>(G)});
    py::array_t<int32_t> fold(std::vector<py::ssize_t>{static_cast<py::ssize_t>(n),
                                                        static_cast<py::ssize_t>(G)});
    double* o = out.mutable_data();
    int32_t* f = fold.mutable_data();
    const double* vp = vel.data();
    const double* rp = has_ref ? ref.data() : nullptr;
    {
        py::gil_scoped_release release;
        parallel_blocks(n, n_threads, 16, [&](int64_t i0, int64_t i1) {
            for (int64_t i = i0; i < i1; ++i) {
                double last = kNaN;
                int64_t missing = 0;  // invalid gates since the last valid one
                for (int64_t g = 0; g < G; ++g) {
                    const int64_t idx = i * G + g;
                    const double v = vp[idx];
                    f[idx] = 0;
                    o[idx] = v;
                    if (!std::isfinite(v)) {
                        ++missing;
                        continue;
                    }
                    double target = kNaN;
                    if (rp != nullptr && std::isfinite(rp[idx]))
                        target = rp[idx];
                    else if (std::isfinite(last) && missing <= max_gap)
                        target = last;
                    if (std::isfinite(target)) {
                        const double nf = std::floor((target - v) / (2.0 * va) + 0.5);
                        f[idx] = static_cast<int32_t>(nf);
                        o[idx] = v + 2.0 * va * nf;
                    }
                    last = o[idx];
                    missing = 0;
                }
            }
        });
    }
    return py::make_tuple(out, fold);
}

PYBIND11_MODULE(_moments, m) {
    m.doc() = "Noise, moments, notch filter and velocity unfolding of Doppler spectra";
    m.def("noise_rows", &noise_rows_py, py::arg("spectra"), py::arg("n_avg"), py::arg("min_bins"),
          py::arg("n_threads") = 0);
    m.def("moments", &moments_py, py::arg("spectra"), py::arg("noise"), py::arg("n_avg"),
          py::arg("n_sigma"), py::arg("mode"), py::arg("dv"), py::arg("win_var"),
          py::arg("n_threads") = 0);
    m.def("notch", &notch_py, py::arg("spectra"), py::arg("noise"), py::arg("n_avg"),
          py::arg("n_sigma"), py::arg("half_width"), py::arg("fill"), py::arg("n_threads") = 0);
    m.def("unfold", &unfold_py, py::arg("vel"), py::arg("ref"), py::arg("va"),
          py::arg("max_gap"), py::arg("n_threads") = 0);
}
