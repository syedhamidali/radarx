// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Noise level, signal-to-noise ratio and noise correction of RHOHV and ZDR.
//
// Two groups of kernels, both threaded over blocks taken from an atomic counter:
//
//  * element-wise kernels on flat arrays (SNR, RHOHV and ZDR corrected for
//    noise), one block of gates per task;
//  * a row kernel that estimates the noise level of every row (a ray, or a whole
//    sweep) from the range-corrected reflectivity of its gates, either with the
//    objective method of Hildebrand and Sekhon (1974) or as a low percentile of
//    the range-corrected reflectivity.
//
// The signal model, the formulas and the radarx choices are in the docstring of
// radarx/retrieve/noise.py. Every step follows the NumPy reference of that
// module, in the same order of operations.

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
constexpr int64_t kBlock = 1 << 16;  // gates per scheduling block

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

inline double from_db(double x) { return std::pow(10.0, 0.1 * x); }
inline double to_db(double x) { return 10.0 * std::log10(x); }

void check_size(const DArray& a, int64_t n, const char* name) {
    if (a.ndim() != 1 || a.shape(0) != n)
        throw std::invalid_argument(std::string(name) + " must be 1-D with the size of the data");
}

}  // namespace

// SNR (dB) = dBZ - noise at 1 km - 20 log10(range / km). If the reflectivity still contains
// the noise power (no noise subtraction) the noise is taken off in linear units first.
py::array_t<double> snr_db_py(const DArray& dbz, const DArray& noise, const DArray& range_m,
                              bool includes_noise, int n_threads) {
    const int64_t n = dbz.shape(0);
    check_size(noise, n, "noise");
    check_size(range_m, n, "range_m");
    py::array_t<double> out(n);
    double* o = out.mutable_data();
    const double* z = dbz.data();
    const double* nz = noise.data();
    const double* rm = range_m.data();
    {
        py::gil_scoped_release release;
        parallel_blocks(n, n_threads, kBlock, [&](int64_t g0, int64_t g1) {
            for (int64_t i = g0; i < g1; ++i) {
                double s = kNaN;
                if (rm[i] > 0.0) {
                    s = z[i] - nz[i] - 20.0 * std::log10(rm[i] / 1000.0);
                    if (includes_noise) {
                        const double lin = from_db(s) - 1.0;
                        s = lin > 0.0 ? to_db(lin) : kNaN;
                    }
                }
                o[i] = s;
            }
        });
    }
    return out;
}

// ZDR (dB) corrected for noise. zdr is the measured ZDR, snr the signal-to-noise ratio of the
// horizontal channel (dB), alpha_db the horizontal minus the vertical noise power (dB).
// Measured ZDR = ZDR (1 + 1/s) / (1 + ZDR / (alpha s)) with s the linear SNR, hence
// ZDR = zdr / (1 + 1/s - zdr / (alpha s)); no solution if the denominator is not positive.
static inline double zdr_denoised(double zdr_db, double snr_db, double alpha_db) {
    const double s = from_db(snr_db);
    const double alpha = from_db(alpha_db);
    const double zm = from_db(zdr_db);
    const double denom = 1.0 + 1.0 / s - zm / (alpha * s);
    if (!(denom > 0.0)) return kNaN;
    return to_db(zm / denom);
}

py::array_t<double> correct_zdr_py(const DArray& zdr, const DArray& snr, double alpha_db,
                                   double min_snr, int n_threads) {
    const int64_t n = zdr.shape(0);
    check_size(snr, n, "snr");
    py::array_t<double> out(n);
    double* o = out.mutable_data();
    const double* zp = zdr.data();
    const double* sp = snr.data();
    {
        py::gil_scoped_release release;
        parallel_blocks(n, n_threads, kBlock, [&](int64_t g0, int64_t g1) {
            for (int64_t i = g0; i < g1; ++i) {
                const bool ok = std::isfinite(zp[i]) && std::isfinite(sp[i]) && sp[i] >= min_snr;
                o[i] = ok ? zdr_denoised(zp[i], sp[i], alpha_db) : kNaN;
            }
        });
    }
    return out;
}

// RHOHV corrected for noise (Gourley et al. 2006, Eq. 6, in the form of the signal model):
// rho = rho_m sqrt((1 + 1/s_h)(1 + 1/s_v)), s_v = alpha s_h / ZDR (linear ZDR of the signal).
// Without zdr (size 0) the two channels are taken to have the same SNR.
py::array_t<double> correct_rhohv_py(const DArray& rhohv, const DArray& snr, const DArray& zdr,
                                     double alpha_db, bool zdr_has_noise, double min_snr,
                                     bool clip, int n_threads) {
    const int64_t n = rhohv.shape(0);
    check_size(snr, n, "snr");
    const bool have_zdr = zdr.size() > 0;
    if (have_zdr) check_size(zdr, n, "zdr");
    py::array_t<double> out(n);
    double* o = out.mutable_data();
    const double* rp = rhohv.data();
    const double* sp = snr.data();
    const double* zp = have_zdr ? zdr.data() : nullptr;
    const double alpha = from_db(alpha_db);
    {
        py::gil_scoped_release release;
        parallel_blocks(n, n_threads, kBlock, [&](int64_t g0, int64_t g1) {
            for (int64_t i = g0; i < g1; ++i) {
                double r = kNaN;
                if (std::isfinite(rp[i]) && std::isfinite(sp[i]) && sp[i] >= min_snr) {
                    const double s = from_db(sp[i]);
                    double ratio = 1.0;  // zdr / alpha, the SNR of H over that of V
                    bool ok = true;
                    if (have_zdr) {
                        double z = zp[i];
                        if (zdr_has_noise && std::isfinite(z)) z = zdr_denoised(z, sp[i], alpha_db);
                        if (std::isfinite(z))
                            ratio = from_db(z) / alpha;
                        else
                            ok = false;
                    }
                    if (ok) {
                        r = rp[i] * std::sqrt((1.0 + 1.0 / s) * (1.0 + ratio / s));
                        if (clip && r > 1.0) r = 1.0;
                    }
                }
                o[i] = r;
            }
        });
    }
    return out;
}

// Noise level of each row of dbz (rows, nsamp): the range term of sample s is range_db[s % nr],
// the sweep being laid out (ray, range) with range last.
// method 0: Hildebrand and Sekhon (1974) on the linear range-corrected power;
// method 1: the given percentile of the range-corrected dBZ, minus `offset` (dB).
// Returns (rows, 3): noise level at 1 km (dBZ), samples in the noise, valid samples.
py::array_t<double> noise_rows_py(const DArray& dbz, const DArray& range_db, int method,
                                  double n_samples, int64_t min_count, double percentile,
                                  double offset, int n_threads) {
    if (dbz.ndim() != 2) throw std::invalid_argument("dbz must be 2-D (rows, samples)");
    if (range_db.ndim() != 1 || range_db.shape(0) < 1)
        throw std::invalid_argument("range_db must be 1-D");
    if (method != 0 && method != 1) throw std::invalid_argument("method must be 0 or 1");
    const int64_t rows = dbz.shape(0);
    const int64_t nsamp = dbz.shape(1);
    const int64_t nr = range_db.shape(0);
    if (nsamp % nr != 0) throw std::invalid_argument("samples must be a multiple of the ranges");
    py::array_t<double> out({static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(3)});
    double* o = out.mutable_data();
    const double* dp = dbz.data();
    const double* rd = range_db.data();
    {
        py::gil_scoped_release release;
        parallel_blocks(rows, n_threads, 1, [&](int64_t r0, int64_t r1) {
            std::vector<double> v(nsamp), cs1(nsamp), cs2(nsamp);
            for (int64_t r = r0; r < r1; ++r) {
                const double* row = dp + r * nsamp;
                int64_t n = 0;
                for (int64_t s = 0; s < nsamp; ++s) {
                    const double x = row[s] - rd[s % nr];
                    if (std::isnan(x)) continue;
                    v[n++] = x;
                }
                double noise = kNaN;
                int64_t used = 0;
                if (n >= std::max<int64_t>(min_count, 1)) {
                    if (method == 1) {
                        const double pos = percentile / 100.0 * static_cast<double>(n - 1);
                        const int64_t k = static_cast<int64_t>(std::floor(pos));
                        const double frac = pos - static_cast<double>(k);
                        std::nth_element(v.begin(), v.begin() + k, v.begin() + n);
                        double lo = v[k];
                        double q = lo;
                        if (frac > 0.0 && k + 1 < n) {
                            const double hi = *std::min_element(v.begin() + k + 1, v.begin() + n);
                            q = lo + (hi - lo) * frac;
                        }
                        noise = q - offset;
                        used = n;
                    } else {
                        for (int64_t i = 0; i < n; ++i) v[i] = from_db(v[i]);
                        std::sort(v.begin(), v.begin() + n);
                        double a = 0.0, b = 0.0;
                        for (int64_t i = 0; i < n; ++i) {
                            a += v[i];
                            b += v[i] * v[i];
                            cs1[i] = a;
                            cs2[i] = b;
                        }
                        // the largest set of the lowest powers that looks like noise:
                        // mean^2 / variance >= n_samples
                        for (int64_t k = n; k >= std::max<int64_t>(min_count, 1); --k) {
                            const double mean = cs1[k - 1] / static_cast<double>(k);
                            const double var = cs2[k - 1] / static_cast<double>(k) - mean * mean;
                            if (var * n_samples <= mean * mean) {
                                noise = to_db(mean);
                                used = k;
                                break;
                            }
                        }
                    }
                }
                o[r * 3 + 0] = noise;
                o[r * 3 + 1] = static_cast<double>(used);
                o[r * 3 + 2] = static_cast<double>(n);
            }
        });
    }
    return out;
}

PYBIND11_MODULE(_noise, m) {
    m.doc() = "Noise level, SNR and noise correction of RHOHV and ZDR";
    m.def("snr_db", &snr_db_py, py::arg("dbz"), py::arg("noise"), py::arg("range_m"),
          py::arg("includes_noise"), py::arg("n_threads") = 0);
    m.def("correct_zdr", &correct_zdr_py, py::arg("zdr"), py::arg("snr"), py::arg("alpha_db"),
          py::arg("min_snr"), py::arg("n_threads") = 0);
    m.def("correct_rhohv", &correct_rhohv_py, py::arg("rhohv"), py::arg("snr"), py::arg("zdr"),
          py::arg("alpha_db"), py::arg("zdr_has_noise"), py::arg("min_snr"), py::arg("clip"),
          py::arg("n_threads") = 0);
    m.def("noise_rows", &noise_rows_py, py::arg("dbz"), py::arg("range_db"), py::arg("method"),
          py::arg("n_samples"), py::arg("min_count"), py::arg("percentile"), py::arg("offset"),
          py::arg("n_threads") = 0);
}
