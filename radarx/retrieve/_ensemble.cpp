// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Kernels of the ensemble module: random errors of the radar measurements,
// and the statistics of an ensemble of retrievals.
//
// Random numbers are counter based (Salmon et al. 2011): Philox-4x64-10
// turns a key (seed, stream) and a counter (gate, member, block, 0) into
// four 64-bit words, so that every gate of every member has its numbers
// without any state, whatever the number of threads. The NumPy reference in
// radarx/retrieve/ensemble.py reads the same words from numpy.random.Philox
// (which implements Philox-4x64-10), so the engines draw the same words.
// Uniform numbers come from the upper 53 bits of a word, Gaussian numbers
// from the Box-Muller transform; the engines use their own log, sin and cos,
// so the Gaussian numbers agree to rounding, not bit for bit.
//
// The error model (which variance each variable gets) is computed by the
// caller; this file adds the errors. See radarx/retrieve/ensemble.py.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

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
using FArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using u64 = uint64_t;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr int64_t kBlock = 256;  // gates per scheduling block

// Philox-4x64-10 (Salmon et al. 2011): multipliers and Weyl increments
constexpr u64 kM0 = 0xD2E7470EE14C6C93ULL;
constexpr u64 kM1 = 0xCA5A826395121157ULL;
constexpr u64 kW0 = 0x9E3779B97F4A7C15ULL;
constexpr u64 kW1 = 0xBB67AE8584CAA73BULL;
constexpr double kTwoPi = 6.283185307179586;
constexpr double kScale = 1.0 / 9007199254740992.0;  // 2^-53
// 10 / ln(10): dB per unit of fractional power error
constexpr double kDb = 4.342944819032518;

enum Field { kDbz = 1, kZdr = 2, kPhidp = 4, kRhohv = 8, kVel = 16 };

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
            body(g0, std::min(total, g0 + kBlock));
        }
    };
    std::vector<std::thread> pool;
    for (int i = 1; i < nt; ++i) pool.emplace_back(worker);
    worker();
    for (auto& th : pool) th.join();
}

// high and low 64 bits of a * b: the 128-bit product where the compiler has one,
// otherwise four 32-bit products
#if defined(_MSC_VER) && defined(_M_X64)
inline void mulhilo(u64 a, u64 b, u64& hi, u64& lo) { lo = _umul128(a, b, &hi); }
#elif defined(__SIZEOF_INT128__)
inline void mulhilo(u64 a, u64 b, u64& hi, u64& lo) {
    const unsigned __int128 p = static_cast<unsigned __int128>(a) * b;
    hi = static_cast<u64>(p >> 64);
    lo = static_cast<u64>(p);
}
#else
inline void mulhilo(u64 a, u64 b, u64& hi, u64& lo) {
    const u64 mask = 0xFFFFFFFFULL;
    const u64 a0 = a & mask, a1 = a >> 32, b0 = b & mask, b1 = b >> 32;
    const u64 p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    const u64 mid = (p00 >> 32) + (p01 & mask) + (p10 & mask);
    lo = (mid << 32) | (p00 & mask);
    hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
}
#endif

inline void philox(u64 c0, u64 c1, u64 c2, u64 c3, u64 k0, u64 k1, u64* out) {
    for (int r = 0; r < 10; ++r) {
        u64 h0, l0, h1, l1;
        mulhilo(kM0, c0, h0, l0);
        mulhilo(kM1, c2, h1, l1);
        const u64 n0 = h1 ^ c1 ^ k0;
        const u64 n2 = h0 ^ c3 ^ k1;
        c0 = n0;
        c1 = l1;
        c2 = n2;
        c3 = l0;
        k0 += kW0;
        k1 += kW1;
    }
    out[0] = c0;
    out[1] = c1;
    out[2] = c2;
    out[3] = c3;
}

// two Gaussian numbers from two words
inline void box_muller(u64 w0, u64 w1, double& a, double& b) {
    const double u1 = static_cast<double>((w0 >> 11) + 1) * kScale;  // (0, 1]
    const double u2 = static_cast<double>(w1 >> 11) * kScale;        // [0, 1)
    const double r = std::sqrt(-2.0 * std::log(u1));
    const double t = kTwoPi * u2;
    a = r * std::cos(t);
    b = r * std::sin(t);
}

// ---------------------------------------------------------------------
// perturbation

struct Inputs {
    const double* v[5];
    const double* fac[6];
    int flags;
    int64_t n;
};

inline double clip01(double x) { return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x); }

// The five Gaussian numbers of gate i of member m. Gate numbers start at 1 so
// that numpy.random.Philox(counter=[0, m, block, 0]) gives the same words.
inline void draws(int64_t i, int64_t m, u64 k0, u64 k1, bool vel, double* z) {
    u64 w[4];
    philox(static_cast<u64>(i) + 1, static_cast<u64>(m), 0, 0, k0, k1, w);
    box_muller(w[0], w[1], z[0], z[1]);
    box_muller(w[2], w[3], z[2], z[3]);
    z[4] = 0.0;
    if (vel) {
        double unused;
        philox(static_cast<u64>(i) + 1, static_cast<u64>(m), 1, 0, k0, k1, w);
        box_muller(w[0], w[1], z[4], unused);
    }
}

void perturb_gate(const Inputs& in, int64_t i, const double* z, double bias_dbz,
                  double bias_zdr, double* const* out) {
    const double xh = in.fac[0][i] * z[0];
    const double xv = in.fac[1][i] * z[0] + in.fac[2][i] * z[1];
    int slot = 0;
    if (in.flags & kDbz) out[slot++][i] = in.v[0][i] + bias_dbz + kDb * xh;
    if (in.flags & kZdr) out[slot++][i] = in.v[1][i] + bias_zdr + kDb * (xh - xv);
    if (in.flags & kPhidp) out[slot++][i] = in.v[2][i] + in.fac[3][i] * z[2];
    if (in.flags & kRhohv) out[slot++][i] = clip01(in.v[3][i] + in.fac[4][i] * z[3]);
    if (in.flags & kVel) out[slot++][i] = in.v[4][i] + in.fac[5][i] * z[4];
}

py::array_t<double> perturb_py(DArray dbz, DArray zdr, DArray phidp, DArray rhohv, DArray vel,
                               int flags, DArray fac, DArray bias_dbz, DArray bias_zdr,
                               u64 k0, u64 k1, int64_t m0, int n_threads) {
    if (fac.ndim() != 2 || fac.shape(0) != 6) throw std::invalid_argument("fac must be (6, n)");
    const int64_t n = fac.shape(1);
    const int64_t nm = bias_dbz.size();
    if (bias_zdr.size() != nm) throw std::invalid_argument("bias arrays differ in length");
    const DArray* given[5] = {&dbz, &zdr, &phidp, &rhohv, &vel};
    const int bits[5] = {kDbz, kZdr, kPhidp, kRhohv, kVel};
    Inputs in;
    in.flags = flags;
    in.n = n;
    int nf = 0;
    for (int f = 0; f < 5; ++f) {
        in.v[f] = given[f]->data();
        if (flags & bits[f]) {
            if (given[f]->size() != n) throw std::invalid_argument("a field differs in size from fac");
            ++nf;
        }
    }
    for (int f = 0; f < 6; ++f) in.fac[f] = fac.data() + f * n;
    py::array_t<double> out({nm, static_cast<int64_t>(nf), n});
    double* base = out.mutable_data();
    const double* bd = bias_dbz.data();
    const double* bz = bias_zdr.data();
    {
        py::gil_scoped_release release;
        parallel_blocks(n, n_threads, [&](int64_t g0, int64_t g1) {
            double z[5];
            for (int64_t j = 0; j < nm; ++j) {
                double* slots[5];
                for (int s = 0; s < nf; ++s) slots[s] = base + (j * nf + s) * n;
                for (int64_t i = g0; i < g1; ++i) {
                    draws(i, m0 + j, k0, k1, (flags & kVel) != 0, z);
                    perturb_gate(in, i, z, bd[j], bz[j], slots);
                }
            }
        });
    }
    return out;
}

// ---------------------------------------------------------------------
// statistics over the members of an ensemble

// Mean and standard deviation (n - 1) of the finite values v[0..n), and the
// percentiles (linear interpolation between order statistics) of the sorted
// copy. Gates with fewer than min_valid finite members are NaN.
void gate_statistics(const float* values, int64_t nm, int64_t n, int64_t g, int64_t min_valid,
                     const double* q, int64_t nq, std::vector<double>& buf, double* mean_out,
                     double* std_out, double* pct_out) {
    buf.clear();
    for (int64_t m = 0; m < nm; ++m) {
        const double x = static_cast<double>(values[m * n + g]);
        if (std::isfinite(x)) buf.push_back(x);
    }
    const int64_t cnt = static_cast<int64_t>(buf.size());
    if (cnt < min_valid || cnt == 0) {
        *mean_out = kNaN;
        *std_out = kNaN;
        for (int64_t k = 0; k < nq; ++k) pct_out[k * n + g] = kNaN;
        return;
    }
    double s = 0.0;
    for (int64_t k = 0; k < cnt; ++k) s += buf[k];
    const double mean = s / static_cast<double>(cnt);
    double ss = 0.0;
    for (int64_t k = 0; k < cnt; ++k) {
        const double d = buf[k] - mean;
        ss += d * d;
    }
    *mean_out = mean;
    *std_out = cnt > 1 ? std::sqrt(ss / static_cast<double>(cnt - 1)) : kNaN;
    std::sort(buf.begin(), buf.end());
    for (int64_t k = 0; k < nq; ++k) {
        const double pos = static_cast<double>(cnt - 1) * (q[k] / 100.0);
        const int64_t lo = static_cast<int64_t>(std::floor(pos));
        const int64_t hi = std::min<int64_t>(lo + 1, cnt - 1);
        const double frac = pos - static_cast<double>(lo);
        pct_out[k * n + g] = buf[lo] + (buf[hi] - buf[lo]) * frac;
    }
}

py::tuple statistics_py(FArray values, DArray percentiles, int64_t min_valid, int n_threads) {
    if (values.ndim() != 2) throw std::invalid_argument("values must be (members, n)");
    const int64_t nm = values.shape(0), n = values.shape(1);
    const int64_t nq = percentiles.size();
    py::array_t<double> mean(n), sd(n);
    py::array_t<double> pct({nq, n});
    const float* vp = values.data();
    const double* qp = percentiles.data();
    double* mp = mean.mutable_data();
    double* sp = sd.mutable_data();
    double* pp = pct.mutable_data();
    {
        py::gil_scoped_release release;
        parallel_blocks(n, n_threads, [&](int64_t g0, int64_t g1) {
            std::vector<double> buf;
            buf.reserve(static_cast<size_t>(nm));
            for (int64_t g = g0; g < g1; ++g)
                gate_statistics(vp, nm, n, g, min_valid, qp, nq, buf, mp + g, sp + g, pp);
        });
    }
    return py::make_tuple(mean, sd, pct);
}

// most frequent class (the lowest code on ties) and its share of the finite members
py::tuple mode_py(FArray values, int64_t n_codes, int64_t min_valid, int n_threads) {
    if (values.ndim() != 2) throw std::invalid_argument("values must be (members, n)");
    if (n_codes < 1) throw std::invalid_argument("n_codes must be positive");
    const int64_t nm = values.shape(0), n = values.shape(1);
    py::array_t<double> cls(n), share(n);
    const float* vp = values.data();
    double* cp = cls.mutable_data();
    double* sp = share.mutable_data();
    {
        py::gil_scoped_release release;
        parallel_blocks(n, n_threads, [&](int64_t g0, int64_t g1) {
            std::vector<int64_t> count(static_cast<size_t>(n_codes));
            for (int64_t g = g0; g < g1; ++g) {
                std::fill(count.begin(), count.end(), 0);
                int64_t cnt = 0;
                for (int64_t m = 0; m < nm; ++m) {
                    const float x = vp[m * n + g];
                    if (!std::isfinite(x)) continue;
                    const int64_t c = static_cast<int64_t>(x);
                    if (c < 0 || c >= n_codes) continue;
                    ++count[static_cast<size_t>(c)];
                    ++cnt;
                }
                if (cnt < min_valid || cnt == 0) {
                    cp[g] = kNaN;
                    sp[g] = kNaN;
                    continue;
                }
                int64_t best = 0;
                for (int64_t c = 1; c < n_codes; ++c)
                    if (count[static_cast<size_t>(c)] > count[static_cast<size_t>(best)]) best = c;
                cp[g] = static_cast<double>(best);
                sp[g] = static_cast<double>(count[static_cast<size_t>(best)]) /
                        static_cast<double>(cnt);
            }
        });
    }
    return py::make_tuple(cls, share);
}

py::array_t<uint64_t> words_py(u64 k0, u64 k1, int64_t member, int64_t block, int64_t n) {
    py::array_t<uint64_t> out({n, static_cast<int64_t>(4)});
    uint64_t* p = out.mutable_data();
    for (int64_t i = 0; i < n; ++i) philox(static_cast<u64>(i) + 1, static_cast<u64>(member),
                                           static_cast<u64>(block), 0, k0, k1, p + 4 * i);
    return out;
}

}  // namespace

PYBIND11_MODULE(_ensemble, m) {
    m.doc() = "Random measurement errors and ensemble statistics";
    m.def("perturb", &perturb_py, py::arg("dbz"), py::arg("zdr"), py::arg("phidp"),
          py::arg("rhohv"), py::arg("vel"), py::arg("flags"), py::arg("fac"),
          py::arg("bias_dbz"), py::arg("bias_zdr"), py::arg("k0"), py::arg("k1"),
          py::arg("m0"), py::arg("n_threads") = 0);
    m.def("statistics", &statistics_py, py::arg("values"), py::arg("percentiles"),
          py::arg("min_valid"), py::arg("n_threads") = 0);
    m.def("mode", &mode_py, py::arg("values"), py::arg("n_codes"), py::arg("min_valid"),
          py::arg("n_threads") = 0);
    m.def("words", &words_py, py::arg("k0"), py::arg("k1"), py::arg("member"),
          py::arg("block"), py::arg("n"));
}
