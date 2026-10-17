// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Melting-layer top and bottom of vertical profiles from the gradients of
// Z, ZDR and rhohv and of their combinations (Allabakash et al. 2019, Sect.
// 2.2 and Fig. 2). The algorithm and the choices the paper leaves open are
// described in radarx/retrieve/melting_layer_gradient.py, whose NumPy
// reference follows the same steps with the same formulas.
//
// Every profile is independent, so threads take blocks of profiles from an
// atomic counter and each thread owns its buffers.

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

// The NumPy reference does not fuse multiplications and additions; a fused
// multiply-add here could pick another gate between two almost equal gradients.
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

namespace py = pybind11;
using DArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using IArray = py::array_t<int32_t, py::array::c_style>;

namespace {

constexpr int64_t kBlock = 16;  // profiles per scheduling block

// Derivative along the (possibly irregular) heights: second-order central
// differences inside, first-order differences at the ends.
void gradient(const double* f, const double* h, int64_t n, double* g) {
    g[0] = (f[1] - f[0]) / (h[1] - h[0]);
    g[n - 1] = (f[n - 1] - f[n - 2]) / (h[n - 1] - h[n - 2]);
    for (int64_t i = 1; i < n - 1; ++i) {
        const double hd = h[i] - h[i - 1], hs = h[i + 1] - h[i];
        g[i] = (hs * hs * (f[i + 1] - f[i]) + hd * hd * (f[i] - f[i - 1])) / (hs * hd * (hd + hs));
    }
}

// Index of the first maximum (or minimum) that is not NaN, -1 if all are NaN.
int64_t extreme(const double* g, int64_t n, bool maximum, int64_t lo = 0, int64_t hi = -1) {
    int64_t best = -1;
    if (hi < 0) hi = n - 1;
    for (int64_t i = lo; i <= hi; ++i) {
        if (g[i] != g[i]) continue;
        if (best < 0 || (maximum ? g[i] > g[best] : g[i] < g[best])) best = i;
    }
    return best;
}

// Index at which at least two of the three indices agree, -1 if none does.
int64_t coincident(int64_t a, int64_t b, int64_t c) {
    if (a >= 0 && (a == b || a == c)) return a;
    if (b >= 0 && b == c) return b;
    return -1;
}

// Largest finite value; false if there is none.
bool finite_max(const double* x, int64_t n, double* out) {
    bool any = false;
    double m = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        if (x[i] != x[i]) continue;
        if (!any || x[i] > m) m = x[i];
        any = true;
    }
    *out = m;
    return any;
}

struct Buffers {
    std::vector<double> az, ad, ar, p1, p2, p3, q1, q2, q3, gz, gr;
    explicit Buffers(int64_t n)
        : az(n), ad(n), ar(n), p1(n), p2(n), p3(n), q1(n), q2(n), q3(n), gz(n), gr(n) {}
};

// bottom, top (gate indices, -1 if none) and the step that gave the top.
void detect(const double* zh, const double* zdr, const double* rho, const double* h, int64_t n,
            double window, Buffers& b, int32_t* bottom, int32_t* top, int32_t* step) {
    *bottom = *top = -1;
    *step = 0;
    double mz, md, mr;
    if (!finite_max(zh, n, &mz) || !finite_max(zdr, n, &md) || !finite_max(rho, n, &mr)) return;
    if (!(mz > 0.0) || !(md > 0.0) || !(mr > 0.0)) return;
    for (int64_t i = 0; i < n; ++i) {
        b.az[i] = std::fabs(zh[i] / mz);
        b.ad[i] = std::fabs(zdr[i] / md);
        b.ar[i] = std::fabs(rho[i] / mr);
    }
    // the products Zh (1-rho), Zdr (1-rho) and Zh Zdr (1-rho) of the normalised values
    for (int64_t i = 0; i < n; ++i) {
        const double c = 1.0 - b.ar[i];
        b.p1[i] = b.az[i] * c;
        b.p2[i] = b.ad[i] * c;
        b.p3[i] = b.az[i] * b.ad[i] * c;
    }
    gradient(b.p1.data(), h, n, b.q1.data());
    gradient(b.p2.data(), h, n, b.q2.data());
    gradient(b.p3.data(), h, n, b.q3.data());
    gradient(b.az.data(), h, n, b.gz.data());
    gradient(b.ar.data(), h, n, b.gr.data());
    const int64_t bot =
        coincident(extreme(b.q1.data(), n, true), extreme(b.q2.data(), n, true),
                   extreme(b.q3.data(), n, true));
    const int64_t cmin =
        coincident(extreme(b.q1.data(), n, false), extreme(b.q2.data(), n, false),
                   extreme(b.q3.data(), n, false));
    if (bot < 0 || cmin < 0) return;
    // the gates within `window` of the minima of the products (all gates if infinite)
    int64_t lo = 0, hi = n - 1;
    if (std::isfinite(window)) {
        while (h[lo] < h[cmin] - window) ++lo;
        while (h[hi] > h[cmin] + window) --hi;
    }
    const int64_t zmin = extreme(b.gz.data(), n, false, lo, hi);  // Z gradient minimum
    const int64_t rmax = extreme(b.gr.data(), n, true, lo, hi);   // rho gradient maximum
    int64_t t;
    int32_t s;
    if (zmin == cmin && rmax == cmin) {
        t = cmin;
        s = 3;
    } else if (std::max(zmin, rmax) > cmin) {
        t = std::max(zmin, rmax);
        s = 4;
    } else {
        t = cmin + 1;
        s = 5;
    }
    if (t >= n) return;
    *bottom = static_cast<int32_t>(bot);
    *top = static_cast<int32_t>(t);
    *step = s;
}

py::tuple melting_layer_gradient(DArray zh, DArray zdr, DArray rhohv, DArray height,
                                 double window, int n_threads) {
    if (zh.ndim() != 2 || zdr.ndim() != 2 || rhohv.ndim() != 2)
        throw std::invalid_argument("profiles must be 2-D (profile, height)");
    const int64_t np_ = zh.shape(0), n = zh.shape(1);
    if (zdr.shape(0) != np_ || zdr.shape(1) != n || rhohv.shape(0) != np_ ||
        rhohv.shape(1) != n)
        throw std::invalid_argument("zh, zdr and rhohv shapes do not match");
    if (height.ndim() != 1 || height.shape(0) != n)
        throw std::invalid_argument("height must have one value per gate");
    if (n < 3) throw std::invalid_argument("a profile needs at least three heights");
    if (window != window || window < 0.0) throw std::invalid_argument("window must not be negative");
    const double* H = height.data();
    for (int64_t i = 1; i < n; ++i)
        if (!(H[i] > H[i - 1])) throw std::invalid_argument("height must be strictly ascending");
    IArray bottom(np_), top(np_), step(np_);
    int32_t *pb = bottom.mutable_data(), *pt = top.mutable_data(), *ps = step.mutable_data();
    const double *Z = zh.data(), *D = zdr.data(), *R = rhohv.data();
    {
        py::gil_scoped_release release;
        unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        int64_t nblocks = (np_ + kBlock - 1) / kBlock;
        int64_t nt = n_threads > 0 ? n_threads : static_cast<int64_t>(hw);
        nt = std::max<int64_t>(1, std::min<int64_t>(nt, nblocks));
        std::atomic<int64_t> next{0};
        auto worker = [&]() {
            Buffers buf(n);
            for (;;) {
                const int64_t blk = next.fetch_add(1);
                if (blk >= nblocks) break;
                const int64_t end = std::min(np_, (blk + 1) * kBlock);
                for (int64_t k = blk * kBlock; k < end; ++k)
                    detect(Z + k * n, D + k * n, R + k * n, H, n, window, buf, pb + k, pt + k, ps + k);
            }
        };
        if (nt == 1) {
            worker();
        } else {
            std::vector<std::thread> pool;
            for (int64_t i = 0; i < nt; ++i) pool.emplace_back(worker);
            for (auto& t : pool) t.join();
        }
    }
    return py::make_tuple(bottom, top, step);
}

}  // namespace

PYBIND11_MODULE(_melting_layer_gradients, m) {
    m.doc() = "Compiled melting-layer detection from profile gradients for radarx.";
    m.def("melting_layer_gradient", &melting_layer_gradient, py::arg("zh"), py::arg("zdr"),
          py::arg("rhohv"), py::arg("height"),
          py::arg("window") = std::numeric_limits<double>::infinity(), py::arg("n_threads") = 0);
}
