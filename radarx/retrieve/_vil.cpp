// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Column products of a reflectivity profile: vertically integrated liquid
// (VIL), liquid-phase VIL below a ceiling, and the echo-top height.
//
// Every column is independent. A column is a list of (height, dBZ) samples
// (beams of a polar volume at one ground range, levels of a grid, or the
// bins of a quasi-vertical profile); missing samples are NaN and are left out,
// a sample of -inf is an observed absence of echo. The columns form one pool
// of work that threads take in blocks from an atomic counter.
//
// VIL follows Greene and Clark (1972): the sum over the layers between
// consecutive samples of 3.44e-6 * (0.5 (Z_i + Z_{i+1}))^(4/7) * dh, with Z in
// mm6 m-3 and dh in m (kg m-2). The echo top follows Lakshmanan et al. (2013):
// linear interpolation in dBZ between the highest sample at or above the
// threshold and the next higher one, the top of the beam if there is none.
// Details, references and the radarx choices are in the docstring of
// radarx/retrieve/vil.py. Every step follows the NumPy reference in that
// module, in the same order of operations.

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
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kCoefficient = 3.44e-6;  // Greene and Clark (1972), kg m-2 / (mm6 m-3)^(4/7) m
constexpr double kExponent = 4.0 / 7.0;
constexpr int64_t kBlock = 256;  // columns per scheduling block

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
            const int64_t g1 = std::min(total, g0 + kBlock);
            body(g0, g1);
        }
    };
    std::vector<std::thread> pool;
    for (int i = 1; i < nt; ++i) pool.emplace_back(worker);
    worker();
    for (auto& th : pool) th.join();
}

// Samples of column c that have a finite height and a value, sorted by height with a
// stable insertion sort; returns how many there are.
int64_t gather_sorted(const double* hp, const double* tp, const double* vp, bool shared,
                      int64_t nk, int64_t nc, int64_t c, double* sh, double* st, double* sv) {
    int64_t n = 0;
    for (int64_t k = 0; k < nk; ++k) {
        const double hk = shared ? hp[k] : hp[k * nc + c];
        const double vk = vp[k * nc + c];
        if (!std::isfinite(hk) || std::isnan(vk)) continue;
        const double tk = shared ? tp[k] : tp[k * nc + c];
        int64_t i = n++;
        while (i > 0 && sh[i - 1] > hk) {
            sh[i] = sh[i - 1];
            st[i] = st[i - 1];
            sv[i] = sv[i - 1];
            --i;
        }
        sh[i] = hk;
        st[i] = tk;
        sv[i] = vk;
    }
    return n;
}

struct Integral {
    double total = 0.0, liquid = 0.0;
    bool has = false, has_liquid = false;
};

// Liquid part of the layer between samples i and i + 1 below the ceiling.
void add_liquid_layer(Integral& r, const double* sh, const double* sz, int64_t i, double ceil_h) {
    if (std::isnan(ceil_h) || sh[i] >= ceil_h) return;
    r.has_liquid = true;
    const double dh = sh[i + 1] - sh[i];
    if (sh[i + 1] <= ceil_h) {
        r.liquid += std::pow(0.5 * (sz[i] + sz[i + 1]), kExponent) * dh;
    } else {
        const double zc = sz[i] + (sz[i + 1] - sz[i]) * (ceil_h - sh[i]) / dh;
        r.liquid += std::pow(0.5 * (sz[i] + zc), kExponent) * (ceil_h - sh[i]);
    }
}

// The layer from base_height up to the lowest sample takes the value of that sample.
void add_fill_below(Integral& r, const double* sh, const double* sz, double ceil_h,
                    double base_height) {
    const double fill = std::pow(sz[0], kExponent);
    const double seg = sh[0] - base_height;
    if (seg > 0.0) {
        r.total += fill * seg;
        r.has = true;
    }
    if (std::isnan(ceil_h)) return;
    const double seg_l = std::min(sh[0], ceil_h) - base_height;
    if (seg_l > 0.0) {
        r.liquid += fill * seg_l;
        r.has_liquid = true;
    }
}

// VIL and liquid VIL of a column: the layers between consecutive samples.
Integral integrate_column(const double* sh, const double* sz, int64_t n, double ceil_h,
                          bool fill_below, double base_height) {
    Integral r;
    for (int64_t i = 0; i + 1 < n; ++i) {
        r.total += std::pow(0.5 * (sz[i] + sz[i + 1]), kExponent) * (sh[i + 1] - sh[i]);
        r.has = true;
        add_liquid_layer(r, sh, sz, i, ceil_h);
    }
    if (fill_below && std::isfinite(base_height)) add_fill_below(r, sh, sz, ceil_h, base_height);
    return r;
}

// Echo top: the highest sample at or above the threshold, interpolated in dBZ towards the
// next higher sample, or the top of its beam.
double echo_top_height(const double* sh, const double* st, const double* sv, int64_t n,
                       double top_threshold, double no_echo_dbz, bool interpolate) {
    int64_t b = -1;
    for (int64_t i = n - 1; i >= 0; --i) {
        if (sv[i] >= top_threshold) {
            b = i;
            break;
        }
    }
    if (b < 0) return kNaN;
    if (interpolate && b + 1 < n) {
        const double za = std::max(sv[b + 1], no_echo_dbz);
        const double zb = sv[b];
        return sh[b] + (zb - top_threshold) / (zb - za) * (sh[b + 1] - sh[b]);
    }
    return st[b];
}

void check_shapes(const DArray& h, const DArray& h_top, const DArray& v, const DArray& ceiling,
                  bool shared) {
    if (v.ndim() != 2) throw std::invalid_argument("v must be 2-D (levels, columns)");
    const int64_t nk = v.shape(0);
    const int64_t nc = v.shape(1);
    if (shared) {
        if (h.ndim() != 1 || h.shape(0) != nk || h_top.ndim() != 1 || h_top.shape(0) != nk)
            throw std::invalid_argument("shared heights must be 1-D with one value per level");
    } else if (h.ndim() != 2 || h.shape(0) != nk || h.shape(1) != nc || h_top.ndim() != 2 ||
               h_top.shape(0) != nk || h_top.shape(1) != nc) {
        throw std::invalid_argument("heights must have the shape of v");
    }
    if (ceiling.ndim() != 1 || ceiling.shape(0) != nc)
        throw std::invalid_argument("ceiling must be 1-D with one value per column");
}

}  // namespace

// h, h_top: (nk,) when shared, else (nk, nc); v: (nk, nc); ceiling: (nc,).
// Returns (6, nc): VIL, liquid VIL, lowest height, highest height, echo top and the
// reflectivity (dBZ) of the highest sample.
py::array_t<double> columns_py(const DArray& h, const DArray& h_top, const DArray& v,
                               const DArray& ceiling, bool shared, double dbz_cap,
                               double min_dbz, double top_threshold, double no_echo_dbz,
                               bool fill_below, double base_height, bool interpolate,
                               int n_threads) {
    check_shapes(h, h_top, v, ceiling, shared);
    const int64_t nk = v.shape(0);
    const int64_t nc = v.shape(1);
    const double cap = std::isnan(dbz_cap) ? kInf : dbz_cap;
    const double floor_dbz = std::isnan(min_dbz) ? -kInf : min_dbz;

    py::array_t<double> out({static_cast<py::ssize_t>(6), static_cast<py::ssize_t>(nc)});
    double* o = out.mutable_data();
    const double* hp = h.data();
    const double* tp = h_top.data();
    const double* vp = v.data();
    const double* cp = ceiling.data();

    {
        py::gil_scoped_release release;
        parallel_blocks(nc, n_threads, [&](int64_t c0, int64_t c1) {
            std::vector<double> sh(nk), st(nk), sz(nk), sv(nk);
            for (int64_t c = c0; c < c1; ++c) {
                const int64_t n =
                    gather_sorted(hp, tp, vp, shared, nk, nc, c, sh.data(), st.data(), sv.data());
                double vil = kNaN, liquid = kNaN, lowest = kNaN, highest = kNaN, top = kNaN, last = kNaN;
                if (n > 0) {
                    lowest = sh[0];
                    highest = sh[n - 1];
                    last = sv[n - 1];
                    for (int64_t i = 0; i < n; ++i)
                        sz[i] = (sv[i] < floor_dbz) ? 0.0 : std::pow(10.0, std::min(sv[i], cap) / 10.0);
                    const Integral r =
                        integrate_column(sh.data(), sz.data(), n, cp[c], fill_below, base_height);
                    if (r.has) vil = kCoefficient * r.total;
                    if (r.has_liquid) liquid = kCoefficient * r.liquid;
                    top = echo_top_height(sh.data(), st.data(), sv.data(), n, top_threshold,
                                          no_echo_dbz, interpolate);
                }
                o[0 * nc + c] = vil;
                o[1 * nc + c] = liquid;
                o[2 * nc + c] = lowest;
                o[3 * nc + c] = highest;
                o[4 * nc + c] = top;
                o[5 * nc + c] = last;
            }
        });
    }
    return out;
}

PYBIND11_MODULE(_vil, m) {
    m.doc() = "VIL, liquid VIL and echo-top kernel";
    m.def("columns", &columns_py, py::arg("h"), py::arg("h_top"), py::arg("v"),
          py::arg("ceiling"), py::arg("shared"), py::arg("dbz_cap"), py::arg("min_dbz"),
          py::arg("top_threshold"), py::arg("no_echo_dbz"), py::arg("fill_below"),
          py::arg("base_height"), py::arg("interpolate"), py::arg("n_threads") = 0);
}
