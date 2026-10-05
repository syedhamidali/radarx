// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Quasi-vertical profile (QVP) kernels.
//
// azimuthal_reduce: for every sweep and range gate, reduce all variables over
// the rays in a single pass. A gate takes part only if every quality field
// exceeds its threshold (e.g. rhohv > 0.6 and Z > -10 dBZ, Ryzhkov et al.
// 2016) and the variable itself is finite. Variables in dB are averaged in
// linear units and converted back. Work is split into (sweep, gate block)
// items handed out with an atomic counter; each thread owns its buffers.
//
// melting_layer: melting-layer detection in QVPs from the co-located rhohv
// minimum and ZDR / Z maxima (after Giangrande et al. 2008), one profile per
// work item.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

namespace py = pybind11;
using FArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using DArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr int64_t kBlockMean = 256;   // gates per work item for means
constexpr int64_t kBlockMedian = 32;  // smaller: the median gathers whole columns

enum Mode : int { kMean = 0, kMeanDb = 1, kMedian = 2 };

int n_workers(int n_threads, int64_t n_items) {
    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    int64_t nt = n_threads > 0 ? n_threads : static_cast<int64_t>(hw);
    return static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(nt, n_items)));
}

template <class F>
void run_parallel(int nt, F&& worker) {
    std::vector<std::thread> pool;
    for (int t = 1; t < nt; ++t) pool.emplace_back(worker);
    worker();
    for (auto& th : pool) th.join();
}

// Median as numpy computes it (mean of the two middle values for even n).
double median_inplace(float* v, int64_t n) {
    const int64_t h = n / 2;
    std::nth_element(v, v + h, v + n);
    const double upper = v[h];
    if (n % 2) return upper;
    return 0.5 * (static_cast<double>(*std::max_element(v, v + h)) + upper);
}

// 10^(x/10) in single precision (relative error about 2e-7, the precision
// of the float32 input) for |x| < 300 dB. Branch-free and without library
// calls so that the accumulation loop vectorises: 2^y = 2^n * exp(f ln 2)
// with n = round(y), |f| <= 1/2, exp by its Taylor series to degree 7. n is
// rounded by adding 1.5 * 2^23, which leaves n in the low bits of the sum;
// those bits shifted into the exponent field give 2^n.
inline float db_to_linear(float x) {
    constexpr float kRound = 12582912.0f;  // 1.5 * 2^23
    float y = x * 0.33219280948873623f;    // log2(10) / 10
    y = y < -100.0f ? -100.0f : (y > 100.0f ? 100.0f : y);
    const float big = y + kRound;
    const float t = (y - (big - kRound)) * 0.69314718055994531f;
    float p = 1.0f / 5040.0f;
    p = p * t + 1.0f / 720.0f;
    p = p * t + 1.0f / 120.0f;
    p = p * t + 1.0f / 24.0f;
    p = p * t + 1.0f / 6.0f;
    p = p * t + 0.5f;
    p = p * t + 1.0f;
    p = p * t + 1.0f;
    uint32_t bits;
    std::memcpy(&bits, &big, sizeof bits);
    bits = (bits + 127u) << 23;
    float scale;
    std::memcpy(&scale, &bits, sizeof scale);
    return p * scale;
}

}  // namespace

// data[s][v]: (nray_s, ngate_s) float32 arrays of variable v on sweep s.
// quality[s][c]: (nray_s, ngate_s) quality fields, gate valid if > threshold[c].
// modes[v]: 0 mean, 1 mean in linear units of a dB quantity, 2 median.
// min_valid[s]: minimum number of valid gates for a defined value.
// Returns (values, counts), each a list over sweeps of (nvar, ngate_s) arrays.
py::tuple azimuthal_reduce(const std::vector<std::vector<FArray>>& data,
                           const std::vector<std::vector<FArray>>& quality,
                           const std::vector<double>& thresholds,
                           const std::vector<int>& modes,
                           const std::vector<int64_t>& min_valid, int n_threads) {
    const size_t ns = data.size();
    const size_t nv = modes.size();
    const size_t nc = thresholds.size();
    if (quality.size() != ns || min_valid.size() != ns)
        throw std::invalid_argument("data, quality and min_valid need one entry per sweep");
    for (int m : modes)
        if (m < kMean || m > kMedian) throw std::invalid_argument("unknown reduction mode");

    bool any_median = false;
    for (int m : modes) any_median |= (m == kMedian);
    const int64_t block = any_median ? kBlockMedian : kBlockMean;
    std::vector<int64_t> nray(ns), ngate(ns), first_item(ns + 1, 0);
    std::vector<std::vector<const float*>> pdata(ns), pqual(ns);
    std::vector<py::array_t<double>> values;
    std::vector<py::array_t<int32_t>> counts;
    int64_t max_ray = 0;
    for (size_t s = 0; s < ns; ++s) {
        if (data[s].size() != nv || quality[s].size() != nc)
            throw std::invalid_argument("each sweep needs every variable and quality field");
        if (nv == 0) throw std::invalid_argument("need at least one variable");
        nray[s] = data[s][0].ndim() == 2 ? data[s][0].shape(0) : -1;
        ngate[s] = data[s][0].ndim() == 2 ? data[s][0].shape(1) : -1;
        if (nray[s] < 1 || ngate[s] < 1) throw std::invalid_argument("sweep data must be 2-D (ray, gate)");
        for (const auto* group : {&data[s], &quality[s]})
            for (const auto& a : *group)
                if (a.ndim() != 2 || a.shape(0) != nray[s] || a.shape(1) != ngate[s])
                    throw std::invalid_argument("all fields of a sweep need the same (ray, gate) shape");
        for (const auto& a : data[s]) pdata[s].push_back(a.data());
        for (const auto& a : quality[s]) pqual[s].push_back(a.data());
        max_ray = std::max(max_ray, nray[s]);
        first_item[s + 1] = first_item[s] + (ngate[s] + block - 1) / block;
        values.emplace_back(std::vector<int64_t>{static_cast<int64_t>(nv), ngate[s]});
        counts.emplace_back(std::vector<int64_t>{static_cast<int64_t>(nv), ngate[s]});
    }
    std::vector<double*> pval(ns);
    std::vector<int32_t*> pcnt(ns);
    for (size_t s = 0; s < ns; ++s) {
        pval[s] = values[s].mutable_data();
        pcnt[s] = counts[s].mutable_data();
    }
    const int64_t n_items = first_item[ns];

    {
        py::gil_scoped_release release;
        std::atomic<int64_t> next{0};
        auto worker = [&]() {
            std::vector<double> sum(nv * block);
            std::vector<int32_t> cnt(nv * block);
            std::vector<uint8_t> ok(block);
            std::vector<float> gather(any_median ? nv * block * max_ray : 0);
            for (;;) {
                const int64_t item = next.fetch_add(1);
                if (item >= n_items) break;
                const size_t s = static_cast<size_t>(
                    std::upper_bound(first_item.begin(), first_item.end(), item) -
                    first_item.begin() - 1);
                const int64_t ng = ngate[s], nr = nray[s];
                const int64_t g0 = (item - first_item[s]) * block;
                const int64_t nb = std::min(block, ng - g0);
                std::fill(sum.begin(), sum.end(), 0.0);
                std::fill(cnt.begin(), cnt.end(), 0);
                uint8_t* okp = ok.data();
                for (int64_t r = 0; r < nr; ++r) {
                    const int64_t off = r * ng + g0;
                    for (int64_t g = 0; g < nb; ++g) okp[g] = 1;
                    for (size_t c = 0; c < nc; ++c) {
                        const float* q = pqual[s][c] + off;
                        const double t = thresholds[c];
                        for (int64_t g = 0; g < nb; ++g)
                            okp[g] &= static_cast<uint8_t>(q[g] > t);  // NaN fails
                    }
                    // branch-free loops (one per mode) so that they vectorise
                    for (size_t v = 0; v < nv; ++v) {
                        const float* x = pdata[s][v] + off;
                        double* sm = sum.data() + v * block;
                        int32_t* ct = cnt.data() + v * block;
                        if (modes[v] == kMean) {
                            for (int64_t g = 0; g < nb; ++g) {
                                const float val = x[g];
                                const bool m = okp[g] && val == val;
                                sm[g] += m ? static_cast<double>(val) : 0.0;
                                ct[g] += m;
                            }
                        } else if (modes[v] == kMeanDb) {
                            for (int64_t g = 0; g < nb; ++g) {
                                const float val = x[g];
                                const bool m = okp[g] && val == val;
                                const float lin = db_to_linear(m ? val : 0.0f);
                                sm[g] += m ? static_cast<double>(lin) : 0.0;
                                ct[g] += m;
                            }
                        } else {
                            float* buf = gather.data() + v * block * max_ray;
                            for (int64_t g = 0; g < nb; ++g) {
                                const float val = x[g];
                                if (okp[g] && val == val) buf[g * max_ray + ct[g]++] = val;
                            }
                        }
                    }
                }
                for (size_t v = 0; v < nv; ++v) {
                    double* out = pval[s] + v * ng + g0;
                    int32_t* oc = pcnt[s] + v * ng + g0;
                    const double* sm = sum.data() + v * block;
                    const int32_t* ct = cnt.data() + v * block;
                    for (int64_t g = 0; g < nb; ++g) {
                        const int64_t n = ct[g];
                        oc[g] = static_cast<int32_t>(n);
                        if (n < 1 || n < min_valid[s]) {
                            out[g] = kNaN;
                        } else if (modes[v] == kMean) {
                            out[g] = sm[g] / n;
                        } else if (modes[v] == kMeanDb) {
                            out[g] = 10.0 * std::log10(sm[g] / n);  // once per gate
                        } else {
                            out[g] = median_inplace(gather.data() + (v * block + g) * max_ray, n);
                        }
                    }
                }
            }
        };
        run_parallel(n_workers(n_threads, n_items), worker);
    }
    py::list vals, cnts;
    for (size_t s = 0; s < ns; ++s) {
        vals.append(values[s]);
        cnts.append(counts[s]);
    }
    return py::make_tuple(vals, cnts);
}

namespace {

struct MeltingLayerParams {
    double rho_lo, rho_hi;  // rhohv minimum near the ZDR peak must lie in [rho_lo, rho_hi]
    double zdr_min;         // minimum ZDR peak [dB]
    double dbz_min;         // minimum Z maximum near the ZDR peak [dBZ]
    double window;          // co-location distance [m]
    double depth;           // search distance for the layer edges [m]
    double fraction;        // edge where the anomaly has fallen by this fraction
};

// Edge of a peak (sign = +1) or dip (sign = -1) of x at gate i0: walking away
// from i0 in direction step, the last gate before the anomaly sign * x has
// fallen by `fraction` of its prominence. The prominence is measured against
// the lowest sign * x within `depth` metres. Missing gates are skipped.
// Returns -1 if the anomaly does not fall that far within `depth`.
int64_t layer_edge(const double* x, const double* H, int64_t nh, int64_t i0, int step,
                   double depth, double sign, double fraction) {
    const double peak = sign * x[i0];
    double base = peak;
    for (int64_t j = i0 + step; j >= 0 && j < nh && std::fabs(H[j] - H[i0]) <= depth; j += step)
        if (!std::isnan(x[j])) base = std::min(base, sign * x[j]);
    if (!(base < peak)) return -1;
    const double threshold = peak - fraction * (peak - base);
    int64_t last = i0;
    for (int64_t j = i0 + step; j >= 0 && j < nh && std::fabs(H[j] - H[i0]) <= depth; j += step) {
        if (std::isnan(x[j])) continue;
        if (sign * x[j] <= threshold) return last;
        last = j;
    }
    return -1;
}

// Highest height with a reflectivity value (echo top), or +inf.
double echo_top(const double* z, const double* H, int64_t nh) {
    for (int64_t j = nh - 1; j >= 0; --j)
        if (!std::isnan(z[j])) return H[j];
    return std::numeric_limits<double>::infinity();
}

// Gate of the rhohv minimum in [lo, hi) if it lies in [rho_lo, rho_hi] and Z
// reaches dbz_min there, else -1.
int64_t colocated_dip(const double* z, const double* r, int64_t lo, int64_t hi,
                      const MeltingLayerParams& c) {
    int64_t jr = -1;
    double zmax = -std::numeric_limits<double>::infinity();
    for (int64_t q = lo; q < hi; ++q) {
        if (!std::isnan(r[q]) && (jr < 0 || r[q] < r[jr])) jr = q;
        if (!std::isnan(z[q])) zmax = std::max(zmax, z[q]);
    }
    if (jr < 0 || r[jr] < c.rho_lo || r[jr] > c.rho_hi || zmax < c.dbz_min) return -1;
    return jr;
}

// The largest ZDR peak in [hmin, hmax] with a co-located rhohv dip and Z
// maximum. Returns its gate (or -1) and the gate of the dip in *dip.
int64_t find_anchor(const double* z, const double* r, const double* d, const double* H,
                    int64_t nh, double hmin, double hmax, const MeltingLayerParams& c,
                    int64_t* dip) {
    int64_t best = -1;
    int64_t lo = 0, hi = 0;  // window [lo, hi) around j, moved monotonically
    for (int64_t j = 0; j < nh; ++j) {
        while (lo < nh && H[lo] < H[j] - c.window) ++lo;
        while (hi < nh && H[hi] <= H[j] + c.window) ++hi;
        const bool candidate = H[j] >= hmin && H[j] <= hmax && !std::isnan(d[j]) &&
                               d[j] >= c.zdr_min && (best < 0 || d[j] > d[best]);
        if (!candidate) continue;
        const int64_t jr = colocated_dip(z, r, lo, hi, c);
        if (jr < 0) continue;
        best = j;
        *dip = jr;
    }
    return best;
}

// One profile. out = (top, bottom, zdr peak height); NaN if not detected.
void detect_profile(const double* z, const double* r, const double* d, const double* H,
                    int64_t nh, double hmin, double hmax, const MeltingLayerParams& c,
                    double* out) {
    out[0] = out[1] = out[2] = kNaN;
    int64_t dip = -1;
    const int64_t best =
        find_anchor(z, r, d, H, nh, hmin, std::min(hmax, echo_top(z, H, nh)), c, &dip);
    if (best < 0) return;
    // layer edges: where the ZDR peak (and the rhohv dip) return towards background
    const int64_t up = layer_edge(d, H, nh, best, +1, c.depth, 1.0, c.fraction);
    const int64_t dn = layer_edge(d, H, nh, best, -1, c.depth, 1.0, c.fraction);
    if (up < 0 || dn < 0) return;  // ZDR anomaly not bounded: no layer
    const int64_t rup = layer_edge(r, H, nh, dip, +1, c.depth, -1.0, c.fraction);
    const int64_t rdn = layer_edge(r, H, nh, dip, -1, c.depth, -1.0, c.fraction);
    out[0] = rup >= 0 ? std::max(H[up], H[rup]) : H[up];
    out[1] = rdn >= 0 ? std::min(H[dn], H[rdn]) : H[dn];
    out[2] = H[best];
}

}  // namespace

// Melting layer in QVPs. zh, rhohv, zdr: (nprof, nh) profiles on ascending
// heights (nh,); hmin, hmax: (nprof,) search range for the ZDR peak.
// Returns (top, bottom, peak) heights per profile (NaN where not detected).
py::tuple melting_layer(const DArray& zh, const DArray& rhohv, const DArray& zdr,
                        const DArray& height, const DArray& hmin, const DArray& hmax,
                        double rho_lo, double rho_hi, double zdr_min, double dbz_min,
                        double window, double depth, double fraction, int n_threads) {
    if (zh.ndim() != 2 || rhohv.ndim() != 2 || zdr.ndim() != 2 || height.ndim() != 1)
        throw std::invalid_argument("profiles must be 2-D (profile, height)");
    const int64_t np_ = zh.shape(0), nh = zh.shape(1);
    for (const auto* a : {&rhohv, &zdr})
        if (a->shape(0) != np_ || a->shape(1) != nh)
            throw std::invalid_argument("zh, rhohv and zdr shapes do not match");
    if (height.shape(0) != nh || hmin.size() != np_ || hmax.size() != np_)
        throw std::invalid_argument("height, hmin or hmax shape does not match");
    const MeltingLayerParams params{rho_lo, rho_hi, zdr_min, dbz_min, window, depth, fraction};
    py::array_t<double> top(np_), bottom(np_), peak(np_);
    double *pt = top.mutable_data(), *pb = bottom.mutable_data(), *pk = peak.mutable_data();
    const double *Z = zh.data(), *R = rhohv.data(), *D = zdr.data(), *H = height.data();
    const double *lo = hmin.data(), *hi = hmax.data();
    {
        py::gil_scoped_release release;
        std::atomic<int64_t> next{0};
        auto worker = [&]() {
            double out[3];
            for (int64_t i = next.fetch_add(1); i < np_; i = next.fetch_add(1)) {
                detect_profile(Z + i * nh, R + i * nh, D + i * nh, H, nh, lo[i], hi[i], params,
                               out);
                pt[i] = out[0];
                pb[i] = out[1];
                pk[i] = out[2];
            }
        };
        run_parallel(n_workers(n_threads, np_), worker);
    }
    return py::make_tuple(top, bottom, peak);
}

PYBIND11_MODULE(_qvp, m) {
    m.doc() = "Compiled quasi-vertical profile kernels for radarx.";
    m.def("azimuthal_reduce", &azimuthal_reduce, py::arg("data"), py::arg("quality"),
          py::arg("thresholds"), py::arg("modes"), py::arg("min_valid"),
          py::arg("n_threads") = 0);
    m.def("melting_layer", &melting_layer, py::arg("zh"), py::arg("rhohv"), py::arg("zdr"),
          py::arg("height"), py::arg("hmin"), py::arg("hmax"), py::arg("rho_lo") = 0.80,
          py::arg("rho_hi") = 0.97, py::arg("zdr_min") = 0.5, py::arg("dbz_min") = 20.0,
          py::arg("window") = 500.0, py::arg("depth") = 1000.0, py::arg("fraction") = 0.5,
          py::arg("n_threads") = 0);
}
