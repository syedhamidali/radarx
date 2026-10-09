// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Polar patch extraction and reassembly for machine-learning models.
//
// A sweep is an (L, A, R) array: L stacked fields (channels, times, ...),
// A rays in acquisition order and R gates. A patch of h rays and w gates
// starts at ray a0 and gate r0 of sweep k; the patch table holds one
// (k, a0, r0) row per patch. Rays wrap around (a0 + i taken modulo A) when
// `wrap` is set, so patches cross north seamlessly; gates never wrap. Pixels
// outside the sweep are set to `fill` on extraction and ignored on
// reassembly.
//
// Reassembly is the weighted mean of every patch pixel that covers an output
// gate, with separable weights wa[i] * wr[j] (cosine, linear or uniform
// windows built by the Python layer; these windows are radarx choices without
// a published source, see radarx/ml/patches.py). It is written as a gather: each work
// item is one output ray of one field of one sweep and visits the (patch,
// patch row) pairs covering that ray, so threads never write to the same
// memory and no locks are needed. Sums are accumulated in double precision.
// Where every pixel covering a gate holds the same value, that value is
// returned as is (the weighted mean of equal values), so reassembling
// unchanged patches gives back the input bit for bit. NaN patch pixels are
// skipped; a gate no finite pixel covers is NaN.
//
// All sweeps of a volume are processed in one call; the threads take chunks
// of work items from an atomic counter.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

template <class T>
using Arr = py::array_t<T, py::array::c_style | py::array::forcecast>;
using IArr = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;
using DArr = py::array_t<double, py::array::c_style | py::array::forcecast>;

int thread_count(int n_threads, int64_t work) {
    const int nt = n_threads > 0
                       ? n_threads
                       : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    return static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(nt, work)));
}

// Runs body(item) for item in [0, total) on nt threads (dynamic chunks).
template <class Body>
void parallel_for(int64_t total, int nt, int64_t chunk, Body&& body) {
    std::atomic<int64_t> next{0};
    auto worker = [&]() {
        for (int64_t u0; (u0 = next.fetch_add(chunk)) < total;) {
            const int64_t u1 = std::min(total, u0 + chunk);
            for (int64_t u = u0; u < u1; ++u) body(u);
        }
    };
    std::vector<std::thread> pool;
    for (int t = 1; t < nt; ++t) pool.emplace_back(worker);
    worker();
    for (auto& th : pool) th.join();
}

// Ray of the sweep for patch row a0 + i, or -1 outside the sweep.
inline int64_t source_ray(int64_t a, int64_t nray, bool wrap) {
    if (wrap) {
        a %= nray;
        return a < 0 ? a + nray : a;
    }
    return (a >= 0 && a < nray) ? a : -1;
}

struct Table {
    const int64_t* rows = nullptr;
    int64_t n = 0;
};

Table check_table(const IArr& table, size_t nsweep) {
    if (table.ndim() != 2 || table.shape(1) != 3)
        throw std::invalid_argument("table must be (n_patches, 3): sweep, ray, gate");
    Table t{table.data(), table.shape(0)};
    for (int64_t n = 0; n < t.n; ++n) {
        const int64_t k = t.rows[3 * n];
        if (k < 0 || k >= static_cast<int64_t>(nsweep))
            throw std::invalid_argument("table refers to a sweep that does not exist");
    }
    return t;
}

}  // namespace

// data[k] is the (L, A_k, R_k) sweep k. Returns (n_patches, L, h, w).
template <class T>
py::array_t<T> extract(const std::vector<Arr<T>>& data, const IArr& table, int64_t h,
                       int64_t w, bool wrap, double fill, int n_threads) {
    if (data.empty()) throw std::invalid_argument("no sweeps given");
    if (h < 1 || w < 1) throw std::invalid_argument("patch size must be positive");
    const size_t nk = data.size();
    std::vector<const T*> src(nk);
    std::vector<int64_t> nray(nk), ngate(nk);
    const int64_t L = data[0].ndim() == 3 ? data[0].shape(0) : -1;
    for (size_t k = 0; k < nk; ++k) {
        if (data[k].ndim() != 3 || data[k].shape(0) != L)
            throw std::invalid_argument("every sweep must be (fields, rays, gates) with the same fields");
        if (data[k].shape(1) < 1 || data[k].shape(2) < 1)
            throw std::invalid_argument("empty sweep");
        src[k] = data[k].data();
        nray[k] = data[k].shape(1);
        ngate[k] = data[k].shape(2);
    }
    const Table t = check_table(table, nk);
    py::array_t<T> out(std::vector<int64_t>{t.n, L, h, w});
    T* dst = out.mutable_data();
    const T fv = static_cast<T>(fill);
    {
        py::gil_scoped_release release;
        const int64_t total = t.n * L * h;  // one patch row per item
        parallel_for(total, thread_count(n_threads, total), 64, [&](int64_t u) {
            const int64_t i = u % h, nl = u / h, l = nl % L, n = nl / L;
            const int64_t k = t.rows[3 * n], a0 = t.rows[3 * n + 1], r0 = t.rows[3 * n + 2];
            T* row = dst + u * w;
            const int64_t a = source_ray(a0 + i, nray[k], wrap);
            if (a < 0) {
                std::fill(row, row + w, fv);
                return;
            }
            const T* s = src[k] + (l * nray[k] + a) * ngate[k];
            const int64_t j0 = std::max<int64_t>(0, -r0);
            const int64_t j1 = std::max<int64_t>(j0, std::min<int64_t>(w, ngate[k] - r0));
            std::fill(row, row + j0, fv);
            if (j1 > j0) std::copy(s + r0 + j0, s + r0 + j1, row + j0);
            std::fill(row + j1, row + w, fv);
        });
    }
    return out;
}

// patches is (n_patches, L, h, w); shapes[k] = (A_k, R_k). Returns one
// (L, A_k, R_k) array per sweep.
template <class T>
std::vector<py::array_t<T>> reassemble(const Arr<T>& patches, const IArr& table,
                                       const std::vector<std::pair<int64_t, int64_t>>& shapes,
                                       const DArr& wa, const DArr& wr, bool wrap,
                                       int n_threads) {
    if (patches.ndim() != 4) throw std::invalid_argument("patches must be (n, fields, h, w)");
    const size_t nk = shapes.size();
    if (nk == 0) throw std::invalid_argument("no sweeps given");
    const int64_t L = patches.shape(1), h = patches.shape(2), w = patches.shape(3);
    if (wa.size() != h || wr.size() != w)
        throw std::invalid_argument("blending weights do not match the patch size");
    const Table t = check_table(table, nk);
    if (patches.shape(0) != t.n)
        throw std::invalid_argument("patches and table hold a different number of patches");

    std::vector<py::array_t<T>> results;
    std::vector<T*> dst(nk);
    std::vector<int64_t> first(nk + 1, 0);  // first work item of each sweep
    int64_t max_gate = 0;
    for (size_t k = 0; k < nk; ++k) {
        const auto [A, R] = shapes[k];
        if (A < 1 || R < 1) throw std::invalid_argument("empty sweep shape");
        results.emplace_back(std::vector<int64_t>{L, A, R});
        dst[k] = results.back().mutable_data();
        first[k + 1] = first[k] + L * A;
        max_gate = std::max(max_gate, R);
    }

    // CSR list of the (patch, patch row) pairs covering each ray of each sweep
    std::vector<int64_t> ray0(nk + 1, 0);
    for (size_t k = 0; k < nk; ++k) ray0[k + 1] = ray0[k] + shapes[k].first;
    std::vector<int64_t> offset(ray0[nk] + 1, 0);
    for (int64_t n = 0; n < t.n; ++n) {
        const int64_t k = t.rows[3 * n], a0 = t.rows[3 * n + 1];
        for (int64_t i = 0; i < h; ++i) {
            const int64_t a = source_ray(a0 + i, shapes[k].first, wrap);
            if (a >= 0) ++offset[ray0[k] + a + 1];
        }
    }
    for (size_t q = 1; q < offset.size(); ++q) offset[q] += offset[q - 1];
    std::vector<std::pair<int64_t, int64_t>> cover(offset.back());
    {
        std::vector<int64_t> fill_at(offset.begin(), offset.end() - 1);
        for (int64_t n = 0; n < t.n; ++n) {
            const int64_t k = t.rows[3 * n], a0 = t.rows[3 * n + 1];
            for (int64_t i = 0; i < h; ++i) {
                const int64_t a = source_ray(a0 + i, shapes[k].first, wrap);
                if (a >= 0) cover[fill_at[ray0[k] + a]++] = {n, i};
            }
        }
    }

    const T* src = patches.data();
    const double* pa = wa.data();
    const double* pr = wr.data();
    {
        py::gil_scoped_release release;
        const int64_t total = first[nk];
        const int nt = thread_count(n_threads, total);
        std::atomic<int64_t> next{0};
        constexpr int64_t chunk = 8;
        auto worker = [&]() {
            std::vector<double> num(max_gate), den(max_gate), value(max_gate);  // per thread
            std::vector<unsigned char> same(max_gate);
            size_t k = 0;
            for (int64_t u0; (u0 = next.fetch_add(chunk)) < total;) {
                const int64_t u1 = std::min(total, u0 + chunk);
                for (int64_t u = u0; u < u1; ++u) {
                    while (u >= first[k + 1]) ++k;  // items only increase
                    const int64_t A = shapes[k].first, R = shapes[k].second;
                    const int64_t l = (u - first[k]) / A, a = (u - first[k]) % A;
                    std::fill(num.begin(), num.begin() + R, 0.0);
                    std::fill(den.begin(), den.begin() + R, 0.0);
                    const int64_t q = ray0[k] + a;
                    for (int64_t c = offset[q]; c < offset[q + 1]; ++c) {
                        const int64_t n = cover[c].first, i = cover[c].second;
                        const int64_t r0 = t.rows[3 * n + 2];
                        const T* p = src + ((n * L + l) * h + i) * w;
                        const int64_t j0 = std::max<int64_t>(0, -r0);
                        const int64_t j1 = std::min<int64_t>(w, R - r0);
                        for (int64_t j = j0; j < j1; ++j) {
                            const double v = static_cast<double>(p[j]);
                            if (std::isnan(v)) continue;
                            const double wt = pa[i] * pr[j];
                            const int64_t g = r0 + j;
                            if (den[g] == 0.0) {
                                value[g] = v;
                                same[g] = 1;
                            } else if (v != value[g]) {
                                same[g] = 0;
                            }
                            num[g] += wt * v;
                            den[g] += wt;
                        }
                    }
                    T* out = dst[k] + (l * A + a) * R;
                    for (int64_t g = 0; g < R; ++g)
                        out[g] = !(den[g] > 0.0) ? std::numeric_limits<T>::quiet_NaN()
                                 : same[g]       ? static_cast<T>(value[g])
                                                 : static_cast<T>(num[g] / den[g]);
                }
            }
        };
        std::vector<std::thread> pool;
        for (int th = 1; th < nt; ++th) pool.emplace_back(worker);
        worker();
        for (auto& th : pool) th.join();
    }
    return results;
}

PYBIND11_MODULE(_patches, m) {
    m.doc() = "Compiled polar patch extraction and reassembly kernels for radarx.ml.";
    m.def("extract_float32", &extract<float>, py::arg("data"), py::arg("table"), py::arg("h"),
          py::arg("w"), py::arg("wrap"), py::arg("fill"), py::arg("n_threads") = 0);
    m.def("extract_float64", &extract<double>, py::arg("data"), py::arg("table"), py::arg("h"),
          py::arg("w"), py::arg("wrap"), py::arg("fill"), py::arg("n_threads") = 0);
    m.def("reassemble_float32", &reassemble<float>, py::arg("patches"), py::arg("table"),
          py::arg("shapes"), py::arg("wa"), py::arg("wr"), py::arg("wrap"),
          py::arg("n_threads") = 0);
    m.def("reassemble_float64", &reassemble<double>, py::arg("patches"), py::arg("table"),
          py::arg("shapes"), py::arg("wa"), py::arg("wr"), py::arg("wrap"),
          py::arg("n_threads") = 0);
}
