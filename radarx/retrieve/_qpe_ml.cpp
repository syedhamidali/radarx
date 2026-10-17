// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Feature matrix of the machine-learning rain rate (radarx/retrieve/qpe_ml.py).
//
// Most gates of a radar volume have no echo, so the network is only run on
// the gates where every input is finite (and a mask allows it). `compact`
// gathers those gates into a (m, F) float32 matrix and returns their flat
// positions; `scatter` writes the outputs of the network back to the flat
// positions of the gates, NaN elsewhere. Both are one threaded pass over the
// gates in blocks (compact counts the valid gates of every block first, so
// that the blocks write to disjoint parts of the matrix, in gate order).

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
#include <vector>

namespace py = pybind11;
using FArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using BArray = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;
using IArray = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;

namespace {

constexpr int64_t kBlock = 1 << 16;  // gates per block

template <class F>
void parallel_blocks(int64_t nblocks, int n_threads, F&& body) {
    if (nblocks <= 0) return;
    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    int nt = n_threads > 0 ? n_threads : static_cast<int>(hw);
    nt = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(nt, nblocks)));
    std::atomic<int64_t> next{0};
    auto worker = [&]() {
        for (;;) {
            const int64_t b = next.fetch_add(1);
            if (b >= nblocks) break;
            body(b);
        }
    };
    std::vector<std::thread> pool;
    for (int i = 1; i < nt; ++i) pool.emplace_back(worker);
    worker();
    for (auto& th : pool) th.join();
}

}  // namespace

// Gates where all columns are finite and the mask (empty: none) is non-zero.
// Returns the features (m, F) and the flat gate positions (m).
py::tuple compact_py(const std::vector<FArray>& cols, const BArray& mask, int n_threads) {
    const int64_t nf = static_cast<int64_t>(cols.size());
    if (nf < 1) throw std::invalid_argument("at least one feature column is needed");
    const int64_t n = cols[0].size();
    std::vector<const float*> c(nf);
    for (int64_t k = 0; k < nf; ++k) {
        if (cols[k].size() != n)
            throw std::invalid_argument("all feature columns must have the same size");
        c[k] = cols[k].data();
    }
    const bool use_mask = mask.size() > 0;
    if (use_mask && mask.size() != n)
        throw std::invalid_argument("the mask must have one value per gate");
    const uint8_t* pm = use_mask ? mask.data() : nullptr;
    const int64_t nblocks = (n + kBlock - 1) / kBlock;
    std::vector<int64_t> count(nblocks + 1, 0);
    auto valid = [&](int64_t i) {
        if (use_mask && pm[i] == 0) return false;
        for (int64_t k = 0; k < nf; ++k)
            if (!std::isfinite(c[k][i])) return false;
        return true;
    };
    {
        py::gil_scoped_release release;
        parallel_blocks(nblocks, n_threads, [&](int64_t b) {
            const int64_t i0 = b * kBlock, i1 = std::min(n, i0 + kBlock);
            int64_t m = 0;
            for (int64_t i = i0; i < i1; ++i) m += valid(i) ? 1 : 0;
            count[b + 1] = m;
        });
    }
    for (int64_t b = 0; b < nblocks; ++b) count[b + 1] += count[b];
    const int64_t total = count[nblocks];
    py::array_t<float> feats({total, nf});
    py::array_t<int64_t> index(total);
    float* pf = feats.mutable_data();
    int64_t* pi = index.mutable_data();
    {
        py::gil_scoped_release release;
        parallel_blocks(nblocks, n_threads, [&](int64_t b) {
            const int64_t i0 = b * kBlock, i1 = std::min(n, i0 + kBlock);
            int64_t j = count[b];
            for (int64_t i = i0; i < i1; ++i) {
                if (!valid(i)) continue;
                for (int64_t k = 0; k < nf; ++k) pf[j * nf + k] = c[k][i];
                pi[j] = i;
                ++j;
            }
        });
    }
    return py::make_tuple(feats, index);
}

// values (m, K) at the flat positions `index` into an array (K, n), NaN elsewhere.
py::array_t<float> scatter_py(const IArray& index, const FArray& values, int64_t n,
                              int n_threads) {
    if (values.ndim() != 2 || values.shape(0) != index.size())
        throw std::invalid_argument("values must have shape (len(index), K)");
    if (n < 0) throw std::invalid_argument("n must not be negative");
    const int64_t m = index.size(), nk = values.shape(1);
    py::array_t<float> out({nk, n});
    float* po = out.mutable_data();
    const int64_t* pi = index.data();
    const float* pv = values.data();
    for (int64_t i = 0; i < m; ++i)
        if (pi[i] < 0 || pi[i] >= n) throw std::invalid_argument("index out of range");
    const int64_t nblocks = (n + kBlock - 1) / kBlock;
    {
        py::gil_scoped_release release;
        const float nan = std::numeric_limits<float>::quiet_NaN();
        parallel_blocks(nblocks, n_threads, [&](int64_t b) {
            const int64_t i0 = b * kBlock, i1 = std::min(n, i0 + kBlock);
            for (int64_t k = 0; k < nk; ++k) std::fill(po + k * n + i0, po + k * n + i1, nan);
        });
        const int64_t mblocks = (m + kBlock - 1) / kBlock;
        parallel_blocks(mblocks, n_threads, [&](int64_t b) {
            const int64_t j0 = b * kBlock, j1 = std::min(m, j0 + kBlock);
            for (int64_t j = j0; j < j1; ++j)
                for (int64_t k = 0; k < nk; ++k) po[k * n + pi[j]] = pv[j * nk + k];
        });
    }
    return out;
}

PYBIND11_MODULE(_qpe_ml, m) {
    m.doc() = "Feature matrix of the machine-learning rain rate";
    m.def("compact", &compact_py, py::arg("columns"), py::arg("mask"), py::arg("n_threads") = 0);
    m.def("scatter", &scatter_py, py::arg("index"), py::arg("values"), py::arg("n"),
          py::arg("n_threads") = 0);
}
