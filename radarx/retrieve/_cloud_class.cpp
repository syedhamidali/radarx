// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Rule-based classification of the radar echoes of a cloud radar by reflectivity,
// temperature and, where given, the linear depolarization ratio and the dual-frequency
// ratio. The rules and their order are described in the docstring of
// radarx/retrieve/cloud_classification.py; the NumPy reference there follows the
// same order. Every gate is independent; threads take blocks of gates from an
// atomic counter.
//
// Codes: 0 no echo, 1 cloud droplets, 2 drizzle or rain, 3 ice, 4 melting ice,
// 5 insects, 6 large ice (aggregates), 7 echo of unknown phase.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

namespace py = pybind11;
using DArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using I8Array = py::array_t<int8_t, py::array::c_style>;

namespace {

constexpr int64_t kBlock = 8192;

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
            const int64_t i0 = next.fetch_add(kBlock);
            if (i0 >= total) break;
            body(i0, std::min(total, i0 + kBlock));
        }
    };
    std::vector<std::thread> pool;
    for (int i = 1; i < nt; ++i) pool.emplace_back(worker);
    worker();
    for (auto& th : pool) th.join();
}

// params: min_dbz, melt_lo, melt_hi, min_drizzle_dbz, min_insect_ldr, max_insect_dbz, min_large_dfr
// (a NaN test is skipped); melt_lo and melt_hi are the bounds of the melting zone in K
I8Array classify_py(const DArray& dbz, const DArray& temperature, const DArray& ldr, const DArray& dfr,
                    const DArray& params, int n_threads) {
    if (dbz.ndim() != 1 || temperature.ndim() != 1 || ldr.ndim() != 1 || dfr.ndim() != 1)
        throw std::invalid_argument("the fields must be 1-D");
    const int64_t n = dbz.shape(0);
    if (temperature.shape(0) != n || ldr.shape(0) != n || dfr.shape(0) != n)
        throw std::invalid_argument("the fields must have the same length");
    if (params.ndim() != 1 || params.shape(0) != 7) throw std::invalid_argument("params must have 7 values");
    const double* p = params.data();
    const double min_dbz = p[0], melt_lo = p[1], melt_hi = p[2], drizzle = p[3], insect_ldr = p[4],
                 insect_dbz = p[5], large_dfr = p[6];
    I8Array out(std::vector<py::ssize_t>{static_cast<py::ssize_t>(n)});
    const double *pz = dbz.data(), *pt = temperature.data(), *pl = ldr.data(), *pd = dfr.data();
    int8_t* po = out.mutable_data();
    {
        py::gil_scoped_release release;
        parallel_blocks(n, n_threads, [&](int64_t i0, int64_t i1) {
            for (int64_t i = i0; i < i1; ++i) {
                const double z = pz[i], t = pt[i], l = pl[i], d = pd[i];
                if (!std::isfinite(z) || (std::isfinite(min_dbz) && z < min_dbz)) {
                    po[i] = 0;
                    continue;
                }
                if (!std::isfinite(t)) {
                    po[i] = 7;
                    continue;
                }
                if (t >= melt_lo && t <= melt_hi) {
                    po[i] = 4;
                    continue;
                }
                if (t > melt_hi) {
                    if (std::isfinite(l) && std::isfinite(insect_ldr) && l >= insect_ldr && z < insect_dbz) {
                        po[i] = 5;
                    } else if (z >= drizzle) {
                        po[i] = 2;
                    } else {
                        po[i] = 1;
                    }
                    continue;
                }
                po[i] = (std::isfinite(d) && std::isfinite(large_dfr) && d >= large_dfr) ? 6 : 3;
            }
        });
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(_cloud_class, m) {
    m.doc() = "Rule-based classification of cloud radar echoes";
    m.def("classify", &classify_py, py::arg("dbz"), py::arg("temperature"), py::arg("ldr"), py::arg("dfr"),
          py::arg("params"), py::arg("n_threads") = 0);
}
