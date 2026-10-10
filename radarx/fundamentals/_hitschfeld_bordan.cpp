// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Two-way path-integrated attenuation of a reflectivity ray by the
// Hitschfeld-Bordan solution.
//
// With the specific attenuation k = a Z^b (dB km-1, Z in mm6 m-3) and the
// measured reflectivity Zm = Z 10^(-0.2 A) (A = the one-way integral of k in
// dB), the true reflectivity is Z = Zm (1 - c S)^(-1/b), with
// c = 0.2 b ln(10) and S the range integral of a Zm^b in dB. The correction in
// dB is therefore PIA = -(10 / b) log10(1 - c S). S is taken to the centre of
// each gate (the gates before it in full, half of the gate itself), a gate
// without a value adds nothing, and 1 - c S is not allowed below
// 10^(-b max_pia / 10), which caps PIA at max_pia dB where the solution is
// unstable. Details and the radarx choices are in the docstring of
// radarx/fundamentals/reflectivity.py. Every step follows the NumPy
// reference in that module, in the same order of operations.
//
// Every ray is independent; threads take blocks of rays from an atomic
// counter.

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
constexpr int64_t kBlock = 64;  // rays per scheduling block

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

DArray pia_py(const DArray& dbz, const DArray& dr_km, double a, double b, double max_pia,
              int n_threads) {
    if (dbz.ndim() != 2) throw std::invalid_argument("dbz must be 2-D (rays, gates)");
    if (dr_km.ndim() != 1) throw std::invalid_argument("dr_km must be 1-D");
    const int64_t nr = dbz.shape(0), ng = dbz.shape(1);
    if (dr_km.shape(0) != ng) throw std::invalid_argument("dr_km must have one value per gate");
    DArray out({nr, ng});
    const double* zp = dbz.data();
    const double* dp = dr_km.data();
    double* o = out.mutable_data();
    const double c = 0.2 * b * std::log(10.0);
    const double floor_value = std::pow(10.0, -b * max_pia / 10.0);
    py::gil_scoped_release release;
    parallel_blocks(nr, n_threads, [&](int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; ++r) {
            const double* z = zp + r * ng;
            double* p = o + r * ng;
            double before = 0.0;  // integral of a Zm^b up to the start of the gate, dB
            for (int64_t g = 0; g < ng; ++g) {
                if (std::isnan(z[g])) {
                    p[g] = kNaN;
                    continue;
                }
                const double term = a * std::pow(std::pow(10.0, z[g] / 10.0), b) * dp[g];
                const double centre = before + 0.5 * term;
                const double rest = std::max(1.0 - c * centre, floor_value);
                p[g] = -(10.0 / b) * std::log10(rest);
                before += term;
            }
        }
    });
    return out;
}

}  // namespace

PYBIND11_MODULE(_hitschfeld_bordan, m) {
    m.doc() = "Hitschfeld-Bordan path-integrated attenuation kernel";
    m.def("pia", &pia_py, py::arg("dbz"), py::arg("dr_km"), py::arg("a"), py::arg("b"),
          py::arg("max_pia"), py::arg("n_threads") = 0);
}
