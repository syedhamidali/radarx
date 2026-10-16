// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Trapping layers of profiles of the radio refractivity (radarx/retrieve/ducts.py), after Bean and
// Dutton (1966): a layer where the modified refractivity M = N + 1e6 h / a decreases with height
// (dM/dh < 0, Table 1.9 and Section 4.4.4) traps rays whose elevation is below the angle of
// penetration, cos(theta_p) = n_top r_top / (n r) of the launch point (Eq. 4.17). Profiles are
// independent and run on all cores (dynamic chunking with an atomic counter). Every step follows
// radarx/retrieve/_ducts_numpy.py in the same order.

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

const int kFields = 7;

template <class F>
void parallel_for(int64_t n, int n_threads, F&& f) {
    int nt = n_threads > 0 ? n_threads : static_cast<int>(std::thread::hardware_concurrency());
    nt = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(nt, n)));
    if (nt == 1) {
        for (int64_t i = 0; i < n; ++i) f(i);
        return;
    }
    std::atomic<int64_t> next{0};
    std::vector<std::thread> pool;
    for (int t = 0; t < nt; ++t)
        pool.emplace_back([&] {
            for (int64_t i = next++; i < n; i = next++) f(i);
        });
    for (auto& t : pool) t.join();
}

double interp(const std::vector<double>& h, const std::vector<double>& n, double z) {
    if (z <= h.front()) return n.front();
    if (z >= h.back()) return n.back();
    const size_t i = std::upper_bound(h.begin(), h.end(), z) - h.begin() - 1;
    const double w = (z - h[i]) / (h[i + 1] - h[i]);
    return n[i] + w * (n[i + 1] - n[i]);
}

py::tuple trapping_layers(const DArray& height, const DArray& refractivity, double radar_height,
                          double earth_radius, int64_t max_layers, double min_deficit, int n_threads) {
    if (height.ndim() != 1 || refractivity.ndim() != 2) throw std::invalid_argument("bad array dimensions");
    if (refractivity.shape(1) != height.shape(0))
        throw std::invalid_argument("refractivity must have the levels of height");
    if (max_layers < 1) throw std::invalid_argument("max_layers must be at least 1");
    if (!(earth_radius > 0.0)) throw std::invalid_argument("earth_radius must be positive");
    const int64_t nlev = height.shape(0), nprof = refractivity.shape(0);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    py::array_t<double> fields({nprof, max_layers, static_cast<int64_t>(kFields)});
    py::array_t<int64_t> count(std::vector<int64_t>{nprof});
    double* pf = fields.mutable_data();
    int64_t* pc = count.mutable_data();
    const double* ph = height.data();
    const double* pn = refractivity.data();
    const double scale = 1.0e6 / earth_radius;
    {
        py::gil_scoped_release release;
        parallel_for(nprof, n_threads, [&](int64_t p) {
            double* out = pf + p * max_layers * kFields;
            for (int64_t k = 0; k < max_layers * kFields; ++k) out[k] = nan;
            pc[p] = 0;
            std::vector<double> h, n, m;
            for (int64_t k = 0; k < nlev; ++k) {
                const double v = pn[p * nlev + k];
                if (std::isfinite(v)) {
                    h.push_back(ph[k]);
                    n.push_back(v);
                    m.push_back(v + ph[k] * scale);
                }
            }
            if (h.size() < 2) return;
            size_t i = 0;
            while (i + 1 < h.size()) {
                if (!(m[i + 1] < m[i])) {
                    ++i;
                    continue;
                }
                const size_t b = i;
                while (i + 1 < h.size() && m[i + 1] < m[i]) ++i;
                const size_t t = i;
                const double deficit = m[b] - m[t];
                if (!(deficit > min_deficit)) continue;
                if (pc[p] < max_layers) {
                    const double nb = 1.0 + 1.0e-6 * n[b], nt = 1.0 + 1.0e-6 * n[t];
                    const double rb = earth_radius + h[b], rt = earth_radius + h[t];
                    const double angle = std::acos(std::min(nt * rt / (nb * rb), 1.0));
                    double radar = nan;
                    if (std::isfinite(radar_height) && h[b] <= radar_height && radar_height <= h[t]) {
                        const double nr = 1.0 + 1.0e-6 * interp(h, n, radar_height);
                        const double rr = earth_radius + radar_height;
                        radar = std::acos(std::min(nt * rt / (nr * rr), 1.0));
                    }
                    double* o = out + pc[p] * kFields;
                    o[0] = h[b];
                    o[1] = h[t];
                    o[2] = h[t] - h[b];
                    o[3] = deficit;
                    o[4] = angle;
                    o[5] = radar;
                    o[6] = b == 0 ? 1.0 : 0.0;
                }
                ++pc[p];
            }
        });
    }
    return py::make_tuple(fields, count);
}

}  // namespace

PYBIND11_MODULE(_ducts, m) {
    m.doc() = "Trapping layers of refractivity profiles";
    m.def("trapping_layers", &trapping_layers, py::arg("height"), py::arg("refractivity"),
          py::arg("radar_height"), py::arg("earth_radius"), py::arg("max_layers"), py::arg("min_deficit"),
          py::arg("n_threads") = 0);
}
