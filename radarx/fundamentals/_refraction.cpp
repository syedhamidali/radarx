// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Ray tracing in a spherically stratified atmosphere (radarx/fundamentals/refraction.py).
//
// A ray leaves the radar at the height h0 with the elevation angle e0 above the local
// horizontal. With r = a + h the distance from the centre of the Earth, n(h) the refractive
// index, theta the elevation of the ray against the local horizontal, phi the central angle
// and s the path length, geometry and Snell's law n r cos(theta) = const (Bean and Dutton
// 1966, Eq. 3.1) give
//
//     dr/ds = sin(theta),   dphi/ds = cos(theta) / r,
//     dtheta/ds = cos(theta) (1 / r + n'(h) / n(h)),
//
// where the curvature of the ray relative to the local horizontal is the curvature of the Earth
// term 1/r plus the ray curvature -n'/n cos(theta) of Bean and Dutton Eq. (1.27), written with the
// sign of theta increasing upward. The system is integrated with the classical fourth order
// Runge-Kutta method from gate to gate; between two gates the path is cut in equal steps of at
// most `step`, and a step is shortened to end where the ray reaches the next level of the profile,
// so that no step spans a change of the gradient (the refractive index is n = 1 + 1e-6 N with N
// piecewise linear in h; above the last level and below the first it is constant). A step is
// not shortened below 1 mm. Rays at the same elevation share one path, so
// the unit of work is one elevation. Same operations, in the same order, as
// radarx/fundamentals/_refraction_numpy.py.

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
constexpr double kPi = 3.14159265358979323846;
constexpr double kDeg = kPi / 180.0;

struct Profile {
    const double* h;
    const double* n;  // refractive index n = 1 + 1e-6 N
    int64_t nl;
};

// The layer a step runs in: n(h) = na + dn (h - ha) there. Piecewise linear between the levels,
// constant below the first and from the last level on. A ray that starts exactly on a level and
// travels downward is in the layer below it. All stages of a Runge-Kutta step use the layer of
// its start, so that a step that ends on a level is not given the gradient of the next layer.
struct Layer {
    double ha, na, dn;
};

inline Layer layer_of(const Profile& p, double h, double sin_th) {
    int64_t i = std::upper_bound(p.h, p.h + p.nl, h) - p.h - 1;  // last level at or below h
    if (sin_th < 0.0 && i >= 0 && h == p.h[i]) --i;
    if (i < 0) return {p.h[0], p.n[0], 0.0};
    if (i >= p.nl - 1) return {p.h[p.nl - 1], p.n[p.nl - 1], 0.0};
    return {p.h[i], p.n[i], (p.n[i + 1] - p.n[i]) / (p.h[i + 1] - p.h[i])};
}

struct Y {
    double r, phi, th;
};

inline Y deriv(const Layer& l, double a, const Y& y) {
    const double n = l.na + l.dn * (y.r - a - l.ha);
    const double c = std::cos(y.th);
    return {std::sin(y.th), c / y.r, c * (1.0 / y.r + l.dn / n)};
}

inline Y step_rk4(const Profile& p, double a, const Y& y, double ds) {
    const Layer l = layer_of(p, y.r - a, std::sin(y.th));
    const Y k1 = deriv(l, a, y);
    const Y k2 = deriv(l, a, {y.r + 0.5 * ds * k1.r, y.phi + 0.5 * ds * k1.phi, y.th + 0.5 * ds * k1.th});
    const Y k3 = deriv(l, a, {y.r + 0.5 * ds * k2.r, y.phi + 0.5 * ds * k2.phi, y.th + 0.5 * ds * k2.th});
    const Y k4 = deriv(l, a, {y.r + ds * k3.r, y.phi + ds * k3.phi, y.th + ds * k3.th});
    return {y.r + ds / 6.0 * (k1.r + 2.0 * k2.r + 2.0 * k3.r + k4.r),
            y.phi + ds / 6.0 * (k1.phi + 2.0 * k2.phi + 2.0 * k3.phi + k4.phi),
            y.th + ds / 6.0 * (k1.th + 2.0 * k2.th + 2.0 * k3.th + k4.th)};
}

// Path length to the next level of the profile in the direction the ray travels; infinity if
// there is none or the ray runs level.
inline double cut_length(const Profile& p, double h, double sin_th) {
    if (std::fabs(sin_th) < 1e-12) return std::numeric_limits<double>::infinity();
    const int64_t i = std::upper_bound(p.h, p.h + p.nl, h) - p.h - 1;  // last level at or below h
    double target;
    if (sin_th > 0.0) {
        if (i + 1 > p.nl - 1) return std::numeric_limits<double>::infinity();
        target = p.h[i + 1];
    } else if (i >= 0 && h > p.h[i]) {
        target = p.h[i];
    } else if (i - 1 >= 0) {
        target = p.h[i - 1];
    } else {
        return std::numeric_limits<double>::infinity();
    }
    return (target - h) / sin_th;
}

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

// rng: (ng,) ascending path lengths [m]; elev: (ne,) degrees; ph, pn: profile heights [m above
// sea level] and refractive index. Returns (ne, ng, 3): height above sea level [m], distance on
// the ground at sea level a * phi [m] and elevation of the ray at the gate [degrees].
py::array_t<double> trace(const DArray& rng, const DArray& elev, const DArray& ph, const DArray& pn,
                          double radar_height, double earth_radius, double step, int n_threads) {
    if (rng.ndim() != 1 || elev.ndim() != 1 || ph.ndim() != 1 || pn.ndim() != 1)
        throw std::invalid_argument("rng, elev, h and n must be one-dimensional");
    if (ph.size() != pn.size() || ph.size() < 1)
        throw std::invalid_argument("the profile needs matching heights and indices");
    if (!(step > 0.0)) throw std::invalid_argument("step must be positive");
    const int64_t ng = rng.size(), ne = elev.size();
    const double* rp = rng.data();
    const double* ep = elev.data();
    for (int64_t k = 1; k < ng; ++k)
        if (!(rp[k] >= rp[k - 1])) throw std::invalid_argument("rng must not decrease");
    const Profile prof{ph.data(), pn.data(), ph.size()};
    py::array_t<double> out({ne, ng, static_cast<int64_t>(3)});
    double* po = out.mutable_data();
    {
        py::gil_scoped_release release;
        parallel_for(ne, n_threads, [&](int64_t q) {
            double* row = po + q * ng * 3;
            if (!std::isfinite(ep[q])) {
                for (int64_t k = 0; k < 3 * ng; ++k) row[k] = kNaN;
                return;
            }
            Y y{earth_radius + radar_height, 0.0, ep[q] * kDeg};
            double s = 0.0;
            for (int64_t k = 0; k < ng; ++k) {
                const double len = rp[k] - s;
                if (len > 0.0) {
                    const int64_t m = static_cast<int64_t>(std::ceil(len / step));
                    const double nominal = len / static_cast<double>(m);
                    double rem = len;
                    while (rem > 0.0) {
                        double ds = nominal;
                        const double sin_th = std::sin(y.th);
                        const double cut = cut_length(prof, y.r - earth_radius, sin_th);
                        // not shorter than 1 mm, nor than a path that changes r by 1e-8 m
                        // (r is known to 1e-9 m), so that a step always crosses the level
                        const double least = std::max(1e-3, 1e-8 / std::max(std::fabs(sin_th), 1e-300));
                        if (std::max(cut, least) < ds) ds = std::max(cut, least);
                        const bool last = ds >= rem;
                        if (last) ds = rem;
                        y = step_rk4(prof, earth_radius, y, ds);
                        rem = last ? 0.0 : rem - ds;
                    }
                    s = rp[k];
                }
                row[3 * k] = y.r - earth_radius;
                row[3 * k + 1] = earth_radius * y.phi;
                row[3 * k + 2] = y.th / kDeg;
            }
        });
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(_refraction, m) {
    m.doc() = "Ray tracing in a spherically stratified atmosphere";
    m.def("trace", &trace, py::arg("rng"), py::arg("elev"), py::arg("h"), py::arg("n"),
          py::arg("radar_height"), py::arg("earth_radius"), py::arg("step") = 100.0,
          py::arg("n_threads") = 0);
}
