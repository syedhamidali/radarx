// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Horizontal-last ordered linear interpolation of a radar volume, the second
// half of the mixed-order interpolation of Collis, Protat and Chung (2010, J.
// Atmos. Oceanic Technol. 27, 1239-1246).
//
// The cone gridding of _cone.cpp interpolates inside each sweep first and
// between the sweeps in height last ("vertical last"): at the ground position
// of an output point every sweep gives a value at its own beam height, and the
// values are interpolated in height. Here the order is reversed ("horizontal
// last"): for an output point at ground distance s, azimuth a and height z,
// each sweep gives the value at the point of its beam that is at height z (the
// beam crosses z at the ground distance s_k(z), which is a function of the
// sweep and the height alone), with the azimuth interpolated between the
// rays; the values of the two sweeps whose crossing points enclose s are then
// interpolated linearly in ground distance. A field that does not change
// horizontally is reproduced exactly however steeply it changes in the
// vertical, which is what the vertical-last order cannot do between beams that
// are far apart; the price is a horizontal smoothing over the distance between
// the two crossing points, which grows with elevation. Points that are not
// enclosed by the crossing points of two sweeps (below the lowest beam, above
// the highest, beyond the range of a sweep) stay empty. The weighting of the
// two orders of interpolation is done by the caller.
//
// The beam geometry (effective Earth radius k_e * a) and the handling of rays
// and gates are those of _cone.cpp, repeated here so that the two kernels are
// independent. max_gap and min_weight are radarx choices.

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
using DArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kPi = 3.14159265358979323846;  // M_PI is not standard (MSVC)
constexpr double kDeg = kPi / 180.0;

struct Sweep {
    const double* data = nullptr;  // (nray, ngate), row-major
    int64_t nray = 0, ngate = 0;
    std::vector<double> range;     // gate ranges [m]
    std::vector<double> ground;    // ground distance of each gate [m]
    std::vector<double> sin_el;    // per ray
    std::vector<double> az_ext;    // sorted ray azimuths, wrapped at both ends
    std::vector<int64_t> ray_ext;  // ray index for each az_ext entry
    double spacing = 0.0;          // median azimuth step [deg]
    double sin_med = 0.0, cos_med = 1.0;  // median elevation of the sweep
};

inline double beam_height(double r, double sin_el, double R, double sr) {
    return std::sqrt(2.0 * sin_el * r * sr + r * r + sr * sr) - R;
}

double median(std::vector<double> v) {
    const size_t n = v.size(), h = n / 2;
    std::nth_element(v.begin(), v.begin() + h, v.end());
    double upper = v[h];
    if (n % 2) return upper;
    double lower = *std::max_element(v.begin(), v.begin() + h);
    return 0.5 * (lower + upper);
}

// Largest i with a[i] <= x, clipped to [0, n - 2].
inline int64_t bisect(const double* a, int64_t n, double x) {
    int64_t lo = 0, hi = n - 1;
    while (hi - lo > 1) {
        int64_t mid = (lo + hi) >> 1;
        if (a[mid] <= x) lo = mid; else hi = mid;
    }
    return lo;
}

void set_geometry(Sweep& sw, const double* elevation, double R, double sr) {
    std::vector<double> el(elevation, elevation + sw.nray);
    sw.sin_el.resize(sw.nray);
    for (int64_t q = 0; q < sw.nray; ++q) sw.sin_el[q] = std::sin(el[q] * kDeg);
    const double el_med = median(el) * kDeg;
    const double sin_m = std::sin(el_med), cos_m = std::cos(el_med);
    sw.sin_med = sin_m;
    sw.cos_med = cos_m;
    sw.ground.resize(sw.ngate);
    for (int64_t g = 0; g < sw.ngate; ++g) {
        const double r = sw.range[g];
        sw.ground[g] = R * std::asin(r * cos_m / (R + beam_height(r, sin_m, R, sr)));
    }
}

void set_azimuth_order(Sweep& sw, const double* azimuth) {
    std::vector<std::pair<double, int64_t>> rays(sw.nray);
    for (int64_t q = 0; q < sw.nray; ++q) {
        double a = std::fmod(azimuth[q], 360.0);
        rays[q] = {a < 0 ? a + 360.0 : a, q};
    }
    std::stable_sort(rays.begin(), rays.end(),
                     [](const auto& p, const auto& q) { return p.first < q.first; });
    std::vector<double> az;
    std::vector<int64_t> idx;
    for (const auto& p : rays) {
        if (!az.empty() && p.first - az.back() <= 1e-6) continue;
        az.push_back(p.first);
        idx.push_back(p.second);
    }
    if (az.size() < 3) throw std::invalid_argument("sweep needs at least 3 distinct azimuths");
    std::vector<double> steps(az.size() - 1);
    for (size_t q = 0; q + 1 < az.size(); ++q) steps[q] = az[q + 1] - az[q];
    sw.spacing = median(steps);
    sw.az_ext.assign(1, az.back() - 360.0);
    sw.az_ext.insert(sw.az_ext.end(), az.begin(), az.end());
    sw.az_ext.push_back(az.front() + 360.0);
    sw.ray_ext.assign(1, idx.back());
    sw.ray_ext.insert(sw.ray_ext.end(), idx.begin(), idx.end());
    sw.ray_ext.push_back(idx.front());
}

Sweep prepare(const DArray& data, const DArray& azimuth, const DArray& elevation,
              const DArray& range, double R, double sr) {
    if (data.ndim() != 2) throw std::invalid_argument("sweep data must be 2-D (ray, gate)");
    Sweep sw;
    sw.nray = data.shape(0);
    sw.ngate = data.shape(1);
    if (azimuth.size() != sw.nray || elevation.size() != sw.nray || range.size() != sw.ngate)
        throw std::invalid_argument("azimuth/elevation/range do not match sweep data shape");
    if (sw.nray < 3 || sw.ngate < 2) throw std::invalid_argument("sweep too small");
    sw.data = data.data();
    sw.range.assign(range.data(), range.data() + sw.ngate);
    set_geometry(sw, elevation.data(), R, sr);
    set_azimuth_order(sw, azimuth.data());
    return sw;
}

// Ground distance at which the beam of the sweep (its median elevation) is at
// height z, or NaN if it never is.
inline double crossing(const Sweep& sw, double z, double R, double sr) {
    const double rad = sr * sr * (sw.sin_med * sw.sin_med - 1.0) + (R + z) * (R + z);
    if (!(rad >= 0.0)) return kNaN;
    const double r = -sr * sw.sin_med + std::sqrt(rad);
    if (!(r > 0.0)) return kNaN;
    return R * std::asin(std::min(1.0, r * sw.cos_med / (R + z)));
}

// Value of the sweep at ground distance sc and azimuth ac: the bilinear
// interpolation of the four surrounding gates, NaN if the point is outside the
// sweep, between rays that are too far apart, or has too little valid weight.
double value_at(const Sweep& sw, double sc, double ac, double max_gap, double min_weight) {
    const int64_t ng = sw.ngate;
    if (!(sc >= sw.ground[0] && sc <= sw.ground[ng - 1])) return kNaN;  // also NaN
    const int64_t i = bisect(sw.ground.data(), ng, sc);
    const double fr = (sc - sw.ground[i]) / (sw.ground[i + 1] - sw.ground[i]);
    const int64_t na = static_cast<int64_t>(sw.az_ext.size());
    const int64_t j = bisect(sw.az_ext.data(), na, ac);
    const double da = sw.az_ext[j + 1] - sw.az_ext[j];
    if (da > max_gap * sw.spacing) return kNaN;
    const double fa = (ac - sw.az_ext[j]) / da;
    const int64_t r0 = sw.ray_ext[j], r1 = sw.ray_ext[j + 1];
    const int64_t o0 = r0 * ng + i, o1 = r1 * ng + i;
    const double w[4] = {(1 - fa) * (1 - fr), (1 - fa) * fr, fa * (1 - fr), fa * fr};
    const double v[4] = {sw.data[o0], sw.data[o0 + 1], sw.data[o1], sw.data[o1 + 1]};
    double num = 0.0, den = 0.0;
    for (int q = 0; q < 4; ++q) {
        if (!std::isnan(v[q])) {
            num += w[q] * v[q];
            den += w[q];
        }
    }
    return den >= min_weight ? num / den : kNaN;
}

// cross holds the crossing ground distance of every sweep at every level
// (nz rows of nk), which does not depend on the column.
void mixed_column(const std::vector<Sweep>& sweeps, double sc, double ac, int64_t nz,
                  double max_gap, double min_weight, const std::vector<double>& cross_table,
                  float* out, int64_t stride) {
    const size_t nk = sweeps.size();
    for (int64_t iz = 0; iz < nz; ++iz) {
        double res = kNaN;
        const double* cross = cross_table.data() + static_cast<size_t>(iz) * nk;
        for (size_t k = 0; k + 1 < nk; ++k) {
            const double s0 = cross[k], s1 = cross[k + 1];
            if (!(std::fabs(s1 - s0) > 1e-9)) continue;  // also NaN
            if ((sc - s0) * (sc - s1) > 0.0) continue;    // not enclosed
            const double v0 = value_at(sweeps[k], s0, ac, max_gap, min_weight);
            const double v1 = value_at(sweeps[k + 1], s1, ac, max_gap, min_weight);
            if (!std::isnan(v0) && !std::isnan(v1)) res = v0 + (sc - s0) / (s1 - s0) * (v1 - v0);
            break;
        }
        out[iz * stride] = static_cast<float>(res);
    }
}

int thread_count(int n_threads) {
    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    return n_threads > 0 ? n_threads : static_cast<int>(hw);
}

}  // namespace

// Horizontal-last interpolation of one field of one radar onto columns given by
// their ground distance and azimuth from the radar; sweeps sorted by elevation,
// lowest first. The result is float32 on (nz,) + ground.shape.
py::array_t<float> mixed_columns(const DArray& ground, const DArray& column_azimuth,
                                 const DArray& z, const std::vector<DArray>& data,
                                 const std::vector<DArray>& azimuth,
                                 const std::vector<DArray>& elevation,
                                 const std::vector<DArray>& range, double site_altitude,
                                 double earth_radius, double max_gap, double min_weight,
                                 int n_threads, double k_e) {
    const size_t nk = data.size();
    if (nk < 1 || azimuth.size() != nk || elevation.size() != nk || range.size() != nk)
        throw std::invalid_argument("need the same number of data/azimuth/elevation/range arrays");
    if (ground.size() != column_azimuth.size())
        throw std::invalid_argument("ground and column_azimuth need the same shape");
    const double R = earth_radius * k_e;
    const double sr = R + site_altitude;
    std::vector<Sweep> sweeps;
    sweeps.reserve(nk);
    for (size_t k = 0; k < nk; ++k)
        sweeps.push_back(prepare(data[k], azimuth[k], elevation[k], range[k], R, sr));

    const int64_t ncol = ground.size();
    const int64_t nz = z.size();
    std::vector<py::ssize_t> shape(ground.shape(), ground.shape() + ground.ndim());
    shape.insert(shape.begin(), static_cast<py::ssize_t>(nz));
    py::array_t<float> result(shape);
    float* out = result.mutable_data();
    const double* pg = ground.data();
    const double* pa = column_azimuth.data();
    const double* pz = z.data();

    std::vector<double> cross_table(static_cast<size_t>(nz) * nk);
    for (int64_t iz = 0; iz < nz; ++iz)
        for (size_t k = 0; k < nk; ++k)
            cross_table[static_cast<size_t>(iz) * nk + k] = crossing(sweeps[k], pz[iz], R, sr);

    {
        py::gil_scoped_release release;
        const int nt = thread_count(n_threads);
        const int64_t block = 1024;
        std::atomic<int64_t> next{0};
        auto worker = [&]() {
            for (;;) {
                const int64_t c0 = next.fetch_add(block);
                if (c0 >= ncol) break;
                const int64_t c1 = std::min(ncol, c0 + block);
                for (int64_t c = c0; c < c1; ++c) {
                    double ac = std::fmod(pa[c], 360.0);
                    if (ac < 0) ac += 360.0;
                    mixed_column(sweeps, pg[c], ac, nz, max_gap, min_weight, cross_table, out + c,
                                 ncol);
                }
            }
        };
        std::vector<std::thread> pool;
        for (int t = 1; t < nt; ++t) pool.emplace_back(worker);
        worker();
        for (auto& th : pool) th.join();
    }
    return result;
}

PYBIND11_MODULE(_mixed, m) {
    m.doc() = "Horizontal-last cone interpolation kernel for radarx.";
    m.def("mixed_columns", &mixed_columns, py::arg("ground"), py::arg("column_azimuth"),
          py::arg("z"), py::arg("data"), py::arg("azimuth"), py::arg("elevation"),
          py::arg("range"), py::arg("site_altitude"), py::arg("earth_radius") = 6371000.0,
          py::arg("max_gap") = 2.0, py::arg("min_weight") = 0.5, py::arg("n_threads") = 0,
          py::arg("k_e") = 4.0 / 3.0);
}
