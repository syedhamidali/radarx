// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Cone gridding kernel.
//
// Every output column (x, y) is mapped to ground distance and azimuth. On each
// sweep (a cone of constant elevation) the value and beam height at that
// column are interpolated bilinearly from the four surrounding gates, using
// the measured ray azimuths and elevations. Each output level is then
// interpolated linearly in height between the two cones that bracket it.
// Levels below the lowest or above the highest cone, or between cones where
// one has no value, stay NaN (optionally the lowest cone fills below itself).
//
// Beam geometry follows xradar.georeference.antenna_to_cartesian (4/3 Earth).

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
};

// Beam height above sea level (4/3 Earth), as in xradar: sr = R + site altitude.
inline double beam_height(double r, double sin_el, double R, double sr) {
    return std::sqrt(2.0 * sin_el * r * sr + r * r + sr * sr) - R;
}

// Median as numpy computes it (mean of the two middle values for even n).
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

// Per-ray sin(elevation), and each gate's ground distance at the median elevation.
void set_geometry(Sweep& sw, const double* elevation, double R, double sr) {
    std::vector<double> el(elevation, elevation + sw.nray);
    sw.sin_el.resize(sw.nray);
    for (int64_t q = 0; q < sw.nray; ++q) sw.sin_el[q] = std::sin(el[q] * kDeg);
    const double el_med = median(el) * kDeg;
    const double sin_m = std::sin(el_med), cos_m = std::cos(el_med);
    sw.ground.resize(sw.ngate);
    for (int64_t g = 0; g < sw.ngate; ++g) {
        const double r = sw.range[g];
        sw.ground[g] = R * std::asin(r * cos_m / (R + beam_height(r, sin_m, R, sr)));
    }
}

// Rays sorted by azimuth, duplicates dropped, wrapped at both ends.
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

}  // namespace

// Grid one field from sweeps sorted by elevation (lowest first).
py::array_t<float> grid_cones(const DArray& x, const DArray& y, const DArray& z,
                               const std::vector<DArray>& data,
                               const std::vector<DArray>& azimuth,
                               const std::vector<DArray>& elevation,
                               const std::vector<DArray>& range, double site_altitude,
                               double earth_radius, double max_gap, double min_weight,
                               bool fill_below, int n_threads) {
    const size_t nk = data.size();
    if (nk < 1 || azimuth.size() != nk || elevation.size() != nk || range.size() != nk)
        throw std::invalid_argument("need the same number of data/azimuth/elevation/range arrays");
    const double R = earth_radius * 4.0 / 3.0;
    const double sr = R + site_altitude;

    std::vector<Sweep> sweeps;
    sweeps.reserve(nk);
    for (size_t k = 0; k < nk; ++k)
        sweeps.push_back(prepare(data[k], azimuth[k], elevation[k], range[k], R, sr));

    const int64_t nx = x.size(), ny = y.size(), nz = z.size();
    const int64_t ncol = nx * ny;
    py::array_t<float> result({nz, ny, nx});  // computed in double, stored as float32
    float* out = result.mutable_data();
    const double* px = x.data();
    const double* py_ = y.data();
    const double* pz = z.data();

    {
        py::gil_scoped_release release;
        unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        int nt = n_threads > 0 ? n_threads : static_cast<int>(hw);
        const int64_t block = 2048;
        std::atomic<int64_t> next{0};

        auto worker = [&]() {
            std::vector<double> vals(nk), hts(nk);
            for (;;) {
                int64_t c0 = next.fetch_add(block);
                if (c0 >= ncol) break;
                int64_t c1 = std::min(ncol, c0 + block);
                for (int64_t c = c0; c < c1; ++c) {
                    const double xc = px[c % nx], yc = py_[c / nx];
                    const double sc = std::hypot(xc, yc);
                    double ac = std::atan2(xc, yc) / kDeg;
                    if (ac < 0) ac += 360.0;

                    for (size_t k = 0; k < nk; ++k) {
                        vals[k] = kNaN;
                        hts[k] = kNaN;
                        const Sweep& sw = sweeps[k];
                        const int64_t ng = sw.ngate;
                        if (sc < sw.ground[0] || sc > sw.ground[ng - 1]) continue;
                        const int64_t i = bisect(sw.ground.data(), ng, sc);
                        const double fr = (sc - sw.ground[i]) / (sw.ground[i + 1] - sw.ground[i]);
                        const int64_t na = static_cast<int64_t>(sw.az_ext.size());
                        const int64_t j = bisect(sw.az_ext.data(), na, ac);
                        const double da = sw.az_ext[j + 1] - sw.az_ext[j];
                        if (da > max_gap * sw.spacing) continue;  // do not bridge missing rays
                        const double fa = (ac - sw.az_ext[j]) / da;
                        const int64_t r0 = sw.ray_ext[j], r1 = sw.ray_ext[j + 1];
                        const double* d0 = sw.data + r0 * ng;
                        const double* d1 = sw.data + r1 * ng;
                        const double w[4] = {(1 - fa) * (1 - fr), (1 - fa) * fr, fa * (1 - fr), fa * fr};
                        const double v[4] = {d0[i], d0[i + 1], d1[i], d1[i + 1]};
                        const double rg[4] = {sw.range[i], sw.range[i + 1], sw.range[i], sw.range[i + 1]};
                        const double se[4] = {sw.sin_el[r0], sw.sin_el[r0], sw.sin_el[r1], sw.sin_el[r1]};
                        double num = 0.0, den = 0.0, h = 0.0;
                        for (int q = 0; q < 4; ++q) {
                            h += w[q] * beam_height(rg[q], se[q], R, sr);
                            if (!std::isnan(v[q])) {
                                num += w[q] * v[q];
                                den += w[q];
                            }
                        }
                        hts[k] = h;  // already above sea level (sr includes the site)
                        if (den >= min_weight) vals[k] = num / den;
                    }

                    // Only cones that reach this column take part; low tilts can
                    // start farther out than high ones, so compact them first.
                    int64_t nc = 0;
                    for (size_t k = 0; k < nk; ++k) {
                        if (std::isnan(hts[k])) continue;
                        hts[nc] = hts[k];
                        vals[nc] = vals[k];
                        ++nc;
                    }
                    int64_t lo = -1;
                    for (int64_t iz = 0; iz < nz; ++iz) {
                        const double zz = pz[iz];
                        double res = kNaN;
                        while (lo + 1 < nc && hts[lo + 1] <= zz) ++lo;
                        if (lo == -1) {
                            if (fill_below && nc > 0) res = vals[0];
                        } else if (lo + 1 < nc) {
                            const double t = (zz - hts[lo]) / (hts[lo + 1] - hts[lo]);
                            res = vals[lo] + t * (vals[lo + 1] - vals[lo]);
                        }
                        out[iz * ncol + c] = static_cast<float>(res);
                    }
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

PYBIND11_MODULE(_cone, m) {
    m.doc() = "Compiled cone gridding kernel for radarx.";
    m.def("grid_cones", &grid_cones, py::arg("x"), py::arg("y"), py::arg("z"), py::arg("data"),
          py::arg("azimuth"), py::arg("elevation"), py::arg("range"),
          py::arg("site_altitude"), py::arg("earth_radius") = 6371000.0,
          py::arg("max_gap") = 2.0, py::arg("min_weight") = 0.5, py::arg("fill_below") = false,
          py::arg("n_threads") = 0);
}
