// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Partial and cumulative beam blockage of a polar sweep by terrain.
//
// Every ray is independent. A ray is a list of gates at slant range r and elevation e.
// The beam centre is placed with the effective Earth radius model
// (reff = k_e * a_earth, the radar at reff + altitude from the centre of the
// effective Earth, as in radarx.fundamentals.geometry and the gridding kernels):
//
//     b = sqrt(r^2 + A^2 + 2 r A sin e),   A = reff + altitude
//     h = b - reff                        (beam centre height above sea level)
//     s = reff * asin(r cos e / b)        (distance on the ground)
//
// The ground position of a gate follows from s and the azimuth: the direct problem
// on a sphere for a geographic DEM (latitude, longitude), or a plane
// (x = x0 + s sin az, y = y0 + s cos az) for a DEM in projected metres. The terrain
// height is interpolated bilinearly in the DEM; outside the DEM or next to a missing
// cell it is NaN.
//
// With y = terrain height - h and the beam radius a = r * beamwidth / 2, the partial
// beam blockage (PBB) of Bech et al. (2003, Eq. A3) is the part of the beam cross
// section below the terrain. "uniform": the circle of radius a with uniform power,
//     PBB = (x sqrt(1 - x^2) + asin(x) + pi / 2) / pi,  x = y / a clipped to [-1, 1].
// "gaussian": the Gaussian beam pattern of the paper's Fig. A1 and of Lang et al.
// (2009), the half-power width of the beam being the beamwidth, so that the power
// below a horizontal edge is the normal distribution function of y / sigma,
//     PBB = 0.5 erfc(-y / (sigma sqrt(2))),  sigma = a / sqrt(2 ln 2) (one way pattern).
// The two way pattern is narrower by sqrt(2). The cumulative beam blockage (CBB) of a
// gate is the maximum of the PBB along the ray up to and including that gate; with
// oversample > 1 the terrain is also sampled at that many points per gate length,
// ending at the gate centre. A gate is flagged as terrain clutter where the beam
// centre is at most clutter_margin above the terrain.
// Details, references and the radarx choices are in the docstring of
// radarx/retrieve/beam_blockage.py. Every step follows the NumPy reference in that
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
#include <tuple>
#include <vector>

namespace py = pybind11;
using DArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kPi = 3.14159265358979323846;
constexpr double kDeg = kPi / 180.0;
constexpr int64_t kBlock = 8;  // rays per scheduling block

enum Pattern { kGaussian = 0, kGaussianTwoWay = 1, kUniform = 2 };

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

// Index i of the cell [axis[i], axis[i + 1]] that holds q (axis ascending, n >= 2), the
// last cell for q at the upper end; -1 outside the axis or for NaN.
inline int64_t find_cell(const double* axis, int64_t n, double q) {
    if (!(q >= axis[0]) || !(q <= axis[n - 1])) return -1;
    const int64_t i = std::upper_bound(axis, axis + n, q) - axis - 1;
    return std::min(i, n - 2);
}

struct Dem {
    const double* z;
    const double* ya;
    const double* xa;
    int64_t ny, nx;

    // Bilinear interpolation; NaN outside the axes or if a corner is NaN.
    double sample(double yq, double xq) const {
        const int64_t i = find_cell(ya, ny, yq);
        const int64_t j = find_cell(xa, nx, xq);
        if (i < 0 || j < 0) return kNaN;
        const double ty = (yq - ya[i]) / (ya[i + 1] - ya[i]);
        const double tx = (xq - xa[j]) / (xa[j + 1] - xa[j]);
        const double z00 = z[i * nx + j], z01 = z[i * nx + j + 1];
        const double z10 = z[(i + 1) * nx + j], z11 = z[(i + 1) * nx + j + 1];
        const double lo = z00 + (z01 - z00) * tx;
        const double hi = z10 + (z11 - z10) * tx;
        return lo + (hi - lo) * ty;
    }
};

struct Geometry {
    bool geographic;
    double site_a, site_b;  // latitude, longitude [deg] or y, x [m]
    double altitude, reff, earth_radius;
    double lon_center;      // centre of the longitude axis (geographic)
    double sin_site_lat = 0.0, cos_site_lat = 1.0;

    // Terrain position of a point at ground distance s and azimuth az.
    void locate(double s, double sin_az, double cos_az, double& yq, double& xq) const {
        if (!geographic) {
            yq = site_a + s * cos_az;
            xq = site_b + s * sin_az;
            return;
        }
        const double delta = s / earth_radius;
        const double sd = std::sin(delta), cd = std::cos(delta);
        const double sin_lat = sin_site_lat * cd + cos_site_lat * sd * cos_az;
        yq = std::asin(std::max(-1.0, std::min(1.0, sin_lat))) / kDeg;
        const double lat2 = yq * kDeg;
        double lon = site_b + std::atan2(sin_az * sd * cos_site_lat,
                                         cd - sin_site_lat * std::sin(lat2)) / kDeg;
        lon += 360.0 * std::floor((lon_center - lon) / 360.0 + 0.5);
        xq = lon;
    }
};

inline double blocked_fraction(double y, double beam_radius, int pattern) {
    const double x = y / beam_radius;
    if (pattern == kUniform) {
        if (x <= -1.0) return 0.0;
        if (x >= 1.0) return 1.0;
        return (x * std::sqrt(1.0 - x * x) + std::asin(x) + 0.5 * kPi) / kPi;
    }
    const double sigma_over_a = (pattern == kGaussianTwoWay ? std::sqrt(0.5) : 1.0) / std::sqrt(2.0 * std::log(2.0));
    return 0.5 * std::erfc(-x / (sigma_over_a * std::sqrt(2.0)));
}

}  // namespace

// rng: (ng,) slant ranges [m], > 0; az, el: (nr,) [deg]; dem: (ny, nx) with the
// ascending axes ya (ny) and xa (nx). Returns PBB, CBB, the terrain height at the gate,
// all (nr, ng) float64, and the clutter flag (nr, ng) uint8.
std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<uint8_t>>
blockage_py(const DArray& rng, const DArray& az, const DArray& el, const DArray& dem,
            const DArray& ya, const DArray& xa, bool geographic, double site_a, double site_b,
            double altitude, double k_e, double earth_radius, double beamwidth, int pattern,
            int oversample, double clutter_margin, int n_threads) {
    if (rng.ndim() != 1 || az.ndim() != 1 || el.ndim() != 1 || az.shape(0) != el.shape(0))
        throw std::invalid_argument("rng, az and el must be 1-D, az and el of the same length");
    if (dem.ndim() != 2 || ya.ndim() != 1 || xa.ndim() != 1 || dem.shape(0) != ya.shape(0) ||
        dem.shape(1) != xa.shape(0) || ya.shape(0) < 2 || xa.shape(0) < 2)
        throw std::invalid_argument("dem must be 2-D (at least 2 by 2) with one axis value per row and column");
    if (pattern < kGaussian || pattern > kUniform) throw std::invalid_argument("unknown pattern");
    if (oversample < 1) throw std::invalid_argument("oversample must be at least 1");
    const int64_t nr = az.shape(0);
    const int64_t ng = rng.shape(0);

    py::array_t<double> pbb({static_cast<py::ssize_t>(nr), static_cast<py::ssize_t>(ng)});
    py::array_t<double> cbb({static_cast<py::ssize_t>(nr), static_cast<py::ssize_t>(ng)});
    py::array_t<double> terrain({static_cast<py::ssize_t>(nr), static_cast<py::ssize_t>(ng)});
    py::array_t<uint8_t> flag({static_cast<py::ssize_t>(nr), static_cast<py::ssize_t>(ng)});
    double* po = pbb.mutable_data();
    double* co = cbb.mutable_data();
    double* to = terrain.mutable_data();
    uint8_t* fo = flag.mutable_data();

    const double* rp = rng.data();
    const double* ap = az.data();
    const double* ep = el.data();
    const Dem map{dem.data(), ya.data(), xa.data(), dem.shape(0), dem.shape(1)};
    Geometry geo{geographic, site_a, site_b, altitude, k_e * earth_radius, earth_radius, 0.0};
    geo.lon_center = 0.5 * (xa.data()[0] + xa.data()[xa.shape(0) - 1]);
    if (geographic) {
        geo.sin_site_lat = std::sin(site_a * kDeg);
        geo.cos_site_lat = std::cos(site_a * kDeg);
    }
    const double reff = geo.reff;
    const double centre = reff + altitude;
    const double half_width = 0.5 * beamwidth * kDeg;

    {
        py::gil_scoped_release release;
        parallel_blocks(nr, n_threads, [&](int64_t q0, int64_t q1) {
            for (int64_t q = q0; q < q1; ++q) {
                double* prow = po + q * ng;
                double* crow = co + q * ng;
                double* trow = to + q * ng;
                uint8_t* frow = fo + q * ng;
                if (!std::isfinite(ap[q]) || !std::isfinite(ep[q])) {
                    for (int64_t k = 0; k < ng; ++k) {
                        prow[k] = crow[k] = trow[k] = kNaN;
                        frow[k] = 0;
                    }
                    continue;
                }
                const double sin_e = std::sin(ep[q] * kDeg), cos_e = std::cos(ep[q] * kDeg);
                const double sin_a = std::sin(ap[q] * kDeg), cos_a = std::cos(ap[q] * kDeg);
                double running = kNaN;
                for (int64_t k = 0; k < ng; ++k) {
                    const double step = k > 0 ? rp[k] - rp[k - 1] : (ng > 1 ? rp[1] - rp[0] : 0.0);
                    double p = kNaN, t = kNaN, hgt = kNaN;
                    for (int m = oversample - 1; m >= 0; --m) {
                        const double r = rp[k] - step * m / oversample;
                        if (!(r > 0.0)) continue;
                        const double b = std::sqrt(r * r + centre * centre + 2.0 * r * centre * sin_e);
                        const double h = b - reff;
                        const double s = reff * std::asin(r * cos_e / b);
                        double yq, xq;
                        geo.locate(s, sin_a, cos_a, yq, xq);
                        const double tm = map.sample(yq, xq);
                        const double pm = std::isnan(tm) ? kNaN : blocked_fraction(tm - h, r * half_width, pattern);
                        if (!std::isnan(pm) && (std::isnan(running) || pm > running)) running = pm;
                        if (m == 0) {
                            p = pm;
                            t = tm;
                            hgt = h;
                        }
                    }
                    prow[k] = p;
                    crow[k] = running;
                    trow[k] = t;
                    frow[k] = (!std::isnan(t) && hgt - t <= clutter_margin) ? 1 : 0;
                }
            }
        });
    }
    return {pbb, cbb, terrain, flag};
}

PYBIND11_MODULE(_beam_blockage, m) {
    m.doc() = "Partial and cumulative beam blockage kernel";
    m.def("blockage", &blockage_py, py::arg("rng"), py::arg("az"), py::arg("el"), py::arg("dem"),
          py::arg("ya"), py::arg("xa"), py::arg("geographic"), py::arg("site_a"),
          py::arg("site_b"), py::arg("altitude"), py::arg("k_e"), py::arg("earth_radius"),
          py::arg("beamwidth"), py::arg("pattern"), py::arg("oversample") = 1,
          py::arg("clutter_margin") = 0.0, py::arg("n_threads") = 0);
}
