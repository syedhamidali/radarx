// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Multi-radar kernels: beam geometry from every radar to every grid cell, the
// weighted merge of several radars' gridded fields, and the pairwise
// difference histograms used to estimate relative calibration biases.
//
// Geometry follows the 4/3 effective Earth radius model used by
// xradar.georeference.antenna_to_cartesian and the cone gridding kernel: a
// cell at ground distance s and height z (above sea level) sits at central
// angle theta = s / R from the radar, where R = 4/3 Earth radius, and the
// beam is a straight line in that frame.

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
#include <tuple>
#include <vector>

namespace py = pybind11;
using DArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using FArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kPi = 3.14159265358979323846;  // M_PI is not standard (MSVC)
constexpr double kDeg = kPi / 180.0;
const double kLn005 = std::log(0.005);

int thread_count(int n_threads) {
    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    return n_threads > 0 ? n_threads : static_cast<int>(hw);
}

template <class F>
void parallel_for(int64_t nwork, int n_threads, F&& body) {
    const int nt = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(thread_count(n_threads), nwork)));
    std::atomic<int64_t> next{0};
    auto worker = [&](int tid) {
        for (;;) {
            const int64_t w = next.fetch_add(1);
            if (w >= nwork) break;
            body(w, tid);
        }
    };
    std::vector<std::thread> pool;
    for (int t = 1; t < nt; ++t) pool.emplace_back(worker, t);
    worker(0);
    for (auto& th : pool) th.join();
}

// Slant range and the elevation of the beam (deg) at the cell (local) and at
// the antenna, for a radar at a = R + site altitude and a cell at b = R + z.
inline void geometry(double s, double a, double b, double R, double& range,
                     double& local_el, double& antenna_el) {
    const double th = s / R;
    const double c = std::cos(th), sn = std::sin(th);
    range = std::sqrt(std::max(0.0, a * a + b * b - 2.0 * a * b * c));
    local_el = std::atan2(b - a * c, a * sn) / kDeg;
    antenna_el = std::atan2(b * c - a, b * sn) / kDeg;
}

// Weight of a cell at antenna elevation e (deg) between the sweeps of sorted
// elevations el[0..n): 1 on a beam axis, 0.5 half way to the next beam (or
// half a beamwidth), below 0.01 one spacing away (Lakshmanan et al. 2006).
inline double beam_weight(double e, const std::vector<double>& el, double beamwidth) {
    const int64_t n = static_cast<int64_t>(el.size());
    if (n == 0) return 1.0;
    int64_t k = std::lower_bound(el.begin(), el.end(), e) - el.begin();  // el[k] >= e
    // nearest beam axis and the spacing towards the cell
    int64_t i;
    if (k == 0) i = 0;
    else if (k == n) i = n - 1;
    else i = (e - el[k - 1] <= el[k] - e) ? k - 1 : k;
    double spacing = beamwidth;
    const int64_t nb = (e >= el[i]) ? i + 1 : i - 1;
    if (nb >= 0 && nb < n) spacing = std::max(spacing, std::abs(el[nb] - el[i]));
    if (!(spacing > 0)) return 1.0;
    const double alpha = std::abs(e - el[i]) / spacing;
    return std::exp(alpha * alpha * alpha * kLn005);
}

// --- geodesics on the WGS84 ellipsoid (Vincenty 1975) ----------------------

constexpr double kA = 6378137.0;
constexpr double kF = 1.0 / 298.257223563;
constexpr double kB = kA * (1.0 - kF);

inline double delta_sigma(double B, double sin_s, double cos_s, double c2sm) {
    return B * sin_s *
           (c2sm + B / 4.0 *
                       (cos_s * (-1.0 + 2.0 * c2sm * c2sm) -
                        B / 6.0 * c2sm * (-3.0 + 4.0 * sin_s * sin_s) * (-3.0 + 4.0 * c2sm * c2sm)));
}

inline void series(double cos2a, double& A, double& B) {
    const double u2 = cos2a * (kA * kA - kB * kB) / (kB * kB);
    A = 1.0 + u2 / 16384.0 * (4096.0 + u2 * (-768.0 + u2 * (320.0 - 175.0 * u2)));
    B = u2 / 1024.0 * (256.0 + u2 * (-128.0 + u2 * (74.0 - 47.0 * u2)));
}

// Inverse problem: distance s and the azimuths (rad) at both ends of the
// geodesic from (lat1, lon1) to (lat2, lon2), all in radians.
void vincenty_inverse(double lat1, double lon1, double lat2, double lon2, double& s,
                      double& az1, double& az2) {
    const double L = lon2 - lon1;
    const double U1 = std::atan((1.0 - kF) * std::tan(lat1));
    const double U2 = std::atan((1.0 - kF) * std::tan(lat2));
    const double sU1 = std::sin(U1), cU1 = std::cos(U1);
    const double sU2 = std::sin(U2), cU2 = std::cos(U2);
    double lam = L, sin_s = 0, cos_s = 1, sigma = 0, cos2a = 1, c2sm = 0, sin_l = 0, cos_l = 1;
    for (int it = 0; it < 200; ++it) {
        sin_l = std::sin(lam);
        cos_l = std::cos(lam);
        const double t1 = cU2 * sin_l, t2 = cU1 * sU2 - sU1 * cU2 * cos_l;
        sin_s = std::sqrt(t1 * t1 + t2 * t2);
        if (sin_s == 0.0) {  // coincident points
            s = 0.0;
            az1 = az2 = 0.0;
            return;
        }
        cos_s = sU1 * sU2 + cU1 * cU2 * cos_l;
        sigma = std::atan2(sin_s, cos_s);
        const double sin_a = cU1 * cU2 * sin_l / sin_s;
        cos2a = 1.0 - sin_a * sin_a;
        c2sm = cos2a != 0.0 ? cos_s - 2.0 * sU1 * sU2 / cos2a : 0.0;
        const double C = kF / 16.0 * cos2a * (4.0 + kF * (4.0 - 3.0 * cos2a));
        const double prev = lam;
        lam = L + (1.0 - C) * kF * sin_a *
                      (sigma + C * sin_s * (c2sm + C * cos_s * (-1.0 + 2.0 * c2sm * c2sm)));
        if (std::abs(lam - prev) < 1e-13) break;
    }
    double A, B;
    series(cos2a, A, B);
    s = kB * A * (sigma - delta_sigma(B, sin_s, cos_s, c2sm));
    az1 = std::atan2(cU2 * sin_l, cU1 * sU2 - sU1 * cU2 * cos_l);
    az2 = std::atan2(cU1 * sin_l, -sU1 * cU2 + cU1 * sU2 * cos_l);
}

// Direct problem: end point and final azimuth of the geodesic of length s
// leaving (lat1, lon1) with azimuth az1 (radians).
void vincenty_direct(double lat1, double lon1, double az1, double s, double& lat2,
                     double& lon2, double& az2) {
    const double tU1 = (1.0 - kF) * std::tan(lat1);
    const double cU1 = 1.0 / std::sqrt(1.0 + tU1 * tU1), sU1 = tU1 * cU1;
    const double sin_a1 = std::sin(az1), cos_a1 = std::cos(az1);
    const double sigma1 = std::atan2(tU1, cos_a1);
    const double sin_a = cU1 * sin_a1;
    const double cos2a = 1.0 - sin_a * sin_a;
    double A, B;
    series(cos2a, A, B);
    double sigma = s / (kB * A), c2sm = 0, sin_s = 0, cos_s = 1;
    for (int it = 0; it < 200; ++it) {
        c2sm = std::cos(2.0 * sigma1 + sigma);
        sin_s = std::sin(sigma);
        cos_s = std::cos(sigma);
        const double next = s / (kB * A) + delta_sigma(B, sin_s, cos_s, c2sm);
        const bool done = std::abs(next - sigma) < 1e-13;
        sigma = next;
        if (done) break;
    }
    c2sm = std::cos(2.0 * sigma1 + sigma);
    sin_s = std::sin(sigma);
    cos_s = std::cos(sigma);
    const double tmp = sU1 * sin_s - cU1 * cos_s * cos_a1;
    lat2 = std::atan2(sU1 * cos_s + cU1 * sin_s * cos_a1,
                      (1.0 - kF) * std::sqrt(sin_a * sin_a + tmp * tmp));
    const double lam = std::atan2(sin_s * sin_a1, cU1 * cos_s - sU1 * sin_s * cos_a1);
    const double C = kF / 16.0 * cos2a * (4.0 + kF * (4.0 - 3.0 * cos2a));
    const double L = lam - (1.0 - C) * kF * sin_a *
                               (sigma + C * sin_s * (c2sm + C * cos_s * (-1.0 + 2.0 * c2sm * c2sm)));
    lon2 = lon1 + L;
    az2 = std::atan2(sin_a, -tmp);
}

}  // namespace

// Geometry of the columns of a grid in the azimuthal equidistant projection
// centred on (origin_lat, origin_lon), as seen from every radar. In that
// projection a column at (x, y) lies at geodesic distance hypot(x, y) and
// azimuth atan2(x, y) from the origin. For every radar the geodesic to the
// column gives the ground distance, the antenna azimuth (initial azimuth) and
// the beam direction at the column (final azimuth), which is rotated into the
// grid frame through the geodesic from the origin, whose grid direction is
// known. Returns lat, lon (ny, nx) and ground, antenna_azimuth, azimuth
// (nradar, ny, nx), angles in degrees.
std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>,
           py::array_t<double>>
column_geometry(const DArray& x, const DArray& y, double origin_lat, double origin_lon,
                const std::vector<double>& radar_lat, const std::vector<double>& radar_lon,
                int n_threads) {
    const int64_t nx = x.size(), ny = y.size();
    const int64_t nr = static_cast<int64_t>(radar_lat.size());
    if (static_cast<int64_t>(radar_lon.size()) != nr)
        throw std::invalid_argument("radar_lat and radar_lon differ in length");
    const int64_t ncol = nx * ny;
    py::array_t<double> lat({ny, nx}), lon({ny, nx});
    py::array_t<double> ground({nr, ny, nx}), antenna({nr, ny, nx}), heading({nr, ny, nx});
    double* plat = lat.mutable_data();
    double* plon = lon.mutable_data();
    double* pg = ground.mutable_data();
    double* pa = antenna.mutable_data();
    double* ph = heading.mutable_data();
    const double* px = x.data();
    const double* py_ = y.data();
    const double lat0 = origin_lat * kDeg, lon0 = origin_lon * kDeg;
    const double Rm = (2.0 * kA + kB) / 3.0;  // mean radius for the AEQD scale
    {
        py::gil_scoped_release release;
        const int64_t block = 1024;
        const int64_t nblock = (ncol + block - 1) / block;
        parallel_for(nblock, n_threads, [&](int64_t w, int) {
            const int64_t c0 = w * block, c1 = std::min(ncol, c0 + block);
            for (int64_t c = c0; c < c1; ++c) {
                const double xc = px[c % nx], yc = py_[c / nx];
                const double s0 = std::hypot(xc, yc);
                const double a0 = s0 > 0 ? std::atan2(xc, yc) : 0.0;
                double la, lo, a0f;
                if (s0 > 0) {
                    vincenty_direct(lat0, lon0, a0, s0, la, lo, a0f);
                } else {
                    la = lat0;
                    lo = lon0;
                    a0f = 0.0;
                }
                double lon_deg = std::fmod(lo / kDeg + 540.0, 360.0) - 180.0;
                plat[c] = la / kDeg;
                plon[c] = lon_deg;
                // transverse scale of the azimuthal equidistant projection
                const double th = s0 / Rm;
                const double k = th > 1e-12 ? th / std::sin(th) : 1.0;
                for (int64_t r = 0; r < nr; ++r) {
                    double s, a1, a2;
                    vincenty_inverse(radar_lat[r] * kDeg, radar_lon[r] * kDeg, la, lo, s, a1, a2);
                    const double phi = a2 - a0f;  // beam direction from the origin radial
                    double h = a0 + std::atan2(k * std::sin(phi), std::cos(phi));
                    double az = a1 / kDeg, hd = h / kDeg;
                    az = std::fmod(az + 720.0, 360.0);
                    hd = std::fmod(hd + 720.0, 360.0);
                    const int64_t o = r * ncol + c;
                    pg[o] = s;
                    pa[o] = az;
                    ph[o] = hd;
                }
            }
        });
    }
    return {lat, lon, ground, antenna, heading};
}

// Slant range (m) and local beam elevation (deg) from every radar to every
// cell: ground[r] is radar r's (ny, nx) ground distance to each column.
// Returns float32 arrays (nradar, nz, ny, nx).
std::tuple<py::array_t<float>, py::array_t<float>> beam_geometry(
    const std::vector<DArray>& ground, const DArray& z, const std::vector<double>& site_altitude,
    double earth_radius, int n_threads) {
    const int64_t nr = static_cast<int64_t>(ground.size());
    if (nr == 0 || static_cast<int64_t>(site_altitude.size()) != nr)
        throw std::invalid_argument("need one ground array and site altitude per radar");
    const int64_t ncol = ground[0].size();
    for (const auto& g : ground)
        if (g.size() != ncol) throw std::invalid_argument("every radar needs the same columns");
    const int64_t nz = z.size();
    std::vector<py::ssize_t> shape{nr, nz};
    shape.insert(shape.end(), ground[0].shape(), ground[0].shape() + ground[0].ndim());
    py::array_t<float> rng(shape), elev(shape);
    float* pr = rng.mutable_data();
    float* pe = elev.mutable_data();
    std::vector<const double*> pg(nr);
    for (int64_t r = 0; r < nr; ++r) pg[r] = ground[r].data();
    const double* pz = z.data();
    const double R = earth_radius * 4.0 / 3.0;
    {
        py::gil_scoped_release release;
        const int64_t block = 4096;
        const int64_t nblock = (ncol + block - 1) / block;
        parallel_for(nr * nblock, n_threads, [&](int64_t w, int) {
            const int64_t r = w / nblock;
            const int64_t c0 = (w % nblock) * block, c1 = std::min(ncol, c0 + block);
            const double a = R + site_altitude[r];
            for (int64_t iz = 0; iz < nz; ++iz) {
                const double b = R + pz[iz];
                const int64_t off = (r * nz + iz) * ncol;
                for (int64_t c = c0; c < c1; ++c) {
                    double rr, le, ae;
                    const double s = pg[r][c];
                    if (std::isnan(s)) {
                        pr[off + c] = pe[off + c] = static_cast<float>(kNaN);
                        continue;
                    }
                    geometry(s, a, b, R, rr, le, ae);
                    pr[off + c] = static_cast<float>(rr);
                    pe[off + c] = static_cast<float>(le);
                }
            }
        });
    }
    return {rng, elev};
}

// Weighted mean over radars of values (nradar, nz, ncol...) with weight
//   exp(-(range / range_scale)^2) * beam_weight * exp(-(dt / time_scale)^2),
// a scale <= 0 switching that factor off. Returns the merged field and the
// weight sum, float32 (nz, ncol...).
std::tuple<py::array_t<float>, py::array_t<float>> merge(
    const FArray& values, const std::vector<DArray>& ground, const DArray& z,
    const std::vector<double>& site_altitude, const std::vector<std::vector<double>>& elevations,
    const std::vector<double>& beamwidth, const std::vector<double>& time_offset,
    double range_scale, double time_scale, double earth_radius, int n_threads) {
    const int64_t nr = static_cast<int64_t>(ground.size());
    if (nr == 0 || values.ndim() < 2 || values.shape(0) != nr)
        throw std::invalid_argument("values must be (nradar, nz, ...) with one ground per radar");
    if (static_cast<int64_t>(site_altitude.size()) != nr ||
        static_cast<int64_t>(elevations.size()) != nr ||
        static_cast<int64_t>(beamwidth.size()) != nr ||
        static_cast<int64_t>(time_offset.size()) != nr)
        throw std::invalid_argument("need site altitude, elevations, beamwidth, time per radar");
    const int64_t nz = z.size();
    if (values.shape(1) != nz) throw std::invalid_argument("values do not match z");
    const int64_t ncol = ground[0].size();
    if (values.size() != nr * nz * ncol) throw std::invalid_argument("values do not match ground");
    for (const auto& g : ground)
        if (g.size() != ncol) throw std::invalid_argument("every radar needs the same columns");
    std::vector<std::vector<double>> el(elevations);
    for (auto& e : el) std::sort(e.begin(), e.end());
    std::vector<double> wt(nr);
    for (int64_t r = 0; r < nr; ++r) {
        const double q = time_scale > 0 ? time_offset[r] / time_scale : 0.0;
        wt[r] = std::exp(-q * q);
    }
    std::vector<py::ssize_t> shape(values.shape() + 1, values.shape() + values.ndim());
    py::array_t<float> merged(shape), weight(shape);
    float* pm = merged.mutable_data();
    float* pw = weight.mutable_data();
    const float* pv = values.data();
    std::vector<const double*> pg(nr);
    for (int64_t r = 0; r < nr; ++r) pg[r] = ground[r].data();
    const double* pz = z.data();
    const double R = earth_radius * 4.0 / 3.0;
    {
        py::gil_scoped_release release;
        const int64_t block = 4096;
        const int64_t nblock = (ncol + block - 1) / block;
        parallel_for(nz * nblock, n_threads, [&](int64_t w, int) {
            const int64_t iz = w / nblock;
            const int64_t c0 = (w % nblock) * block, c1 = std::min(ncol, c0 + block);
            const double b = R + pz[iz];
            for (int64_t c = c0; c < c1; ++c) {
                double num = 0.0, den = 0.0;
                for (int64_t r = 0; r < nr; ++r) {
                    const float v = pv[(r * nz + iz) * ncol + c];
                    if (std::isnan(v)) continue;
                    double rr, le, ae;
                    geometry(pg[r][c], R + site_altitude[r], b, R, rr, le, ae);
                    double q = range_scale > 0 ? rr / range_scale : 0.0;
                    const double wgt = std::exp(-q * q) * beam_weight(ae, el[r], beamwidth[r]) * wt[r];
                    num += wgt * v;
                    den += wgt;
                }
                const int64_t o = iz * ncol + c;
                pm[o] = static_cast<float>(den > 0 ? num / den : kNaN);
                pw[o] = static_cast<float>(den);
            }
        });
    }
    return {merged, weight};
}

// Histograms of the differences values[i] - values[j] for every radar pair
// i < j over the cells (last axis) where both are defined, optionally only
// where max(range) / min(range) <= max_range_ratio. Bins of bin_width span
// [-max_difference, max_difference). Returns counts (npair, nbin) and the
// sum and sum of squares of the differences inside that span (npair).
std::tuple<py::array_t<int64_t>, py::array_t<double>, py::array_t<double>> pair_histograms(
    const FArray& values, const FArray& range, double max_range_ratio, double bin_width,
    double max_difference, int n_threads) {
    if (values.ndim() != 2) throw std::invalid_argument("values must be (nradar, ncell)");
    const int64_t nr = values.shape(0), ncell = values.shape(1);
    const bool use_range = max_range_ratio > 0;
    if (use_range && (range.ndim() != 2 || range.shape(0) != nr || range.shape(1) != ncell))
        throw std::invalid_argument("range must match values");
    if (!(bin_width > 0) || !(max_difference > 0))
        throw std::invalid_argument("bin_width and max_difference must be positive");
    const int64_t nbin = static_cast<int64_t>(std::ceil(2.0 * max_difference / bin_width));
    const int64_t npair = nr * (nr - 1) / 2;
    py::array_t<int64_t> counts({npair, nbin});
    py::array_t<double> sums(npair), sumsq(npair);
    int64_t* pc = counts.mutable_data();
    double* ps = sums.mutable_data();
    double* pq = sumsq.mutable_data();
    std::fill(pc, pc + npair * nbin, 0);
    std::fill(ps, ps + npair, 0.0);
    std::fill(pq, pq + npair, 0.0);
    const float* pv = values.data();
    const float* prg = use_range ? range.data() : nullptr;
    {
        py::gil_scoped_release release;
        const int64_t block = 16384;
        const int64_t nblock = (ncell + block - 1) / block;
        const int nt = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(thread_count(n_threads), nblock)));
        // per-thread accumulators, allocated once before the parallel loop
        std::vector<std::vector<int64_t>> hist(nt, std::vector<int64_t>(npair * nbin, 0));
        std::vector<std::vector<double>> s1(nt, std::vector<double>(npair, 0.0));
        std::vector<std::vector<double>> s2(nt, std::vector<double>(npair, 0.0));
        parallel_for(nblock, nt, [&](int64_t w, int tid) {
            const int64_t c0 = w * block, c1 = std::min(ncell, c0 + block);
            int64_t* h = hist[tid].data();
            double* a1 = s1[tid].data();
            double* a2 = s2[tid].data();
            for (int64_t c = c0; c < c1; ++c) {
                int64_t p = 0;
                for (int64_t i = 0; i < nr; ++i) {
                    const float vi = pv[i * ncell + c];
                    if (std::isnan(vi)) {
                        p += nr - 1 - i;
                        continue;
                    }
                    for (int64_t j = i + 1; j < nr; ++j, ++p) {
                        const float vj = pv[j * ncell + c];
                        if (std::isnan(vj)) continue;
                        if (use_range) {
                            const double ri = prg[i * ncell + c], rj = prg[j * ncell + c];
                            const double lo = std::min(ri, rj), hi = std::max(ri, rj);
                            if (!(hi <= max_range_ratio * lo)) continue;
                        }
                        const double d = static_cast<double>(vi) - static_cast<double>(vj);
                        const int64_t b = static_cast<int64_t>(std::floor((d + max_difference) / bin_width));
                        if (b < 0 || b >= nbin) continue;
                        h[p * nbin + b] += 1;
                        a1[p] += d;
                        a2[p] += d * d;
                    }
                }
            }
        });
        for (int t = 0; t < nt; ++t) {
            for (int64_t k = 0; k < npair * nbin; ++k) pc[k] += hist[t][k];
            for (int64_t p = 0; p < npair; ++p) {
                ps[p] += s1[t][p];
                pq[p] += s2[t][p];
            }
        }
    }
    return {counts, sums, sumsq};
}

PYBIND11_MODULE(_multi, m) {
    m.doc() = "Compiled multi-radar geometry, merging and comparison kernels for radarx.";
    m.def("column_geometry", &column_geometry, py::arg("x"), py::arg("y"),
          py::arg("origin_lat"), py::arg("origin_lon"), py::arg("radar_lat"),
          py::arg("radar_lon"), py::arg("n_threads") = 0);
    m.def("beam_geometry", &beam_geometry, py::arg("ground"), py::arg("z"),
          py::arg("site_altitude"), py::arg("earth_radius") = 6371000.0, py::arg("n_threads") = 0);
    m.def("merge", &merge, py::arg("values"), py::arg("ground"), py::arg("z"),
          py::arg("site_altitude"), py::arg("elevations"), py::arg("beamwidth"),
          py::arg("time_offset"), py::arg("range_scale"), py::arg("time_scale"),
          py::arg("earth_radius") = 6371000.0, py::arg("n_threads") = 0);
    m.def("pair_histograms", &pair_histograms, py::arg("values"), py::arg("range"),
          py::arg("max_range_ratio"), py::arg("bin_width"), py::arg("max_difference"),
          py::arg("n_threads") = 0);
}
