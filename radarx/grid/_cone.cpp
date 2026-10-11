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
// Optionally every cone is sampled at the position displaced by its own shift
// (the advection of the storm from the time of the sweep to the analysis time).
// Sweeps of the same elevation (repeated low tilts, SAILS) can be merged into
// one cone first: the mean of the repeats that have a value, or the first repeat
// that has one when they are ordered by priority (latest, or nearest in time).
// Levels below the lowest or above the highest cone, or between cones where
// one has no value, stay NaN (optionally the lowest cone fills below itself).
//
// grid_cones grids one field of one radar on a regular (x, y) grid centred on
// the radar; grid_columns grids many fields of many radars in one call onto
// columns given by their ground distance and azimuth from each radar
// (multi-radar gridding on a shared grid).
//
// grid_gates is a different scheme, the radius of influence mapping of
// objective analysis: every valid gate is placed in Cartesian space (the same
// beam geometry), the gates are sorted into a coarse 3-D bucket grid, and every
// output point takes the weighted mean of the gates closer than its radius of
// influence R. The weights are Cressman (1959), (R^2 - d^2) / (R^2 + d^2), or
// Barnes (1964), exp(-4 d^2 / R^2), with d the Cartesian distance between the
// gate and the point; "nearest" takes the closest gate. R grows linearly with the
// distance from the radar and the height above it (the "dist" function of
// Py-ART). Scan geometry plays no role, so PPI, RHI, sector and repeated sweeps
// are handled alike.
//
// Beam geometry follows xradar.georeference.antenna_to_cartesian (effective
// Earth radius k_e * a, k_e = 4/3 by default; Doviak and Zrnic 1993, Doppler
// Radar and Weather Observations, 2nd ed., Eqs. 2.28b-d, pp. 21-22). The vertical step is analogous to the VI scheme of
// Zhang et al. (2005, J. Atmos. Oceanic Technol. 22, 30-42, Eqs. 5-7, p. 36) but
// interpolates in beam height, not elevation angle. max_gap and min_weight are
// radarx choices, not values from a paper.

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
#include <type_traits>
#include <vector>

namespace py = pybind11;
using DArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using FArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kPi = 3.14159265358979323846;  // M_PI is not standard (MSVC)
constexpr double kDeg = kPi / 180.0;

struct Sweep {
    const double* data = nullptr;  // (nray, ngate), row-major, float64 input
    const float* data32 = nullptr; // or float32 input (multi-radar gridding)
    int64_t nray = 0, ngate = 0;
    double value(int64_t k) const {
        return data ? data[k] : static_cast<double>(data32[k]);
    }
    std::vector<double> range;     // gate ranges [m]
    std::vector<double> ground;    // ground distance of each gate [m]
    std::vector<double> sin_el;    // per ray
    std::vector<double> az_ext;    // sorted ray azimuths, wrapped at both ends
    std::vector<int64_t> ray_ext;  // ray index for each az_ext entry
    double spacing = 0.0;          // median azimuth step [deg]
    // The cone is sampled at (x - dx, y - dy) instead of (x, y), relative to
    // the radar [m]: echoes seen at the time of the sweep moved to the
    // analysis time by the storm motion times the time difference.
    double dx = 0.0, dy = 0.0;
};

// Beam height above sea level (effective Earth radius R = k_e a), as in
// xradar: sr = R + site altitude.
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

template <class Array>
Sweep prepare(const Array& data, const DArray& azimuth, const DArray& elevation,
              const DArray& range, double R, double sr) {
    if (data.ndim() != 2) throw std::invalid_argument("sweep data must be 2-D (ray, gate)");
    Sweep sw;
    sw.nray = data.shape(0);
    sw.ngate = data.shape(1);
    if (azimuth.size() != sw.nray || elevation.size() != sw.nray || range.size() != sw.ngate)
        throw std::invalid_argument("azimuth/elevation/range do not match sweep data shape");
    if (sw.nray < 3 || sw.ngate < 2) throw std::invalid_argument("sweep too small");
    if constexpr (std::is_same_v<typename Array::value_type, float>)
        sw.data32 = data.data();
    else
        sw.data = data.data();
    sw.range.assign(range.data(), range.data() + sw.ngate);
    set_geometry(sw, elevation.data(), R, sr);
    set_azimuth_order(sw, azimuth.data());
    return sw;
}

// Value and beam height of every sweep at ground distance sc, azimuth ac (deg),
// then the column's levels interpolated in height between bracketing cones.
// vals/hts are scratch buffers of at least sweeps.size(); out has stride ncol.
void grid_column(const std::vector<Sweep>& sweeps, double sc, double ac, const double* pz,
                 int64_t nz, double R, double sr, double max_gap, double min_weight,
                 bool fill_below, double fill_depth, std::vector<double>& vals,
                 std::vector<double>& hts, float* out, int64_t stride,
                 const std::vector<int64_t>& groups, bool group_mean) {
    const size_t nk = sweeps.size();
    for (size_t k = 0; k < nk; ++k) {
        vals[k] = kNaN;
        hts[k] = kNaN;
        const Sweep& sw = sweeps[k];
        const int64_t ng = sw.ngate;
        double sk = sc, ak = ac;
        if (sw.dx != 0.0 || sw.dy != 0.0) {
            const double px = sc * std::sin(ac * kDeg) - sw.dx;
            const double py = sc * std::cos(ac * kDeg) - sw.dy;
            sk = std::hypot(px, py);
            ak = std::atan2(px, py) / kDeg;
            if (ak < 0) ak += 360.0;
        }
        if (!(sk >= sw.ground[0] && sk <= sw.ground[ng - 1])) continue;  // also NaN
        const int64_t i = bisect(sw.ground.data(), ng, sk);
        const double fr = (sk - sw.ground[i]) / (sw.ground[i + 1] - sw.ground[i]);
        const int64_t na = static_cast<int64_t>(sw.az_ext.size());
        const int64_t j = bisect(sw.az_ext.data(), na, ak);
        const double da = sw.az_ext[j + 1] - sw.az_ext[j];
        if (da > max_gap * sw.spacing) continue;  // do not bridge missing rays
        const double fa = (ak - sw.az_ext[j]) / da;
        const int64_t r0 = sw.ray_ext[j], r1 = sw.ray_ext[j + 1];
        const int64_t o0 = r0 * ng + i, o1 = r1 * ng + i;
        const double w[4] = {(1 - fa) * (1 - fr), (1 - fa) * fr, fa * (1 - fr), fa * fr};
        const double v[4] = {sw.value(o0), sw.value(o0 + 1), sw.value(o1), sw.value(o1 + 1)};
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

    // Sweeps of one elevation (repeated low tilts) form a group, which becomes one
    // cone: the mean of the members that have a value, or the first member that
    // has one (the members come ordered by priority).
    size_t nu = nk;
    if (!groups.empty()) {
        nu = groups.size();
        size_t k0 = 0;
        for (size_t g = 0; g < nu; ++g) {
            const size_t k1 = k0 + static_cast<size_t>(groups[g]);
            double v = kNaN, h = kNaN;
            if (group_mean) {
                double sv = 0.0, sh = 0.0;
                int nv = 0, nh = 0;
                for (size_t k = k0; k < k1; ++k) {
                    if (!std::isnan(vals[k])) {
                        sv += vals[k];
                        ++nv;
                    }
                    if (!std::isnan(hts[k])) {
                        sh += hts[k];
                        ++nh;
                    }
                }
                if (nv) v = sv / nv;
                if (nh) h = sh / nh;
            } else {
                for (size_t k = k0; k < k1; ++k) {
                    if (std::isnan(v) && (!std::isnan(vals[k]) || std::isnan(h))) {
                        v = vals[k];
                        h = hts[k];
                    }
                }
            }
            vals[g] = v;
            hts[g] = h;
            k0 = k1;
        }
    }

    // Only cones that reach this column take part; low tilts can start
    // farther out than high ones, so compact them first.
    int64_t nc = 0;
    for (size_t k = 0; k < nu; ++k) {
        if (std::isnan(hts[k])) continue;
        hts[nc] = hts[k];
        vals[nc] = vals[k];
        ++nc;
    }
    // With a finite fill_depth the lowest cone fills at most that far below
    // itself and never below the radar's altitude; infinity fills every level.
    const bool limited = std::isfinite(fill_depth);
    const double site_altitude = sr - R;
    int64_t lo = -1;
    for (int64_t iz = 0; iz < nz; ++iz) {
        const double zz = pz[iz];
        double res = kNaN;
        while (lo + 1 < nc && hts[lo + 1] <= zz) ++lo;
        if (lo == -1) {
            if (fill_below && nc > 0 &&
                (!limited || (zz >= hts[0] - fill_depth && zz >= site_altitude)))
                res = vals[0];
        } else if (lo + 1 < nc) {
            const double t = (zz - hts[lo]) / (hts[lo + 1] - hts[lo]);
            res = vals[lo] + t * (vals[lo + 1] - vals[lo]);
        }
        out[iz * stride] = static_cast<float>(res);
    }
}

int thread_count(int n_threads) {
    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    return n_threads > 0 ? n_threads : static_cast<int>(hw);
}

template <class Array>
std::vector<Sweep> prepare_all(const std::vector<Array>& data, const std::vector<DArray>& azimuth,
                               const std::vector<DArray>& elevation,
                               const std::vector<DArray>& range, double R, double sr) {
    const size_t nk = data.size();
    if (nk < 1 || azimuth.size() != nk || elevation.size() != nk || range.size() != nk)
        throw std::invalid_argument("need the same number of data/azimuth/elevation/range arrays");
    std::vector<Sweep> sweeps;
    sweeps.reserve(nk);
    for (size_t k = 0; k < nk; ++k)
        sweeps.push_back(prepare(data[k], azimuth[k], elevation[k], range[k], R, sr));
    return sweeps;
}

// Sweeps from float32 or float64 data without copying (other dtypes are
// converted to float64); the arrays used are appended to keep (alive).
std::vector<Sweep> prepare_any(const std::vector<py::array>& data,
                               const std::vector<DArray>& azimuth,
                               const std::vector<DArray>& elevation,
                               const std::vector<DArray>& range, double R, double sr,
                               std::vector<py::array>& keep) {
    const size_t nk = data.size();
    if (nk < 1 || azimuth.size() != nk || elevation.size() != nk || range.size() != nk)
        throw std::invalid_argument("need the same number of data/azimuth/elevation/range arrays");
    std::vector<Sweep> sweeps;
    sweeps.reserve(nk);
    for (size_t k = 0; k < nk; ++k) {
        if (data[k].dtype().is(py::dtype::of<float>())) {
            FArray a = FArray::ensure(data[k]);
            if (!a) throw std::invalid_argument("could not read sweep data");
            keep.push_back(a);
            sweeps.push_back(prepare(a, azimuth[k], elevation[k], range[k], R, sr));
        } else {
            DArray a = DArray::ensure(data[k]);
            if (!a) throw std::invalid_argument("could not read sweep data");
            keep.push_back(a);
            sweeps.push_back(prepare(a, azimuth[k], elevation[k], range[k], R, sr));
        }
    }
    return sweeps;
}

}  // namespace

// Grid one field from sweeps sorted by elevation (lowest first).
py::array_t<float> grid_cones(const DArray& x, const DArray& y, const DArray& z,
                               const std::vector<DArray>& data,
                               const std::vector<DArray>& azimuth,
                               const std::vector<DArray>& elevation,
                               const std::vector<DArray>& range, double site_altitude,
                               double earth_radius, double max_gap, double min_weight,
                               bool fill_below, int n_threads, double k_e, double fill_depth,
                               const std::vector<double>& shift,
                               const std::vector<int64_t>& groups, bool group_mean) {
    const double R = earth_radius * k_e;
    const double sr = R + site_altitude;
    std::vector<Sweep> sweeps = prepare_all(data, azimuth, elevation, range, R, sr);
    const size_t nk = sweeps.size();
    if (!groups.empty()) {
        int64_t total = 0;
        for (int64_t g : groups) {
            if (g < 1) throw std::invalid_argument("every group needs at least one sweep");
            total += g;
        }
        if (total != static_cast<int64_t>(nk))
            throw std::invalid_argument("the group sizes must add up to the number of sweeps");
    }
    if (!shift.empty() && shift.size() != 2 * nk)
        throw std::invalid_argument("shift needs a (dx, dy) pair per sweep");
    for (size_t k = 0; k < shift.size() / 2; ++k) {
        sweeps[k].dx = shift[2 * k];
        sweeps[k].dy = shift[2 * k + 1];
    }

    const int64_t nx = x.size(), ny = y.size(), nz = z.size();
    const int64_t ncol = nx * ny;
    py::array_t<float> result({nz, ny, nx});  // computed in double, stored as float32
    float* out = result.mutable_data();
    const double* px = x.data();
    const double* py_ = y.data();
    const double* pz = z.data();

    {
        py::gil_scoped_release release;
        const int nt = thread_count(n_threads);
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
                    grid_column(sweeps, sc, ac, pz, nz, R, sr, max_gap, min_weight, fill_below,
                                fill_depth, vals, hts, out + c, ncol, groups, group_mean);
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

// Grid many fields of many radars in one call onto columns given by their
// ground distance and azimuth from each radar. Task t grids the sweeps data[t]
// (sorted by elevation, lowest first) of radar radar[t]; ground[r] and
// column_azimuth[r] hold radar r's (ny, nx) column geometry. The threads share
// (task, block of columns) work items.
std::vector<py::array_t<float>> grid_columns(
    const std::vector<DArray>& ground, const std::vector<DArray>& column_azimuth,
    const DArray& z, const std::vector<int64_t>& radar,
    const std::vector<std::vector<py::array>>& data,
    const std::vector<std::vector<DArray>>& azimuth,
    const std::vector<std::vector<DArray>>& elevation,
    const std::vector<std::vector<DArray>>& range, const std::vector<double>& site_altitude,
    double earth_radius, double max_gap, double min_weight, bool fill_below, int n_threads,
    double k_e, double fill_depth, const std::vector<std::vector<double>>& shift) {
    const size_t nr = ground.size();
    const size_t ntask = radar.size();
    if (nr == 0) throw std::invalid_argument("need at least one radar");
    if (column_azimuth.size() != nr || site_altitude.size() != nr)
        throw std::invalid_argument("need one ground/azimuth array and site altitude per radar");
    if (data.size() != ntask || azimuth.size() != ntask || elevation.size() != ntask ||
        range.size() != ntask)
        throw std::invalid_argument("need data/azimuth/elevation/range for every task");
    const int64_t ncol = ground[0].size();
    for (size_t r = 0; r < nr; ++r)
        if (ground[r].size() != ncol || column_azimuth[r].size() != ncol)
            throw std::invalid_argument("every radar needs the same number of columns");
    const double R = earth_radius * k_e;
    std::vector<std::vector<Sweep>> sweeps(ntask);
    std::vector<py::array> keep;  // converted inputs stay alive during the call
    std::vector<double> srs(ntask);
    size_t nk_max = 1;
    for (size_t t = 0; t < ntask; ++t) {
        if (radar[t] < 0 || static_cast<size_t>(radar[t]) >= nr)
            throw std::invalid_argument("task radar index out of range");
        srs[t] = R + site_altitude[radar[t]];
        sweeps[t] = prepare_any(data[t], azimuth[t], elevation[t], range[t], R, srs[t], keep);
        nk_max = std::max(nk_max, sweeps[t].size());
        if (!shift.empty()) {
            if (shift.size() != ntask || shift[t].size() != 2 * sweeps[t].size())
                throw std::invalid_argument("shift needs a (dx, dy) pair per sweep and task");
            for (size_t k = 0; k < sweeps[t].size(); ++k) {
                sweeps[t][k].dx = shift[t][2 * k];
                sweeps[t][k].dy = shift[t][2 * k + 1];
            }
        }
    }
    const int64_t nz = z.size();
    const std::vector<int64_t> no_groups;
    std::vector<py::ssize_t> shape(ground[0].shape(), ground[0].shape() + ground[0].ndim());
    shape.insert(shape.begin(), static_cast<py::ssize_t>(nz));
    std::vector<py::array_t<float>> results;
    std::vector<float*> outs;
    for (size_t t = 0; t < ntask; ++t) {
        results.emplace_back(shape);
        outs.push_back(results.back().mutable_data());
    }
    std::vector<const double*> pg(nr), pa(nr);
    for (size_t r = 0; r < nr; ++r) {
        pg[r] = ground[r].data();
        pa[r] = column_azimuth[r].data();
    }
    const double* pz = z.data();

    {
        py::gil_scoped_release release;
        const int nt = thread_count(n_threads);
        const int64_t block = 2048;
        const int64_t nblock = (ncol + block - 1) / block;
        const int64_t nwork = nblock * static_cast<int64_t>(ntask);
        std::atomic<int64_t> next{0};

        auto worker = [&]() {
            std::vector<double> vals(nk_max), hts(nk_max);
            for (;;) {
                const int64_t w = next.fetch_add(1);
                if (w >= nwork) break;
                const int64_t t = w / nblock;
                const int64_t c0 = (w % nblock) * block;
                const int64_t c1 = std::min(ncol, c0 + block);
                const int64_t r = radar[t];
                for (int64_t c = c0; c < c1; ++c) {
                    double ac = std::fmod(pa[r][c], 360.0);
                    if (ac < 0) ac += 360.0;
                    grid_column(sweeps[t], pg[r][c], ac, pz, nz, R, srs[t], max_gap,
                                min_weight, fill_below, fill_depth, vals, hts, outs[t] + c,
                                ncol, no_groups, false);
                }
            }
        };
        std::vector<std::thread> pool;
        for (int t = 1; t < nt; ++t) pool.emplace_back(worker);
        worker();
        for (auto& th : pool) th.join();
    }
    return results;
}

// ---------------------------------------------------------------------------
// Radius of influence gridding of gates
// ---------------------------------------------------------------------------
namespace {

enum Weighting { kCressman = 0, kBarnes = 1, kNearest = 2 };

struct GateCloud {
    std::vector<double> x, y, z, v;
    void append(const GateCloud& o) {
        x.insert(x.end(), o.x.begin(), o.x.end());
        y.insert(y.end(), o.y.begin(), o.y.end());
        z.insert(z.end(), o.z.begin(), o.z.end());
        v.insert(v.end(), o.v.begin(), o.v.end());
    }
};

struct GateSweep {
    const double* data;
    int64_t nray, ngate;
    const double* azimuth;
    const double* elevation;
    const double* range;
    double dx, dy;
};

}  // namespace

py::array_t<float> grid_gates(const DArray& cx, const DArray& cy, const DArray& z,
                              const std::vector<DArray>& data,
                              const std::vector<DArray>& azimuth,
                              const std::vector<DArray>& elevation,
                              const std::vector<DArray>& range, double site_altitude,
                              double earth_radius, double k_e, int weighting, double roi,
                              double roi_range_factor, double roi_height_factor,
                              int64_t min_gates, int n_threads,
                              const std::vector<double>& shift) {
    const size_t nk = data.size();
    if (azimuth.size() != nk || elevation.size() != nk || range.size() != nk)
        throw std::invalid_argument("need the same number of data/azimuth/elevation/range arrays");
    if (!shift.empty() && shift.size() != 2 * nk)
        throw std::invalid_argument("shift needs a (dx, dy) pair per sweep");
    if (weighting < kCressman || weighting > kNearest)
        throw std::invalid_argument("unknown weighting");
    if (!(roi > 0.0) || roi_range_factor < 0.0 || roi_height_factor < 0.0)
        throw std::invalid_argument("roi must be positive and the factors not negative");
    if (min_gates < 1) throw std::invalid_argument("min_gates must be at least 1");
    const int64_t ncol = cx.size(), nz = z.size();
    if (cy.size() != ncol) throw std::invalid_argument("cx and cy differ in length");

    std::vector<GateSweep> sweeps(nk);
    for (size_t k = 0; k < nk; ++k) {
        if (data[k].ndim() != 2) throw std::invalid_argument("sweep data must be 2-D (ray, gate)");
        const int64_t nray = data[k].shape(0), ngate = data[k].shape(1);
        if (azimuth[k].size() != nray || elevation[k].size() != nray || range[k].size() != ngate)
            throw std::invalid_argument("azimuth/elevation/range do not match sweep data shape");
        sweeps[k] = {data[k].data(),      nray,
                     ngate,               azimuth[k].data(),
                     elevation[k].data(), range[k].data(),
                     shift.empty() ? 0.0 : shift[2 * k], shift.empty() ? 0.0 : shift[2 * k + 1]};
    }

    const double R = earth_radius * k_e;
    const double sr = R + site_altitude;
    const double* pcx = cx.data();
    const double* pcy = cy.data();
    const double* pz = z.data();
    py::array_t<float> result({nz, ncol});
    float* out = result.mutable_data();
    if (ncol == 0 || nz == 0) return result;

    {
        py::gil_scoped_release release;
        const int nt = thread_count(n_threads);

        // Extent of the output points and the largest radius of influence.
        double xmin = pcx[0], xmax = pcx[0], ymin = pcy[0], ymax = pcy[0], smax = 0.0;
        for (int64_t c = 0; c < ncol; ++c) {
            xmin = std::min(xmin, pcx[c]);
            xmax = std::max(xmax, pcx[c]);
            ymin = std::min(ymin, pcy[c]);
            ymax = std::max(ymax, pcy[c]);
            smax = std::max(smax, std::hypot(pcx[c], pcy[c]));
        }
        double zmin = pz[0], zmax = pz[0];
        for (int64_t k = 0; k < nz; ++k) {
            zmin = std::min(zmin, pz[k]);
            zmax = std::max(zmax, pz[k]);
        }
        const double rmax =
            std::max(roi, roi_range_factor * smax +
                              roi_height_factor * std::max(zmax - site_altitude, 0.0));
        const double x0 = xmin - rmax, y0 = ymin - rmax, z0 = zmin - rmax;
        const double ex = xmax + rmax - x0, ey = ymax + rmax - y0, ez = zmax + rmax - z0;

        // Gates, in chunks of rays so that the order does not depend on the threads.
        struct Chunk {
            size_t sweep;
            int64_t r0, r1;
        };
        std::vector<Chunk> chunks;
        const int64_t rays_per_chunk = 32;
        for (size_t k = 0; k < nk; ++k)
            for (int64_t r0 = 0; r0 < sweeps[k].nray; r0 += rays_per_chunk)
                chunks.push_back({k, r0, std::min(sweeps[k].nray, r0 + rays_per_chunk)});
        std::vector<GateCloud> parts(chunks.size());
        {
            std::atomic<int64_t> next{0};
            auto worker = [&]() {
                for (;;) {
                    const int64_t w = next.fetch_add(1);
                    if (w >= static_cast<int64_t>(chunks.size())) break;
                    const Chunk& ch = chunks[w];
                    const GateSweep& sw = sweeps[ch.sweep];
                    GateCloud& cloud = parts[w];
                    for (int64_t q = ch.r0; q < ch.r1; ++q) {
                        const double el = sw.elevation[q] * kDeg, az = sw.azimuth[q] * kDeg;
                        if (!std::isfinite(el) || !std::isfinite(az)) continue;
                        const double sin_el = std::sin(el), cos_el = std::cos(el);
                        const double sin_az = std::sin(az), cos_az = std::cos(az);
                        for (int64_t g = 0; g < sw.ngate; ++g) {
                            const double val = sw.data[q * sw.ngate + g];
                            if (std::isnan(val)) continue;
                            const double r = sw.range[g];
                            const double b =
                                std::sqrt(2.0 * sin_el * r * sr + r * r + sr * sr);
                            const double gs = R * std::asin(r * cos_el / b);
                            const double gx = gs * sin_az + sw.dx, gy = gs * cos_az + sw.dy;
                            const double gz = b - R;
                            if (!(gx >= x0 && gx <= x0 + ex && gy >= y0 && gy <= y0 + ey &&
                                  gz >= z0 && gz <= z0 + ez))
                                continue;
                            cloud.x.push_back(gx);
                            cloud.y.push_back(gy);
                            cloud.z.push_back(gz);
                            cloud.v.push_back(val);
                        }
                    }
                }
            };
            std::vector<std::thread> pool;
            for (int t = 1; t < nt; ++t) pool.emplace_back(worker);
            worker();
            for (auto& th : pool) th.join();
        }
        GateCloud cloud;
        size_t total = 0;
        for (const auto& p : parts) total += p.x.size();
        cloud.x.reserve(total);
        cloud.y.reserve(total);
        cloud.z.reserve(total);
        cloud.v.reserve(total);
        for (auto& p : parts) {
            cloud.append(p);
            p = GateCloud();
        }

        // Bucket grid: the cell is at least the largest radius, and coarser when the
        // domain would need too many buckets.
        double h = rmax;
        auto nb = [&](double e) { return static_cast<int64_t>(std::floor(e / h)) + 1; };
        while (static_cast<double>(nb(ex)) * nb(ey) * nb(ez) > 8.0e6) h *= 1.5;
        const int64_t nbx = nb(ex), nby = nb(ey), nbz = nb(ez);
        auto bucket = [&](double v, double v0, int64_t n) {
            return std::min(n - 1, std::max<int64_t>(0, static_cast<int64_t>((v - v0) / h)));
        };
        const size_t ng = cloud.x.size();
        std::vector<int64_t> start(static_cast<size_t>(nbx * nby * nbz) + 1, 0);
        std::vector<int64_t> key(ng);
        for (size_t i = 0; i < ng; ++i) {
            key[i] = (bucket(cloud.z[i], z0, nbz) * nby + bucket(cloud.y[i], y0, nby)) * nbx +
                     bucket(cloud.x[i], x0, nbx);
            ++start[key[i] + 1];
        }
        for (size_t b = 1; b < start.size(); ++b) start[b] += start[b - 1];
        std::vector<double> sx(ng), sy(ng), sz(ng), sv(ng);
        {
            std::vector<int64_t> fill(start.begin(), start.end() - 1);
            for (size_t i = 0; i < ng; ++i) {
                const int64_t at = fill[key[i]]++;
                sx[at] = cloud.x[i];
                sy[at] = cloud.y[i];
                sz[at] = cloud.z[i];
                sv[at] = cloud.v[i];
            }
        }

        std::atomic<int64_t> next{0};
        const int64_t block = 256;
        auto worker = [&]() {
            for (;;) {
                const int64_t c0 = next.fetch_add(block);
                if (c0 >= ncol) break;
                const int64_t c1 = std::min(ncol, c0 + block);
                for (int64_t c = c0; c < c1; ++c) {
                    const double xc = pcx[c], yc = pcy[c];
                    const double sc = std::hypot(xc, yc);
                    for (int64_t k = 0; k < nz; ++k) {
                        const double zc = pz[k];
                        const double rad = std::max(
                            roi, roi_range_factor * sc + roi_height_factor * (zc - site_altitude));
                        const double r2 = rad * rad;
                        const int64_t ix0 = bucket(xc - rad, x0, nbx), ix1 = bucket(xc + rad, x0, nbx);
                        const int64_t iy0 = bucket(yc - rad, y0, nby), iy1 = bucket(yc + rad, y0, nby);
                        const int64_t iz0 = bucket(zc - rad, z0, nbz), iz1 = bucket(zc + rad, z0, nbz);
                        double num = 0.0, den = 0.0, best = r2, nearest = kNaN;
                        int64_t count = 0;
                        for (int64_t iz = iz0; iz <= iz1; ++iz)
                            for (int64_t iy = iy0; iy <= iy1; ++iy) {
                                const int64_t row = (iz * nby + iy) * nbx;
                                for (int64_t i = start[row + ix0]; i < start[row + ix1 + 1]; ++i) {
                                    const double ddx = sx[i] - xc, ddy = sy[i] - yc,
                                                 ddz = sz[i] - zc;
                                    const double d2 = ddx * ddx + ddy * ddy + ddz * ddz;
                                    if (!(d2 < r2)) continue;
                                    ++count;
                                    if (weighting == kNearest) {
                                        if (d2 < best) {
                                            best = d2;
                                            nearest = sv[i];
                                        }
                                    } else {
                                        const double w = weighting == kCressman
                                                             ? (r2 - d2) / (r2 + d2)
                                                             : std::exp(-4.0 * d2 / r2);
                                        num += w * sv[i];
                                        den += w;
                                    }
                                }
                            }
                        double res = kNaN;
                        if (count >= min_gates) {
                            if (weighting == kNearest)
                                res = nearest;
                            else if (den > 0.0)
                                res = num / den;
                        }
                        out[k * ncol + c] = static_cast<float>(res);
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
          py::arg("n_threads") = 0, py::arg("k_e") = 4.0 / 3.0,
          py::arg("fill_depth") = std::numeric_limits<double>::infinity(),
          py::arg("shift") = std::vector<double>(),
          py::arg("groups") = std::vector<int64_t>(), py::arg("group_mean") = false);
    m.def("grid_gates", &grid_gates, py::arg("cx"), py::arg("cy"), py::arg("z"),
          py::arg("data"), py::arg("azimuth"), py::arg("elevation"), py::arg("range"),
          py::arg("site_altitude"), py::arg("earth_radius") = 6371000.0,
          py::arg("k_e") = 4.0 / 3.0, py::arg("weighting") = 0, py::arg("roi") = 1000.0,
          py::arg("roi_range_factor") = 0.0, py::arg("roi_height_factor") = 0.0,
          py::arg("min_gates") = 1, py::arg("n_threads") = 0,
          py::arg("shift") = std::vector<double>());
    m.def("grid_columns", &grid_columns, py::arg("ground"), py::arg("column_azimuth"),
          py::arg("z"), py::arg("radar"), py::arg("data"), py::arg("azimuth"),
          py::arg("elevation"), py::arg("range"), py::arg("site_altitude"),
          py::arg("earth_radius") = 6371000.0, py::arg("max_gap") = 2.0,
          py::arg("min_weight") = 0.5, py::arg("fill_below") = false, py::arg("n_threads") = 0,
          py::arg("k_e") = 4.0 / 3.0,
          py::arg("fill_depth") = std::numeric_limits<double>::infinity(),
          py::arg("shift") = std::vector<std::vector<double>>());
}
