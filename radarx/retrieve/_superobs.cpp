// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Super-observation kernel: bins the gates of one sweep into columns and
// height layers and sums what the averages are made of.
//
// A gate is placed with the 4/3 effective Earth model (Doviak and Zrnic 1993,
// Doppler Radar and Weather Observations, 2nd ed., Eqs. 2.28b-d, pp. 21-22):
// its height above sea level, its ground distance, its latitude and longitude
// on the great circle of the ray, and the unit vector of the ray at the gate
// in east, north and up components. The column of a gate is either a box of a
// regular grid of the plane around the radar (mode 0) or a block of cells x
// cells model cells found by a nearest-cell search (mode 1). The layer comes
// from the layer edges. For every (column, layer) bin the kernel sums gate
// counts, linear reflectivity, reflectivity and velocity (and their squares),
// the radial unit vector and the position; the means, spreads and error
// estimates are made from these sums in radarx/retrieve/superobs.py.
//
// Rays are cut into fixed blocks that threads take from an atomic counter.
// Each block sums into its own small table, so the result does not depend on
// the number of threads; the tables are added in block order. Every step
// follows the NumPy reference in radarx/retrieve/superobs.py, in the same order
// of operations.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;
using DArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kPi = 3.14159265358979323846;  // M_PI is not standard (MSVC)
constexpr double kDeg = kPi / 180.0;
constexpr int64_t kBlockRays = 8;  // rays per scheduling block

// Columns of the table of sums.
enum Stat {
    N_ALL = 0,  // gates in the bin
    N_ZS,       // gates of sweeps that have a reflectivity field
    N_Z,        // gates with a valid reflectivity
    S_Z,        // sum of Z [mm6 m-3]
    S_D,        // sum of dBZ
    S_D2,       // sum of dBZ squared
    N_VS,       // gates of sweeps that have a velocity field
    N_V,        // gates with a valid velocity
    S_V,        // sum of velocity [m s-1]
    S_V2,       // sum of velocity squared
    S_EX,       // sums of the east, north and up components of the ray
    S_EY,
    S_EZ,
    S_LAT,      // sum of latitudes [deg]
    S_DLON,     // sum of longitude minus radar longitude [deg]
    S_H,        // sum of heights above sea level [m]
    S_R,        // sum of slant ranges [m]
    N_STATS
};

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

// Nearest model cell of a point, by buckets of the size of the search radius.
class GridIndex {
public:
    GridIndex(const DArray& x, const DArray& y, double max_dist) : max_dist_(max_dist) {
        if (x.ndim() != 2 || y.ndim() != 2 || x.shape(0) != y.shape(0) || x.shape(1) != y.shape(1))
            throw std::invalid_argument("x and y must be 2-D arrays of the same shape");
        if (!(max_dist > 0.0)) throw std::invalid_argument("max_dist must be positive");
        ny_ = x.shape(0);
        nx_ = x.shape(1);
        const int64_t n = ny_ * nx_;
        x_.assign(x.data(), x.data() + n);
        y_.assign(y.data(), y.data() + n);
        double xmax = -std::numeric_limits<double>::infinity();
        double ymax = xmax;
        xmin_ = std::numeric_limits<double>::infinity();
        ymin_ = xmin_;
        for (int64_t k = 0; k < n; ++k) {
            if (!std::isfinite(x_[k]) || !std::isfinite(y_[k])) continue;
            xmin_ = std::min(xmin_, x_[k]);
            xmax = std::max(xmax, x_[k]);
            ymin_ = std::min(ymin_, y_[k]);
            ymax = std::max(ymax, y_[k]);
        }
        if (!std::isfinite(xmin_)) throw std::invalid_argument("the grid has no finite cell");
        const double wx = std::floor((xmax - xmin_) / max_dist) + 1.0;
        const double wy = std::floor((ymax - ymin_) / max_dist) + 1.0;
        if (wx * wy > 1.0e8) throw std::invalid_argument("the grid is too large for its spacing");
        nbx_ = static_cast<int64_t>(wx);
        nby_ = static_cast<int64_t>(wy);
        start_.assign(nbx_ * nby_ + 1, 0);
        std::vector<int64_t> bucket(n, -1);
        for (int64_t k = 0; k < n; ++k) {
            if (!std::isfinite(x_[k]) || !std::isfinite(y_[k])) continue;
            const int64_t bx = static_cast<int64_t>(std::floor((x_[k] - xmin_) / max_dist));
            const int64_t by = static_cast<int64_t>(std::floor((y_[k] - ymin_) / max_dist));
            bucket[k] = by * nbx_ + bx;
            ++start_[bucket[k] + 1];
        }
        for (int64_t b = 0; b < nbx_ * nby_; ++b) start_[b + 1] += start_[b];
        item_.resize(start_[nbx_ * nby_]);
        std::vector<int64_t> fill(start_.begin(), start_.end() - 1);
        for (int64_t k = 0; k < n; ++k)
            if (bucket[k] >= 0) item_[fill[bucket[k]]++] = k;
    }

    // Flat index of the nearest cell within max_dist, else -1. Ties take the lower index.
    int64_t nearest(double qx, double qy) const {
        const double fx = std::floor((qx - xmin_) / max_dist_);
        const double fy = std::floor((qy - ymin_) / max_dist_);
        if (!(fx >= -1.0 && fx <= static_cast<double>(nbx_) && fy >= -1.0 &&
              fy <= static_cast<double>(nby_)))
            return -1;
        const int64_t bx = static_cast<int64_t>(fx);
        const int64_t by = static_cast<int64_t>(fy);
        int64_t best = -1;
        double best_d2 = max_dist_ * max_dist_;
        for (int64_t j = std::max<int64_t>(by - 1, 0); j <= std::min(by + 1, nby_ - 1); ++j) {
            for (int64_t i = std::max<int64_t>(bx - 1, 0); i <= std::min(bx + 1, nbx_ - 1); ++i) {
                const int64_t b = j * nbx_ + i;
                for (int64_t p = start_[b]; p < start_[b + 1]; ++p) {
                    const int64_t k = item_[p];
                    const double dx = x_[k] - qx;
                    const double dy = y_[k] - qy;
                    const double d2 = dx * dx + dy * dy;
                    if (d2 < best_d2 || (d2 == best_d2 && (best < 0 || k < best))) {
                        best_d2 = d2;
                        best = k;
                    }
                }
            }
        }
        return best;
    }

    int64_t ny() const { return ny_; }
    int64_t nx() const { return nx_; }

private:
    double max_dist_, xmin_, ymin_;
    int64_t ny_, nx_, nbx_, nby_;
    std::vector<double> x_, y_;
    std::vector<int64_t> start_, item_;
};

namespace {

struct Table {
    std::vector<int64_t> bins;
    std::vector<double> stats;  // N_STATS per bin
};

// One block of rays: sums the gates into a table sorted by bin.
Table sum_block(int64_t r0, int64_t r1, const double* az, const double* el, const double* rng,
                int64_t ngate, const double* dbz, const double* vr, const double* rho,
                double lat0, double lon0, double alt0, double reff, double rearth,
                double max_range, double min_ground, const double* edges, int64_t nedge,
                bool clear, double vmax, double rho_min, int mode, double box, int64_t half,
                int64_t cells, int64_t nbx_blocks, double lat_ref, double lon_ref,
                const GridIndex* index) {
    const int64_t nlev = nedge - 1;
    const int64_t width = 2 * half + 1;
    const double slat0 = std::sin(lat0), clat0 = std::cos(lat0);
    const double cos_ref = std::cos(lat_ref), sin_ref = std::sin(lat_ref);
    const bool has_z = dbz != nullptr, has_v = vr != nullptr;
    std::unordered_map<int64_t, int64_t> slot;
    slot.reserve(256);
    std::vector<int64_t> bins;
    std::vector<double> sums;

    for (int64_t ray = r0; ray < r1; ++ray) {
        const double a = az[ray] * kDeg, e = el[ray] * kDeg;
        const double saz = std::sin(a), caz = std::cos(a), sel = std::sin(e), cel = std::cos(e);
        const double* zrow = has_z ? dbz + ray * ngate : nullptr;
        const double* vrow = has_v ? vr + ray * ngate : nullptr;
        const double* rrow = rho ? rho + ray * ngate : nullptr;
        for (int64_t g = 0; g < ngate; ++g) {
            const double r = rng[g];
            if (!(r > 0.0) || r > max_range) continue;
            bool zok = false, vok = false;
            double zd = 0.0, vv = 0.0;
            if (has_z) {
                zd = zrow[g];
                zok = std::isfinite(zd);
            }
            if (has_v) {
                vv = vrow[g];
                vok = std::isfinite(vv) && std::fabs(vv) <= vmax;
            }
            if (rrow && std::isfinite(rrow[g]) && rrow[g] < rho_min) zok = vok = false;
            if (!clear && !zok && !vok) continue;

            // site altitude in the Earth radius, as in xradar and radarx.fundamentals.geometry
            const double ra = reff + alt0;
            const double bd = std::sqrt(r * r + ra * ra + 2.0 * r * ra * sel);
            const double hgt = bd - ra;
            const double hm = hgt + alt0;
            if (!(hm >= edges[0] && hm < edges[nedge - 1])) continue;
            const double s = reff * std::asin(r * cel / bd);
            if (!(s > min_ground)) continue;
            const int64_t level = (std::upper_bound(edges, edges + nedge, hm) - edges) - 1;

            const double d = s / rearth;
            const double sd = std::sin(d), cd = std::cos(d);
            double sl = slat0 * cd + clat0 * sd * caz;
            sl = std::min(1.0, std::max(-1.0, sl));
            const double lat = std::asin(sl);
            const double dlon = std::atan2(saz * sd * clat0, cd - slat0 * sl);

            int64_t col;
            if (mode == 0) {
                const double fx = std::floor(s * saz / box);
                const double fy = std::floor(s * caz / box);
                const int64_t ix = static_cast<int64_t>(fx) + half;
                const int64_t iy = static_cast<int64_t>(fy) + half;
                if (ix < 0 || ix >= width || iy < 0 || iy >= width) continue;
                col = iy * width + ix;
            } else {
                // azimuthal equidistant projection about the reference point
                const double dl = lon0 + dlon - lon_ref;
                const double cl = std::cos(lat);
                const double pa = cl * std::sin(dl);
                const double pb = cos_ref * sl - sin_ref * cl * std::cos(dl);
                const double pc = sin_ref * sl + cos_ref * cl * std::cos(dl);
                const double pr = std::hypot(pa, pb);
                const double kp = pr > 0.0 ? std::atan2(pr, pc) / pr : 1.0;
                const int64_t k = index->nearest(rearth * kp * pa, rearth * kp * pb);
                if (k < 0) continue;
                col = ((k / index->nx()) / cells) * nbx_blocks + ((k % index->nx()) / cells);
            }
            const int64_t bin = col * nlev + level;

            int64_t pos;
            auto it = slot.find(bin);
            if (it == slot.end()) {
                pos = static_cast<int64_t>(bins.size());
                slot.emplace(bin, pos);
                bins.push_back(bin);
                sums.resize(sums.size() + N_STATS, 0.0);
            } else {
                pos = it->second;
            }
            double* t = sums.data() + pos * N_STATS;
            t[N_ALL] += 1.0;
            t[S_LAT] += lat / kDeg;
            t[S_DLON] += dlon / kDeg;
            t[S_H] += hm;
            t[S_R] += r;
            if (has_z) {
                t[N_ZS] += 1.0;
                if (zok) {
                    t[N_Z] += 1.0;
                    t[S_Z] += std::pow(10.0, zd / 10.0);
                    t[S_D] += zd;
                    t[S_D2] += zd * zd;
                }
            }
            if (has_v) {
                t[N_VS] += 1.0;
                if (vok) {
                    t[N_V] += 1.0;
                    t[S_V] += vv;
                    t[S_V2] += vv * vv;
                    // ray direction at the gate: bearing of the great circle there and the
                    // elevation relative to the local horizontal (4/3 Earth)
                    const double aa = std::sin(dlon) * clat0;
                    const double bb = std::cos(lat) * slat0 - sl * clat0 * std::cos(dlon);
                    const double nrm = std::hypot(aa, bb);
                    const double eg = e + s / reff;
                    const double ceg = std::cos(eg);
                    t[S_EX] += (aa / nrm) * ceg;
                    t[S_EY] += (-bb / nrm) * ceg;
                    t[S_EZ] += std::sin(eg);
                }
            }
        }
    }

    // sort the bins of the block
    std::vector<int64_t> order(bins.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int64_t>(i);
    std::sort(order.begin(), order.end(), [&](int64_t p, int64_t q) { return bins[p] < bins[q]; });
    Table out;
    out.bins.resize(bins.size());
    out.stats.resize(sums.size());
    for (size_t i = 0; i < order.size(); ++i) {
        out.bins[i] = bins[order[i]];
        std::copy(sums.begin() + order[i] * N_STATS, sums.begin() + (order[i] + 1) * N_STATS,
                  out.stats.begin() + i * N_STATS);
    }
    return out;
}

}  // namespace

// Returns (bins, stats): the table of the sweep sorted by bin, bins (n,) int64 and
// stats (n, N_STATS) float64. An empty array for dbz, vr or rho means that the sweep has no
// such field.
py::tuple bin_sweep_py(const DArray& azimuth, const DArray& elevation, const DArray& range,
                       const DArray& dbz, const DArray& vr, const DArray& rho, double lat0,
                       double lon0, double alt0, double reff, double rearth, double max_range,
                       double min_ground, const DArray& edges, bool clear, double vmax,
                       double rho_min, int mode, double box, int64_t half, int64_t cells,
                       int64_t nbx_blocks, double lat_ref, double lon_ref,
                       const GridIndex* index, int n_threads) {
    const int64_t nray = azimuth.shape(0);
    const int64_t ngate = range.shape(0);
    if (elevation.ndim() != 1 || elevation.shape(0) != nray)
        throw std::invalid_argument("elevation must have one value per ray");
    const int64_t expected = nray * ngate;
    for (const DArray* f : {&dbz, &vr, &rho})
        if (f->size() != 0 && f->size() != expected)
            throw std::invalid_argument("a field must have the shape (rays, gates)");
    if (edges.ndim() != 1 || edges.shape(0) < 2)
        throw std::invalid_argument("edges must hold at least two layer edges");
    if (mode != 0 && mode != 1) throw std::invalid_argument("mode must be 0 (boxes) or 1 (grid)");
    if (mode == 1 && (index == nullptr || cells < 1))
        throw std::invalid_argument("grid mode needs a grid index and cells >= 1");
    if (mode == 0 && !(box > 0.0)) throw std::invalid_argument("box must be positive");

    const int64_t nblocks = (nray + kBlockRays - 1) / kBlockRays;
    std::vector<Table> tables(nblocks);
    const double* dp = dbz.size() ? dbz.data() : nullptr;
    const double* vp = vr.size() ? vr.data() : nullptr;
    const double* rp = rho.size() ? rho.data() : nullptr;
    {
        py::gil_scoped_release release;
        parallel_blocks(nblocks, n_threads, [&](int64_t b) {
            const int64_t r0 = b * kBlockRays;
            const int64_t r1 = std::min(nray, r0 + kBlockRays);
            tables[b] = sum_block(r0, r1, azimuth.data(), elevation.data(), range.data(), ngate,
                                  dp, vp, rp, lat0 * kDeg, lon0 * kDeg, alt0, reff, rearth,
                                  max_range, min_ground, edges.data(), edges.shape(0), clear,
                                  vmax, rho_min, mode, box, half, cells, nbx_blocks,
                                  lat_ref * kDeg, lon_ref * kDeg, index);
        });
    }
    // add the tables of the blocks, in block order, into one table sorted by bin
    struct Ref {
        int64_t bin;
        int32_t table;
        int32_t row;
    };
    std::vector<Ref> refs;
    for (size_t b = 0; b < tables.size(); ++b)
        for (size_t i = 0; i < tables[b].bins.size(); ++i)
            refs.push_back({tables[b].bins[i], static_cast<int32_t>(b), static_cast<int32_t>(i)});
    std::stable_sort(refs.begin(), refs.end(),
                     [](const Ref& p, const Ref& q) { return p.bin < q.bin; });
    int64_t total = 0;
    for (size_t i = 0; i < refs.size(); ++i)
        if (i == 0 || refs[i].bin != refs[i - 1].bin) ++total;
    py::array_t<int64_t> bins(total);
    py::array_t<double> stats({total, static_cast<int64_t>(N_STATS)});
    int64_t* bo = bins.mutable_data();
    double* so = stats.mutable_data();
    std::fill(so, so + total * N_STATS, 0.0);
    int64_t at = -1;
    for (size_t i = 0; i < refs.size(); ++i) {
        if (i == 0 || refs[i].bin != refs[i - 1].bin) {
            ++at;
            bo[at] = refs[i].bin;
        }
        const double* src = tables[refs[i].table].stats.data() + refs[i].row * N_STATS;
        for (int c = 0; c < N_STATS; ++c) so[at * N_STATS + c] += src[c];
    }
    return py::make_tuple(bins, stats);
}

PYBIND11_MODULE(_superobs, m) {
    m.doc() = "Super-observation binning kernel";
    py::class_<GridIndex>(m, "GridIndex")
        .def(py::init<const DArray&, const DArray&, double>(), py::arg("x"), py::arg("y"),
             py::arg("max_dist"))
        .def("nearest",
             [](const GridIndex& g, double x, double y) { return g.nearest(x, y); })
        .def_property_readonly("shape",
                               [](const GridIndex& g) { return py::make_tuple(g.ny(), g.nx()); });
    m.attr("N_STATS") = static_cast<int>(N_STATS);
    m.def("bin_sweep", &bin_sweep_py, py::arg("azimuth"), py::arg("elevation"), py::arg("range"),
          py::arg("dbz"), py::arg("vr"), py::arg("rho"), py::arg("lat0"), py::arg("lon0"),
          py::arg("alt0"), py::arg("reff"), py::arg("rearth"), py::arg("max_range"),
          py::arg("min_ground"), py::arg("edges"), py::arg("clear"), py::arg("vmax"),
          py::arg("rho_min"), py::arg("mode"), py::arg("box"), py::arg("half"), py::arg("cells"),
          py::arg("nbx_blocks"), py::arg("lat_ref"), py::arg("lon_ref"),
          py::arg("index").none(true), py::arg("n_threads") = 0);
}
