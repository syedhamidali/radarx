// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Lightning (Lightning Mapping Array) kernel.
//
// cluster(): VHF sources, sorted by time, are grouped into flashes. Two
// sources are linked when their normalized space-time separation
// (d / distance)^2 + (dt / time)^2 <= 1; flashes are the connected components
// of these links (DBSCAN with a minimum of one point). Every source only looks
// forward in time up to `time` seconds, so the sources form one pool of work
// that threads take in blocks from an atomic counter; links go into a
// lock-free union-find (compare-and-swap, root of the smaller index wins), so
// the components do not depend on the thread schedule.
//
// flash_stats(): per flash, the number of sources, its first and last source
// and the area of the convex hull of its sources (Andrew's monotone chain).
//
// grid(): source density (sources per pixel), flash extent density (flashes
// with at least one source in the pixel) and flash initiation density (first
// source of each flash) on rectilinear (t, z, y, x) bins. Flashes are the pool
// of work; pixel counters are atomic.
//
// cells(): flash and source counts per cell of a cell-label mask that changes
// in time (e.g. a tracked-storm segmentation), per time bin (and height bin).
//
// Every function follows the NumPy reference in radarx/retrieve/lightning.py.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace py = pybind11;
using DArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using LArray = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;
using IArray = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;
using BArray = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;

namespace {

constexpr int64_t kBlock = 256;  // work items per scheduling block

int thread_count(int n_threads, int64_t nblocks) {
    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    int nt = n_threads > 0 ? n_threads : static_cast<int>(hw);
    return static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(nt, nblocks)));
}

// Run body(begin, end, scratch) over [0, total) in blocks taken from an atomic
// counter. Each thread owns one Scratch object for the whole run.
template <class Scratch, class F>
void parallel_blocks(int64_t total, int n_threads, F&& body) {
    if (total <= 0) return;
    const int64_t nblocks = (total + kBlock - 1) / kBlock;
    const int nt = thread_count(n_threads, nblocks);
    std::atomic<int64_t> next{0};
    auto worker = [&]() {
        Scratch scratch;
        for (;;) {
            const int64_t g0 = next.fetch_add(kBlock);
            if (g0 >= total) break;
            body(g0, std::min(total, g0 + kBlock), scratch);
        }
    };
    std::vector<std::thread> pool;
    for (int i = 1; i < nt; ++i) pool.emplace_back(worker);
    worker();
    for (auto& th : pool) th.join();
}

struct NoScratch {};

// ---------------------------------------------------------------- union-find

struct UnionFind {
    std::unique_ptr<std::atomic<int64_t>[]> parent;
    explicit UnionFind(int64_t n) : parent(new std::atomic<int64_t>[n]) {
        for (int64_t i = 0; i < n; ++i) parent[i].store(i, std::memory_order_relaxed);
    }
    int64_t find(int64_t i) {
        for (;;) {
            int64_t p = parent[i].load(std::memory_order_relaxed);
            if (p == i) return i;
            int64_t gp = parent[p].load(std::memory_order_relaxed);
            if (gp != p) parent[i].compare_exchange_weak(p, gp, std::memory_order_relaxed);
            i = gp;
        }
    }
    void unite(int64_t a, int64_t b) {
        for (;;) {
            a = find(a);
            b = find(b);
            if (a == b) return;
            if (a < b) std::swap(a, b);  // link the larger root below the smaller
            int64_t expected = a;
            if (parent[a].compare_exchange_strong(expected, b, std::memory_order_acq_rel))
                return;
        }
    }
};

void check_1d(const py::array& a, int64_t n, const char* name) {
    if (a.ndim() != 1 || a.shape(0) != n)
        throw std::invalid_argument(std::string(name) + " must be 1-D with one value per source");
}

// Sources grouped by flash: order[offsets[f] .. offsets[f + 1]) are the
// sources of flash f in increasing index (time) order. Labels < 0 or >= nflash
// belong to no flash.
void group_by_flash(const int64_t* labels, int64_t n, int64_t nflash,
                    std::vector<int64_t>& offsets, std::vector<int64_t>& order) {
    offsets.assign(nflash + 1, 0);
    for (int64_t i = 0; i < n; ++i) {
        const int64_t f = labels[i];
        if (f >= 0 && f < nflash) ++offsets[f + 1];
    }
    for (int64_t f = 0; f < nflash; ++f) offsets[f + 1] += offsets[f];
    order.resize(offsets[nflash]);
    std::vector<int64_t> fill(offsets.begin(), offsets.end() - 1);
    for (int64_t i = 0; i < n; ++i) {
        const int64_t f = labels[i];
        if (f >= 0 && f < nflash) order[fill[f]++] = i;
    }
}

// Bin of v in the sorted edges e[0..n] (n bins, [e_k, e_k+1)), or -1.
inline int64_t bin_of(const double* e, int64_t n, double v) {
    if (!(v >= e[0]) || !(v < e[n])) return -1;  // also rejects NaN
    const double* p = std::upper_bound(e, e + n + 1, v);
    return static_cast<int64_t>(p - e) - 1;
}

struct Axis {
    const double* e;
    int64_t n;
};

Axis axis_of(const DArray& edges, const char* name) {
    if (edges.ndim() != 1 || edges.shape(0) < 2)
        throw std::invalid_argument(std::string(name) + " edges need at least two values");
    return Axis{edges.data(), edges.shape(0) - 1};
}

// Vectors reused by one thread for the pixels of one flash.
struct PixelScratch {
    std::vector<int64_t> keys;
};

// Andrew's monotone chain: area of the convex hull of the points (2-D).
struct HullScratch {
    std::vector<std::pair<double, double>> pts, hull;
};

inline double cross(const std::pair<double, double>& o, const std::pair<double, double>& a,
                    const std::pair<double, double>& b) {
    return (a.first - o.first) * (b.second - o.second) -
           (a.second - o.second) * (b.first - o.first);
}

double hull_area(HullScratch& s) {
    auto& p = s.pts;
    std::sort(p.begin(), p.end());
    p.erase(std::unique(p.begin(), p.end()), p.end());
    const size_t n = p.size();
    if (n < 3) return 0.0;
    auto& h = s.hull;
    h.assign(2 * n, {0.0, 0.0});
    size_t k = 0;
    for (size_t i = 0; i < n; ++i) {
        while (k >= 2 && cross(h[k - 2], h[k - 1], p[i]) <= 0.0) --k;
        h[k++] = p[i];
    }
    for (size_t i = n - 1, t = k + 1; i > 0; --i) {
        while (k >= t && cross(h[k - 2], h[k - 1], p[i - 1]) <= 0.0) --k;
        h[k++] = p[i - 1];
    }
    // h[0 .. k-1] closes on itself (h[k-1] == h[0])
    double a = 0.0;
    for (size_t i = 0; i + 1 < k; ++i)
        a += h[i].first * h[i + 1].second - h[i + 1].first * h[i].second;
    return 0.5 * std::abs(a);
}

template <class T>
py::array_t<T> from_atomic(const std::unique_ptr<std::atomic<T>[]>& a,
                           const std::vector<py::ssize_t>& shape, int64_t size) {
    py::array_t<T> out(shape);
    T* o = out.mutable_data();
    for (int64_t i = 0; i < size; ++i) o[i] = a[i].load(std::memory_order_relaxed);
    return out;
}

}  // namespace

// Flash label (0 .. nflash - 1, numbered by first source) of every source.
// x, y, z in metres, t in seconds, sorted by t.
py::array_t<int64_t> cluster_py(const DArray& x, const DArray& y, const DArray& z,
                                const DArray& t, double distance, double time,
                                int n_threads) {
    const int64_t n = t.ndim() == 1 ? t.shape(0) : -1;
    if (n < 0) throw std::invalid_argument("t must be 1-D");
    check_1d(x, n, "x");
    check_1d(y, n, "y");
    check_1d(z, n, "z");
    if (!(distance > 0.0) || !(time > 0.0))
        throw std::invalid_argument("distance and time must be positive");
    const double *px = x.data(), *py_ = y.data(), *pz = z.data(), *pt = t.data();
    for (int64_t i = 1; i < n; ++i)
        if (!(pt[i] >= pt[i - 1])) throw std::invalid_argument("sources must be sorted by time");
    py::array_t<int64_t> labels(n);
    int64_t* out = labels.mutable_data();
    {
        py::gil_scoped_release release;
        UnionFind uf(std::max<int64_t>(n, 1));
        const double id2 = 1.0 / (distance * distance), it2 = 1.0 / (time * time);
        parallel_blocks<NoScratch>(n, n_threads, [&](int64_t i0, int64_t i1, NoScratch&) {
            for (int64_t i = i0; i < i1; ++i) {
                for (int64_t j = i + 1; j < n; ++j) {
                    const double dt = pt[j] - pt[i];
                    const double tt = dt * dt * it2;
                    if (tt > 1.0) break;
                    const double dx = px[j] - px[i], dy = py_[j] - py_[i], dz = pz[j] - pz[i];
                    if ((dx * dx + dy * dy + dz * dz) * id2 + tt <= 1.0) uf.unite(i, j);
                }
            }
        });
        // number the flashes in the order of their first source
        std::vector<int64_t> id(n, -1);
        int64_t next = 0;
        for (int64_t i = 0; i < n; ++i) {
            const int64_t r = uf.find(i);
            if (id[r] < 0) id[r] = next++;
            out[i] = id[r];
        }
    }
    return labels;
}

// Number of sources, first and last source and convex-hull area of (x, y) of
// every flash. Returns (count, first, last, area).
py::tuple flash_stats_py(const LArray& labels, int64_t nflash, const DArray& x,
                         const DArray& y, int n_threads) {
    const int64_t n = labels.ndim() == 1 ? labels.shape(0) : -1;
    if (n < 0) throw std::invalid_argument("labels must be 1-D");
    check_1d(x, n, "x");
    check_1d(y, n, "y");
    if (nflash < 0) throw std::invalid_argument("nflash must not be negative");
    py::array_t<int64_t> count(nflash), first(nflash), last(nflash);
    py::array_t<double> area(nflash);
    int64_t *pc = count.mutable_data(), *pf = first.mutable_data(), *pl = last.mutable_data();
    double* pa = area.mutable_data();
    const int64_t* lab = labels.data();
    const double *px = x.data(), *py_ = y.data();
    {
        py::gil_scoped_release release;
        std::vector<int64_t> offsets, order;
        group_by_flash(lab, n, nflash, offsets, order);
        parallel_blocks<HullScratch>(nflash, n_threads,
                                     [&](int64_t f0, int64_t f1, HullScratch& s) {
            for (int64_t f = f0; f < f1; ++f) {
                const int64_t a = offsets[f], b = offsets[f + 1];
                pc[f] = b - a;
                pf[f] = b > a ? order[a] : -1;
                pl[f] = b > a ? order[b - 1] : -1;
                s.pts.clear();
                for (int64_t k = a; k < b; ++k) {
                    const int64_t i = order[k];
                    if (std::isfinite(px[i]) && std::isfinite(py_[i]))
                        s.pts.emplace_back(px[i], py_[i]);
                }
                pa[f] = hull_area(s);
            }
        });
    }
    return py::make_tuple(count, first, last, area);
}

// Source, flash extent and flash initiation counts on (t, z, y, x) bins given
// by their edges. Flashes with flash_ok == 0 are left out of the flash
// products; `first` is the initiating source of each flash.
py::tuple grid_py(const DArray& x, const DArray& y, const DArray& z, const DArray& t,
                  const LArray& labels, const LArray& first, const BArray& flash_ok,
                  const DArray& xe, const DArray& ye, const DArray& ze, const DArray& te,
                  int n_threads) {
    const int64_t n = t.ndim() == 1 ? t.shape(0) : -1;
    if (n < 0) throw std::invalid_argument("t must be 1-D");
    check_1d(x, n, "x");
    check_1d(y, n, "y");
    check_1d(z, n, "z");
    check_1d(labels, n, "labels");
    const int64_t nflash = first.ndim() == 1 ? first.shape(0) : -1;
    if (nflash < 0 || flash_ok.ndim() != 1 || flash_ok.shape(0) != nflash)
        throw std::invalid_argument("first and flash_ok need one value per flash");
    const Axis ax = axis_of(xe, "x"), ay = axis_of(ye, "y"), az = axis_of(ze, "z"),
               at = axis_of(te, "t");
    const int64_t size = at.n * az.n * ay.n * ax.n;
    const std::vector<py::ssize_t> shape{at.n, az.n, ay.n, ax.n};
    const double *px = x.data(), *py_ = y.data(), *pz = z.data(), *pt = t.data();
    const int64_t *lab = labels.data(), *pfirst = first.data();
    const uint8_t* ok = flash_ok.data();
    std::unique_ptr<std::atomic<int32_t>[]> src(new std::atomic<int32_t>[size]());
    std::unique_ptr<std::atomic<int32_t>[]> fed(new std::atomic<int32_t>[size]());
    std::unique_ptr<std::atomic<int32_t>[]> fid(new std::atomic<int32_t>[size]());
    {
        py::gil_scoped_release release;
        auto pixel = [&](int64_t i) -> int64_t {
            const int64_t it = bin_of(at.e, at.n, pt[i]);
            if (it < 0) return -1;
            const int64_t iz = bin_of(az.e, az.n, pz[i]);
            if (iz < 0) return -1;
            const int64_t iy = bin_of(ay.e, ay.n, py_[i]);
            if (iy < 0) return -1;
            const int64_t ix = bin_of(ax.e, ax.n, px[i]);
            if (ix < 0) return -1;
            return ((it * az.n + iz) * ay.n + iy) * ax.n + ix;
        };
        parallel_blocks<NoScratch>(n, n_threads, [&](int64_t i0, int64_t i1, NoScratch&) {
            for (int64_t i = i0; i < i1; ++i) {
                const int64_t k = pixel(i);
                if (k >= 0) src[k].fetch_add(1, std::memory_order_relaxed);
            }
        });
        std::vector<int64_t> offsets, order;
        group_by_flash(lab, n, nflash, offsets, order);
        parallel_blocks<PixelScratch>(nflash, n_threads,
                                      [&](int64_t f0, int64_t f1, PixelScratch& s) {
            for (int64_t f = f0; f < f1; ++f) {
                if (!ok[f]) continue;
                if (pfirst[f] >= 0 && pfirst[f] < n) {
                    const int64_t k = pixel(pfirst[f]);
                    if (k >= 0) fid[k].fetch_add(1, std::memory_order_relaxed);
                }
                s.keys.clear();
                for (int64_t j = offsets[f]; j < offsets[f + 1]; ++j) {
                    const int64_t k = pixel(order[j]);
                    if (k >= 0) s.keys.push_back(k);
                }
                std::sort(s.keys.begin(), s.keys.end());
                const auto end = std::unique(s.keys.begin(), s.keys.end());
                for (auto it = s.keys.begin(); it != end; ++it)
                    fed[*it].fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    return py::make_tuple(from_atomic(src, shape, size), from_atomic(fed, shape, size),
                          from_atomic(fid, shape, size));
}

// Flash and source counts per cell of a time-dependent label mask.
//
// mask: (nframe, ny, nx) cell index (0 .. ncell - 1, < 0 for no cell) on the
// bins given by xe, ye; frame: the mask frame of every source (< 0: none);
// labels, first, flash_ok: flashes as in grid(); te: time bin edges; ze:
// height bin edges of the source counts. A flash is counted in the time bin
// of its first source, in the cell of its first source (extent = false) or in
// every cell any of its sources falls in (extent = true). Returns
// (flash counts (ncell, nt), source counts (ncell, nt, nz)).
py::tuple cells_py(const IArray& mask, int64_t ncell, const DArray& xe, const DArray& ye,
                   const LArray& frame, const DArray& x, const DArray& y, const DArray& z,
                   const DArray& t, const LArray& labels, const LArray& first,
                   const BArray& flash_ok, const DArray& te, const DArray& ze, bool extent,
                   int n_threads) {
    if (mask.ndim() != 3) throw std::invalid_argument("mask must be (frame, y, x)");
    const Axis ax = axis_of(xe, "x"), ay = axis_of(ye, "y"), at = axis_of(te, "t"),
               az = axis_of(ze, "z");
    const int64_t nframe = mask.shape(0);
    if (mask.shape(1) != ay.n || mask.shape(2) != ax.n)
        throw std::invalid_argument("mask shape does not match the x and y edges");
    if (ncell < 0) throw std::invalid_argument("ncell must not be negative");
    const int64_t n = t.ndim() == 1 ? t.shape(0) : -1;
    if (n < 0) throw std::invalid_argument("t must be 1-D");
    check_1d(frame, n, "frame");
    check_1d(x, n, "x");
    check_1d(y, n, "y");
    check_1d(z, n, "z");
    check_1d(labels, n, "labels");
    const int64_t nflash = first.ndim() == 1 ? first.shape(0) : -1;
    if (nflash < 0 || flash_ok.ndim() != 1 || flash_ok.shape(0) != nflash)
        throw std::invalid_argument("first and flash_ok need one value per flash");
    const int32_t* pm = mask.data();
    const int64_t *pfr = frame.data(), *lab = labels.data(), *pfirst = first.data();
    const double *px = x.data(), *py_ = y.data(), *pz = z.data(), *pt = t.data();
    const uint8_t* ok = flash_ok.data();
    const int64_t nflash_out = ncell * at.n, nsrc_out = ncell * at.n * az.n;
    std::unique_ptr<std::atomic<int64_t>[]> fl(new std::atomic<int64_t>[std::max<int64_t>(nflash_out, 1)]());
    std::unique_ptr<std::atomic<int64_t>[]> sc(new std::atomic<int64_t>[std::max<int64_t>(nsrc_out, 1)]());
    {
        py::gil_scoped_release release;
        auto cell_of = [&](int64_t i) -> int64_t {
            const int64_t f = pfr[i];
            if (f < 0 || f >= nframe) return -1;
            const int64_t iy = bin_of(ay.e, ay.n, py_[i]);
            if (iy < 0) return -1;
            const int64_t ix = bin_of(ax.e, ax.n, px[i]);
            if (ix < 0) return -1;
            const int64_t c = pm[(f * ay.n + iy) * ax.n + ix];
            return c >= 0 && c < ncell ? c : -1;
        };
        if (az.n > 0) {
            parallel_blocks<NoScratch>(n, n_threads, [&](int64_t i0, int64_t i1, NoScratch&) {
                for (int64_t i = i0; i < i1; ++i) {
                    const int64_t c = cell_of(i);
                    if (c < 0) continue;
                    const int64_t it = bin_of(at.e, at.n, pt[i]);
                    const int64_t iz = bin_of(az.e, az.n, pz[i]);
                    if (it < 0 || iz < 0) continue;
                    sc[(c * at.n + it) * az.n + iz].fetch_add(1, std::memory_order_relaxed);
                }
            });
        }
        std::vector<int64_t> offsets, order;
        group_by_flash(lab, n, nflash, offsets, order);
        parallel_blocks<PixelScratch>(nflash, n_threads,
                                      [&](int64_t f0, int64_t f1, PixelScratch& s) {
            for (int64_t f = f0; f < f1; ++f) {
                const int64_t i0 = pfirst[f];
                if (!ok[f] || i0 < 0 || i0 >= n) continue;
                const int64_t it = bin_of(at.e, at.n, pt[i0]);
                if (it < 0) continue;
                if (!extent) {
                    const int64_t c = cell_of(i0);
                    if (c >= 0) fl[c * at.n + it].fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                s.keys.clear();
                for (int64_t j = offsets[f]; j < offsets[f + 1]; ++j) {
                    const int64_t c = cell_of(order[j]);
                    if (c >= 0) s.keys.push_back(c);
                }
                std::sort(s.keys.begin(), s.keys.end());
                const auto end = std::unique(s.keys.begin(), s.keys.end());
                for (auto k = s.keys.begin(); k != end; ++k)
                    fl[*k * at.n + it].fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    return py::make_tuple(
        from_atomic(fl, std::vector<py::ssize_t>{ncell, at.n}, nflash_out),
        from_atomic(sc, std::vector<py::ssize_t>{ncell, at.n, az.n}, nsrc_out));
}

PYBIND11_MODULE(_lightning, m) {
    m.doc() = "Compiled lightning (LMA) kernel for radarx.";
    m.def("cluster", &cluster_py, py::arg("x"), py::arg("y"), py::arg("z"), py::arg("t"),
          py::arg("distance"), py::arg("time"), py::arg("n_threads") = 0);
    m.def("flash_stats", &flash_stats_py, py::arg("labels"), py::arg("nflash"), py::arg("x"),
          py::arg("y"), py::arg("n_threads") = 0);
    m.def("grid", &grid_py, py::arg("x"), py::arg("y"), py::arg("z"), py::arg("t"),
          py::arg("labels"), py::arg("first"), py::arg("flash_ok"), py::arg("x_edges"),
          py::arg("y_edges"), py::arg("z_edges"), py::arg("t_edges"),
          py::arg("n_threads") = 0);
    m.def("cells", &cells_py, py::arg("mask"), py::arg("ncell"), py::arg("x_edges"),
          py::arg("y_edges"), py::arg("frame"), py::arg("x"), py::arg("y"), py::arg("z"),
          py::arg("t"), py::arg("labels"), py::arg("first"), py::arg("flash_ok"),
          py::arg("t_edges"), py::arg("z_edges"), py::arg("extent"),
          py::arg("n_threads") = 0);
}
