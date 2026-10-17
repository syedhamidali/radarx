// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Object kernels on stacks of 2-D frames: connected-component labelling, the nested
// multi-threshold feature regions of tobac (Heikenfeld et al. 2019), marker-controlled
// watershed segmentation, per-label sums and extremes, and the overlap counts of two
// label images that are used to link cells in time (Feng et al. 2023). Every frame (or
// pair of frames) is independent, so threads take frames from an atomic counter. The
// NumPy references in radarx/retrieve/tracking.py follow the same steps and tie
// breaks, so that both engines give the same labels.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <vector>

namespace py = pybind11;
using DArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using IArray = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;
using UArray = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

template <class F>
void parallel_items(int64_t total, int n_threads, F&& body) {
    if (total <= 0) return;
    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    int nt = n_threads > 0 ? n_threads : static_cast<int>(hw);
    nt = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(nt, total)));
    std::atomic<int64_t> next{0};
    auto worker = [&]() {
        for (;;) {
            const int64_t i = next.fetch_add(1);
            if (i >= total) break;
            body(i);
        }
    };
    std::vector<std::thread> pool;
    for (int i = 1; i < nt; ++i) pool.emplace_back(worker);
    worker();
    for (auto& th : pool) th.join();
}

void check_connectivity(int c) {
    if (c != 4 && c != 8) throw std::invalid_argument("connectivity must be 4 or 8");
}

// Neighbour offsets (dy, dx) in the order up, left, right, down, then the diagonals.
const int kDy[8] = {-1, 0, 0, 1, -1, -1, 1, 1};
const int kDx[8] = {0, -1, 1, 0, -1, 1, -1, 1};

// Gives the label to the component of the pixel s; returns its number of pixels.
int64_t flood(const uint8_t* on, int64_t ny, int64_t nx, int conn, int32_t* lab, int64_t s,
              int32_t label, std::vector<int64_t>& stack) {
    int64_t members = 0;
    lab[s] = label;
    stack.push_back(s);
    while (!stack.empty()) {
        const int64_t p = stack.back();
        stack.pop_back();
        ++members;
        const int64_t y = p / nx, x = p % nx;
        for (int k = 0; k < conn; ++k) {
            const int64_t yy = y + kDy[k], xx = x + kDx[k];
            if (yy < 0 || yy >= ny || xx < 0 || xx >= nx) continue;
            const int64_t q = yy * nx + xx;
            if (on[q] && !lab[q]) {
                lab[q] = label;
                stack.push_back(q);
            }
        }
    }
    return members;
}

// Labels the true pixels of one frame in raster order of their first pixel (the order of
// scipy.ndimage.label). Returns the number of components; `size` and `first` get their pixel
// counts and first pixels.
int64_t label_frame(const uint8_t* on, int64_t ny, int64_t nx, int conn, int32_t* lab,
                    std::vector<int64_t>* size, std::vector<int64_t>* first) {
    const int64_t n = ny * nx;
    std::fill(lab, lab + n, 0);
    std::vector<int64_t> stack;
    int64_t count = 0;
    if (size) size->clear();
    if (first) first->clear();
    for (int64_t s = 0; s < n; ++s) {
        if (!on[s] || lab[s]) continue;
        const int64_t members = flood(on, ny, nx, conn, lab, s, static_cast<int32_t>(++count), stack);
        if (size) size->push_back(members);
        if (first) first->push_back(s);
    }
    return count;
}

// mask: (nt, ny, nx) -> labels (nt, ny, nx), counts (nt)
py::tuple label_py(const UArray& mask, int conn, int n_threads) {
    check_connectivity(conn);
    if (mask.ndim() != 3) throw std::invalid_argument("mask must be 3-D (time, y, x)");
    const int64_t nt = mask.shape(0), ny = mask.shape(1), nx = mask.shape(2);
    IArray labels(std::vector<py::ssize_t>{nt, ny, nx});
    py::array_t<int32_t> counts(std::vector<py::ssize_t>{nt});
    const uint8_t* pm = mask.data();
    int32_t* pl = labels.mutable_data();
    int32_t* pc = counts.mutable_data();
    {
        py::gil_scoped_release release;
        parallel_items(nt, n_threads, [&](int64_t t) {
            pc[t] = static_cast<int32_t>(
                label_frame(pm + t * ny * nx, ny, nx, conn, pl + t * ny * nx, nullptr, nullptr));
        });
    }
    return py::make_tuple(labels, counts);
}

// Nested multi-threshold regions. For the thresholds in ascending order, the regions of
// field >= threshold with at least min_size pixels are the candidates; the regions that
// contain no candidate of the next higher threshold are the features (the leaves of the
// nesting). Features are numbered in raster order of their first pixel.
struct Levels {
    std::vector<std::vector<int32_t>> lab;     // component of every pixel, per threshold
    std::vector<std::vector<int64_t>> size;    // pixels of every component
    std::vector<std::vector<int64_t>> first;   // first pixel of every component
    std::vector<std::vector<uint8_t>> kept;    // component has at least min_size pixels
    std::vector<std::vector<uint8_t>> has_child;  // component holds a kept component above
};

void label_levels(const double* f, int64_t ny, int64_t nx, const double* th, int64_t nthr,
                  int64_t min_size, int conn, Levels& lv) {
    const int64_t n = ny * nx;
    lv.lab.assign(nthr, std::vector<int32_t>(n));
    lv.size.assign(nthr, {});
    lv.first.assign(nthr, {});
    lv.kept.assign(nthr, {});
    lv.has_child.assign(nthr, {});
    std::vector<uint8_t> on(n);
    for (int64_t l = 0; l < nthr; ++l) {
        for (int64_t i = 0; i < n; ++i) on[i] = std::isfinite(f[i]) && f[i] >= th[l];
        label_frame(on.data(), ny, nx, conn, lv.lab[l].data(), &lv.size[l], &lv.first[l]);
        const int64_t m = static_cast<int64_t>(lv.size[l].size());
        lv.kept[l].assign(m, 0);
        lv.has_child[l].assign(m, 0);
        for (int64_t c = 0; c < m; ++c) {
            if (lv.size[l][c] < min_size) continue;
            lv.kept[l][c] = 1;
            if (l > 0) lv.has_child[l - 1][lv.lab[l - 1][lv.first[l][c]] - 1] = 1;
        }
    }
}

// Writes the leaf regions as markers numbered by their first pixel; returns their levels.
std::vector<int32_t> leaf_markers(const Levels& lv, int64_t nthr, int64_t n, int32_t* out) {
    std::vector<std::tuple<int64_t, int64_t, int64_t>> leaves;  // first pixel, level, component
    for (int64_t l = 0; l < nthr; ++l)
        for (int64_t c = 0; c < static_cast<int64_t>(lv.size[l].size()); ++c)
            if (lv.kept[l][c] && !lv.has_child[l][c]) leaves.emplace_back(lv.first[l][c], l, c);
    std::sort(leaves.begin(), leaves.end());
    std::vector<std::vector<int32_t>> number(nthr);
    for (int64_t l = 0; l < nthr; ++l) number[l].assign(lv.size[l].size(), 0);
    std::vector<int32_t> level_of;
    int32_t k = 0;
    for (const auto& lf : leaves) {
        number[std::get<1>(lf)][std::get<2>(lf)] = ++k;
        level_of.push_back(static_cast<int32_t>(std::get<1>(lf)));
    }
    std::fill(out, out + n, 0);
    for (int64_t l = 0; l < nthr; ++l)
        for (int64_t i = 0; i < n; ++i) {
            const int32_t c = lv.lab[l][i];
            if (c && number[l][c - 1]) out[i] = number[l][c - 1];
        }
    return level_of;
}

// Returns markers (nt, ny, nx), counts (nt) and the threshold index of every marker
// (concatenated over the frames).
py::tuple features_py(const DArray& field, const DArray& thresholds, int64_t min_size, int conn,
                      int n_threads) {
    check_connectivity(conn);
    if (field.ndim() != 3) throw std::invalid_argument("field must be 3-D (time, y, x)");
    const int64_t nt = field.shape(0), ny = field.shape(1), nx = field.shape(2);
    const int64_t n = ny * nx, nthr = thresholds.size();
    if (nthr < 1) throw std::invalid_argument("at least one threshold is needed");
    IArray markers(std::vector<py::ssize_t>{nt, ny, nx});
    py::array_t<int32_t> counts(std::vector<py::ssize_t>{nt});
    const double* pf = field.data();
    const double* th = thresholds.data();
    int32_t* pk = markers.mutable_data();
    int32_t* pc = counts.mutable_data();
    std::vector<std::vector<int32_t>> level_of(nt);
    {
        py::gil_scoped_release release;
        parallel_items(nt, n_threads, [&](int64_t t) {
            Levels lv;
            label_levels(pf + t * n, ny, nx, th, nthr, min_size, conn, lv);
            level_of[t] = leaf_markers(lv, nthr, n, pk + t * n);
            pc[t] = static_cast<int32_t>(level_of[t].size());
        });
    }
    int64_t total = 0;
    for (int64_t t = 0; t < nt; ++t) total += pc[t];
    py::array_t<int32_t> levels(std::vector<py::ssize_t>{total});
    int32_t* pv = levels.mutable_data();
    for (int64_t t = 0; t < nt; ++t)
        for (int32_t v : level_of[t]) *pv++ = v;
    return py::make_tuple(markers, counts, levels);
}

// Marker-controlled watershed on a stack of frames: the markers flood downhill (towards
// lower values of the field) through the pixels of the mask. The pixel with the highest
// value is taken first; equal values are taken in the order in which they were queued.
IArray watershed_py(const DArray& field, const IArray& markers, const UArray& mask, int conn,
                    int n_threads) {
    check_connectivity(conn);
    if (field.ndim() != 3) throw std::invalid_argument("field must be 3-D (time, y, x)");
    const int64_t nt = field.shape(0), ny = field.shape(1), nx = field.shape(2);
    const int64_t n = ny * nx;
    if (markers.ndim() != 3 || mask.ndim() != 3 || markers.shape(0) != nt ||
        markers.shape(1) != ny || markers.shape(2) != nx || mask.shape(0) != nt ||
        mask.shape(1) != ny || mask.shape(2) != nx)
        throw std::invalid_argument("markers and mask must have the shape of the field");
    IArray labels(std::vector<py::ssize_t>{nt, ny, nx});
    const double* pf = field.data();
    const int32_t* pk = markers.data();
    const uint8_t* pm = mask.data();
    int32_t* pl = labels.mutable_data();
    {
        py::gil_scoped_release release;
        parallel_items(nt, n_threads, [&](int64_t t) {
            const double* f = pf + t * n;
            const int32_t* mk = pk + t * n;
            const uint8_t* ms = pm + t * n;
            int32_t* out = pl + t * n;
            using Item = std::tuple<double, int64_t, int64_t>;  // value, -age, pixel
            auto cmp = [](const Item& a, const Item& b) {
                if (std::get<0>(a) != std::get<0>(b)) return std::get<0>(a) < std::get<0>(b);
                return std::get<1>(a) < std::get<1>(b);
            };
            std::priority_queue<Item, std::vector<Item>, decltype(cmp)> heap(cmp);
            int64_t age = 0;
            for (int64_t i = 0; i < n; ++i) {
                out[i] = 0;
                if (mk[i] > 0 && ms[i] && std::isfinite(f[i])) {
                    out[i] = mk[i];
                    heap.emplace(f[i], -(age++), i);
                }
            }
            while (!heap.empty()) {
                const int64_t p = std::get<2>(heap.top());
                heap.pop();
                const int64_t y = p / nx, x = p % nx;
                for (int k = 0; k < conn; ++k) {
                    const int64_t yy = y + kDy[k], xx = x + kDx[k];
                    if (yy < 0 || yy >= ny || xx < 0 || xx >= nx) continue;
                    const int64_t q = yy * nx + xx;
                    if (out[q] || !ms[q] || !std::isfinite(f[q])) continue;
                    out[q] = out[p];
                    heap.emplace(f[q], -(age++), q);
                }
            }
        });
    }
    return labels;
}

// Gaussian smoothing of every frame with the separable kernel of scipy.ndimage.gaussian_filter
// (radius int(4 sigma + 0.5), weights normalised, edge values repeated).
DArray smooth_py(const DArray& field, double sigma, int n_threads) {
    if (field.ndim() != 3) throw std::invalid_argument("field must be 3-D (time, y, x)");
    if (!(sigma > 0.0)) throw std::invalid_argument("sigma must be positive");
    const int64_t nt = field.shape(0), ny = field.shape(1), nx = field.shape(2);
    const int64_t n = ny * nx;
    const int64_t radius = static_cast<int64_t>(4.0 * sigma + 0.5);
    std::vector<double> w(2 * radius + 1);
    double total = 0.0;
    for (int64_t k = -radius; k <= radius; ++k) {
        w[k + radius] = std::exp(-0.5 * double(k) * double(k) / (sigma * sigma));
        total += w[k + radius];
    }
    for (auto& v : w) v /= total;
    DArray out(std::vector<py::ssize_t>{nt, ny, nx});
    const double* pf = field.data();
    double* po = out.mutable_data();
    {
        py::gil_scoped_release release;
        parallel_items(nt, n_threads, [&](int64_t t) {
            const double* f = pf + t * n;
            double* o = po + t * n;
            std::vector<double> tmp(n);
            for (int64_t y = 0; y < ny; ++y)
                for (int64_t x = 0; x < nx; ++x) {
                    double acc = 0.0;
                    for (int64_t k = -radius; k <= radius; ++k) {
                        const int64_t xx = std::min<int64_t>(std::max<int64_t>(x + k, 0), nx - 1);
                        acc += w[k + radius] * f[y * nx + xx];
                    }
                    tmp[y * nx + x] = acc;
                }
            for (int64_t y = 0; y < ny; ++y)
                for (int64_t x = 0; x < nx; ++x) {
                    double acc = 0.0;
                    for (int64_t k = -radius; k <= radius; ++k) {
                        const int64_t yy = std::min<int64_t>(std::max<int64_t>(y + k, 0), ny - 1);
                        acc += w[k + radius] * tmp[yy * nx + x];
                    }
                    o[y * nx + x] = acc;
                }
        });
    }
    return out;
}

// Sums, counts, maxima and minima of the finite values of fields (nf, nt, ny, nx) over the
// labels (nt, ny, nx), each as (nf, nt, nlab).
py::tuple region_sums_py(const IArray& labels, const DArray& values, int64_t nlab, int n_threads) {
    if (labels.ndim() != 3 || values.ndim() != 4) throw std::invalid_argument("bad dimensions");
    const int64_t nf = values.shape(0), nt = values.shape(1), ny = values.shape(2),
                  nx = values.shape(3);
    if (labels.shape(0) != nt || labels.shape(1) != ny || labels.shape(2) != nx)
        throw std::invalid_argument("labels must have the shape of the fields");
    const int64_t n = ny * nx;
    const std::vector<py::ssize_t> shape{nf, nt, nlab};
    DArray sum(shape), cnt(shape), vmax(shape), vmin(shape);
    const int32_t* pl = labels.data();
    const double* pv = values.data();
    double* ps = sum.mutable_data();
    double* pc = cnt.mutable_data();
    double* px = vmax.mutable_data();
    double* pn = vmin.mutable_data();
    {
        py::gil_scoped_release release;
        parallel_items(nf * nt, n_threads, [&](int64_t job) {
            const int64_t t = job % nt;
            const int32_t* lab = pl + t * n;
            const double* v = pv + job * n;
            double* s = ps + job * nlab;
            double* c = pc + job * nlab;
            double* hi = px + job * nlab;
            double* lo = pn + job * nlab;
            for (int64_t k = 0; k < nlab; ++k) {
                s[k] = 0.0;
                c[k] = 0.0;
                hi[k] = kNaN;
                lo[k] = kNaN;
            }
            for (int64_t i = 0; i < n; ++i) {
                const int32_t l = lab[i];
                if (l <= 0 || l > nlab || !std::isfinite(v[i])) continue;
                const int64_t k = l - 1;
                s[k] += v[i];
                c[k] += 1.0;
                if (std::isnan(hi[k]) || v[i] > hi[k]) hi[k] = v[i];
                if (std::isnan(lo[k]) || v[i] < lo[k]) lo[k] = v[i];
            }
        });
    }
    return py::make_tuple(sum, cnt, vmax, vmin);
}

// Overlap counts of pairs of label images. Pixel (y, x) of l0 with label a > 0 moves to
// (y + dy, x + dx) and counts for the label b > 0 of l1 there. dy and dx are (np, sy, sx)
// with sy, sx equal to 1 (one shift for the frame) or to ny, nx (a shift per pixel).
// n0 and n1 hold the number of labels of every frame; the counts of pair p are the
// n0[p] * n1[p] values from offset sum(n0[:p] * n1[:p]), row a, column b.
DArray overlap_py(const IArray& l0, const IArray& l1, const IArray& n0, const IArray& n1,
                  const IArray& dy, const IArray& dx, int n_threads) {
    if (l0.ndim() != 3 || l1.ndim() != 3 || dy.ndim() != 3 || dx.ndim() != 3)
        throw std::invalid_argument("bad dimensions");
    const int64_t np = l0.shape(0), ny = l0.shape(1), nx = l0.shape(2);
    if (l1.shape(0) != np || l1.shape(1) != ny || l1.shape(2) != nx || n0.size() != np ||
        n1.size() != np || dy.shape(0) != np || dx.shape(0) != np)
        throw std::invalid_argument("inconsistent shapes");
    const int64_t sy = dy.shape(1), sx = dy.shape(2);
    if ((sy != 1 && sy != ny) || (sx != 1 && sx != nx) || dx.shape(1) != sy || dx.shape(2) != sx)
        throw std::invalid_argument("dy and dx must be (np, 1 or ny, 1 or nx)");
    std::vector<int64_t> offset(np + 1, 0);
    for (int64_t p = 0; p < np; ++p) offset[p + 1] = offset[p] + int64_t(n0.data()[p]) * n1.data()[p];
    DArray out(std::vector<py::ssize_t>{offset[np]});
    double* po = out.mutable_data();
    std::fill(po, po + offset[np], 0.0);
    const int32_t* a_all = l0.data();
    const int32_t* b_all = l1.data();
    const int32_t* pdy = dy.data();
    const int32_t* pdx = dx.data();
    const int32_t* pn0 = n0.data();
    const int32_t* pn1 = n1.data();
    {
        py::gil_scoped_release release;
        parallel_items(np, n_threads, [&](int64_t p) {
            const int32_t* a = a_all + p * ny * nx;
            const int32_t* b = b_all + p * ny * nx;
            const int32_t* sdy = pdy + p * sy * sx;
            const int32_t* sdx = pdx + p * sy * sx;
            double* c = po + offset[p];
            const int64_t m0 = pn0[p], m1 = pn1[p];
            for (int64_t y = 0; y < ny; ++y)
                for (int64_t x = 0; x < nx; ++x) {
                    const int32_t la = a[y * nx + x];
                    if (la <= 0 || la > m0) continue;
                    const int64_t s = (sy == 1 ? 0 : y) * sx + (sx == 1 ? 0 : x);
                    const int64_t yy = y + sdy[s], xx = x + sdx[s];
                    if (yy < 0 || yy >= ny || xx < 0 || xx >= nx) continue;
                    const int32_t lb = b[yy * nx + xx];
                    if (lb <= 0 || lb > m1) continue;
                    c[int64_t(la - 1) * m1 + (lb - 1)] += 1.0;
                }
        });
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(_objects, m) {
    m.doc() = "Labelling, feature regions, watershed, region sums and overlaps of 2-D frames";
    m.def("label", &label_py, py::arg("mask"), py::arg("connectivity") = 8,
          py::arg("n_threads") = 0);
    m.def("features", &features_py, py::arg("field"), py::arg("thresholds"), py::arg("min_size"),
          py::arg("connectivity") = 8, py::arg("n_threads") = 0);
    m.def("watershed", &watershed_py, py::arg("field"), py::arg("markers"), py::arg("mask"),
          py::arg("connectivity") = 8, py::arg("n_threads") = 0);
    m.def("smooth", &smooth_py, py::arg("field"), py::arg("sigma"), py::arg("n_threads") = 0);
    m.def("region_sums", &region_sums_py, py::arg("labels"), py::arg("values"), py::arg("nlab"),
          py::arg("n_threads") = 0);
    m.def("overlap", &overlap_py, py::arg("l0"), py::arg("l1"), py::arg("n0"), py::arg("n1"),
          py::arg("dy"), py::arg("dx"), py::arg("n_threads") = 0);
}
