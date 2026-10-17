// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Azimuthal statistics of a polar field on rings around a centre: the mean, the
// wavenumbers 1 to 3 and the coverage of the ring, for the eyewall and rainband
// diagnostics of tropical cyclones (radarx/retrieve/tc_diagnostics.py).
//
// The data are scattered gates: position (x, y) in m east and north of the radar, a value
// v and a weight w (the gate area, proportional to the ground range). Around the centre
// (cx, cy) a ring of radius R takes every gate within the half width hw of the ring with
// the triangular weight w (1 - |d - R| / hw), d being the distance from the centre. The
// azimuth g of a gate is atan2(x - cx, y - cy) (clockwise from north, seen from the
// centre), and the weighted least squares fit is
//
//     v = A0 + sum_{n=1..L} (A_n cos(n g) + B_n sin(n g)),   L <= 3,
//
// with L reduced where the largest angular gap of the data is wide: the wavenumber n is
// fitted for gaps up to 180 / n degrees. The gaps are measured on bins of one degree.
// The module has the NumPy implementation with the same steps. The gates are sorted by
// the distance from the centre once (counting sort in bins of the half width), so a ring
// only visits the bins that it overlaps; the rings run on a pool of threads.

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
constexpr int kMaxOrder = 3;
constexpr int kCoef = 2 * kMaxOrder + 1;
constexpr int kBins = 360;
constexpr int kColumns = kCoef + 6;  // coefficients, then n, gap, order, rms, std, echo fraction
constexpr double kPivotTolerance = 1.0e-10;

template <class F>
void parallel_blocks(int64_t total, int n_threads, int64_t block, F&& body) {
    if (total <= 0) return;
    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    int nt = n_threads > 0 ? n_threads : static_cast<int>(hw);
    const int64_t nblocks = (total + block - 1) / block;
    nt = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(nt, nblocks)));
    std::atomic<int64_t> next{0};
    auto worker = [&]() {
        for (;;) {
            const int64_t g0 = next.fetch_add(block);
            if (g0 >= total) break;
            const int64_t g1 = std::min(total, g0 + block);
            body(g0, g1);
        }
    };
    std::vector<std::thread> pool;
    for (int i = 1; i < nt; ++i) pool.emplace_back(worker);
    worker();
    for (auto& th : pool) th.join();
}

struct Settings {
    double half_width;
    int max_order;
    int min_points;
    double max_gap;      // degrees
    double threshold;    // NaN: no echo fraction
};

class Gates {
public:
    Gates(const double* x, const double* y, const double* v, const double* w, int64_t n, double cx,
          double cy, double bin)
        : x_(x), y_(y), v_(v), w_(w), cx_(cx), cy_(cy), bin_(bin) {
        dist_.resize(n);
        int64_t nb = 1;
        for (int64_t i = 0; i < n; ++i) {
            dist_[i] = std::hypot(x[i] - cx, y[i] - cy);
            nb = std::max<int64_t>(nb, static_cast<int64_t>(std::floor(dist_[i] / bin)) + 1);
        }
        nbins_ = nb;
        start_.assign(nb + 1, 0);
        for (int64_t i = 0; i < n; ++i) start_[static_cast<int64_t>(std::floor(dist_[i] / bin)) + 1]++;
        for (int64_t b = 0; b < nb; ++b) start_[b + 1] += start_[b];
        order_.resize(n);
        std::vector<int64_t> fill(start_.begin(), start_.end() - 1);
        for (int64_t i = 0; i < n; ++i)
            order_[fill[static_cast<int64_t>(std::floor(dist_[i] / bin))]++] = i;
    }

    bool fit(double radius, const Settings& s, double* out) const {
        const double hw = s.half_width;
        int64_t b0 = static_cast<int64_t>(std::floor((radius - hw) / bin_));
        int64_t b1 = static_cast<int64_t>(std::floor((radius + hw) / bin_));
        b0 = std::max<int64_t>(b0, 0);
        b1 = std::min<int64_t>(b1, nbins_ - 1);
        // gates of the ring: moments of the weights for the plain statistics and the
        // occupancy of the one-degree bins for the gaps
        std::vector<int64_t> members;
        bool seen[kBins];
        for (int k = 0; k < kBins; ++k) seen[k] = false;
        double sw = 0.0, swv = 0.0, swvv = 0.0, swe = 0.0;
        int64_t count = 0;
        for (int64_t b = b0; b <= b1; ++b)
            for (int64_t q = start_[b]; q < start_[b + 1]; ++q) {
                const int64_t i = order_[q];
                const double off = std::fabs(dist_[i] - radius);
                if (off >= hw) continue;
                const double wt = w_[i] * (1.0 - off / hw);
                if (!(wt > 0.0)) continue;
                members.push_back(i);
                sw += wt;
                swv += wt * v_[i];
                swvv += wt * v_[i] * v_[i];
                if (!std::isnan(s.threshold) && v_[i] >= s.threshold) swe += wt;
                const double g = std::atan2(x_[i] - cx_, y_[i] - cy_) * 180.0 / kPi;
                int k = static_cast<int>(std::floor(g < 0.0 ? g + 360.0 : g));
                seen[std::min(std::max(k, 0), kBins - 1)] = true;
                ++count;
            }
        if (count < s.min_points) return false;
        double gap = largest_gap(seen);
        if (gap > s.max_gap) return false;
        int order = 0;
        for (int n = 1; n <= s.max_order; ++n)
            if (gap <= 180.0 / n) order = n;
        const int m = 2 * order + 1;
        if (count <= m) return false;
        double N[kCoef][kCoef], b[kCoef];
        for (int a = 0; a < m; ++a) {
            b[a] = 0.0;
            for (int c = 0; c < m; ++c) N[a][c] = 0.0;
        }
        double basis[kCoef];
        for (const int64_t i : members) {
            const double wt = w_[i] * (1.0 - std::fabs(dist_[i] - radius) / hw);
            const double g = std::atan2(x_[i] - cx_, y_[i] - cy_);
            basis[0] = 1.0;
            for (int n = 1; n <= order; ++n) {
                basis[2 * n - 1] = std::cos(n * g);
                basis[2 * n] = std::sin(n * g);
            }
            for (int a = 0; a < m; ++a) {
                b[a] += wt * basis[a] * v_[i];
                for (int c = 0; c <= a; ++c) N[a][c] += wt * basis[a] * basis[c];
            }
        }
        double G[kCoef][kCoef];
        for (int a = 0; a < m; ++a)
            for (int c = 0; c <= a; ++c) {
                double sum = N[a][c];
                for (int r = 0; r < c; ++r) sum -= G[a][r] * G[c][r];
                if (a == c) {
                    if (!(sum > kPivotTolerance * N[a][a])) return false;
                    G[a][a] = std::sqrt(sum);
                } else {
                    G[a][c] = sum / G[c][c];
                }
            }
        double t[kCoef], K[kCoef];
        for (int a = 0; a < m; ++a) {
            double sum = b[a];
            for (int r = 0; r < a; ++r) sum -= G[a][r] * t[r];
            t[a] = sum / G[a][a];
        }
        for (int a = m - 1; a >= 0; --a) {
            double sum = t[a];
            for (int r = a + 1; r < m; ++r) sum -= G[r][a] * K[r];
            K[a] = sum / G[a][a];
        }
        double rss = swvv;
        for (int a = 0; a < m; ++a) rss -= K[a] * b[a];
        rss = std::max(rss, 0.0);
        for (int a = 0; a < kCoef; ++a) out[a] = a < m ? K[a] : kNaN;
        out[kCoef + 0] = static_cast<double>(count);
        out[kCoef + 1] = gap;
        out[kCoef + 2] = static_cast<double>(order);
        out[kCoef + 3] = std::sqrt(rss / sw);
        const double mean = swv / sw;
        out[kCoef + 4] = std::sqrt(std::max(swvv / sw - mean * mean, 0.0));
        out[kCoef + 5] = std::isnan(s.threshold) ? kNaN : swe / sw;
        return true;
    }

private:
    // largest run of empty one-degree bins (degrees); 360 for an empty ring
    static double largest_gap(const bool* seen) {
        int first = -1;
        for (int k = 0; k < kBins; ++k)
            if (seen[k]) {
                first = k;
                break;
            }
        if (first < 0) return 360.0;
        int best = 0, run = 0;
        for (int j = 1; j <= kBins; ++j) {
            if (seen[(first + j) % kBins]) {
                best = std::max(best, run);
                run = 0;
            } else {
                ++run;
            }
        }
        return static_cast<double>(best) + 1.0;
    }

    const double *x_, *y_, *v_, *w_;
    double cx_, cy_, bin_;
    int64_t nbins_;
    std::vector<double> dist_;
    std::vector<int64_t> start_, order_;
};

// Rings of the radii: (nr, 13) = A0, A1, B1, A2, B2, A3, B3, n, largest gap (deg), order,
// weighted rms residual, weighted standard deviation of the gates, fraction of the weight at
// or above the threshold. NaN for a ring that cannot be fitted.
py::array_t<double> rings_py(const DArray& x, const DArray& y, const DArray& v, const DArray& w,
                             double cx, double cy, const DArray& radii, double half_width,
                             int max_order, int min_points, double max_gap_deg, double threshold,
                             int n_threads) {
    const int64_t n = x.shape(0);
    for (const DArray* a : {&x, &y, &v, &w})
        if (a->ndim() != 1 || a->shape(0) != n)
            throw std::invalid_argument("x, y, v and w must be 1-D arrays of the same length");
    if (radii.ndim() != 1) throw std::invalid_argument("radii must be 1-D");
    if (!(half_width > 0.0)) throw std::invalid_argument("half_width must be positive");
    if (max_order < 0 || max_order > kMaxOrder)
        throw std::invalid_argument("max_order must be between 0 and 3");
    const int64_t nr = radii.shape(0);
    py::array_t<double> result({static_cast<py::ssize_t>(nr), static_cast<py::ssize_t>(kColumns)});
    double* res = result.mutable_data();
    {
        py::gil_scoped_release release;
        const Gates gates(x.data(), y.data(), v.data(), w.data(), n, cx, cy, half_width);
        const Settings s{half_width, max_order, min_points, max_gap_deg, threshold};
        const double* rp = radii.data();
        parallel_blocks(nr, n_threads, 1, [&](int64_t r0, int64_t r1) {
            for (int64_t r = r0; r < r1; ++r) {
                double* row = res + r * kColumns;
                if (!gates.fit(rp[r], s, row))
                    for (int k = 0; k < kColumns; ++k) row[k] = kNaN;
            }
        });
    }
    return result;
}

}  // namespace

PYBIND11_MODULE(_tc_rings, m) {
    m.doc() = "Azimuthal ring statistics of a polar field around a vortex centre";
    m.def("rings", &rings_py, py::arg("x"), py::arg("y"), py::arg("v"), py::arg("w"),
          py::arg("cx"), py::arg("cy"), py::arg("radii"), py::arg("half_width"),
          py::arg("max_order") = 2, py::arg("min_points") = 30, py::arg("max_gap") = 180.0,
          py::arg("threshold") = kNaN, py::arg("n_threads") = 0);
}
