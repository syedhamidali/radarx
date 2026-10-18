// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Check of the ray angles of a sweep: outliers, duplicates, gaps and reversals, and a repair
// of the angles of the bad rays by interpolation between the nearest good rays.
//
// A ray is an outlier when it deviates by more than `tolerance` ray spacings from the
// positions that the rays on its left predict and also from the positions that the rays on
// its right predict (the median of the up to `half_window` neighbours on each side,
// extrapolated with the median spacing). Requiring both sides keeps a persistent jump (a gap)
// from being reported as a run of outliers. The steps between the rays that are not outliers
// then give the duplicates (step of less than `duplicate_fraction` spacings), the reversals
// (a step against the scan direction) and the gaps (more than `gap_factor` spacings forward;
// the ray before the gap is flagged). Every sweep is independent, so threads take sweeps from
// an atomic counter. The NumPy reference in radarx/retrieve/_azimuth_qc_numpy.py follows the
// same steps in the same order.

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
using UArray = py::array_t<uint8_t, py::array::c_style>;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kStationary = 0.01;  // degrees per ray: a smaller typical step is no scan
constexpr uint8_t kOutlier = 1, kDuplicate = 2, kGap = 4, kReversal = 8, kMissing = 16;

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

inline double wrap180(double x) { return x - 360.0 * std::floor((x + 180.0) / 360.0); }
inline double wrap360(double x) { return x - 360.0 * std::floor(x / 360.0); }

double median_of(std::vector<double>& v) {
    const size_t n = v.size(), h = n / 2;
    std::nth_element(v.begin(), v.begin() + h, v.end());
    const double upper = v[h];
    if (n % 2) return upper;
    const double lower = *std::max_element(v.begin(), v.begin() + h);
    return 0.5 * (lower + upper);
}

// Median deviation of ray i from the positions that the rays `members` predict.
double side_residual(const std::vector<double>& a, const std::vector<int64_t>& p, int64_t i,
                     const std::vector<int64_t>& members, double delta) {
    std::vector<double> r;
    r.reserve(members.size());
    for (int64_t j : members) r.push_back(wrap180(a[i] - a[j] - static_cast<double>(p[i] - p[j]) * delta));
    return median_of(r);
}

void flag_one(const double* angle, int64_t n, double tolerance, int64_t half_window,
              double duplicate_fraction, double gap_factor, uint8_t* flags, double* fixed,
              double& delta_out) {
    for (int64_t i = 0; i < n; ++i) {
        flags[i] = 0;
        fixed[i] = angle[i];
    }
    delta_out = kNaN;
    std::vector<int64_t> p;  // positions of the finite rays
    for (int64_t i = 0; i < n; ++i) {
        if (std::isfinite(angle[i])) p.push_back(i);
        else flags[i] |= kMissing;
    }
    const int64_t m = static_cast<int64_t>(p.size());
    if (m < 5) return;
    std::vector<double> a(m);
    for (int64_t i = 0; i < m; ++i) a[i] = angle[p[i]];
    std::vector<double> step(m - 1);
    for (int64_t i = 0; i + 1 < m; ++i) step[i] = wrap180(a[i + 1] - a[i]) / static_cast<double>(p[i + 1] - p[i]);
    const double delta = median_of(step);
    delta_out = delta;
    if (std::fabs(delta) < kStationary) {  // the antenna does not scan
        delta_out = 0.0;
        return;
    }
    const double tol = tolerance * std::fabs(delta);

    std::vector<char> outlier(m, 0), duplicate(m, 0);
    for (int64_t i = 0; i < m; ++i) {
        std::vector<int64_t> left, right;
        for (int64_t j = i - 1; j >= 0 && p[i] - p[j] <= half_window; --j) left.push_back(j);
        for (int64_t j = i + 1; j < m && p[j] - p[i] <= half_window; ++j) right.push_back(j);
        const bool has_left = left.size() >= 2, has_right = right.size() >= 2;
        double e_left = 0.0, e_right = 0.0;
        if (has_left) e_left = side_residual(a, p, i, left, delta);
        if (has_right) e_right = side_residual(a, p, i, right, delta);
        if (has_left && has_right) outlier[i] = std::min(std::fabs(e_left), std::fabs(e_right)) > tol;
        else if (has_left) outlier[i] = std::fabs(e_left) > tol;
        else if (has_right) outlier[i] = std::fabs(e_right) > tol;
        if (outlier[i]) flags[p[i]] |= kOutlier;
    }

    int64_t prev = -1;
    for (int64_t i = 0; i < m; ++i) {
        if (outlier[i]) continue;
        if (prev >= 0) {
            const double ratio = wrap180(a[i] - a[prev]) / (delta * static_cast<double>(p[i] - p[prev]));
            if (ratio > gap_factor) flags[p[prev]] |= kGap;
            else if (std::fabs(ratio) < duplicate_fraction) duplicate[i] = 1;
            else if (ratio < -duplicate_fraction) flags[p[i]] |= kReversal;
        }
        prev = i;
    }
    for (int64_t i = 0; i < m; ++i)
        if (duplicate[i]) flags[p[i]] |= kDuplicate;

    // repair by interpolation in the ray number between the good rays
    std::vector<char> bad(n, 1);
    for (int64_t i = 0; i < m; ++i)
        if (!outlier[i] && !duplicate[i]) bad[p[i]] = 0;
    std::vector<int64_t> before(n, -1), after(n, -1);
    int64_t last = -1;
    for (int64_t i = 0; i < n; ++i) {
        before[i] = last;
        if (!bad[i]) last = i;
    }
    last = -1;
    for (int64_t i = n - 1; i >= 0; --i) {
        after[i] = last;
        if (!bad[i]) last = i;
    }
    for (int64_t i = 0; i < n; ++i) {
        if (!bad[i]) continue;
        const int64_t lo = before[i], hi = after[i];
        double value;
        if (lo >= 0 && hi >= 0)
            value = angle[lo] + wrap180(angle[hi] - angle[lo]) * static_cast<double>(i - lo) / static_cast<double>(hi - lo);
        else if (lo >= 0) value = angle[lo] + static_cast<double>(i - lo) * delta;
        else if (hi >= 0) value = angle[hi] - static_cast<double>(hi - i) * delta;
        else continue;
        fixed[i] = wrap360(value);
    }
}

py::list flag_azimuth_py(const py::list& angles, double tolerance, int64_t half_window,
                         double duplicate_fraction, double gap_factor, int n_threads) {
    if (!(tolerance > 0.0)) throw std::invalid_argument("tolerance must be positive");
    if (half_window < 2) throw std::invalid_argument("half_window must be at least 2");
    const int64_t count = static_cast<int64_t>(angles.size());
    std::vector<DArray> in;
    std::vector<UArray> flags;
    std::vector<DArray> fixed;
    std::vector<double> deltas(count, kNaN);
    for (int64_t s = 0; s < count; ++s) {
        DArray a = py::cast<DArray>(angles[s]);
        if (a.ndim() != 1) throw std::invalid_argument("every sweep needs a 1-D array of ray angles");
        const std::vector<py::ssize_t> shape{a.shape(0)};
        in.push_back(a);
        flags.emplace_back(shape);
        fixed.emplace_back(shape);
    }
    {
        py::gil_scoped_release release;
        parallel_items(count, n_threads, [&](int64_t s) {
            flag_one(in[s].data(), static_cast<int64_t>(in[s].shape(0)), tolerance, half_window,
                     duplicate_fraction, gap_factor, flags[s].mutable_data(),
                     fixed[s].mutable_data(), deltas[s]);
        });
    }
    py::list out;
    for (int64_t s = 0; s < count; ++s) out.append(py::make_tuple(flags[s], fixed[s], deltas[s]));
    return out;
}

}  // namespace

PYBIND11_MODULE(_azimuth_qc, m) {
    m.doc() = "Outliers, duplicates, gaps and reversals in the ray angles of a sweep";
    m.def("flag_azimuth", &flag_azimuth_py, py::arg("angles"), py::arg("tolerance"),
          py::arg("half_window"), py::arg("duplicate_fraction"), py::arg("gap_factor"),
          py::arg("n_threads") = 0);
}
