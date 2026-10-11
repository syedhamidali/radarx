// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// UNRAVEL dealiasing driver for PPI sweeps (Louf et al. 2020, JTECH 37,
// 741-758), implemented from the paper. The NumPy/Python module
// _unravel_numpy.py is the specification of every step and gives the same
// values; the choices the paper leaves open are listed there.
//
// A sweep is processed in the order of the default driver of the paper
// (its Fig. 4): the modified z-score pre-processing with a first unfolding of
// each ray, the search for two reference radials, the unfolding of their
// neighbours and of four sectors by azimuthal continuity, two passes of range
// and azimuthal continuity with a growing window, four passes of the box
// module, the closest-reference module, and the checks (least-squares line
// along each ray, box check). The 3-D check between neighbouring elevation
// angles is a separate function. Gates carry the flags -3 (missing), 0 (not
// processed), 1 (processed, unchanged) and 2 (processed, unfolded); only the
// checks change a gate that has been processed.
//
// Sweeps are independent and run in parallel; inside a sweep the passes are
// sequential, as in the paper, because a gate that has been processed is the
// reference of the next one. Products of two doubles are stored before they
// are added so that no fused multiply-add changes the last digit: the values
// are the same as those of the Python version.

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
#include <utility>
#include <vector>

namespace py = pybind11;
using DArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using IArray = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

inline double mul(double a, double b) {
    volatile double r = a * b;
    return r;
}

inline double fold_to(double vref, double v, double vn) {
    const double k = std::floor((vref - v) / (2.0 * vn) + 0.5);
    return v + mul(2.0 * vn, k);
}

// Eq. (3): the nearest unfolding of v within alpha vn of vref, else NaN.
inline bool accept(double vref, double v, double vn, double alpha, double& u) {
    u = fold_to(vref, v, vn);
    return std::fabs(u - vref) < alpha * vn;
}

double median_of(std::vector<double>& s) {
    std::sort(s.begin(), s.end());
    const size_t n = s.size();
    if (n % 2) return s[n / 2];
    return (s[n / 2 - 1] + s[n / 2]) * 0.5;
}

double mean_of(const std::vector<double>& s) {
    double t = 0.0;
    for (double x : s) t += x;
    return t / static_cast<double>(s.size());
}

// Ray index with wrapping, or -1 outside the sweep.
inline int64_t ray_of(int64_t i, int64_t nray, bool wrap) {
    if (wrap) {
        int64_t r = i % nray;
        return r < 0 ? r + nray : r;
    }
    return (i >= 0 && i < nray) ? i : -1;
}

struct Params {
    double alpha, alpha_first, alpha_pre, z_cutoff, beta;
    int pre_window, closest_half;
    std::vector<int> windows;
    std::vector<std::pair<int, int>> boxes;
    std::pair<int, int> check_box;
};

struct Sweep {
    int64_t nray, ngate;
    const double* vel;
    double vn;
    const double* az;
    const double* rng;
    const double* x;
    const double* y;
    bool wrap;
    std::vector<double> v, out;  // preprocessed and current values
    std::vector<int8_t> flag;
    inline int64_t at(int64_t i, int64_t g) const { return i * ngate + g; }
};

// ---------------------------------------------------------------- initialisation

void preprocess(Sweep& s, const Params& P) {
    s.v.assign(s.vel, s.vel + s.nray * s.ngate);
    std::vector<int64_t> idx;
    std::vector<double> vals, dev, pos, neg;
    for (int64_t i = 0; i < s.nray; ++i) {
        double* row = s.v.data() + i * s.ngate;
        for (int64_t n = 0; n < s.ngate; ++n) {
            idx.clear();
            vals.clear();
            for (int64_t k = n; k < std::min<int64_t>(n + P.pre_window, s.ngate); ++k)
                if (!std::isnan(row[k])) {
                    idx.push_back(k);
                    vals.push_back(row[k]);
                }
            if (idx.empty()) continue;
            std::vector<double> sorted = vals;
            const double m = median_of(sorted);
            dev.clear();
            pos.clear();
            neg.clear();
            for (double x : vals) {
                dev.push_back(std::fabs(x - m));
                if (x > 0) pos.push_back(x);
                if (x < 0) neg.push_back(x);
            }
            const double mad = median_of(dev);
            const double mean_pos = pos.empty() ? kNaN : mean_of(pos);
            const double mean_neg = neg.empty() ? kNaN : mean_of(neg);
            const double first = m >= 0 ? mean_pos : mean_neg;
            const double second = m >= 0 ? mean_neg : mean_pos;
            for (int64_t k : idx) {
                const double x = row[k];
                double z;
                if (mad > 0)
                    z = (x - m) / mad;
                else
                    z = (x == m) ? 0.0 : kInf;
                if (std::fabs(z) <= P.z_cutoff) continue;
                double u = kNaN;
                bool done = false;
                for (double ref : {first, second}) {
                    if (!std::isnan(ref) && accept(ref, x, s.vn, P.alpha_pre, u)) {
                        done = true;
                        break;
                    }
                }
                row[k] = done ? u : kNaN;
            }
        }
    }
}

inline double circ_diff(double a, double b, bool wrap) {
    const double d = std::fabs(a - b);
    return wrap ? std::min(d, 360.0 - d) : d;
}

std::pair<int64_t, int64_t> reference_radials(const Sweep& s, const Params& P) {
    const int64_t nray = s.nray, ngate = s.ngate;
    std::vector<int64_t> count(nray, 0), near(nray, 0);
    std::vector<double> curve(nray, kNaN);
    int64_t total = 0;
    for (int64_t i = 0; i < nray; ++i) {
        double sum = 0.0;
        for (int64_t g = 0; g < ngate; ++g) {
            const double x = s.v[s.at(i, g)];
            if (std::isnan(x)) continue;
            ++count[i];
            sum += std::fabs(x);
            if (std::fabs(x) >= P.beta * s.vn) ++near[i];
        }
        total += count[i];
        if (count[i]) curve[i] = sum / static_cast<double>(count[i]);
    }
    if (total == 0) return {0, 0};
    const double average = static_cast<double>(total) / static_cast<double>(nray);
    std::vector<char> eligible(nray);
    auto any_of = [&]() {
        for (char c : eligible)
            if (c) return true;
        return false;
    };
    for (int64_t i = 0; i < nray; ++i)
        eligible[i] = count[i] >= (2.0 / 3.0) * average && count[i] > 0 && near[i] <= 3;
    if (!any_of())
        for (int64_t i = 0; i < nray; ++i) eligible[i] = count[i] > 0 && near[i] <= 3;
    if (!any_of())
        for (int64_t i = 0; i < nray; ++i) eligible[i] = count[i] > 0;
    std::vector<int64_t> rays;
    std::vector<double> sc;
    for (int64_t i = 0; i < nray; ++i)
        if (eligible[i]) {
            rays.push_back(i);
            sc.push_back(curve[i]);
        }
    const int64_t e = static_cast<int64_t>(rays.size());
    int64_t k1 = 0;
    for (int64_t k = 1; k < e; ++k)
        if (sc[k] < sc[k1]) k1 = k;
    const int64_t ref1 = rays[k1];
    double best = -1.0;
    int64_t ref2 = ref1;
    for (int64_t p = 0; p < e; ++p) {
        const double left = (s.wrap || p > 0) ? sc[((p - 1) % e + e) % e] : kInf;
        const double right = (s.wrap || p < e - 1) ? sc[(p + 1) % e] : kInf;
        if (!(sc[p] <= left && sc[p] <= right)) continue;
        if (circ_diff(s.az[rays[p]], s.az[ref1], s.wrap) < 120.0) continue;
        double highs[2];
        int h = 0;
        for (int step : {1, -1}) {
            double top = sc[p];
            int64_t q = p, steps = 0;
            while (steps < e - 1) {
                q = s.wrap ? ((q + step) % e + e) % e : q + step;
                if (q < 0 || q >= e) break;
                ++steps;
                if (sc[q] < sc[p]) break;
                top = std::max(top, sc[q]);
            }
            highs[h++] = top;
        }
        const double depth = std::min(highs[0], highs[1]) - sc[p];
        if (depth > best) {
            best = depth;
            ref2 = rays[p];
        }
    }
    return {ref1, ref2};
}

inline void set_gate(Sweep& s, int64_t i, int64_t g, double u) {
    s.out[s.at(i, g)] = u;
    s.flag[s.at(i, g)] = (u == s.v[s.at(i, g)]) ? 1 : 2;
}

void init_flags(Sweep& s, int64_t ref1, int64_t ref2, const Params& P) {
    s.out = s.v;
    s.flag.assign(s.nray * s.ngate, 0);
    for (int64_t j = 0; j < s.nray * s.ngate; ++j)
        if (std::isnan(s.v[j])) s.flag[j] = -3;
    std::vector<int64_t> refs{ref1};
    if (ref2 != ref1) refs.push_back(ref2);
    for (int64_t r : refs)
        for (int64_t g = 0; g < s.ngate; ++g)
            if (s.flag[s.at(r, g)] == 0 && std::fabs(s.v[s.at(r, g)]) < P.beta * s.vn)
                s.flag[s.at(r, g)] = 1;
    for (int64_t r : refs)
        for (int64_t nb : {r - 1, r + 1}) {
            const int64_t b = ray_of(nb, s.nray, s.wrap);
            if (b < 0) continue;
            for (int64_t g = 0; g < s.ngate; ++g)
                if (s.flag[s.at(b, g)] == 0 && s.flag[s.at(r, g)] > 0) {
                    double u;
                    if (accept(s.out[s.at(r, g)], s.v[s.at(b, g)], s.vn, P.alpha_first, u))
                        set_gate(s, b, g, u);
                }
        }
}

// ---------------------------------------------------------------- dealiasing modules

void first_pass(Sweep& s, int64_t ref1, int64_t ref2, const Params& P) {
    const int64_t nray = s.nray;
    std::pair<int64_t, int64_t> c1, c2;
    if (s.wrap) {
        const int64_t dcw = ref1 == ref2 ? nray : (((ref2 - ref1) % nray) + nray) % nray;
        const int64_t dccw = nray - dcw;
        c1 = {dcw / 2, std::max<int64_t>(0, dccw - 1 - dccw / 2)};
        c2 = {dccw / 2, std::max<int64_t>(0, dcw - 1 - dcw / 2)};
    } else {
        const int64_t lo = std::min(ref1, ref2), hi = std::max(ref1, ref2), d = hi - lo;
        if (ref1 == ref2) {
            c1 = {nray - 1 - ref1, ref1};
            c2 = {0, 0};
        } else {
            const std::pair<int64_t, int64_t> clo{d / 2, lo};
            const std::pair<int64_t, int64_t> chi{nray - 1 - hi, std::max<int64_t>(0, d - 1 - d / 2)};
            c1 = ref1 == lo ? clo : chi;
            c2 = ref2 == lo ? clo : chi;
        }
    }
    std::vector<std::tuple<int64_t, int, int64_t>> plan{{ref1, 1, c1.first}, {ref1, -1, c1.second}};
    if (ref2 != ref1 || s.wrap) {
        plan.emplace_back(ref2, 1, c2.first);
        plan.emplace_back(ref2, -1, c2.second);
    }
    std::vector<double> refs;
    for (const auto& [start, dir, count] : plan)
        for (int64_t k = 1; k <= count; ++k) {
            const int64_t i = ray_of(start + dir * k, nray, s.wrap);
            for (int64_t g = 0; g < s.ngate; ++g) {
                if (s.flag[s.at(i, g)] != 0) continue;
                refs.clear();
                for (int j = 1; j <= 3; ++j) {
                    const int64_t p = ray_of(i - dir * j, nray, s.wrap);
                    if (p >= 0 && s.flag[s.at(p, g)] > 0) refs.push_back(s.out[s.at(p, g)]);
                }
                if (refs.size() < 2) continue;
                double u;
                if (accept(mean_of(refs), s.v[s.at(i, g)], s.vn, P.alpha_first, u))
                    set_gate(s, i, g, u);
            }
        }
}

void range_pass(Sweep& s, double alpha, int window) {
    std::vector<double> refs;
    for (int64_t i = 0; i < s.nray; ++i) {
        for (int64_t g = 0; g < s.ngate; ++g) {
            if (s.flag[s.at(i, g)] != 0) continue;
            refs.clear();
            for (int64_t k = std::max<int64_t>(0, g - window); k < g; ++k)
                if (s.flag[s.at(i, k)] > 0) refs.push_back(s.out[s.at(i, k)]);
            double u;
            if (!refs.empty() && accept(mean_of(refs), s.v[s.at(i, g)], s.vn, alpha, u))
                set_gate(s, i, g, u);
        }
        for (int64_t g = s.ngate - 1; g >= 0; --g) {
            if (s.flag[s.at(i, g)] != 0) continue;
            refs.clear();
            for (int64_t k = g + 1; k < std::min<int64_t>(s.ngate, g + window + 1); ++k)
                if (s.flag[s.at(i, k)] > 0) refs.push_back(s.out[s.at(i, k)]);
            double u;
            if (!refs.empty() && accept(mean_of(refs), s.v[s.at(i, g)], s.vn, alpha, u))
                set_gate(s, i, g, u);
        }
    }
}

void azimuth_pass(Sweep& s, int64_t ref1, double alpha, int window) {
    const int64_t nray = s.nray;
    std::vector<double> refs;
    for (int dir : {1, -1}) {
        std::vector<int64_t> rays;
        if (s.wrap) {
            for (int64_t k = 1; k < nray; ++k) rays.push_back(ray_of(ref1 + dir * k, nray, true));
        } else if (dir == 1) {
            for (int64_t i = ref1 + 1; i < nray; ++i) rays.push_back(i);
        } else {
            for (int64_t i = ref1 - 1; i >= 0; --i) rays.push_back(i);
        }
        for (int64_t i : rays)
            for (int64_t g = 0; g < s.ngate; ++g) {
                if (s.flag[s.at(i, g)] != 0) continue;
                refs.clear();
                for (int j = 1; j <= window; ++j) {
                    const int64_t p = ray_of(i - dir * j, nray, s.wrap);
                    if (p >= 0 && s.flag[s.at(p, g)] > 0) refs.push_back(s.out[s.at(p, g)]);
                }
                double u;
                if (!refs.empty() && accept(median_of(refs), s.v[s.at(i, g)], s.vn, alpha, u))
                    set_gate(s, i, g, u);
            }
    }
}

void box_pass(Sweep& s, double alpha, std::pair<int, int> size) {
    const int64_t half_g = size.first / 2, half_a = size.second / 2;
    std::vector<double> refs;
    for (int64_t i = 0; i < s.nray; ++i)
        for (int64_t g = 0; g < s.ngate; ++g) {
            if (s.flag[s.at(i, g)] != 0) continue;
            refs.clear();
            for (int64_t a = i - half_a; a <= i + half_a; ++a) {
                const int64_t p = ray_of(a, s.nray, s.wrap);
                if (p < 0) continue;
                for (int64_t k = std::max<int64_t>(0, g - half_g);
                     k < std::min<int64_t>(s.ngate, g + half_g + 1); ++k)
                    if (s.flag[s.at(p, k)] > 0) refs.push_back(s.out[s.at(p, k)]);
            }
            double u;
            if (!refs.empty() && accept(median_of(refs), s.v[s.at(i, g)], s.vn, alpha, u))
                set_gate(s, i, g, u);
        }
}

void closest_pass(Sweep& s, int half) {
    std::vector<int64_t> done_idx;
    for (int64_t j = 0; j < s.nray * s.ngate; ++j)
        if (s.flag[j] > 0) done_idx.push_back(j);
    if (done_idx.empty()) return;
    const std::vector<double> snapshot = s.out;
    const std::vector<int8_t> done = s.flag;
    std::vector<double> refs;
    for (int64_t i = 0; i < s.nray; ++i)
        for (int64_t g = 0; g < s.ngate; ++g) {
            if (done[s.at(i, g)] != 0) continue;
            const double xi = s.x[s.at(i, g)], yi = s.y[s.at(i, g)];
            double best = kInf;
            int64_t bj = done_idx[0];
            for (int64_t j : done_idx) {
                const double dx = s.x[j] - xi, dy = s.y[j] - yi;
                const double d2 = mul(dx, dx) + mul(dy, dy);
                if (d2 < best) {
                    best = d2;
                    bj = j;
                }
            }
            const int64_t ci = bj / s.ngate, cg = bj % s.ngate;
            refs.clear();
            for (int64_t a = ci - half; a <= ci + half; ++a) {
                const int64_t p = ray_of(a, s.nray, s.wrap);
                if (p < 0) continue;
                for (int64_t q = std::max<int64_t>(0, cg - half); q < std::min<int64_t>(s.ngate, cg + half + 1); ++q)
                    if (done[s.at(p, q)] > 0) refs.push_back(snapshot[s.at(p, q)]);
            }
            set_gate(s, i, g, fold_to(median_of(refs), s.v[s.at(i, g)], s.vn));
        }
}

// ---------------------------------------------------------------- checks

inline double readjust(double vref, double x, double vn, double alpha) {
    if (std::fabs(x - vref) >= alpha * vn) {
        double u;
        if (accept(vref, x, vn, alpha, u)) return u;
    }
    return x;
}

void radial_check(Sweep& s, double alpha) {
    std::vector<double> next = s.out;
    std::vector<int64_t> idx;
    for (int64_t i = 0; i < s.nray; ++i) {
        idx.clear();
        for (int64_t k = 0; k < s.ngate; ++k)
            if (s.flag[s.at(i, k)] > 0) idx.push_back(k);
        if (idx.size() < 10) continue;
        const double n = static_cast<double>(idx.size());
        double sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (int64_t k : idx) {
            const double r = s.rng[k], y = s.out[s.at(i, k)];
            sx += r;
            sy += y;
            sxx += mul(r, r);
            sxy += mul(r, y);
        }
        const double den = mul(n, sxx) - mul(sx, sx);
        if (den == 0.0) continue;
        const double a = (mul(n, sxy) - mul(sx, sy)) / den;
        const double b = (sy - mul(a, sx)) / n;
        for (int64_t k : idx)
            next[s.at(i, k)] = readjust(mul(a, s.rng[k]) + b, s.out[s.at(i, k)], s.vn, alpha);
    }
    s.out = next;
}

// Mean of the values within one standard deviation of their mean; false if none.
bool within_sigma(const std::vector<double>& vals, double& ref) {
    const double mu = mean_of(vals);
    double var = 0.0;
    for (double x : vals) var += mul(x - mu, x - mu);
    const double sigma = std::sqrt(var / static_cast<double>(vals.size()));
    std::vector<double> kept;
    for (double x : vals)
        if (std::fabs(x - mu) <= sigma) kept.push_back(x);
    if (kept.empty()) return false;
    ref = mean_of(kept);
    return true;
}

void box_check(Sweep& s, double alpha, std::pair<int, int> size) {
    const int64_t half_g = size.first / 2, half_a = size.second / 2;
    std::vector<double> next = s.out, vals;
    for (int64_t i = 0; i < s.nray; ++i)
        for (int64_t g = 0; g < s.ngate; ++g) {
            if (s.flag[s.at(i, g)] <= 0) continue;
            vals.clear();
            for (int64_t a = i - half_a; a <= i + half_a; ++a) {
                const int64_t p = ray_of(a, s.nray, s.wrap);
                if (p < 0) continue;
                for (int64_t k = std::max<int64_t>(0, g - half_g);
                     k < std::min<int64_t>(s.ngate, g + half_g + 1); ++k)
                    if (s.flag[s.at(p, k)] > 0 && !(p == i && k == g)) vals.push_back(s.out[s.at(p, k)]);
            }
            double ref;
            if (!vals.empty() && within_sigma(vals, ref))
                next[s.at(i, g)] = readjust(ref, s.out[s.at(i, g)], s.vn, alpha);
        }
    s.out = next;
}

// ---------------------------------------------------------------- driver

bool any_unprocessed(const Sweep& s) {
    for (int8_t f : s.flag)
        if (f == 0) return true;
    return false;
}

void run_sweep(Sweep& s, const Params& P) {
    preprocess(s, P);
    const auto [ref1, ref2] = reference_radials(s, P);
    init_flags(s, ref1, ref2, P);
    first_pass(s, ref1, ref2, P);
    for (int w : P.windows)
        if (any_unprocessed(s)) {
            range_pass(s, P.alpha, w);
            azimuth_pass(s, ref1, P.alpha, w);
        }
    for (const auto& size : P.boxes)
        if (any_unprocessed(s)) box_pass(s, P.alpha, size);
    if (any_unprocessed(s)) closest_pass(s, P.closest_half);
    radial_check(s, P.alpha);
    box_check(s, P.alpha, P.check_box);
}

template <class Body>
void parallel_for(int64_t n, int n_threads, Body&& body) {
    const int hw = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    const int nt = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(n_threads > 0 ? n_threads : hw, n)));
    std::atomic<int64_t> next{0};
    auto worker = [&]() {
        for (int64_t i; (i = next.fetch_add(1)) < n;) body(i);
    };
    std::vector<std::thread> pool;
    for (int t = 1; t < nt; ++t) pool.emplace_back(worker);
    worker();
    for (auto& th : pool) th.join();
}

}  // namespace

// One (dealiased velocity, flag) pair per sweep. Gates that are not processed
// (flag -3) are NaN in the velocity.
std::vector<std::pair<py::array_t<double>, py::array_t<int8_t>>> unravel(
    const std::vector<DArray>& velocity, const std::vector<double>& nyquist,
    const std::vector<DArray>& azimuth, const std::vector<DArray>& range,
    const std::vector<DArray>& x, const std::vector<DArray>& y, const std::vector<bool>& wrap,
    double alpha, double alpha_first, double alpha_pre, double z_cutoff, int pre_window, double beta,
    const std::vector<int>& windows, const std::vector<std::pair<int, int>>& boxes, int closest_half,
    std::pair<int, int> check_box, int n_threads) {
    const size_t ns = velocity.size();
    if (nyquist.size() != ns || azimuth.size() != ns || range.size() != ns || x.size() != ns ||
        y.size() != ns || wrap.size() != ns)
        throw std::invalid_argument("need one array of each kind per sweep");
    Params P{alpha, alpha_first, alpha_pre, z_cutoff, beta, pre_window, closest_half, windows, boxes, check_box};
    std::vector<Sweep> sweeps(ns);
    for (size_t k = 0; k < ns; ++k) {
        Sweep& s = sweeps[k];
        if (velocity[k].ndim() != 2) throw std::invalid_argument("velocity must be 2-D (ray, gate)");
        s.nray = velocity[k].shape(0);
        s.ngate = velocity[k].shape(1);
        if (azimuth[k].size() != s.nray || range[k].size() != s.ngate || x[k].size() != s.nray * s.ngate ||
            y[k].size() != s.nray * s.ngate)
            throw std::invalid_argument("azimuth, range, x or y do not match the velocity");
        if (!(nyquist[k] > 0)) throw std::invalid_argument("nyquist must be positive");
        s.vel = velocity[k].data();
        s.vn = nyquist[k];
        s.az = azimuth[k].data();
        s.rng = range[k].data();
        s.x = x[k].data();
        s.y = y[k].data();
        s.wrap = wrap[k];
    }
    {
        py::gil_scoped_release release;
        parallel_for(static_cast<int64_t>(ns), n_threads, [&](int64_t k) { run_sweep(sweeps[k], P); });
    }
    std::vector<std::pair<py::array_t<double>, py::array_t<int8_t>>> result;
    for (size_t k = 0; k < ns; ++k) {
        Sweep& s = sweeps[k];
        py::array_t<double> out({s.nray, s.ngate});
        py::array_t<int8_t> flag({s.nray, s.ngate});
        double* po = out.mutable_data();
        int8_t* pf = flag.mutable_data();
        for (int64_t j = 0; j < s.nray * s.ngate; ++j) {
            const bool done = s.flag[j] > 0;
            po[j] = done ? s.out[j] : kNaN;
            pf[j] = done ? (s.out[j] == s.vel[j] ? 1 : 2) : -3;
        }
        result.emplace_back(out, flag);
    }
    return result;
}

// Continuity with the sweep below (3-D check): returns the readjusted velocity.
py::array_t<double> check_3d(const DArray& out, const py::array_t<int8_t, py::array::c_style | py::array::forcecast>& flag,
                             const DArray& below,
                             const py::array_t<int8_t, py::array::c_style | py::array::forcecast>& below_flag,
                             const IArray& iray, const IArray& igate, double vn, double alpha, int window_gates,
                             int window_rays, bool wrap) {
    if (out.ndim() != 2 || below.ndim() != 2) throw std::invalid_argument("sweeps must be 2-D");
    const int64_t nray = out.shape(0), ngate = out.shape(1), nbr = below.shape(0), nbg = below.shape(1);
    if (flag.size() != out.size() || iray.size() != out.size() || igate.size() != out.size() ||
        below_flag.size() != below.size())
        throw std::invalid_argument("flag and index arrays must match their sweep");
    py::array_t<double> result({nray, ngate});
    double* pr = result.mutable_data();
    const double* po = out.data();
    const double* pb = below.data();
    const int8_t* pf = flag.data();
    const int8_t* pbf = below_flag.data();
    const int32_t* ir = iray.data();
    const int32_t* ig = igate.data();
    {
        py::gil_scoped_release release;
        std::vector<double> vals;
        for (int64_t i = 0; i < nray; ++i)
            for (int64_t g = 0; g < ngate; ++g) {
                const int64_t j = i * ngate + g;
                pr[j] = po[j];
                if (pf[j] <= 0 || ir[j] < 0 || ig[j] < 0) continue;
                vals.clear();
                for (int64_t a = ir[j] - window_rays; a <= ir[j] + window_rays; ++a) {
                    const int64_t p = ray_of(a, nbr, wrap);
                    if (p < 0) continue;
                    for (int64_t k = std::max<int64_t>(0, ig[j] - window_gates);
                         k < std::min<int64_t>(nbg, ig[j] + window_gates + 1); ++k)
                        if (pbf[p * nbg + k] > 0) vals.push_back(pb[p * nbg + k]);
                }
                double ref;
                if (!vals.empty() && within_sigma(vals, ref)) pr[j] = readjust(ref, po[j], vn, alpha);
            }
    }
    return result;
}

PYBIND11_MODULE(_unravel, m) {
    m.doc() = "Compiled UNRAVEL dealiasing driver (Louf et al. 2020) for radarx.";
    m.def("unravel", &unravel, py::arg("velocity"), py::arg("nyquist"), py::arg("azimuth"), py::arg("range"),
          py::arg("x"), py::arg("y"), py::arg("wrap"), py::arg("alpha"), py::arg("alpha_first"),
          py::arg("alpha_pre"), py::arg("z_cutoff"), py::arg("pre_window"), py::arg("beta"),
          py::arg("windows"), py::arg("boxes"), py::arg("closest_half"), py::arg("check_box"),
          py::arg("n_threads") = 0);
    m.def("check_3d", &check_3d, py::arg("out"), py::arg("flag"), py::arg("below"), py::arg("below_flag"),
          py::arg("iray"), py::arg("igate"), py::arg("vn"), py::arg("alpha"), py::arg("window_gates"),
          py::arg("window_rays"), py::arg("wrap"));
}
