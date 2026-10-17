// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Kernels of radarx/retrieve/dealias_methods.py:
//
// r2d2_sweep: region-based recursive dealiasing of one PPI sweep after Feldmann
// et al. (2020), steps 2 to 9 of the paper (the steps are described in the module
// docstring, and radarx/retrieve/_dealias_methods_numpy.py follows them line by
// line). The velocities are integers in steps of 1/256 m/s, so every decision is
// made with integer arithmetic and the result does not depend on the number of
// threads. The work inside one sweep (maximum differences, buffer zones, the
// scanning passes along rays and along range, the windows) is shared by the
// threads; the steps of the scanning passes are sequential along a ray or a
// range gate, as the passes of the paper are.
//
// fold_margin: regions of a dealiased sweep (union-find over neighbouring gates
// that differ by less than a threshold) and, for every region, the number of
// gate pairs on its boundary and the sum of the velocity differences across it.
// Sweeps are independent and go to the threads through an atomic counter.

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
using IArray = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;

namespace {

constexpr int32_t kMissing = std::numeric_limits<int32_t>::min();

inline int64_t floor_div(int64_t a, int64_t b) {
    int64_t q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
    return q;
}
// round(num / den), halves up (den > 0)
inline int64_t round_div(int64_t num, int64_t den) { return floor_div(2 * num + den, 2 * den); }

int resolve_threads(int n_threads) {
    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    return n_threads > 0 ? n_threads : static_cast<int>(hw);
}

// body(chunk, begin, end) for chunks of `block` items, shared by the threads
template <class F>
void parallel_chunks(int64_t n, int64_t block, int n_threads, F&& body) {
    if (n <= 0) return;
    block = std::max<int64_t>(1, block);
    const int64_t nchunks = (n + block - 1) / block;
    int nt = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(resolve_threads(n_threads), nchunks)));
    std::atomic<int64_t> next{0};
    auto worker = [&]() {
        for (;;) {
            const int64_t c = next.fetch_add(1);
            if (c >= nchunks) break;
            body(c, c * block, std::min(n, (c + 1) * block));
        }
    };
    if (nt == 1) {
        worker();
        return;
    }
    std::vector<std::thread> pool;
    for (int i = 1; i < nt; ++i) pool.emplace_back(worker);
    worker();
    for (auto& t : pool) t.join();
}

struct Grid {
    int64_t nray, ngate;
    bool wrap;
    int64_t size() const { return nray * ngate; }
    // ray i + d, or -1 outside the sweep
    int64_t ray(int64_t i, int64_t d) const {
        int64_t r = i + d;
        if (wrap) {
            r %= nray;
            if (r < 0) r += nray;
            return r;
        }
        return (r < 0 || r >= nray) ? -1 : r;
    }
};

// ---------------------------------------------------------------- R2D2

struct Params {
    int64_t nyq2;
    int64_t beta;
    double gamma;
    int64_t min_overlap;
    int64_t buffer_half;
    std::vector<int64_t> thresholds;
    int64_t window_half;
    int64_t window_min;
    int64_t max_iterations;
    int64_t patience;
    bool ref_fallback;
    int n_threads;
};

// largest absolute difference of every valid gate to its valid 8 neighbours; returns the sum
int64_t max_difference(const Grid& g, const std::vector<int32_t>& cur, std::vector<int32_t>& dmax, int nt) {
    const int64_t rows_per_chunk = std::max<int64_t>(1, 65536 / std::max<int64_t>(1, g.ngate));
    const int64_t nchunks = (g.nray + rows_per_chunk - 1) / rows_per_chunk;
    std::vector<int64_t> partial(static_cast<size_t>(nchunks), 0);
    parallel_chunks(g.nray, rows_per_chunk, nt, [&](int64_t c, int64_t i0, int64_t i1) {
        int64_t sum = 0;
        for (int64_t i = i0; i < i1; ++i) {
            for (int64_t j = 0; j < g.ngate; ++j) {
                const int64_t idx = i * g.ngate + j;
                const int32_t v = cur[idx];
                if (v == kMissing) {
                    dmax[idx] = 0;
                    continue;
                }
                int64_t best = 0;
                for (int di = -1; di <= 1; ++di) {
                    const int64_t i2 = g.ray(i, di);
                    if (i2 < 0) continue;
                    for (int dj = -1; dj <= 1; ++dj) {
                        if (di == 0 && dj == 0) continue;
                        const int64_t j2 = j + dj;
                        if (j2 < 0 || j2 >= g.ngate) continue;
                        const int32_t w = cur[i2 * g.ngate + j2];
                        if (w == kMissing) continue;
                        best = std::max<int64_t>(best, std::llabs(static_cast<int64_t>(v) - w));
                    }
                }
                dmax[idx] = static_cast<int32_t>(best);
                sum += best;
            }
        }
        partial[static_cast<size_t>(c)] = sum;
    });
    int64_t total = 0;
    for (int64_t s : partial) total += s;
    return total;
}

// gates with a gate of `mask` within `half` rays and gates
void dilate(const Grid& g, const std::vector<uint8_t>& mask, int64_t half, std::vector<uint8_t>& out, int nt) {
    std::vector<uint8_t> rows(static_cast<size_t>(g.size()), 0);
    parallel_chunks(g.nray, 16, nt, [&](int64_t, int64_t i0, int64_t i1) {
        for (int64_t i = i0; i < i1; ++i) {
            for (int64_t j = 0; j < g.ngate; ++j) {
                uint8_t any = 0;
                const int64_t a = std::max<int64_t>(0, j - half), b = std::min<int64_t>(g.ngate - 1, j + half);
                for (int64_t j2 = a; j2 <= b && !any; ++j2) any = mask[i * g.ngate + j2];
                rows[i * g.ngate + j] = any;
            }
        }
    });
    parallel_chunks(g.nray, 16, nt, [&](int64_t, int64_t i0, int64_t i1) {
        for (int64_t i = i0; i < i1; ++i) {
            for (int64_t j = 0; j < g.ngate; ++j) {
                uint8_t any = 0;
                for (int64_t d = -half; d <= half && !any; ++d) {
                    const int64_t i2 = g.ray(i, d);
                    if (i2 >= 0) any = rows[i2 * g.ngate + j];
                }
                out[i * g.ngate + j] = any;
            }
        }
    });
}

inline int32_t find_root(std::vector<int32_t>& parent, int32_t x) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    return x;
}

// connected regions (8 neighbours) of `mask`; ids from 0, -1 outside; returns their number
int32_t label_regions(const Grid& g, const std::vector<uint8_t>& mask, std::vector<int32_t>& labels) {
    const int64_t n = g.size();
    std::vector<int32_t> parent(static_cast<size_t>(n));
    for (int64_t k = 0; k < n; ++k) parent[k] = static_cast<int32_t>(k);
    auto unite = [&](int64_t a, int64_t b) {
        int32_t ra = find_root(parent, static_cast<int32_t>(a)), rb = find_root(parent, static_cast<int32_t>(b));
        if (ra != rb) parent[std::max(ra, rb)] = std::min(ra, rb);
    };
    for (int64_t i = 0; i < g.nray; ++i) {
        const int64_t i2 = g.ray(i, 1);
        for (int64_t j = 0; j < g.ngate; ++j) {
            const int64_t k = i * g.ngate + j;
            if (!mask[k]) continue;
            if (j + 1 < g.ngate && mask[k + 1]) unite(k, k + 1);
            if (i2 < 0) continue;
            for (int dj = -1; dj <= 1; ++dj) {
                const int64_t j2 = j + dj;
                if (j2 < 0 || j2 >= g.ngate) continue;
                const int64_t k2 = i2 * g.ngate + j2;
                if (mask[k2]) unite(k, k2);
            }
        }
    }
    std::vector<int32_t> id(static_cast<size_t>(n), -1);
    int32_t count = 0;
    for (int64_t k = 0; k < n; ++k) {
        if (!mask[k]) {
            labels[k] = -1;
            continue;
        }
        const int32_t r = find_root(parent, static_cast<int32_t>(k));
        if (id[r] < 0) id[r] = count++;
        labels[k] = id[r];
    }
    return count;
}

// fold step of every region from the guidance fields; decided[r] and delta[r]
void region_decisions(const Grid& g, const std::vector<int32_t>& cur, const std::vector<int32_t>& labels, int32_t nreg,
                      const std::vector<std::pair<const int32_t*, bool>>& guides, const Params& p, std::vector<uint8_t>& decided,
                      std::vector<int64_t>& delta) {
    decided.assign(static_cast<size_t>(nreg), 0);
    delta.assign(static_cast<size_t>(nreg), 0);
    if (nreg == 0 || guides.empty()) return;
    std::vector<int64_t> have(static_cast<size_t>(nreg), 0), first(static_cast<size_t>(nreg), 0);
    std::vector<uint8_t> fail(static_cast<size_t>(nreg), 0);
    std::vector<int64_t> cnt(static_cast<size_t>(nreg)), s1(static_cast<size_t>(nreg)), s2(static_cast<size_t>(nreg));
    const int64_t n = g.size();
    for (const auto& entry : guides) {
        const int32_t* guide = entry.first;
        const bool fallback = entry.second;
        std::fill(cnt.begin(), cnt.end(), 0);
        std::fill(s1.begin(), s1.end(), 0);
        std::fill(s2.begin(), s2.end(), 0);
        for (int64_t k = 0; k < n; ++k) {
            const int32_t r = labels[k];
            if (r < 0 || guide[k] == kMissing) continue;
            const int64_t d = static_cast<int64_t>(guide[k]) - cur[k];
            ++cnt[r];
            s1[r] += d;
            s2[r] += d * d;
        }
        for (int32_t r = 0; r < nreg; ++r) {
            if (cnt[r] < p.min_overlap || cnt[r] == 0) continue;
            if (fallback && have[r] > 0) continue;
            const double safe = static_cast<double>(cnt[r]);
            const double mean = static_cast<double>(s1[r]) / safe;
            const double var = static_cast<double>(s2[r]) / safe - mean * mean;
            const int64_t k = round_div(s1[r], cnt[r] * p.nyq2);
            if (var > p.gamma * p.gamma) fail[r] = 1;
            if (have[r] > 0 && k != first[r]) fail[r] = 1;
            if (have[r] == 0) first[r] = k;
            ++have[r];
        }
    }
    for (int32_t r = 0; r < nreg; ++r) {
        decided[r] = (have[r] > 0 && !fail[r]) ? 1 : 0;
        delta[r] = decided[r] ? first[r] : 0;
    }
}

// one scanning pass of the spatial dealiasing. axis 1: along every ray (range gates in
// order); axis 0: along every range gate (rays in order, two laps for a full circle).
// The operation on one gate is the same in both: it takes the fold that brings it within
// s_q of its resolved predecessor in the scanning direction. The rays are walked in
// order for blocks of range gates, so that the memory is read along the rows.
inline void scan_step(std::vector<int32_t>& cur, std::vector<int32_t>& fold, std::vector<uint8_t>& res, int64_t a,
                      int64_t b, int64_t nyq2, int64_t s_q) {
    if (cur[a] == kMissing || res[a] || !res[b]) return;
    const int64_t m = round_div(static_cast<int64_t>(cur[b]) - cur[a], nyq2);
    const int64_t nw = cur[a] + m * nyq2;
    if (std::llabs(nw - cur[b]) < s_q) {
        cur[a] = static_cast<int32_t>(nw);
        fold[a] += static_cast<int32_t>(m);
        res[a] = 1;
    }
}

void scan(const Grid& g, std::vector<int32_t>& cur, std::vector<int32_t>& fold, std::vector<uint8_t>& res,
          int64_t nyq2, int64_t s_q, int axis, bool reverse, int nt) {
    const int64_t step = reverse ? 1 : -1;  // the predecessor lies at position + step
    if (axis == 1) {
        parallel_chunks(g.nray, 8, nt, [&](int64_t, int64_t i0, int64_t i1) {
            for (int64_t i = i0; i < i1; ++i) {
                const int64_t base = i * g.ngate;
                for (int64_t t = 0; t < g.ngate; ++t) {
                    const int64_t pos = reverse ? g.ngate - 1 - t : t;
                    const int64_t pre = pos + step;
                    if (pre < 0 || pre >= g.ngate) continue;
                    scan_step(cur, fold, res, base + pos, base + pre, nyq2, s_q);
                }
            }
        });
        return;
    }
    const int64_t laps = g.wrap ? 2 : 1;
    parallel_chunks(g.ngate, 256, nt, [&](int64_t, int64_t j0, int64_t j1) {
        for (int64_t lap = 0; lap < laps; ++lap) {
            for (int64_t t = 0; t < g.nray; ++t) {
                const int64_t pos = reverse ? g.nray - 1 - t : t;
                int64_t pre = pos + step;
                if (g.wrap) {
                    pre %= g.nray;
                    if (pre < 0) pre += g.nray;
                } else if (pre < 0 || pre >= g.nray) {
                    continue;
                }
                const int64_t ra = pos * g.ngate, rb = pre * g.ngate;
                for (int64_t j = j0; j < j1; ++j) scan_step(cur, fold, res, ra + j, rb + j, nyq2, s_q);
            }
        }
    });
}

// unresolved gates follow the median of the resolved gates around them
void window_pass(const Grid& g, std::vector<int32_t>& cur, std::vector<int32_t>& fold, std::vector<uint8_t>& res,
                 const Params& p) {
    std::vector<int64_t> todo;
    for (int64_t k = 0; k < g.size(); ++k)
        if (cur[k] != kMissing && !res[k]) todo.push_back(k);
    if (todo.empty()) return;
    const std::vector<int32_t> snap = cur;
    const std::vector<uint8_t> snap_res = res;
    const int64_t half = p.window_half;
    parallel_chunks(static_cast<int64_t>(todo.size()), 256, p.n_threads, [&](int64_t, int64_t t0, int64_t t1) {
        std::vector<int32_t> values;
        for (int64_t t = t0; t < t1; ++t) {
            const int64_t k = todo[static_cast<size_t>(t)];
            const int64_t i = k / g.ngate, j = k % g.ngate;
            values.clear();
            const int64_t j0 = std::max<int64_t>(0, j - half), j1 = std::min<int64_t>(g.ngate - 1, j + half);
            for (int64_t d = -half; d <= half; ++d) {
                const int64_t i2 = g.ray(i, d);
                if (i2 < 0) continue;
                for (int64_t j2 = j0; j2 <= j1; ++j2) {
                    const int64_t k2 = i2 * g.ngate + j2;
                    if (snap_res[k2]) values.push_back(snap[k2]);
                }
            }
            if (static_cast<int64_t>(values.size()) < p.window_min || values.empty()) continue;
            const size_t mid = (values.size() - 1) / 2;
            std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid), values.end());
            const int64_t m = round_div(static_cast<int64_t>(values[mid]) - cur[k], p.nyq2);
            cur[k] = static_cast<int32_t>(cur[k] + m * p.nyq2);
            fold[k] += static_cast<int32_t>(m);
            res[k] = 1;
        }
    });
}

std::tuple<py::array_t<int32_t>, std::vector<int64_t>> r2d2_sweep(
    const IArray& q_in, const IArray& ref_in, const IArray& above_in, const IArray& prev_in, bool wrap, int64_t nyq2,
    int64_t beta, double gamma, int64_t min_overlap, int64_t buffer_half, const std::vector<int64_t>& thresholds,
    int64_t window_half, int64_t window_min, int64_t max_iterations, int64_t patience, bool ref_fallback,
    int n_threads) {
    if (q_in.ndim() != 2) throw std::invalid_argument("velocity must be 2-D (ray, gate)");
    if (nyq2 <= 0) throw std::invalid_argument("nyq2 must be positive");
    const Grid g{q_in.shape(0), q_in.shape(1), wrap};
    const int64_t n = g.size();
    const int32_t* guide_src[3] = {above_in.data(), prev_in.data(), ref_in.data()};  // above, prev, ref
    const bool present[3] = {above_in.size() == n, prev_in.size() == n, ref_in.size() == n};
    for (const IArray* a : {&above_in, &prev_in, &ref_in})
        if (a->size() != 0 && a->size() != n) throw std::invalid_argument("guidance must have the shape of the velocity");
    Params p{nyq2, beta, gamma, min_overlap, buffer_half, thresholds, window_half, window_min, max_iterations, patience, ref_fallback, n_threads};
    py::array_t<int32_t> out({g.nray, g.ngate});
    std::vector<int64_t> info(4, 0);
    {
        py::gil_scoped_release release;
        std::vector<int32_t> cur(q_in.data(), q_in.data() + n);
        std::vector<int32_t> fold(static_cast<size_t>(n), 0), best_fold(static_cast<size_t>(n), 0);
        std::vector<int32_t> dmax(static_cast<size_t>(n), 0);
        int64_t nvalid = 0;
        int64_t cover[3] = {0, 0, 0};
        for (int64_t k = 0; k < n; ++k) {
            if (cur[k] == kMissing) continue;
            ++nvalid;
            for (int c = 0; c < 3; ++c)
                if (present[c] && guide_src[c][k] != kMissing) ++cover[c];
        }
        int64_t best_sum = max_difference(g, cur, dmax, n_threads);
        int64_t best_it = 0, iterations = 0, buffer_gates = 0;
        std::vector<uint8_t> res(static_cast<size_t>(n), 0), best_res(static_cast<size_t>(n), 0);
        const bool far = std::max(cover[0], cover[1]) * 10 < nvalid;
        if (nvalid > 0) {
            std::vector<uint8_t> high(static_cast<size_t>(n)), buffer(static_cast<size_t>(n)), mask(static_cast<size_t>(n));
            std::vector<int32_t> labels(static_cast<size_t>(n));
            std::vector<uint8_t> decided;
            std::vector<int64_t> delta;
            for (int64_t it = 1; it <= max_iterations; ++it) {
                max_difference(g, cur, dmax, n_threads);
                for (int64_t k = 0; k < n; ++k) high[k] = (cur[k] != kMissing && dmax[k] >= beta) ? 1 : 0;
                dilate(g, high, buffer_half, buffer, n_threads);
                for (int64_t k = 0; k < n; ++k) mask[k] = (cur[k] != kMissing && !buffer[k]) ? 1 : 0;
                const int32_t nreg = label_regions(g, mask, labels);
                std::vector<std::pair<const int32_t*, bool>> guides;
                if (present[0]) guides.emplace_back(guide_src[0], false);
                if (present[1]) guides.emplace_back(guide_src[1], false);
                if (present[2] && (ref_fallback || far || it > 1)) guides.emplace_back(guide_src[2], ref_fallback);
                region_decisions(g, cur, labels, nreg, guides, p, decided, delta);
                const std::vector<int32_t> fold_before = fold;
                std::fill(res.begin(), res.end(), 0);
                for (int64_t k = 0; k < n; ++k) {
                    const int32_t r = labels[k];
                    if (r < 0 || !decided[r]) continue;
                    cur[k] = static_cast<int32_t>(cur[k] + delta[r] * nyq2);
                    fold[k] += static_cast<int32_t>(delta[r]);
                    res[k] = 1;
                }
                buffer_gates = 0;
                for (int64_t k = 0; k < n; ++k)
                    if (cur[k] != kMissing && !res[k]) ++buffer_gates;
                if (buffer_gates > 0) {
                    for (int64_t s_q : thresholds) {
                        scan(g, cur, fold, res, nyq2, s_q, 1, false, n_threads);
                        scan(g, cur, fold, res, nyq2, s_q, 1, true, n_threads);
                        scan(g, cur, fold, res, nyq2, s_q, 0, false, n_threads);
                        scan(g, cur, fold, res, nyq2, s_q, 0, true, n_threads);
                    }
                }
                window_pass(g, cur, fold, res, p);
                if (present[2]) {
                    const int32_t* ref = guide_src[2];
                    for (int64_t k = 0; k < n; ++k) {
                        if (cur[k] == kMissing || res[k] || ref[k] == kMissing) continue;
                        const int64_t m = round_div(static_cast<int64_t>(ref[k]) - cur[k], nyq2);
                        cur[k] = static_cast<int32_t>(cur[k] + m * nyq2);
                        fold[k] += static_cast<int32_t>(m);
                        res[k] = 1;
                    }
                }
                const int64_t total = max_difference(g, cur, dmax, n_threads);
                iterations = it;
                if (total < best_sum) {
                    best_sum = total;
                    best_fold = fold;
                    best_res = res;
                    best_it = it;
                } else if (it - best_it >= patience) {
                    break;
                }
                if (fold == fold_before) break;
            }
        }
        int64_t unresolved = nvalid;
        if (best_it > 0) {
            unresolved = 0;
            for (int64_t k = 0; k < n; ++k)
                if (q_in.data()[k] != kMissing && !best_res[k]) ++unresolved;
        }
        std::copy(best_fold.begin(), best_fold.end(), out.mutable_data());
        info = {iterations, best_it, unresolved, buffer_gates};
    }
    return {out, info};
}

// ---------------------------------------------------------------- fold margin

struct MarginOut {
    std::vector<int32_t> region;
    std::vector<int64_t> size, pairs, total;
};

MarginOut margin_one(const int32_t* q, const Grid& g, int64_t threshold) {
    const int64_t n = g.size();
    std::vector<int32_t> parent(static_cast<size_t>(n));
    for (int64_t k = 0; k < n; ++k) parent[k] = static_cast<int32_t>(k);
    auto unite = [&](int64_t a, int64_t b) {
        int32_t ra = find_root(parent, static_cast<int32_t>(a)), rb = find_root(parent, static_cast<int32_t>(b));
        if (ra != rb) parent[std::max(ra, rb)] = std::min(ra, rb);
    };
    for (int64_t i = 0; i < g.nray; ++i) {
        const int64_t i2 = g.ray(i, 1);
        for (int64_t j = 0; j < g.ngate; ++j) {
            const int64_t k = i * g.ngate + j;
            if (q[k] == kMissing) continue;
            if (j + 1 < g.ngate && q[k + 1] != kMissing && std::llabs(static_cast<int64_t>(q[k]) - q[k + 1]) < threshold)
                unite(k, k + 1);
            if (i2 >= 0) {
                const int64_t k2 = i2 * g.ngate + j;
                if (q[k2] != kMissing && std::llabs(static_cast<int64_t>(q[k]) - q[k2]) < threshold) unite(k, k2);
            }
        }
    }
    std::vector<int32_t> id(static_cast<size_t>(n), -1);
    MarginOut out;
    out.region.assign(static_cast<size_t>(n), -1);
    int32_t count = 0;
    for (int64_t k = 0; k < n; ++k) {
        if (q[k] == kMissing) continue;
        const int32_t r = find_root(parent, static_cast<int32_t>(k));
        if (id[r] < 0) id[r] = count++;
        out.region[k] = id[r];
    }
    out.size.assign(static_cast<size_t>(count), 0);
    out.pairs.assign(static_cast<size_t>(count), 0);
    out.total.assign(static_cast<size_t>(count), 0);
    for (int64_t k = 0; k < n; ++k)
        if (out.region[k] >= 0) ++out.size[out.region[k]];
    auto pair = [&](int64_t a, int64_t b) {
        if (q[a] == kMissing || q[b] == kMissing || out.region[a] == out.region[b]) return;
        const int64_t d = static_cast<int64_t>(q[b]) - q[a];
        ++out.pairs[out.region[a]];
        out.total[out.region[a]] += d;
        ++out.pairs[out.region[b]];
        out.total[out.region[b]] -= d;
    };
    for (int64_t i = 0; i < g.nray; ++i) {
        const int64_t i2 = g.ray(i, 1);
        for (int64_t j = 0; j < g.ngate; ++j) {
            const int64_t k = i * g.ngate + j;
            if (j + 1 < g.ngate) pair(k, k + 1);
            if (i2 >= 0) pair(k, i2 * g.ngate + j);
        }
    }
    return out;
}

using MarginTuple =
    std::tuple<py::array_t<int32_t>, py::array_t<int64_t>, py::array_t<int64_t>, py::array_t<int64_t>>;

template <class T>
py::array_t<T> to_array(const std::vector<T>& v, std::vector<py::ssize_t> shape) {
    py::array_t<T> a(shape);
    std::copy(v.begin(), v.end(), a.mutable_data());
    return a;
}

std::vector<MarginTuple> fold_margin(const std::vector<IArray>& q, const std::vector<bool>& wrap,
                                     const std::vector<int64_t>& threshold, int n_threads) {
    const size_t ns = q.size();
    if (wrap.size() != ns || threshold.size() != ns)
        throw std::invalid_argument("one velocity, wrap flag and threshold per sweep");
    for (size_t s = 0; s < ns; ++s)
        if (q[s].ndim() != 2) throw std::invalid_argument("velocity must be 2-D (ray, gate)");
    std::vector<MarginOut> results(ns);
    {
        py::gil_scoped_release release;
        parallel_chunks(static_cast<int64_t>(ns), 1, n_threads, [&](int64_t, int64_t s0, int64_t s1) {
            for (int64_t s = s0; s < s1; ++s) {
                const Grid g{q[s].shape(0), q[s].shape(1), static_cast<bool>(wrap[s])};
                results[s] = margin_one(q[s].data(), g, threshold[s]);
            }
        });
    }
    std::vector<MarginTuple> out;
    for (size_t s = 0; s < ns; ++s) {
        auto& r = results[s];
        out.emplace_back(to_array(r.region, {q[s].shape(0), q[s].shape(1)}),
                         to_array(r.size, {static_cast<py::ssize_t>(r.size.size())}),
                         to_array(r.pairs, {static_cast<py::ssize_t>(r.pairs.size())}),
                         to_array(r.total, {static_cast<py::ssize_t>(r.total.size())}));
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(_dealias_methods, m) {
    m.doc() = "R2D2 dealiasing and fold margin kernels";
    m.def("r2d2_sweep", &r2d2_sweep, py::arg("velocity"), py::arg("ref"), py::arg("above"), py::arg("prev"),
          py::arg("wrap"), py::arg("nyq2"), py::arg("beta"), py::arg("gamma"), py::arg("min_overlap"),
          py::arg("buffer_half"), py::arg("thresholds"), py::arg("window_half"), py::arg("window_min"),
          py::arg("max_iterations"), py::arg("patience"), py::arg("ref_fallback") = false, py::arg("n_threads") = 0);
    m.def("fold_margin", &fold_margin, py::arg("velocity"), py::arg("wrap"), py::arg("threshold"),
          py::arg("n_threads") = 0);
}
