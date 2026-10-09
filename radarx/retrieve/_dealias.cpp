// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Region-based Doppler velocity dealiasing kernel for PPI sweeps. The method
// is radarx's own combination of published concepts (cited below; details not
// checked against the papers), not an implementation of one
// paper. Thresholds and limits are radarx choices.
//
// region_folds (all sweeps of a volume in one call, in parallel):
// 1. Segmentation: neighbouring gates (along the ray and between adjacent
//    rays) whose velocities differ by less than a threshold are joined with a
//    union-find. Within a region the field is continuous, so all its gates
//    share one Nyquist fold.
// 2. Region adjacency: for every pair of touching regions (also across
//    short gaps along and across rays) the number of boundary gate pairs
//    and the summed velocity jump across the boundary (in units of the
//    Nyquist interval 2 Vn, fixed point) are accumulated.
// 3. Fold offsets: a maximum spanning tree (unambiguous boundaries first,
//    then by length; Kruskal with an offset-carrying union-find) gives an
//    initial integer fold per region;
//    integer coordinate descent then minimises the summed squared velocity
//    jump over all region boundaries (a least-squares criterion attributed to
//    Jing and Wiener 1993, paper not checked, restricted to integers and
//    solved here by radarx's own coordinate descent), alternating
//    single-region moves with moves of whole blocks of consistently joined
//    regions, so that
//    groups that are offset together are corrected as well.
//
// absolute_folds:
// 4. The largest component is shifted to a reference velocity (most common
//    gate vote) where one exists: a wind profile (Eilts and Smith 1990) or
//    the dealiased sweep below (volume continuity, James and Houze 2001);
//    otherwise its mean velocity is brought closest to zero. A VAD fit of it
//    (Browning and Wexler 1968) is the reference for all other components.
//    Finally, gates that differ by more than Vn from all their neighbours are
//    moved to the fold closest to the neighbours' mean.
//
// All sums that decide a fold are integers, so the result does not depend
// on the number of threads and matches the NumPy reference exactly.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace py = pybind11;
using DArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using BArray = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;
using IArray = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;

namespace {

constexpr int64_t kScale = int64_t(1) << 20;  // fixed point for jumps / (2 Vn)
constexpr int64_t kMaxVote = 127;             // folds beyond this are not physical

// floor(a / b) for b > 0
inline int64_t floor_div(int64_t a, int64_t b) {
    int64_t q = a / b;
    if ((a % b != 0) && (a < 0)) --q;
    return q;
}

// round(num / den) with halves rounded up, den > 0
inline int64_t round_div(int64_t num, int64_t den) { return floor_div(2 * num + den, 2 * den); }

inline int32_t find(int32_t* parent, int32_t x) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];  // path halving
        x = parent[x];
    }
    return x;
}

// The smaller index becomes the root, so a root is the first gate of its region.
inline void unite(int32_t* parent, int32_t a, int32_t b) {
    a = find(parent, a);
    b = find(parent, b);
    if (a == b) return;
    if (a < b)
        parent[b] = a;
    else
        parent[a] = b;
}

struct Pair {
    uint64_t key;  // (low label << 32) | high label
    int64_t q;     // jump high - low, in units of 2 Vn / kScale
    bool operator<(const Pair& o) const { return key < o.key; }
};

struct Edge {
    int32_t a, b;  // a < b
    int64_t n;     // boundary gate pairs
    int64_t s;     // summed jump v_b - v_a, fixed point
};

int resolve_threads(int n_threads) {
    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    return std::max(1, n_threads > 0 ? n_threads : static_cast<int>(hw));
}

// Persistent worker threads for many short parallel passes (e.g. the
// sweep-by-sweep continuity chain), so threads are not created for each pass.
class Pool {
  public:
    explicit Pool(int n) {
        for (int i = 1; i < n; ++i) threads_.emplace_back([this, i] { loop(i); });
    }
    ~Pool() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : threads_) t.join();
    }
    int size() const { return static_cast<int>(threads_.size()) + 1; }
    // run f(tid) for tid in [0, nt) on the calling thread and nt - 1 workers
    void run(int nt, const std::function<void(int)>& f) {
        {
            std::lock_guard<std::mutex> lk(m_);
            job_ = &f;
            active_ = nt;
            remaining_ = nt - 1;
            ++generation_;
        }
        cv_.notify_all();
        f(0);
        std::unique_lock<std::mutex> lk(m_);
        done_.wait(lk, [&] { return remaining_ == 0; });
        job_ = nullptr;
    }

  private:
    void loop(int id) {
        int64_t seen = 0;
        std::unique_lock<std::mutex> lk(m_);
        for (;;) {
            cv_.wait(lk, [&] { return stop_ || generation_ != seen; });
            if (stop_) return;
            seen = generation_;
            if (id >= active_) continue;
            const std::function<void(int)>* f = job_;
            lk.unlock();
            (*f)(id);
            lk.lock();
            if (--remaining_ == 0) done_.notify_one();
        }
    }
    std::vector<std::thread> threads_;
    std::mutex m_;
    std::condition_variable cv_, done_;
    const std::function<void(int)>* job_ = nullptr;
    int64_t generation_ = 0;
    int active_ = 0, remaining_ = 0;
    bool stop_ = false;
};

// Pool of the calling thread, if any (worker threads never have one, so
// nested parallel passes start their own threads).
thread_local Pool* tl_pool = nullptr;

// Dynamic scheduling: workers take blocks [i0, i1) of [0, n) from an atomic counter.
template <class F>
void parallel_blocks(int64_t n, int64_t block, int nt, F&& body) {
    std::atomic<int64_t> next{0};
    auto worker = [&](int tid) {
        for (;;) {
            const int64_t i0 = next.fetch_add(block);
            if (i0 >= n) break;
            body(tid, i0, std::min(n, i0 + block));
        }
    };
    nt = static_cast<int>(std::min<int64_t>(nt, (n + block - 1) / block));
    if (nt <= 1) {
        worker(0);
        return;
    }
    if (tl_pool && nt <= tl_pool->size()) {
        tl_pool->run(nt, worker);
        return;
    }
    std::vector<std::thread> pool;
    for (int t = 1; t < nt; ++t) pool.emplace_back(worker, t);
    worker(0);
    for (auto& th : pool) th.join();
}

// Integer coordinate descent (Gauss-Seidel) on sum_e n_e (k_a - k_b - s_e / (n_e S))^2:
// each node in turn takes the integer fold that minimises the cost given its
// neighbours, largest nodes first, until no node changes.
void icm(int32_t nnode, const std::vector<Edge>& edges, const std::vector<int64_t>& size,
         std::vector<int64_t>& k, int max_iterations) {
    std::vector<int64_t> start(nnode + 1, 0);
    for (const Edge& E : edges) {
        ++start[E.a + 1];
        ++start[E.b + 1];
    }
    for (int32_t r = 0; r < nnode; ++r) start[r + 1] += start[r];
    std::vector<int32_t> nb(start[nnode]);
    std::vector<int64_t> nbn(start[nnode]), den(nnode, 0), ssum(nnode, 0);
    {
        std::vector<int64_t> fill(start.begin(), start.end() - 1);
        for (const Edge& E : edges) {
            // k_a ~ k_b + s / (n S);  k_b ~ k_a - s / (n S)
            int64_t i = fill[E.a]++;
            nb[i] = E.b;
            nbn[i] = E.n;
            i = fill[E.b]++;
            nb[i] = E.a;
            nbn[i] = E.n;
            den[E.a] += E.n;
            den[E.b] += E.n;
            ssum[E.a] += E.s;
            ssum[E.b] -= E.s;
        }
    }
    std::vector<int32_t> order;
    order.reserve(nnode);
    for (int32_t r = 0; r < nnode; ++r)
        if (start[r] < start[r + 1]) order.push_back(r);
    std::stable_sort(order.begin(), order.end(),
                     [&](int32_t x, int32_t y) { return size[x] > size[y]; });
    for (int it = 0; it < max_iterations; ++it) {
        int64_t changed = 0;
        for (int32_t r : order) {
            int64_t nk = 0;
            for (int64_t i = start[r]; i < start[r + 1]; ++i) nk += nbn[i] * k[nb[i]];
            const int64_t kn = round_div(nk * kScale + ssum[r], den[r] * kScale);
            if (kn != k[r]) {
                k[r] = kn;
                ++changed;
            }
        }
        if (!changed) break;
    }
}

struct SweepIn {
    const double* v;
    const uint8_t* link;
    int64_t nray, ngate;
    double nyquist;
};

// Steps 1-3 for one sweep: fold of every gate relative to its component's
// first region (k) and that component's id (comp, -1 for empty gates).
void solve_sweep(const SweepIn& in, double threshold, int64_t max_gap, int64_t max_ray_gap,
                 int max_iterations, int nt,
                 int32_t* kout,
                 int32_t* cout) {
    const int64_t nray = in.nray, ngate = in.ngate, N = nray * ngate;
    const double* v = in.v;
    const uint8_t* link = in.link;
    const double twovn = 2.0 * in.nyquist;
    const double thr = threshold * in.nyquist;
    auto joins = [&](int64_t g, int64_t h) {
        // NaN compares false, so empty gates never join
        return std::fabs(v[g] - v[h]) < thr;
    };

    // ---- 1. segmentation ------------------------------------------------------
    std::vector<int32_t> parent_vec(N);
    int32_t* parent = parent_vec.data();
    for (int64_t g = 0; g < N; ++g) parent[g] = static_cast<int32_t>(g);
    // blocks of rays are disjoint, so threads only touch their own nodes
    const int64_t ray_block = std::max<int64_t>(1, (nray + nt - 1) / nt);
    parallel_blocks(nray, ray_block, nt, [&](int, int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; ++r) {
            const int64_t base = r * ngate;
            for (int64_t j = 0; j + 1 < ngate; ++j)
                if (joins(base + j, base + j + 1))
                    unite(parent, int32_t(base + j), int32_t(base + j + 1));
            if (r + 1 < r1 && link[r]) {
                for (int64_t j = 0; j < ngate; ++j)
                    if (joins(base + j, base + ngate + j))
                        unite(parent, int32_t(base + j), int32_t(base + ngate + j));
            }
        }
    });
    // links across block boundaries and the 360 degree wrap
    for (int64_t r = 0; r < nray; ++r) {
        const int64_t r2 = (r + 1) % nray;
        const bool inner = (r + 1 < nray) && ((r + 1) % ray_block != 0);
        if (!link[r] || inner || r2 == r) continue;
        for (int64_t j = 0; j < ngate; ++j)
            if (joins(r * ngate + j, r2 * ngate + j))
                unite(parent, int32_t(r * ngate + j), int32_t(r2 * ngate + j));
    }

    // canonical labels: regions numbered by their first gate (raster order)
    int32_t* label = cout;  // reuse the output buffer
    int32_t nreg = 0;
    for (int64_t g = 0; g < N; ++g) {
        if (std::isnan(v[g])) {
            label[g] = -1;
            continue;
        }
        const int32_t root = find(parent, int32_t(g));
        label[g] = (root == g) ? nreg++ : label[root];
    }
    std::vector<int32_t>().swap(parent_vec);
    std::vector<int64_t> size(nreg, 0);
    for (int64_t g = 0; g < N; ++g)
        if (label[g] >= 0) ++size[label[g]];

    // ---- 2. region adjacency -----------------------------------------------------
    std::vector<std::vector<Pair>> local(nt);
    const int64_t pair_block = std::max<int64_t>(1, ray_block / 4);
    parallel_blocks(nray, pair_block, nt, [&](int tid, int64_t r0, int64_t r1) {
        auto& buf = local[tid];
        if (buf.capacity() == 0) buf.reserve(static_cast<size_t>((r1 - r0) * ngate / 4 + 64));
        auto add = [&](int64_t g, int64_t h) {
            const int32_t lg = label[g], lh = label[h];
            if (lg < 0 || lh < 0 || lg == lh) return;
            const double d = lg < lh ? v[h] - v[g] : v[g] - v[h];
            const int64_t q = static_cast<int64_t>(std::floor(d / twovn * double(kScale) + 0.5));
            const uint64_t lo = static_cast<uint64_t>(std::min(lg, lh));
            const uint64_t hi = static_cast<uint64_t>(std::max(lg, lh));
            buf.push_back({(lo << 32) | hi, q});
        };
        for (int64_t r = r0; r < r1; ++r) {
            const int64_t base = r * ngate;
            for (int64_t j = 0; j + 1 < ngate; ++j) add(base + j, base + j + 1);
            // bridge short gaps along the ray (e.g. range-folded gates)
            int64_t last = -1;
            for (int64_t j = 0; j < ngate; ++j) {
                if (label[base + j] < 0) continue;
                if (last >= 0 && j - last > 1 && j - last - 1 <= max_gap)
                    add(base + last, base + j);
                last = j;
            }
            const int64_t r2 = (r + 1) % nray;
            if (link[r] && r2 != r)
                for (int64_t j = 0; j < ngate; ++j) add(base + j, r2 * ngate + j);
            // bridge short gaps across rays (missing or censored rays)
            if (link[r] && nray > 2)
                for (int64_t j = 0; j < ngate; ++j) {
                    if (label[base + j] < 0) continue;
                    int64_t cur = (r + 1) % nray, d = 1;
                    while (d <= max_ray_gap && label[cur * ngate + j] < 0 && link[cur] &&
                           d + 1 < nray) {
                        cur = (cur + 1) % nray;
                        ++d;
                    }
                    if (d >= 2 && label[cur * ngate + j] >= 0) add(base + j, cur * ngate + j);
                }
        }
    });
    size_t npair = 0;
    for (auto& b : local) npair += b.size();
    std::vector<Pair> pairs;
    pairs.reserve(npair);
    for (auto& b : local) {
        pairs.insert(pairs.end(), b.begin(), b.end());
        std::vector<Pair>().swap(b);
    }
    std::sort(pairs.begin(), pairs.end());
    std::vector<Edge> edges;
    for (size_t i = 0; i < pairs.size();) {
        size_t k = i;
        int64_t s = 0;
        while (k < pairs.size() && pairs[k].key == pairs[i].key) s += pairs[k++].q;
        edges.push_back({int32_t(pairs[i].key >> 32), int32_t(pairs[i].key & 0xffffffffu),
                         int64_t(k - i), s});
        i = k;
    }
    std::vector<Pair>().swap(pairs);

    // ---- 3a. maximum spanning tree on boundary length ------------------------------
    std::vector<int32_t> order(edges.size());
    for (size_t e = 0; e < edges.size(); ++e) order[e] = int32_t(e);
    // confident boundaries first (mean jump within 0.3 of a whole number of
    // folds), then by length; ambiguous ones only join otherwise separate parts
    std::vector<uint8_t> ambiguous(edges.size());
    for (size_t e = 0; e < edges.size(); ++e) {
        const Edge& E = edges[e];
        const int64_t m = round_div(E.s, E.n * kScale);
        ambiguous[e] = 10 * std::llabs(E.s - m * E.n * kScale) > 3 * E.n * kScale;
    }
    std::stable_sort(order.begin(), order.end(), [&](int32_t x, int32_t y) {
        if (ambiguous[x] != ambiguous[y]) return ambiguous[x] < ambiguous[y];
        return edges[x].n > edges[y].n;
    });
    std::vector<int32_t> rp(nreg);
    std::vector<int64_t> pot(nreg, 0);  // fold relative to parent
    for (int32_t r = 0; r < nreg; ++r) rp[r] = r;
    auto rfind = [&](int32_t x) {
        int32_t root = x;
        int64_t acc = 0;
        while (rp[root] != root) {
            acc += pot[root];
            root = rp[root];
        }
        while (rp[x] != root) {  // point the path straight at the root
            const int32_t next = rp[x];
            const int64_t px = pot[x];
            pot[x] = acc;
            rp[x] = root;
            acc -= px;
            x = next;
        }
        return root;
    };
    for (int32_t e : order) {
        const Edge& E = edges[e];
        const int32_t ra = rfind(E.a), rb = rfind(E.b);
        if (ra == rb) continue;
        const int64_t m = round_div(E.s, E.n * kScale);  // k_a - k_b
        const int64_t pa = (E.a == ra) ? 0 : pot[E.a];
        const int64_t pb = (E.b == rb) ? 0 : pot[E.b];
        const int64_t d = m - pa + pb;  // k_ra - k_rb
        if (ra < rb) {
            rp[rb] = ra;
            pot[rb] = -d;
        } else {
            rp[ra] = rb;
            pot[ra] = d;
        }
    }
    std::vector<int64_t> k(nreg);
    std::vector<int32_t> comp(nreg);
    for (int32_t r = 0; r < nreg; ++r) {
        comp[r] = rfind(r);
        k[r] = (comp[r] == r) ? 0 : pot[r];
    }

    // ---- 3b. integer least squares: coordinate descent on regions and blocks -------
    icm(nreg, edges, size, k, max_iterations);
    // Single-region moves stop in a local minimum when a whole group of
    // regions is offset together. Groups of regions whose mutual boundaries
    // are all consistent are therefore moved as blocks, alternating with
    // region moves, until no block moves (the cost never increases).
    for (int outer = 0; outer < max_iterations; ++outer) {
        std::vector<int32_t> sp(nreg);
        for (int32_t r = 0; r < nreg; ++r) sp[r] = r;
        for (const Edge& E : edges)
            // joined by a boundary that agrees with the folds within 1/4 fold
            if (4 * std::llabs(E.s - (k[E.a] - k[E.b]) * E.n * kScale) <= E.n * kScale)
                unite(sp.data(), E.a, E.b);
        std::vector<int32_t> sid(nreg);
        int32_t nsup = 0;
        for (int32_t r = 0; r < nreg; ++r) {
            const int32_t root = find(sp.data(), r);
            sid[r] = (root == r) ? nsup++ : sid[root];
        }
        if (nsup == nreg) break;  // blocks are single regions: already optimal
        std::vector<int64_t> ssize(nsup, 0);
        for (int32_t r = 0; r < nreg; ++r) ssize[sid[r]] += size[r];
        {
            std::vector<std::pair<uint64_t, std::pair<int64_t, int64_t>>> items;
            for (const Edge& E : edges) {
                const int32_t A = sid[E.a], B = sid[E.b];
                if (A == B) continue;
                const int64_t t = E.n * (k[E.b] - k[E.a]) * kScale + E.s;  // block jump A -> B
                if (A < B)
                    items.push_back({(uint64_t(A) << 32) | uint64_t(B), {E.n, t}});
                else
                    items.push_back({(uint64_t(B) << 32) | uint64_t(A), {E.n, -t}});
            }
            std::sort(items.begin(), items.end(),
                      [](const auto& x, const auto& y) { return x.first < y.first; });
            std::vector<Edge> sedges;
            for (size_t i = 0; i < items.size();) {
                size_t j = i;
                int64_t n = 0, s = 0;
                while (j < items.size() && items[j].first == items[i].first) {
                    n += items[j].second.first;
                    s += items[j].second.second;
                    ++j;
                }
                sedges.push_back({int32_t(items[i].first >> 32),
                                  int32_t(items[i].first & 0xffffffffu), n, s});
                i = j;
            }
            std::vector<int64_t> delta(nsup, 0);
            icm(nsup, sedges, ssize, delta, max_iterations);
            bool moved = false;
            for (int32_t r = 0; r < nreg; ++r)
                if (delta[sid[r]] != 0) {
                    k[r] += delta[sid[r]];
                    moved = true;
                }
            if (!moved) break;
        }
        icm(nreg, edges, size, k, max_iterations);
    }

    for (int64_t g = 0; g < N; ++g) {
        const int32_t l = label[g];
        kout[g] = l < 0 ? 0 : static_cast<int32_t>(k[l]);
        cout[g] = l < 0 ? -1 : comp[l];
    }
}

// ---- step 4: absolute folds ----------------------------------------------------

constexpr double kVelQ = 256.0;     // velocities in fixed point of 1/256 m/s
constexpr double kTrigQ = 16384.0;  // sin/cos of azimuth in fixed point
constexpr int64_t kVadMinGates = 50;
constexpr double kVadMinSpread = 0.02;  // det of the normalised VAD matrix (full circle 0.25)

// x * q rounded; q is a power of two, so the product is exact (FMA or not)
inline int64_t quant(double x, double q) { return static_cast<int64_t>(std::floor(x * q + 0.5)); }

// a * b - c * d without fused multiply-add, so it matches NumPy bit for bit
inline double cross(double a, double b, double c, double d) {
    volatile double x = a * b;
    volatile double y = c * d;
    return x - y;
}

struct AbsIn {
    const double* v;
    const int32_t* k;
    const int32_t* comp;
    const double* ref;  // may be null
    const double* rsin;
    const double* rcos;
    const uint8_t* link;
    int64_t nray, ngate;
    double nyquist;
};

// Parallel pass over the rays of a sweep with one integer accumulator of
// ``width`` entries per thread, summed afterwards (integers: any order is exact).
template <class F>
std::vector<int64_t> reduce_rays(int64_t nray, int64_t width, int nt, int64_t init, F&& body,
                                 bool use_min = false, bool use_max = false) {
    const int64_t block = 16;
    nt = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(nt, (nray + block - 1) / block)));
    std::vector<std::vector<int64_t>> local(nt, std::vector<int64_t>(width, init));
    parallel_blocks(nray, block, nt, [&](int tid, int64_t r0, int64_t r1) {
        int64_t* acc = local[tid].data();
        for (int64_t r = r0; r < r1; ++r) body(acc, r);
    });
    std::vector<int64_t> total(std::move(local[0]));
    for (int t = 1; t < nt; ++t)
        for (int64_t i = 0; i < width; ++i) {
            if (use_min)
                total[i] = std::min(total[i], local[t][i]);
            else if (use_max)
                total[i] = std::max(total[i], local[t][i]);
            else
                total[i] += local[t][i];
        }
    return total;
}

// Most common vote per wanted component; ``refval(g)`` is the reference at a
// gate (NaN: no vote). Components that got votes are flagged in ``voted``.
template <class R>
void mode_votes(const AbsIn& in, int32_t ncomp, const std::vector<uint8_t>& want, R&& refval,
                std::vector<int64_t>& shift, std::vector<uint8_t>& voted, int nt) {
    const int64_t ngate = in.ngate;
    const double twovn = 2.0 * in.nyquist;
    auto vote = [&](int64_t g, double r) {
        double t = (r - in.v[g]) / twovn;
        t = t - double(in.k[g]);
        const double f = std::floor(t + 0.5);
        return static_cast<int64_t>(std::min(std::max(f, double(-kMaxVote)), double(kMaxVote)));
    };
    auto each = [&](int64_t r, auto&& fn) {
        for (int64_t g = r * ngate; g < (r + 1) * ngate; ++g) {
            const int32_t c = in.comp[g];
            if (c < 0 || !want[c]) continue;
            const double x = refval(g);
            if (std::isnan(x)) continue;
            fn(c, vote(g, x));
        }
    };
    const std::vector<int64_t> vmin = reduce_rays(
        in.nray, ncomp, nt, kMaxVote + 1,
        [&](int64_t* acc, int64_t r) {
            each(r, [&](int32_t c, int64_t w) { acc[c] = std::min(acc[c], w); });
        },
        true);
    const std::vector<int64_t> vmax = reduce_rays(
        in.nray, ncomp, nt, -kMaxVote - 1,
        [&](int64_t* acc, int64_t r) {
            each(r, [&](int32_t c, int64_t w) { acc[c] = std::max(acc[c], w); });
        },
        false, true);
    std::vector<int64_t> offset(ncomp + 1, 0);
    for (int32_t c = 0; c < ncomp; ++c)
        offset[c + 1] = offset[c] + (vmax[c] >= vmin[c] ? vmax[c] - vmin[c] + 1 : 0);
    const std::vector<int64_t> hist =
        reduce_rays(in.nray, offset[ncomp], nt, 0, [&](int64_t* acc, int64_t r) {
            each(r, [&](int32_t c, int64_t w) { ++acc[offset[c] + w - vmin[c]]; });
        });
    for (int32_t c = 0; c < ncomp; ++c) {
        if (!want[c] || vmax[c] < vmin[c]) continue;
        int64_t best = 0, best_count = -1;
        for (int64_t x = vmin[c]; x <= vmax[c]; ++x) {
            const int64_t cnt = hist[offset[c] + x - vmin[c]];
            if (cnt == 0) continue;
            // most gates, then smaller |vote|, then the negative one
            if (cnt > best_count || (cnt == best_count && std::llabs(x) < std::llabs(best))) {
                best = x;
                best_count = cnt;
            }
        }
        shift[c] = best;
        voted[c] = 1;
    }
}

// Shift that brings the mean velocity of each wanted component closest to zero.
void zero_mean(const AbsIn& in, int32_t ncomp, const std::vector<uint8_t>& want,
               std::vector<int64_t>& shift, int nt) {
    const int64_t ngate = in.ngate;
    const double twovn = 2.0 * in.nyquist;
    // per component: [sum, count]
    const std::vector<int64_t> acc =
        reduce_rays(in.nray, 2 * int64_t(ncomp), nt, 0, [&](int64_t* a, int64_t r) {
            for (int64_t g = r * ngate; g < (r + 1) * ngate; ++g) {
                const int32_t c = in.comp[g];
                if (c < 0 || !want[c]) continue;
                a[2 * c] += quant(in.v[g] / twovn, double(kScale)) + int64_t(in.k[g]) * kScale;
                ++a[2 * c + 1];
            }
        });
    for (int32_t c = 0; c < ncomp; ++c)
        if (want[c] && acc[2 * c + 1]) shift[c] = round_div(-acc[2 * c], acc[2 * c + 1] * kScale);
}

// VAD fit v = a0 + a1 sin(az) + a2 cos(az) (Browning and Wexler 1968) of the
// gates of component ``anchor`` with folds ``f``, per range gate over +-``window``
// gates. Sums are integers (exact, any order); range gates that cannot be
// fitted take the fit of the nearest fitted gate (ties: the nearer one to the
// radar). Returns false if no gate could be fitted.
bool vad_fit(const AbsIn& in, const int32_t* f, int32_t anchor, int64_t window,
             std::vector<double>& coef, int nt) {
    const int64_t ngate = in.ngate;
    const int64_t twovn_q = quant(2.0 * in.nyquist, kVelQ);
    // per range gate: n, s, c, ss, sc, cc, v, vs, vc
    std::vector<int64_t> acc =
        reduce_rays(in.nray, 9 * (ngate + 1), nt, 0, [&](int64_t* acc_, int64_t r) {
            const int64_t s = quant(in.rsin[r], kTrigQ), c = quant(in.rcos[r], kTrigQ);
            for (int64_t j = 0; j < ngate; ++j) {
                const int64_t g = r * ngate + j;
                if (in.comp[g] != anchor) continue;
                const int64_t u = quant(in.v[g], kVelQ) + int64_t(f[g]) * twovn_q;
                int64_t* a = acc_ + 9 * (j + 1);
                a[0] += 1;
                a[1] += s;
                a[2] += c;
                a[3] += s * s;
                a[4] += s * c;
                a[5] += c * c;
                a[6] += u;
                a[7] += u * s;
                a[8] += u * c;
            }
        });
    for (int64_t j = 1; j <= ngate; ++j)  // prefix sums over range
        for (int q = 0; q < 9; ++q) acc[9 * j + q] += acc[9 * (j - 1) + q];
    coef.assign(3 * ngate, 0.0);
    std::vector<int64_t> good;
    const double T = kTrigQ, V = kVelQ;
    for (int64_t j = 0; j < ngate; ++j) {
        const int64_t lo = std::max<int64_t>(0, j - window);
        const int64_t hi = std::min<int64_t>(ngate, j + window + 1);
        double m[9];
        for (int q = 0; q < 9; ++q) m[q] = double(acc[9 * hi + q] - acc[9 * lo + q]);
        const double n = m[0];
        if (n < double(kVadMinGates)) continue;
        // normal equations in physical units (scaling by powers of two is exact)
        const double s = m[1] / T, c = m[2] / T;
        const double ss = m[3] / (T * T), sc = m[4] / (T * T), cc = m[5] / (T * T);
        const double bv = m[6] / V, bs = m[7] / (V * T), bc = m[8] / (V * T);
        // Cramer's rule for [[n, s, c], [s, ss, sc], [c, sc, cc]] x = [bv, bs, bc]
        const double k1 = cross(ss, cc, sc, sc);
        const double k2 = cross(s, cc, sc, c);
        const double k3 = cross(s, sc, ss, c);
        const double det = cross(n, k1, s, k2) + [&] { volatile double x = c * k3; return x; }();
        const double n3 = n * n * n;
        if (!(det / n3 >= kVadMinSpread)) continue;
        const double e1 = cross(bs, cc, sc, bc);
        const double e2 = cross(bs, sc, ss, bc);
        const double det0 = cross(bv, k1, s, e1) + [&] { volatile double x = c * e2; return x; }();
        const double e3 = cross(s, bc, bs, c);
        const double det1 = cross(n, e1, bv, k2) + [&] { volatile double x = c * e3; return x; }();
        const double e4 = cross(ss, bc, bs, sc);
        const double det2 = cross(n, e4, s, e3) + [&] { volatile double x = bv * k3; return x; }();
        coef[3 * j] = det0 / det;
        coef[3 * j + 1] = det1 / det;
        coef[3 * j + 2] = det2 / det;
        good.push_back(j);
    }
    if (good.empty()) return false;
    size_t p = 0;  // first fitted gate >= j
    for (int64_t j = 0; j < ngate; ++j) {
        while (p < good.size() && good[p] < j) ++p;
        int64_t src;
        if (p == good.size())
            src = good.back();
        else if (p == 0 || good[p] == j)
            src = good[p];
        else
            src = (good[p] - j < j - good[p - 1]) ? good[p] : good[p - 1];
        for (int q = 0; q < 3; ++q) coef[3 * j + q] = coef[3 * src + q];
    }
    return true;
}

// Final gate check: a gate whose velocity differs by more than Vn from every
// valid neighbour (at least ``kMinNeighbours`` of its 8) is moved to the fold
// closest to the neighbours' mean. Jacobi passes over ray blocks in parallel.
constexpr int kMinNeighbours = 3;

// The ray itself and its linked neighbours (-1: none) for the gate check.
inline void neighbour_rays(const AbsIn& in, int64_t r, int64_t* rays) {
    const int64_t nray = in.nray;
    rays[0] = r;
    rays[1] = rays[2] = -1;
    if (nray < 2) return;
    const int64_t rp = (r + nray - 1) % nray, rn = (r + 1) % nray;
    if (in.link[rp] && rp != r) rays[1] = rp;
    if (in.link[r] && rn != r && rn != rays[1]) rays[2] = rn;
}

constexpr int64_t kNoGate = std::numeric_limits<int64_t>::min();

// Fold change of gate g (ray rays[0], gate j) from its unfolded neighbours
// ``uq``: nonzero only if all (>= kMinNeighbours) of them are more than Vn away.
inline int64_t gate_shift(const int64_t* uq, const int64_t* rays, int64_t ngate, int64_t j,
                          int64_t g, int64_t vn_q, int64_t twovn_q) {
    const int64_t u = uq[g];
    if (u == kNoGate) return 0;
    const int64_t j0 = std::max<int64_t>(0, j - 1);
    const int64_t j1 = std::min<int64_t>(ngate - 1, j + 1);
    int64_t cnt = 0, sum = 0;
    for (int q = 0; q < 3; ++q) {
        if (rays[q] < 0) continue;
        for (int64_t jj = j0; jj <= j1; ++jj) {
            const int64_t h = rays[q] * ngate + jj;
            const int64_t w = uq[h];
            if (h == g || w == kNoGate) continue;
            if (std::llabs(w - u) <= vn_q) return 0;  // a close neighbour: keep
            ++cnt;
            sum += w;
        }
    }
    if (cnt < kMinNeighbours) return 0;
    return round_div(sum - cnt * u, cnt * twovn_q);
}

int64_t gate_check(const AbsIn& in, std::vector<int32_t>& f, int passes, int nt) {
    const int64_t nray = in.nray, ngate = in.ngate, N = nray * ngate;
    const int64_t twovn_q = quant(2.0 * in.nyquist, kVelQ);
    const int64_t vn_q = quant(in.nyquist, kVelQ);
    constexpr int64_t kNone = kNoGate;
    std::vector<int64_t> vq(N), uq(N);
    std::vector<int32_t> next(N);
    const int64_t block = 16;
    parallel_blocks(nray, block, nt, [&](int, int64_t r0, int64_t r1) {
        for (int64_t g = r0 * ngate; g < r1 * ngate; ++g)
            vq[g] = in.comp[g] < 0 ? kNone : quant(in.v[g], kVelQ);
    });
    int64_t total = 0;
    for (int it = 0; it < passes; ++it) {
        parallel_blocks(nray, block, nt, [&](int, int64_t r0, int64_t r1) {
            for (int64_t g = r0 * ngate; g < r1 * ngate; ++g)
                uq[g] = vq[g] == kNone ? kNone : vq[g] + int64_t(f[g]) * twovn_q;
        });
        std::atomic<int64_t> changed{0};
        parallel_blocks(nray, block, nt, [&](int, int64_t r0, int64_t r1) {
            int64_t local = 0;
            for (int64_t r = r0; r < r1; ++r) {
                int64_t rays[3];
                neighbour_rays(in, r, rays);
                for (int64_t j = 0; j < ngate; ++j) {
                    const int64_t g = r * ngate + j;
                    const int64_t d = gate_shift(uq.data(), rays, ngate, j, g, vn_q, twovn_q);
                    next[g] = static_cast<int32_t>(f[g] + d);
                    local += d != 0;
                }
            }
            changed += local;
        });
        f.swap(next);
        total += changed.load();
        if (changed.load() == 0) break;
    }
    return total;
}

// Step 4 for one sweep.
// 1. The largest component (the anchor) is shifted to the reference (most
//    common gate vote) if it overlaps one, otherwise to zero mean velocity.
// 2. A VAD fit of the anchor gives an in-sweep reference; every other
//    component is shifted to the external reference where available, else
//    to the VAD (most common vote), else to zero mean.
// 3. Gate check against the neighbours.
void absolute_sweep(const AbsIn& in, int64_t vad_window, int gate_passes, int nt, int32_t* out) {
    const int64_t N = in.nray * in.ngate, ngate = in.ngate;
    const int32_t ncomp = static_cast<int32_t>(
        reduce_rays(
            in.nray, 1, nt, 0,
            [&](int64_t* a, int64_t r) {
                for (int64_t g = r * ngate; g < (r + 1) * ngate; ++g)
                    a[0] = std::max<int64_t>(a[0], in.comp[g] + 1);
            },
            false, true)[0]);
    const std::vector<int64_t> csize =
        reduce_rays(in.nray, ncomp, nt, 0, [&](int64_t* a, int64_t r) {
            for (int64_t g = r * ngate; g < (r + 1) * ngate; ++g)
                if (in.comp[g] >= 0) ++a[in.comp[g]];
        });
    std::vector<int64_t> shift(ncomp, 0);
    std::vector<int32_t> f(N, 0);
    auto fill_folds = [&](bool anchor_only, int32_t anchor) {
        parallel_blocks(in.nray, 16, nt, [&](int, int64_t r0, int64_t r1) {
            for (int64_t g = r0 * ngate; g < r1 * ngate; ++g) {
                const int32_t c = in.comp[g];
                if (c < 0 || (anchor_only && c != anchor)) continue;
                f[g] = static_cast<int32_t>(in.k[g] + shift[c]);
            }
        });
    };
    if (ncomp > 0) {
        int32_t anchor = 0;
        for (int32_t c = 1; c < ncomp; ++c)
            if (csize[c] > csize[anchor]) anchor = c;
        auto ext = [&](int64_t g) {
            return in.ref ? in.ref[g] : std::numeric_limits<double>::quiet_NaN();
        };
        std::vector<uint8_t> want(ncomp, 0), voted(ncomp, 0);
        want[anchor] = 1;
        mode_votes(in, ncomp, want, ext, shift, voted, nt);
        if (!voted[anchor]) zero_mean(in, ncomp, want, shift, nt);
        fill_folds(true, anchor);

        std::vector<double> coef;
        const bool has_vad = vad_fit(in, f.data(), anchor, vad_window, coef, nt);
        auto refval = [&](int64_t g) {
            const double e = ext(g);
            if (!std::isnan(e) || !has_vad) return e;
            const int64_t r = g / in.ngate, j = g % in.ngate;
            volatile double x = coef[3 * j + 1] * in.rsin[r];
            volatile double y = coef[3 * j + 2] * in.rcos[r];
            return (coef[3 * j] + x) + y;
        };
        std::fill(want.begin(), want.end(), 1);
        want[anchor] = 0;
        mode_votes(in, ncomp, want, refval, shift, voted, nt);
        for (int32_t c = 0; c < ncomp; ++c) want[c] = (c != anchor) && !voted[c];
        zero_mean(in, ncomp, want, shift, nt);
        fill_folds(false, anchor);
        gate_check(in, f, gate_passes, nt);
    }
    std::copy(f.begin(), f.end(), out);
}

void check_sweep(const DArray& v, const BArray& link, double nyquist) {
    if (v.ndim() != 2) throw std::invalid_argument("velocity must be 2-D (ray, gate)");
    if (link.size() != v.shape(0)) throw std::invalid_argument("ray_link needs one entry per ray");
    if (!(nyquist > 0)) throw std::invalid_argument("nyquist must be positive");
    if (v.size() >= (int64_t(1) << 31)) throw std::invalid_argument("sweep too large");
}

}  // namespace

// Steps 1-3 for every sweep; sweeps are processed in parallel.
std::vector<std::tuple<py::array_t<int32_t>, py::array_t<int32_t>>> region_folds(
    const std::vector<DArray>& velocity, const std::vector<BArray>& ray_link,
    const std::vector<double>& nyquist, double threshold, int64_t max_gap, int64_t max_ray_gap,
    int max_iterations,
    int n_threads) {
    const size_t ns = velocity.size();
    if (ray_link.size() != ns || nyquist.size() != ns)
        throw std::invalid_argument("need one ray_link and nyquist per sweep");
    std::vector<SweepIn> in(ns);
    std::vector<py::array_t<int32_t>> kout, cout;
    for (size_t i = 0; i < ns; ++i) {
        check_sweep(velocity[i], ray_link[i], nyquist[i]);
        in[i] = {velocity[i].data(), ray_link[i].data(), velocity[i].shape(0),
                 velocity[i].shape(1), nyquist[i]};
        kout.emplace_back(std::vector<py::ssize_t>{in[i].nray, in[i].ngate});
        cout.emplace_back(std::vector<py::ssize_t>{in[i].nray, in[i].ngate});
    }
    std::vector<int32_t*> kp(ns), cp(ns);
    for (size_t i = 0; i < ns; ++i) {
        kp[i] = kout[i].mutable_data();
        cp[i] = cout[i].mutable_data();
    }
    {
        py::gil_scoped_release release;
        const int nt = resolve_threads(n_threads);
        // threads beyond one per sweep work inside the sweeps
        const int outer = static_cast<int>(std::min<int64_t>(nt, int64_t(ns)));
        const int inner = std::max(1, nt / std::max(1, outer));
        // largest sweeps first, so the slowest ones do not start last
        std::vector<size_t> order(ns);
        for (size_t i = 0; i < ns; ++i) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return in[a].nray * in[a].ngate > in[b].nray * in[b].ngate;
        });
        parallel_blocks(int64_t(ns), 1, outer, [&](int, int64_t i0, int64_t) {
            const size_t i = order[i0];
            solve_sweep(in[i], threshold, max_gap, max_ray_gap, max_iterations, inner, kp[i], cp[i]);
        });
    }
    std::vector<std::tuple<py::array_t<int32_t>, py::array_t<int32_t>>> result;
    for (size_t i = 0; i < ns; ++i) result.emplace_back(kout[i], cout[i]);
    return result;
}

// Step 4 for every sweep; references may be None. Without ``previous`` the
// sweeps run in parallel. With ``previous[i] >= 0`` (always < i), sweep i uses
// the dealiased sweep ``previous[i]`` as its reference, at the gates given by
// ``prev_ray`` (per ray) and ``prev_gate`` (per gate; -1: none), filled from
// ``reference``; such chains run in order, each sweep with all threads.
std::vector<py::array_t<int32_t>> absolute_folds(
    const std::vector<DArray>& velocity, const std::vector<IArray>& folds,
    const std::vector<IArray>& components, const std::vector<double>& nyquist,
    const std::vector<py::object>& reference, const std::vector<DArray>& ray_sin,
    const std::vector<DArray>& ray_cos, const std::vector<BArray>& ray_link, int64_t vad_window,
    int gate_passes, const std::vector<int>& previous, const std::vector<IArray>& prev_ray,
    const std::vector<IArray>& prev_gate, int n_threads) {
    const size_t ns = velocity.size();
    if (folds.size() != ns || components.size() != ns || nyquist.size() != ns ||
        reference.size() != ns || ray_sin.size() != ns || ray_cos.size() != ns ||
        ray_link.size() != ns)
        throw std::invalid_argument("need one of each input per sweep");
    std::vector<DArray> refs(ns);
    std::vector<AbsIn> in(ns);
    std::vector<py::array_t<int32_t>> out;
    std::vector<int32_t*> op(ns);
    for (size_t i = 0; i < ns; ++i) {
        check_sweep(velocity[i], ray_link[i], nyquist[i]);
        const py::ssize_t n = velocity[i].size(), nray = velocity[i].shape(0);
        if (folds[i].size() != n || components[i].size() != n)
            throw std::invalid_argument("folds and components must match velocity");
        if (ray_sin[i].size() != nray || ray_cos[i].size() != nray)
            throw std::invalid_argument("ray_sin and ray_cos need one entry per ray");
        const double* rp = nullptr;
        if (!reference[i].is_none()) {
            refs[i] = reference[i].cast<DArray>();
            if (refs[i].size() != n)
                throw std::invalid_argument("reference must have the shape of velocity");
            rp = refs[i].data();
        }
        in[i] = {velocity[i].data(), folds[i].data(),   components[i].data(),
                 rp,                 ray_sin[i].data(), ray_cos[i].data(),
                 ray_link[i].data(), nray,              velocity[i].shape(1),
                 nyquist[i]};
        out.emplace_back(std::vector<py::ssize_t>{nray, velocity[i].shape(1)});
        op[i] = out[i].mutable_data();
    }
    const bool chain = !previous.empty();
    if (chain) {
        if (previous.size() != ns || prev_ray.size() != ns || prev_gate.size() != ns)
            throw std::invalid_argument("need previous, prev_ray and prev_gate per sweep");
        for (size_t i = 0; i < ns; ++i) {
            if (previous[i] >= int(i)) throw std::invalid_argument("previous[i] must be < i");
            if (previous[i] >= 0 && (prev_ray[i].size() != in[i].nray ||
                                     prev_gate[i].size() != in[i].ngate))
                throw std::invalid_argument("prev_ray/prev_gate do not match the sweep");
        }
    }
    {
        py::gil_scoped_release release;
        const int nt = resolve_threads(n_threads);
        if (chain) {
            Pool pool(nt);  // reused by every pass of every sweep in the chain
            struct Use {
                explicit Use(Pool* p) { tl_pool = p; }
                ~Use() { tl_pool = nullptr; }
            } use(&pool);
            std::vector<double> refbuf;
            for (size_t i = 0; i < ns; ++i) {
                AbsIn cur = in[i];
                if (previous[i] >= 0) {
                    const AbsIn& pv = in[previous[i]];
                    const int32_t* pf = op[previous[i]];
                    const int32_t* mr = prev_ray[i].data();
                    const int32_t* mg = prev_gate[i].data();
                    const double ptwovn = 2.0 * pv.nyquist;
                    refbuf.assign(cur.nray * cur.ngate, std::numeric_limits<double>::quiet_NaN());
                    for (int64_t r = 0; r < cur.nray; ++r) {
                        double* row = refbuf.data() + r * cur.ngate;
                        if (mr[r] >= 0)
                            for (int64_t j = 0; j < cur.ngate; ++j) {
                                if (mg[j] < 0) continue;
                                const int64_t h = int64_t(mr[r]) * pv.ngate + mg[j];
                                if (pv.comp[h] < 0) continue;
                                volatile double off = ptwovn * double(pf[h]);
                                row[j] = pv.v[h] + off;
                            }
                        if (cur.ref)
                            for (int64_t j = 0; j < cur.ngate; ++j)
                                if (std::isnan(row[j])) row[j] = cur.ref[r * cur.ngate + j];
                    }
                    cur.ref = refbuf.data();
                }
                absolute_sweep(cur, vad_window, gate_passes, nt, op[i]);
            }
        } else {
            const int outer = static_cast<int>(std::min<int64_t>(nt, std::max<int64_t>(1, ns)));
            const int inner = std::max(1, nt / outer);
            parallel_blocks(int64_t(ns), 1, outer, [&](int, int64_t i0, int64_t) {
                absolute_sweep(in[i0], vad_window, gate_passes, inner, op[i0]);
            });
        }
    }
    return out;
}

PYBIND11_MODULE(_dealias, m) {
    m.doc() = "Compiled region-based Doppler velocity dealiasing kernel for radarx.";
    m.def("region_folds", &region_folds, py::arg("velocity"), py::arg("ray_link"),
          py::arg("nyquist"), py::arg("threshold") = 0.3, py::arg("max_gap") = 20,
          py::arg("max_ray_gap") = 10,
          py::arg("max_iterations") = 100, py::arg("n_threads") = 0);
    m.def("absolute_folds", &absolute_folds, py::arg("velocity"), py::arg("folds"),
          py::arg("components"), py::arg("nyquist"), py::arg("reference"), py::arg("ray_sin"),
          py::arg("ray_cos"), py::arg("ray_link"), py::arg("vad_window") = 20,
          py::arg("gate_passes") = 2, py::arg("previous") = std::vector<int>(),
          py::arg("prev_ray") = std::vector<IArray>(), py::arg("prev_gate") = std::vector<IArray>(),
          py::arg("n_threads") = 0);
}
