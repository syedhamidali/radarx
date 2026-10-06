// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Fuzzy-logic hydrometeor classification kernel.
//
// Every gate is classified independently, so the rows (rays, or rows of a
// grid) of all sweeps of a volume form one pool of work that threads take in
// small blocks from an atomic counter. For each gate a single fused pass
// evaluates every membership function, aggregates the memberships of each
// class, applies the class restrictions and takes the argmax; nothing but
// the outputs is written to memory.
//
// Membership functions (one per class and variable, from a table built in
// radarx/retrieve/hid.py):
//   beta       1 / (1 + ((x - m) / a)^(2 b))         Dolan and Rutledge (2009)
//   trapezoid  0 below x1, 1 on [x2, x3], 0 above x4  Park et al. (2009)
// Trapezoid corners may be functions of the reflectivity (Park et al. 2009,
// their Eqs. 4 and 5). Variables: 0 Z, 1 ZDR, 2 KDP, 3 rhohv, 4 temperature.
//
// Aggregation:
//   additive (Park et al. 2009, Eq. 3; Thompson et al. 2014)
//       A = sum_j W_j Q_j P_j / sum_j W_j Q_j   over the available variables,
//       with the confidence vector Q of Park et al. (2009) if requested;
//   hybrid (Dolan et al. 2013, Eq. 8)
//       A = P_T P_Z (W_zdr P_zdr + W_kdp P_kdp + W_rho P_rho) / (sum of W).
// The winter mode (Thompson et al. 2014) adds a melting-layer detection step
// and a second pass, see classify().
//
// Every step follows the NumPy reference in radarx/retrieve/hid.py.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace py = pybind11;
using DArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using IArray = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;
using U8Array = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr int kNVar = 5;
constexpr int kMaxClass = 32;
constexpr int kZ = 0, kZdr = 1, kKdp = 2, kRho = 3, kT = 4;
enum Kind { kNone = 0, kBeta = 1, kTrap = 2 };
enum Mode { kAdditive = 0, kHybrid = 1, kWinter = 2 };
enum WinterState { kNoMelting = 0, kPartial = 1, kComplete = 2 };

// Reflectivity-dependent corners of the rain trapezoids, Park et al. (2009)
// Eqs. (4) and (5): 1-3 are f1-f3 (ZDR), 4-5 are g1-g2 (LKdp); 0 is zero.
constexpr int kNFunc = 6;
inline void zfuncs(double z, double* f) {
    f[0] = 0.0;
    f[1] = -0.50 + 2.50e-3 * z + 7.50e-4 * z * z;
    f[2] = 0.68 - 4.81e-2 * z + 2.92e-3 * z * z;
    f[3] = 1.42 + 6.67e-2 * z + 4.85e-4 * z * z;
    f[4] = -44.0 + 0.8 * z;
    f[5] = -22.0 + 0.5 * z;
}

// u^n for a small non-negative integer n (by squaring).
inline double ipow(double u, int n) {
    double r = 1.0;
    while (n) {
        if (n & 1) r *= u;
        u *= u;
        n >>= 1;
    }
    return r;
}

// One membership function of a class with its weight.
struct Term {
    int v = 0, kind = kNone;
    double p[4] = {0.0, 0.0, 0.0, 0.0};
    int sel[4] = {0, 0, 0, 0};
    double w = 0.0;
    int n = -1;  // beta slope b as an integer, -1 if not integral
};

struct Rule {
    int v = 0, op = 0, sel = 0;
    double thr = 0.0;
};

// Gate variables (NaN = missing), confidence vector and Park functions of Z.
struct Gate {
    double x[kNVar];
    double q[kNVar];
    double f[kNFunc];
};

struct Table {
    int nc = 0;
    Term term[kMaxClass][kNVar];
    int nterm[kMaxClass] = {};
    Rule rule[kMaxClass][8];
    int nrule[kMaxClass] = {};
    int group[kMaxClass] = {};
    bool has_zones = false;
    int64_t zone_allowed[5] = {0, 0, 0, 0, 0};
    int mode = kAdditive;
    bool kdp_log = false;
    bool quality = false;
    int ws = -1, ot = -1;  // winter: wet snow and "other" classes (group 0)
};

inline double membership(const Term& m, double x, const double* f) {
    if (m.kind == kBeta) {
        const double u = (x - m.p[0]) / m.p[1];
        const double u2 = u * u;
        return 1.0 / (1.0 + (m.n >= 0 ? ipow(u2, m.n) : std::pow(u2, m.p[2])));
    }
    const double x1 = m.p[0] + f[m.sel[0]], x2 = m.p[1] + f[m.sel[1]];
    const double x3 = m.p[2] + f[m.sel[2]], x4 = m.p[3] + f[m.sel[3]];
    if (x < x1 || x > x4) return 0.0;
    if (x < x2) return (x - x1) / (x2 - x1);
    if (x <= x3) return 1.0;
    return (x4 - x) / (x4 - x3);
}

// Aggregated score of class c.
inline double score(const Table& t, int c, const Gate& g) {
    const Term* terms = t.term[c];
    const int nt = t.nterm[c];
    if (t.mode == kHybrid) {
        double pz = 1.0, pt = 1.0, num = 0.0, den = 0.0;
        for (int k = 0; k < nt; ++k) {
            const Term& m = terms[k];
            const double x = g.x[m.v];
            if (std::isnan(x)) continue;
            const double pv = membership(m, x, g.f);
            if (m.v == kZ) pz = pv;
            else if (m.v == kT) pt = pv;
            else {
                num += m.w * pv;
                den += m.w;
            }
        }
        return pt * pz * (den > 0.0 ? num / den : 1.0);
    }
    double num = 0.0, den = 0.0;
    for (int k = 0; k < nt; ++k) {
        const Term& m = terms[k];
        const double x = g.x[m.v];
        if (std::isnan(x)) continue;
        const double wq = m.w * g.q[m.v];
        num += wq * membership(m, x, g.f);
        den += wq;
    }
    return den > 0.0 ? num / den : 0.0;
}

// Confidence vector of Park et al. (2009), Eqs. (14)-(17) and (23), from
// the attenuation (PHIDP), rhohv and partial beam blockage terms.
inline void confidence(double phidp, double rho, double block, double* q) {
    const double fphi = std::isnan(phidp) ? 0.0 : (phidp / 250.0) * (phidp / 250.0);
    const double fblk = std::isnan(block) ? 0.0 : (block / 50.0) * (block / 50.0);
    double chi = 0.0;
    if (!std::isnan(rho) && rho >= 0.8) chi = ((1.0 - rho) / 0.2) * ((1.0 - rho) / 0.2);
    q[kZ] = std::exp(-0.69 * (fphi + fblk));
    q[kZdr] = std::exp(-0.69 * (fphi + chi + fblk));
    q[kKdp] = std::exp(-0.69 * chi);
    q[kRho] = q[kKdp];
    q[kT] = 1.0;
}

// Position of the beam relative to the melting layer, Park et al. (2009),
// Fig. 2 and Eq. (24): 0 beam entirely below the bottom, 1 centre below the
// bottom, 2 centre in the layer, 3 centre above the top but lower edge below
// it, 4 beam entirely above the top.
inline int ml_zone(double h, double half, double bottom, double top) {
    if (h + half < bottom) return 0;
    if (h < bottom) return 1;
    if (h < top) return 2;
    if (h - half < top) return 3;
    return 4;
}

// Hard thresholds of Park et al. (2009), Table 3.
inline bool suppressed(const Table& t, int c, const Gate& g) {
    for (int k = 0; k < t.nrule[c]; ++k) {
        const Rule& r = t.rule[c][k];
        const double v = g.x[r.v];
        if (std::isnan(v)) continue;
        const double thr = r.thr + g.f[r.sel];
        if (r.op == 0 ? v > thr : v < thr) return true;
    }
    return false;
}

struct Block {
    int64_t nrow = 0, ncol = 0, first_row = 0;
    const double* var[kNVar] = {nullptr, nullptr, nullptr, nullptr, nullptr};
    const double* phidp = nullptr;
    const double* block = nullptr;
    const uint8_t* valid = nullptr;
    const double* height = nullptr;
    const double* rng = nullptr;  // per column [m], polar data only
    const double* ml_bottom = nullptr;  // per row
    const double* ml_top = nullptr;     // per row
    int8_t* cls = nullptr;
    float* conf = nullptr;
    float* scores = nullptr;  // (nc, nrow, ncol) or null
};

// Run fn(block, row, thread) over the rows of all blocks with atomic block
// scheduling.
template <class Fn>
void parallel_rows(const std::vector<Block>& blocks, int nt, Fn fn) {
    const int64_t total =
        blocks.empty() ? 0 : blocks.back().first_row + blocks.back().nrow;
    if (total == 0) return;
    const int64_t chunk = 4;
    std::atomic<int64_t> next{0};
    auto worker = [&](int tid) {
        size_t k = 0;
        for (;;) {
            const int64_t r0 = next.fetch_add(chunk);
            if (r0 >= total) break;
            const int64_t r1 = std::min(total, r0 + chunk);
            for (int64_t r = r0; r < r1; ++r) {
                while (r >= blocks[k].first_row + blocks[k].nrow) ++k;
                fn(blocks[k], r - blocks[k].first_row, tid);
            }
        }
    };
    std::vector<std::thread> pool;
    for (int t = 1; t < nt; ++t) pool.emplace_back(worker, t);
    worker(0);
    for (auto& th : pool) th.join();
}

const double* optional(const py::object& obj, int64_t n, std::vector<DArray>& keep,
                       const char* name) {
    if (obj.is_none()) return nullptr;
    keep.push_back(obj.cast<DArray>());
    if (keep.back().size() != n)
        throw std::invalid_argument(std::string(name) + " has the wrong size");
    return keep.back().data();
}

// Load the gate's variables (KDP as LKdp if requested). Returns false for
// gates that are not classified.
inline bool load(const Table& t, const Block& b, int64_t i, Gate& g) {
    if (b.valid && !b.valid[i]) return false;
    for (int v = 0; v < kNVar; ++v) g.x[v] = b.var[v] ? b.var[v][i] : kNaN;
    if (std::isnan(g.x[kZ])) return false;
    if (t.kdp_log && !std::isnan(g.x[kKdp]))
        g.x[kKdp] = g.x[kKdp] > 1e-3 ? 10.0 * std::log10(g.x[kKdp]) : -30.0;
    zfuncs(g.x[kZ], g.f);
    if (t.quality)
        confidence(b.phidp ? b.phidp[i] : kNaN, g.x[kRho], b.block ? b.block[i] : kNaN, g.q);
    else
        for (int v = 0; v < kNVar; ++v) g.q[v] = 1.0;
    return true;
}

}  // namespace

// Classify the gates of a list of blocks (sweeps or grids, each 2-D
// (row, column)) in one call. See radarx/retrieve/hid.py for the arguments.
py::tuple classify(
    const std::vector<DArray>& zh, const std::vector<py::object>& zdr,
    const std::vector<py::object>& kdp, const std::vector<py::object>& rho,
    const std::vector<py::object>& temp, const std::vector<py::object>& phidp,
    const std::vector<py::object>& blockage, const std::vector<py::object>& valid,
    const std::vector<py::object>& height, const std::vector<py::object>& rng,
    const std::vector<py::object>& ml_bottom, const std::vector<py::object>& ml_top,
    const IArray& kind, const DArray& par, const IArray& fsel, const DArray& weight,
    const IArray& group, const IArray& r_class, const IArray& r_var, const IArray& r_op,
    const IArray& r_fsel, const DArray& r_thr, const py::object& zone_allowed, int mode,
    bool kdp_log, bool quality, double sin_half_beam, int ws_class, int ot_class,
    int64_t ml_gates_partial, int64_t ml_gates_complete, double stats_rmin,
    double stats_rmax, double hist_lo, double hist_bin, int64_t hist_n, bool want_scores,
    int n_threads) {
    const size_t nb = zh.size();
    for (const auto* lst : {&zdr, &kdp, &rho, &temp, &phidp, &blockage, &valid, &height,
                            &rng, &ml_bottom, &ml_top})
        if (lst->size() != nb) throw std::invalid_argument("need one entry per block in every list");
    if (mode < kAdditive || mode > kWinter) throw std::invalid_argument("unknown mode");
    if (kind.ndim() != 2 || kind.shape(1) != kNVar)
        throw std::invalid_argument("kind must be (n_class, 5)");
    Table t;
    t.nc = static_cast<int>(kind.shape(0));
    if (t.nc < 1 || t.nc > kMaxClass) throw std::invalid_argument("bad number of classes");
    if (par.size() != t.nc * kNVar * 4 || fsel.size() != t.nc * kNVar * 4 ||
        weight.size() != t.nc * kNVar || group.size() != t.nc)
        throw std::invalid_argument("membership table has inconsistent shapes");
    t.mode = mode;
    for (int c = 0; c < t.nc; ++c) {
        t.group[c] = static_cast<int>(group.data()[c]);
        for (int v = 0; v < kNVar; ++v) {
            const int64_t k = kind.data()[c * kNVar + v];
            const double w = weight.data()[c * kNVar + v];
            if (k < kNone || k > kTrap) throw std::invalid_argument("unknown membership kind");
            if (k == kNone || (mode != kHybrid && !(w > 0.0))) continue;
            Term& m = t.term[c][t.nterm[c]++];
            m.v = v;
            m.kind = static_cast<int>(k);
            m.w = w;
            for (int j = 0; j < 4; ++j) {
                m.p[j] = par.data()[(c * kNVar + v) * 4 + j];
                const int64_t sel = fsel.data()[(c * kNVar + v) * 4 + j];
                if (sel < 0 || sel >= kNFunc) throw std::invalid_argument("bad function selector");
                m.sel[j] = static_cast<int>(sel);
            }
            if (k == kBeta) {
                if (!(m.p[1] != 0.0)) throw std::invalid_argument("beta half-width must not be 0");
                const double b = m.p[2];
                if (b >= 0.0 && b <= 64.0 && b == std::floor(b)) m.n = static_cast<int>(b);
            }
        }
    }
    const py::ssize_t nr = r_class.size();
    if (r_var.size() != nr || r_op.size() != nr || r_fsel.size() != nr || r_thr.size() != nr)
        throw std::invalid_argument("rule arrays have inconsistent sizes");
    for (py::ssize_t r = 0; r < nr; ++r) {
        const int64_t c = r_class.data()[r], v = r_var.data()[r], sel = r_fsel.data()[r];
        if (c < 0 || c >= t.nc || v < 0 || v >= kNVar || sel < 0 || sel >= kNFunc ||
            t.nrule[c] >= 8)
            throw std::invalid_argument("bad rule");
        Rule& rule = t.rule[c][t.nrule[c]++];
        rule.v = static_cast<int>(v);
        rule.op = static_cast<int>(r_op.data()[r]);
        rule.sel = static_cast<int>(sel);
        rule.thr = r_thr.data()[r];
    }
    if (!zone_allowed.is_none()) {
        IArray za = zone_allowed.cast<IArray>();
        if (za.size() != 5) throw std::invalid_argument("zone_allowed needs 5 entries");
        t.has_zones = true;
        for (int k = 0; k < 5; ++k) t.zone_allowed[k] = za.data()[k];
    }
    t.kdp_log = kdp_log;
    t.quality = quality;
    if (mode == kWinter) {
        if (ws_class < 0 || ws_class >= t.nc || ot_class < 0 || ot_class >= t.nc)
            throw std::invalid_argument("winter mode needs the wet snow and other classes");
        if (!(hist_bin > 0) || hist_n < 1) throw std::invalid_argument("bad histogram");
        t.ws = ws_class;
        t.ot = ot_class;
    }

    std::vector<DArray> keep;
    std::vector<U8Array> keep_u8;
    keep.reserve(nb * 12);
    std::vector<Block> blocks(nb);
    std::vector<py::array_t<int8_t>> cls_out;
    std::vector<py::array_t<float>> conf_out;
    std::vector<py::object> score_out;
    int64_t first = 0;
    for (size_t i = 0; i < nb; ++i) {
        if (zh[i].ndim() != 2) throw std::invalid_argument("fields must be 2-D (row, column)");
        Block& b = blocks[i];
        b.nrow = zh[i].shape(0);
        b.ncol = zh[i].shape(1);
        const int64_t n = b.nrow * b.ncol;
        b.first_row = first;
        first += b.nrow;
        b.var[kZ] = zh[i].data();
        b.var[kZdr] = optional(zdr[i], n, keep, "zdr");
        b.var[kKdp] = optional(kdp[i], n, keep, "kdp");
        b.var[kRho] = optional(rho[i], n, keep, "rhohv");
        b.var[kT] = optional(temp[i], n, keep, "temperature");
        b.phidp = optional(phidp[i], n, keep, "phidp");
        b.block = optional(blockage[i], n, keep, "blockage");
        b.height = optional(height[i], n, keep, "height");
        b.rng = optional(rng[i], b.ncol, keep, "range");
        b.ml_bottom = optional(ml_bottom[i], b.nrow, keep, "ml_bottom");
        b.ml_top = optional(ml_top[i], b.nrow, keep, "ml_top");
        if (!valid[i].is_none()) {
            keep_u8.push_back(valid[i].cast<U8Array>());
            if (keep_u8.back().size() != n) throw std::invalid_argument("mask has the wrong size");
            b.valid = keep_u8.back().data();
        }
        if (mode == kWinter && !b.height)
            throw std::invalid_argument("the winter classification needs gate heights");
        cls_out.emplace_back(std::vector<py::ssize_t>{b.nrow, b.ncol});
        conf_out.emplace_back(std::vector<py::ssize_t>{b.nrow, b.ncol});
        b.cls = cls_out.back().mutable_data();
        b.conf = conf_out.back().mutable_data();
        if (want_scores) {
            py::array_t<float> s(std::vector<py::ssize_t>{t.nc, b.nrow, b.ncol});
            b.scores = s.mutable_data();
            score_out.push_back(s);
        } else {
            score_out.push_back(py::none());
        }
    }

    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    int nt = n_threads > 0 ? n_threads : static_cast<int>(hw);
    nt = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(nt, std::max<int64_t>(first, 1))));

    int64_t n_ws = 0;
    double ml_height = kNaN;
    int state = kNoMelting;
    {
        py::gil_scoped_release release;

        if (mode == kWinter) {
            // pass 1: melting-layer detection (wet snow vs other) and a
            // histogram of the heights of wet snow gates within the range
            // interval used for the melting-layer statistics.
            std::vector<int64_t> hist(static_cast<size_t>(nt) * hist_n, 0);
            parallel_rows(blocks, nt, [&](const Block& b, int64_t r, int tid) {
                int64_t* h = hist.data() + static_cast<size_t>(tid) * hist_n;
                Gate gate;
                for (int64_t g = 0; g < b.ncol; ++g) {
                    const int64_t i = r * b.ncol + g;
                    b.cls[i] = 0;
                    if (!load(t, b, i, gate)) continue;
                    const bool ws = score(t, t.ws, gate) > score(t, t.ot, gate);
                    b.cls[i] = ws ? 1 : 0;
                    if (!ws || std::isnan(b.height[i])) continue;
                    if (b.rng && (b.rng[g] < stats_rmin || b.rng[g] > stats_rmax)) continue;
                    int64_t k = static_cast<int64_t>(std::floor((b.height[i] - hist_lo) / hist_bin));
                    k = std::min(std::max<int64_t>(k, 0), hist_n - 1);
                    ++h[k];
                }
            });
            std::vector<int64_t> total(hist_n, 0);
            for (int tid = 0; tid < nt; ++tid)
                for (int64_t k = 0; k < hist_n; ++k) total[k] += hist[tid * hist_n + k];
            for (int64_t k = 0; k < hist_n; ++k) n_ws += total[k];
            if (n_ws >= ml_gates_complete) state = kComplete;
            else if (n_ws >= ml_gates_partial) state = kPartial;
            if (n_ws > 0) {
                // median: the bin holding the element of rank (n - 1) / 2
                const int64_t rank = (n_ws - 1) / 2;
                int64_t cum = 0;
                for (int64_t k = 0; k < hist_n; ++k) {
                    cum += total[k];
                    if (cum > rank) {
                        ml_height = hist_lo + (static_cast<double>(k) + 0.5) * hist_bin;
                        break;
                    }
                }
            }
        }

        parallel_rows(blocks, nt, [&](const Block& b, int64_t r, int) {
            Gate gate;
            double s[kMaxClass];
            const double bottom = b.ml_bottom ? b.ml_bottom[r] : kNaN;
            const double top = b.ml_top ? b.ml_top[r] : kNaN;
            const bool zones = t.has_zones && !std::isnan(bottom) && !std::isnan(top);
            const int64_t plane = b.nrow * b.ncol;
            for (int64_t g = 0; g < b.ncol; ++g) {
                const int64_t i = r * b.ncol + g;
                const bool ws_flag = mode == kWinter && b.cls[i] == 1;
                if (!load(t, b, i, gate)) {
                    b.cls[i] = 0;
                    b.conf[i] = std::numeric_limits<float>::quiet_NaN();
                    if (b.scores)
                        for (int c = 0; c < t.nc; ++c)
                            b.scores[c * plane + i] = std::numeric_limits<float>::quiet_NaN();
                    continue;
                }
                for (int c = 0; c < t.nc; ++c) {
                    s[c] = score(t, c, gate);
                    if (b.scores) b.scores[c * plane + i] = static_cast<float>(s[c]);
                }
                int64_t allowed = ~int64_t(0);
                if (mode == kWinter) {
                    // Thompson et al. (2014): wet snow from the detection
                    // step, below-ML classes under the median melting-layer
                    // height, above-ML classes elsewhere.
                    int want = 2;
                    if (state == kComplete && !std::isnan(b.height[i]) && b.height[i] < ml_height)
                        want = 1;
                    allowed = 0;
                    if (ws_flag && state != kNoMelting) {
                        allowed = int64_t(1) << t.ws;
                    } else {
                        for (int c = 0; c < t.nc; ++c)
                            if (t.group[c] == want) allowed |= int64_t(1) << c;
                    }
                } else if (zones && b.height && !std::isnan(b.height[i])) {
                    const double half = b.rng ? b.rng[g] * sin_half_beam : 0.0;
                    allowed = t.zone_allowed[ml_zone(b.height[i], half, bottom, top)];
                }
                int best = -1;
                double best_s = -1.0;
                for (int c = 0; c < t.nc; ++c) {
                    if (!((allowed >> c) & 1)) continue;
                    if (t.nrule[c] && suppressed(t, c, gate)) continue;
                    if (s[c] > best_s) {
                        best_s = s[c];
                        best = c;
                    }
                }
                b.cls[i] = static_cast<int8_t>(best + 1);
                b.conf[i] = best >= 0 ? static_cast<float>(best_s)
                                      : std::numeric_limits<float>::quiet_NaN();
            }
        });
    }
    py::list cls_list, conf_list, score_list;
    for (size_t i = 0; i < nb; ++i) {
        cls_list.append(cls_out[i]);
        conf_list.append(conf_out[i]);
        score_list.append(score_out[i]);
    }
    py::dict info;
    info["n_wet_snow"] = n_ws;
    info["melting_layer_height"] = ml_height;
    info["melting"] = state;
    return py::make_tuple(cls_list, conf_list, score_list, info);
}

PYBIND11_MODULE(_hid, m) {
    m.doc() = "Compiled fuzzy-logic hydrometeor classification kernel for radarx.";
    m.def("classify", &classify, py::arg("zh"), py::arg("zdr"), py::arg("kdp"),
          py::arg("rhohv"), py::arg("temperature"), py::arg("phidp"), py::arg("blockage"),
          py::arg("valid"), py::arg("height"), py::arg("range"), py::arg("ml_bottom"),
          py::arg("ml_top"), py::arg("kind"), py::arg("par"), py::arg("fsel"),
          py::arg("weight"), py::arg("group"), py::arg("r_class"), py::arg("r_var"),
          py::arg("r_op"), py::arg("r_fsel"), py::arg("r_thr"), py::arg("zone_allowed"),
          py::arg("mode"), py::arg("kdp_log"), py::arg("quality"),
          py::arg("sin_half_beam"), py::arg("ws_class"), py::arg("ot_class"),
          py::arg("ml_gates_partial"), py::arg("ml_gates_complete"),
          py::arg("stats_rmin"), py::arg("stats_rmax"), py::arg("hist_lo"),
          py::arg("hist_bin"), py::arg("hist_n"), py::arg("want_scores") = true,
          py::arg("n_threads") = 0);
}
