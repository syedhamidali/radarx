// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Echo classifier of Gourley, Tabary and Parent du Chatelet (2007, J. Atmos.
// Oceanic Technol. 24, 1439-1451), method "gourley2007" of
// radarx.retrieve.echo_mask. The NumPy reference is _gourley_numpy.py (same
// function, same order of operations).
//
//   1. per gate: texture of ZDR and PhiDP as the root-mean-square difference to
//      the 3 x 3 neighbouring gates (Eq. 1), the membership of rhohv and of
//      both textures in the three classes (tables of Fig. 3), the aggregation
//      value of every class (Eqs. 3 and 4), the suppression thresholds of
//      Table 1 and the class with the largest aggregation value;
//   2. per gate: despeckling over the 8 neighbours (p. 1446).
//
// The rays of all sweeps form one pool of work that threads take in small
// blocks from an atomic counter.

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
using UArray = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;

namespace {

constexpr float kNaNf = std::numeric_limits<float>::quiet_NaN();

// internal type map
enum Type : int8_t { kNone = 0, kPrecip = 1, kClutter = 2, kClear = 3, kUnclassified = 4 };
// output classes
enum Class : int8_t { kNoEcho = 0, kMeteo = 1, kNonMeteo = 2, kSpeckle = 3 };

struct Table {
    std::vector<double> x;
    std::vector<double> y[3];  // precipitation, ground clutter, clear air
};

struct Params {
    double floor_z, floor_zdr, floor_rho, floor_phi, floor_vel;
    double snr_min;
    Table tab[3];  // rhohv, texture of ZDR, texture of PhiDP
    double w[3];
    double rho_min, phi_min, z_precip_min, phitex_max, sig_max, vel_max, z_clear_max;
    double slope_zdr, slope_phi, start_km;
    bool despeckle;
};

struct Sweep {
    const double* z = nullptr;
    const double* zdr = nullptr;
    const double* rho = nullptr;
    const double* phi = nullptr;
    const double* vel = nullptr;
    const double* sig = nullptr;
    const double* snr = nullptr;
    const double* rng = nullptr;  // gate range [km]
    const uint8_t* links = nullptr;
    float* score = nullptr;
    int8_t* cls = nullptr;
    int8_t* typ = nullptr;
    int64_t nray = 0, ngate = 0, first_ray = 0;
};

template <class Fn>
void parallel_rays(const std::vector<Sweep>& sweeps, int nt, Fn fn) {
    const int64_t total = sweeps.empty() ? 0 : sweeps.back().first_ray + sweeps.back().nray;
    if (total == 0) return;
    nt = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(nt, total)));
    const int64_t block = 4;
    std::atomic<int64_t> next{0};
    auto worker = [&]() {
        size_t k = 0;
        for (;;) {
            const int64_t r0 = next.fetch_add(block);
            if (r0 >= total) break;
            const int64_t r1 = std::min(total, r0 + block);
            for (int64_t r = r0; r < r1; ++r) {
                while (r >= sweeps[k].first_ray + sweeps[k].nray) ++k;
                fn(sweeps[k], r - sweeps[k].first_ray);
            }
        }
    };
    std::vector<std::thread> pool;
    for (int t = 1; t < nt; ++t) pool.emplace_back(worker);
    worker();
    for (auto& th : pool) th.join();
}

inline double wrap180(double d) { return d - 360.0 * std::floor(d / 360.0 + 0.5); }

// Previous and next linked ray (-1 if none), as _neighbour_rays in qc.py.
inline void neighbours(const Sweep& sw, int64_t r, int64_t& prv, int64_t& nxt) {
    const int64_t n = sw.nray;
    nxt = sw.links[r] ? (r + 1) % n : -1;
    if (nxt == r) nxt = -1;
    const int64_t q = (r - 1 + n) % n;
    prv = sw.links[q] ? q : -1;
    if (prv == r || prv == nxt) prv = -1;
}

inline bool is_echo(const Params& P, const Sweep& sw, int64_t i) {
    bool e = std::isfinite(sw.z[i]) && sw.z[i] > P.floor_z;
    if (sw.snr != nullptr) e = e && sw.snr[i] >= P.snr_min;
    return e;
}

inline bool has_data(const Params& P, const Sweep& sw, const double* x, double floor, int64_t i) {
    return x != nullptr && is_echo(P, sw, i) && std::isfinite(x[i]) && x[i] > floor;
}

// Linear interpolation of a table, clamped at its ends (numpy.interp).
inline double lookup(const std::vector<double>& xs, const std::vector<double>& ys, double x) {
    const size_t n = xs.size();
    if (x < xs[0]) return ys[0];
    if (x > xs[n - 1]) return ys[n - 1];
    if (x == xs[n - 1]) return ys[n - 1];
    const size_t j = static_cast<size_t>(std::upper_bound(xs.begin(), xs.end(), x) - xs.begin()) - 1;
    const double slope = (ys[j + 1] - ys[j]) / (xs[j + 1] - xs[j]);
    return slope * (x - xs[j]) + ys[j];
}

// Root-mean-square difference of the gate to its 3 x 3 neighbours (Eq. 1).
// NaN with fewer than three neighbours that have data.
double texture(const Params& P, const Sweep& sw, const double* x, double floor, bool wrap,
               int64_t r, int64_t g, int64_t prv, int64_t nxt) {
    const int64_t ng = sw.ngate;
    const int64_t centre = r * ng + g;
    const int64_t rays[3] = {prv, r, nxt};
    double total = 0.0, count = 0.0;
    for (int64_t q : rays) {
        if (q < 0) continue;
        for (int64_t dg = -1; dg <= 1; ++dg) {
            if (q == r && dg == 0) continue;
            const int64_t j = g + dg;
            if (j < 0 || j >= ng) continue;
            const int64_t i = q * ng + j;
            if (!has_data(P, sw, x, floor, i)) continue;
            double d = x[centre] - x[i];
            if (wrap) d = wrap180(d);
            total = total + d * d;
            count = count + 1.0;
        }
    }
    if (count < 3.0) return std::numeric_limits<double>::quiet_NaN();
    return std::sqrt(total / (count + 1.0));
}

inline double corrected(double t, double rng_km, double slope, double start_km) {
    if (slope == 0.0 || !std::isfinite(t)) return t;
    const double extra = slope * std::max(rng_km - start_km, 0.0);
    return std::max(t - extra, 0.0);
}

// Step 1: type of every gate of one ray and the share of the precipitation class.
void classify_ray(const Params& P, const Sweep& sw, int64_t r) {
    const int64_t ng = sw.ngate, off = r * ng;
    int64_t prv, nxt;
    neighbours(sw, r, prv, nxt);
    for (int64_t g = 0; g < ng; ++g) {
        const int64_t i = off + g;
        if (!is_echo(P, sw, i)) {
            sw.score[i] = kNaNf;
            sw.typ[i] = kNone;
            continue;
        }
        double num[3] = {0.0, 0.0, 0.0}, den = 0.0;
        bool sup_p = false, sup_g = false;
        auto add = [&](int k, double v) {
            for (int c = 0; c < 3; ++c)
                num[c] = num[c] + P.w[k] * lookup(P.tab[k].x, P.tab[k].y[c], v);
            den = den + P.w[k];
        };
        if (has_data(P, sw, sw.rho, P.floor_rho, i)) {
            double v = sw.rho[i];
            if (v > 1.0) v = 2.0 - v;  // values above 1 only occur at low SNR
            add(0, v);
            if (v < P.rho_min) sup_p = true;
        }
        if (has_data(P, sw, sw.zdr, P.floor_zdr, i)) {
            const double t = corrected(
                texture(P, sw, sw.zdr, P.floor_zdr, false, r, g, prv, nxt), sw.rng[g],
                P.slope_zdr, P.start_km);
            if (std::isfinite(t)) add(1, t);
        }
        if (has_data(P, sw, sw.phi, P.floor_phi, i)) {
            const double t = corrected(
                texture(P, sw, sw.phi, P.floor_phi, true, r, g, prv, nxt), sw.rng[g],
                P.slope_phi, P.start_km);
            if (std::isfinite(t)) {
                add(2, t);
                if (t > P.phitex_max) sup_p = true;
            }
            if (sw.phi[i] < P.phi_min) sup_p = true;
        }
        const double z = sw.z[i];
        if (z < P.z_precip_min) sup_p = true;
        if (has_data(P, sw, sw.sig, -std::numeric_limits<double>::infinity(), i) &&
            sw.sig[i] > P.sig_max)
            sup_g = true;
        if (has_data(P, sw, sw.vel, P.floor_vel, i) && std::fabs(sw.vel[i]) > P.vel_max)
            sup_g = true;
        const bool sup_c = z > P.z_clear_max;
        if (!(den > 0.0)) {
            sw.score[i] = kNaNf;
            sw.typ[i] = kUnclassified;
            continue;
        }
        double q[3];
        for (int c = 0; c < 3; ++c) q[c] = num[c] / den;
        if (sup_p) q[0] = 0.0;
        if (sup_g) q[1] = 0.0;
        if (sup_c) q[2] = 0.0;
        const double total = q[0] + q[1] + q[2];
        sw.score[i] = static_cast<float>(total > 0.0 ? q[0] / total : 0.0);
        if (q[0] > std::max(q[1], q[2]))
            sw.typ[i] = kPrecip;
        else
            sw.typ[i] = q[1] > q[2] ? kClutter : kClear;
    }
}

// Step 2: despeckling over the 8 neighbours and the output classes of one ray.
void finish_ray(const Params& P, const Sweep& sw, int64_t r) {
    const int64_t ng = sw.ngate, off = r * ng;
    int64_t prv, nxt;
    neighbours(sw, r, prv, nxt);
    const int64_t rays[3] = {prv, r, nxt};
    for (int64_t g = 0; g < ng; ++g) {
        const int64_t i = off + g;
        int8_t t = sw.typ[i];
        bool isolated = false;
        if (P.despeckle && (t == kPrecip || t == kClear)) {
            int count = 0;
            for (int64_t q : rays) {
                if (q < 0) continue;
                for (int64_t dg = -1; dg <= 1; ++dg) {
                    if (q == r && dg == 0) continue;
                    const int64_t j = g + dg;
                    if (j < 0 || j >= ng) continue;
                    if (sw.typ[q * ng + j] == kPrecip) ++count;
                }
            }
            if (t == kPrecip && count < 3) {
                t = kClear;
                isolated = true;
            } else if (t == kClear && count > 6) {
                t = kPrecip;
            }
        }
        int8_t c = kNoEcho;
        if (t == kUnclassified) c = kSpeckle;
        else if (t == kClutter || t == kClear) c = kNonMeteo;
        else if (t == kPrecip) c = kMeteo;
        if (isolated) c = kSpeckle;
        sw.cls[i] = c;
    }
}

const double* optional_data(const py::object& obj, const DArray& ref, DArray& keep,
                            const char* name) {
    if (obj.is_none()) return nullptr;
    keep = obj.cast<DArray>();
    if (keep.ndim() != 2 || keep.shape(0) != ref.shape(0) || keep.shape(1) != ref.shape(1))
        throw std::invalid_argument(std::string(name) + " must have the shape of dbzh");
    return keep.data();
}

}  // namespace

// Classify the sweeps of a volume in one call. Per sweep: reflectivity (ray,
// gate), optional zdr, rhohv, phidp, radial velocity, pulse-to-pulse
// reflectivity variability and snr of the same shape (or None), the ray links
// and the gate ranges [km]. floors: no-data floors of dbzh, zdr, rhohv, phidp
// and velocity. mem_x[k], mem_y[k] (3 x n): membership tables of rhohv, texture
// of ZDR and texture of PhiDP. rules: the seven thresholds of Table 1. slopes:
// range correction of the textures of ZDR and PhiDP per km and its start [km].
py::tuple classify(const std::vector<DArray>& dbz, const std::vector<py::object>& zdr,
                   const std::vector<py::object>& rho, const std::vector<py::object>& phi,
                   const std::vector<py::object>& vel, const std::vector<py::object>& sig,
                   const std::vector<py::object>& snr, const std::vector<UArray>& links,
                   const std::vector<DArray>& rng_km, const DArray& floors, double snr_min,
                   const std::vector<DArray>& mem_x, const std::vector<DArray>& mem_y,
                   const DArray& weights, const DArray& rules, const DArray& slopes,
                   bool despeckle, int n_threads) {
    const size_t ns = dbz.size();
    if (zdr.size() != ns || rho.size() != ns || phi.size() != ns || vel.size() != ns ||
        sig.size() != ns || snr.size() != ns || links.size() != ns || rng_km.size() != ns)
        throw std::invalid_argument("need one entry per sweep in every list");
    if (floors.size() != 5) throw std::invalid_argument("need five no-data floors");
    if (mem_x.size() != 3 || mem_y.size() != 3)
        throw std::invalid_argument("need three membership tables");
    if (weights.size() != 3) throw std::invalid_argument("need three weights");
    if (rules.size() != 7) throw std::invalid_argument("need seven thresholds");
    if (slopes.size() != 3) throw std::invalid_argument("need three range-correction values");
    Params P{};
    const double* fl = floors.data();
    P.floor_z = fl[0];
    P.floor_zdr = fl[1];
    P.floor_rho = fl[2];
    P.floor_phi = fl[3];
    P.floor_vel = fl[4];
    P.snr_min = snr_min;
    for (int k = 0; k < 3; ++k) {
        if (mem_x[k].ndim() != 1 || mem_x[k].shape(0) < 2 || mem_y[k].ndim() != 2 ||
            mem_y[k].shape(0) != 3 || mem_y[k].shape(1) != mem_x[k].shape(0))
            throw std::invalid_argument("a membership table needs x (n) and y (3, n), n >= 2");
        const int64_t n = mem_x[k].shape(0);
        P.tab[k].x.assign(mem_x[k].data(), mem_x[k].data() + n);
        for (int c = 0; c < 3; ++c)
            P.tab[k].y[c].assign(mem_y[k].data() + c * n, mem_y[k].data() + (c + 1) * n);
        P.w[k] = weights.data()[k];
    }
    const double* ru = rules.data();
    P.rho_min = ru[0];
    P.phi_min = ru[1];
    P.z_precip_min = ru[2];
    P.phitex_max = ru[3];
    P.sig_max = ru[4];
    P.vel_max = ru[5];
    P.z_clear_max = ru[6];
    P.slope_zdr = slopes.data()[0];
    P.slope_phi = slopes.data()[1];
    P.start_km = slopes.data()[2];
    P.despeckle = despeckle;

    std::vector<DArray> keep(6 * ns);
    std::vector<py::array_t<float>> scores;
    std::vector<py::array_t<int8_t>> classes;
    std::vector<std::vector<int8_t>> types(ns);
    std::vector<Sweep> sweeps(ns);
    int64_t first = 0;
    for (size_t i = 0; i < ns; ++i) {
        if (dbz[i].ndim() != 2) throw std::invalid_argument("dbzh must be 2-D (ray, gate)");
        Sweep& sw = sweeps[i];
        sw.nray = dbz[i].shape(0);
        sw.ngate = dbz[i].shape(1);
        if (links[i].ndim() != 1 || links[i].shape(0) != sw.nray)
            throw std::invalid_argument("need one link per ray");
        if (rng_km[i].ndim() != 1 || rng_km[i].shape(0) != sw.ngate)
            throw std::invalid_argument("need one range per gate");
        sw.z = dbz[i].data();
        sw.zdr = optional_data(zdr[i], dbz[i], keep[6 * i], "zdr");
        sw.rho = optional_data(rho[i], dbz[i], keep[6 * i + 1], "rhohv");
        sw.phi = optional_data(phi[i], dbz[i], keep[6 * i + 2], "phidp");
        sw.vel = optional_data(vel[i], dbz[i], keep[6 * i + 3], "vrad");
        sw.sig = optional_data(sig[i], dbz[i], keep[6 * i + 4], "zsigma");
        sw.snr = optional_data(snr[i], dbz[i], keep[6 * i + 5], "snr");
        sw.links = links[i].data();
        sw.rng = rng_km[i].data();
        sw.first_ray = first;
        first += sw.nray;
        scores.emplace_back(std::vector<py::ssize_t>{sw.nray, sw.ngate});
        classes.emplace_back(std::vector<py::ssize_t>{sw.nray, sw.ngate});
        types[i].assign(static_cast<size_t>(sw.nray * sw.ngate), 0);
        sw.score = scores.back().mutable_data();
        sw.cls = classes.back().mutable_data();
        sw.typ = types[i].data();
    }
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const int nt = n_threads > 0 ? n_threads : static_cast<int>(hw);
    {
        py::gil_scoped_release release;
        parallel_rays(sweeps, nt, [&](const Sweep& sw, int64_t r) { classify_ray(P, sw, r); });
        parallel_rays(sweeps, nt, [&](const Sweep& sw, int64_t r) { finish_ray(P, sw, r); });
    }
    py::list score_list, class_list;
    for (size_t i = 0; i < ns; ++i) {
        score_list.append(scores[i]);
        class_list.append(classes[i]);
    }
    return py::make_tuple(score_list, class_list);
}

PYBIND11_MODULE(_gourley, m) {
    m.doc() = "Compiled echo classifier of Gourley et al. (2007) for radarx.";
    m.def("classify", &classify, py::arg("dbzh"), py::arg("zdr"), py::arg("rhohv"),
          py::arg("phidp"), py::arg("vrad"), py::arg("zsigma"), py::arg("snr"),
          py::arg("links"), py::arg("rng_km"), py::arg("floors"), py::arg("snr_min"),
          py::arg("mem_x"), py::arg("mem_y"), py::arg("weights"), py::arg("rules"),
          py::arg("slopes"), py::arg("despeckle"), py::arg("n_threads") = 0);
}
