// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Calibration statistics of dual-polarization radar fields: the differential
// reflectivity offset against a reference value (vertical pointing, light rain,
// dry snow), the reflectivity offset from the self-consistency of Z, ZDR and KDP
// in rain, and the system differential phase of every ray.
//
// Every row (a sweep, a profile or a ray) is independent. The rows form one
// pool of work that threads take from an atomic counter. Selection of the gates,
// the reference values and the statistics follow the NumPy reference in
// radarx/retrieve/calibration.py, in the same order of operations. An array of
// size 0 stands for a criterion that is not used.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

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

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kPi = 3.14159265358979323846;

template <class F>
void parallel_rows(int64_t total, int n_threads, F&& body) {
    if (total <= 0) return;
    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    int nt = n_threads > 0 ? n_threads : static_cast<int>(hw);
    nt = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(nt, total)));
    std::atomic<int64_t> next{0};
    auto worker = [&]() {
        for (;;) {
            const int64_t r = next.fetch_add(1);
            if (r >= total) break;
            body(r);
        }
    };
    std::vector<std::thread> pool;
    for (int i = 1; i < nt; ++i) pool.emplace_back(worker);
    worker();
    for (auto& th : pool) th.join();
}

struct Optional2D {
    const double* p = nullptr;
    int64_t n = 0;
    Optional2D(const DArray& a, int64_t rows, int64_t cols, const char* name) {
        if (a.size() == 0) return;
        if (a.ndim() != 2 || a.shape(0) != rows || a.shape(1) != cols)
            throw std::invalid_argument(std::string(name) + " must have the shape (rows, samples)");
        p = a.data();
        n = cols;
    }
    bool used() const { return p != nullptr; }
    double at(int64_t r, int64_t i) const { return p[r * n + i]; }
};

// piecewise-linear interpolation on increasing xs, clipped at the ends
double interp(const double* xs, const double* ys, int64_t n, double x) {
    if (x <= xs[0]) return ys[0];
    if (x >= xs[n - 1]) return ys[n - 1];
    const double* hi = std::upper_bound(xs, xs + n, x);
    const int64_t k = hi - xs;
    const double t = (x - xs[k - 1]) / (xs[k] - xs[k - 1]);
    return ys[k - 1] + (ys[k] - ys[k - 1]) * t;
}

double median_of(std::vector<double>& v, int64_t n) {
    const int64_t m = n / 2;
    std::nth_element(v.begin(), v.begin() + m, v.begin() + n);
    const double upper = v[m];
    if (n % 2 == 1) return upper;
    const double lower = *std::max_element(v.begin(), v.begin() + m);
    return 0.5 * (lower + upper);
}

}  // namespace

// Differential reflectivity offsets. zdr: (rows, n). dbz, rhohv, snr, mask: (rows, n) or empty.
// elev: (rows,) elevation angle in degrees. ref_z, ref_v: reference ZDR (dB) against dBZ, a
// constant if they have one element. Returns (rows, 5): number of gates, mean, standard
// deviation (n - 1) and median of ZDR - reference, and the mean reference.
py::array_t<double> offset_stats_py(const DArray& zdr, const DArray& dbz, const DArray& rhohv,
                                    const DArray& snr, const DArray& mask, const DArray& elev,
                                    const DArray& ref_z, const DArray& ref_v, double z_min,
                                    double z_max, double rhohv_min, double snr_min,
                                    int n_threads) {
    if (zdr.ndim() != 2) throw std::invalid_argument("zdr must be 2-D (rows, samples)");
    const int64_t rows = zdr.shape(0);
    const int64_t n = zdr.shape(1);
    Optional2D Z(dbz, rows, n, "dbz"), R(rhohv, rows, n, "rhohv"), S(snr, rows, n, "snr"),
        M(mask, rows, n, "mask");
    if (elev.ndim() != 1 || elev.shape(0) != rows)
        throw std::invalid_argument("elev must have one value per row");
    if (ref_v.ndim() != 1 || ref_v.shape(0) < 1 || ref_z.shape(0) != ref_v.shape(0))
        throw std::invalid_argument("ref_z and ref_v must be 1-D with the same size");
    const int64_t nref = ref_v.shape(0);
    const bool table = nref > 1;
    if ((table || std::isfinite(z_min) || std::isfinite(z_max)) && !Z.used())
        throw std::invalid_argument("dbz is needed for the reflectivity limits or a table");
    py::array_t<double> out({static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(5)});
    double* o = out.mutable_data();
    const double* zp = zdr.data();
    const double* ep = elev.data();
    const double* rz = ref_z.data();
    const double* rv = ref_v.data();
    {
        py::gil_scoped_release release;
        parallel_rows(rows, n_threads, [&](int64_t r) {
            std::vector<double> dev(n);
            double sum_ref = 0.0;
            int64_t k = 0;
            const double theta = ep[r] * kPi / 180.0;
            const double s2 = std::sin(theta) * std::sin(theta);
            const double c2 = std::cos(theta) * std::cos(theta);
            for (int64_t i = 0; i < n; ++i) {
                const double zd = zp[r * n + i];
                if (!std::isfinite(zd)) continue;
                if (R.used() && !(R.at(r, i) >= rhohv_min)) continue;
                if (S.used() && !(S.at(r, i) >= snr_min)) continue;
                if (M.used() && !(M.at(r, i) != 0.0)) continue;
                double ref = rv[0];
                if (Z.used()) {
                    const double z = Z.at(r, i);
                    if (!std::isfinite(z) || z < z_min || z > z_max) continue;
                    if (table) ref = interp(rz, rv, nref, z);
                }
                if (std::isfinite(theta))
                    ref = ref - 20.0 * std::log10(std::pow(10.0, 0.05 * ref) * s2 + c2);
                dev[k++] = zd - ref;
                sum_ref += ref;
            }
            double mean = kNaN, sd = kNaN, med = kNaN, mref = kNaN;
            if (k > 0) {
                double s = 0.0;
                for (int64_t i = 0; i < k; ++i) s += dev[i];
                mean = s / static_cast<double>(k);
                mref = sum_ref / static_cast<double>(k);
                if (k > 1) {
                    double q = 0.0;
                    for (int64_t i = 0; i < k; ++i) q += (dev[i] - mean) * (dev[i] - mean);
                    sd = std::sqrt(q / static_cast<double>(k - 1));
                }
                med = median_of(dev, k);
            }
            o[r * 5 + 0] = static_cast<double>(k);
            o[r * 5 + 1] = mean;
            o[r * 5 + 2] = sd;
            o[r * 5 + 3] = med;
            o[r * 5 + 4] = mref;
        });
    }
    return out;
}

// Reflectivity offset from the self-consistency of Z, ZDR and KDP in rain. Gates are averaged in
// bins of z (dBZ); from the bin means of ZDR and KDP the reflectivity that the consistency
// relation log10(KDP / Z) = lg(ZDR) expects is compared with the bin mean reflectivity.
// gz (increasing), glg: the relation. Returns (rows, 5): gates used, bins used, median and
// gate-weighted mean of (Z - expected Z), standard deviation of the bins.
py::array_t<double> consistency_stats_py(const DArray& z, const DArray& zdr, const DArray& kdp,
                                         const DArray& rhohv, const DArray& snr,
                                         const DArray& mask, const DArray& gz, const DArray& glg,
                                         double zdr_offset, double z_min, double z_max,
                                         double bin_width, double rhohv_min, double snr_min,
                                         double kdp_min, int64_t min_bin_count, int n_threads) {
    if (z.ndim() != 2) throw std::invalid_argument("z must be 2-D (rows, samples)");
    const int64_t rows = z.shape(0);
    const int64_t n = z.shape(1);
    Optional2D D(zdr, rows, n, "zdr"), K(kdp, rows, n, "kdp"), R(rhohv, rows, n, "rhohv"),
        S(snr, rows, n, "snr"), M(mask, rows, n, "mask");
    if (!D.used() || !K.used()) throw std::invalid_argument("zdr and kdp are needed");
    if (gz.ndim() != 1 || gz.shape(0) < 2 || glg.shape(0) != gz.shape(0))
        throw std::invalid_argument("gz and glg must be 1-D with the same size, at least 2");
    if (!(bin_width > 0.0) || !(z_max > z_min))
        throw std::invalid_argument("need bin_width > 0 and z_max > z_min");
    const int64_t nb = static_cast<int64_t>(std::ceil((z_max - z_min) / bin_width));
    const int64_t ng = gz.shape(0);
    py::array_t<double> out({static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(5)});
    double* o = out.mutable_data();
    const double* zp = z.data();
    const double* gzp = gz.data();
    const double* glp = glg.data();
    {
        py::gil_scoped_release release;
        parallel_rows(rows, n_threads, [&](int64_t r) {
            std::vector<double> cnt(nb, 0.0), sz(nb, 0.0), sd(nb, 0.0), sk(nb, 0.0);
            for (int64_t i = 0; i < n; ++i) {
                const double zv = zp[r * n + i];
                const double dv = D.at(r, i);
                const double kv = K.at(r, i);
                if (!std::isfinite(zv) || !std::isfinite(dv) || !std::isfinite(kv)) continue;
                if (zv < z_min || zv >= z_max) continue;
                if (R.used() && !(R.at(r, i) >= rhohv_min)) continue;
                if (S.used() && !(S.at(r, i) >= snr_min)) continue;
                if (M.used() && !(M.at(r, i) != 0.0)) continue;
                const int64_t b = std::min<int64_t>(
                    nb - 1, static_cast<int64_t>(std::floor((zv - z_min) / bin_width)));
                cnt[b] += 1.0;
                sz[b] += zv;
                sd[b] += dv;
                sk[b] += kv;
            }
            std::vector<double> dev;
            std::vector<double> weight;
            double gates = 0.0;
            for (int64_t b = 0; b < nb; ++b) {
                if (cnt[b] < static_cast<double>(std::max<int64_t>(min_bin_count, 1))) continue;
                const double mz = sz[b] / cnt[b];
                const double md = sd[b] / cnt[b] - zdr_offset;
                const double mk = sk[b] / cnt[b];
                if (!(mk >= kdp_min) || !(mk > 0.0)) continue;
                if (md < gzp[0] || md > gzp[ng - 1]) continue;
                const double lg = interp(gzp, glp, ng, md);
                dev.push_back(mz - 10.0 * (std::log10(mk) - lg));
                weight.push_back(cnt[b]);
                gates += cnt[b];
            }
            const int64_t nbins = static_cast<int64_t>(dev.size());
            double med = kNaN, wmean = kNaN, sdev = kNaN;
            if (nbins > 0) {
                double ws = 0.0, wd = 0.0, s = 0.0;
                for (int64_t i = 0; i < nbins; ++i) {
                    ws += weight[i];
                    wd += weight[i] * dev[i];
                    s += dev[i];
                }
                wmean = wd / ws;
                const double mean = s / static_cast<double>(nbins);
                if (nbins > 1) {
                    double q = 0.0;
                    for (int64_t i = 0; i < nbins; ++i) q += (dev[i] - mean) * (dev[i] - mean);
                    sdev = std::sqrt(q / static_cast<double>(nbins - 1));
                }
                std::vector<double> tmp(dev);
                med = median_of(tmp, nbins);
            }
            o[r * 5 + 0] = gates;
            o[r * 5 + 1] = static_cast<double>(nbins);
            o[r * 5 + 2] = med;
            o[r * 5 + 3] = wmean;
            o[r * 5 + 4] = sdev;
        });
    }
    return out;
}

// System differential phase of every ray: the circular mean of the first n_use valid gates from
// gate first_gate on. phidp: (rows, ngate) in degrees. Returns (rows, 3): offset in (-180, 180],
// gates used, mean resultant length (1: all gates agree, 0: no preferred phase).
py::array_t<double> phidp_rays_py(const DArray& phidp, const DArray& rhohv, const DArray& snr,
                                  const DArray& mask, int64_t first_gate, int64_t n_use,
                                  double rhohv_min, double snr_min, int64_t min_count,
                                  int n_threads) {
    if (phidp.ndim() != 2) throw std::invalid_argument("phidp must be 2-D (rays, gates)");
    const int64_t rows = phidp.shape(0);
    const int64_t n = phidp.shape(1);
    Optional2D R(rhohv, rows, n, "rhohv"), S(snr, rows, n, "snr"), M(mask, rows, n, "mask");
    py::array_t<double> out({static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(3)});
    double* o = out.mutable_data();
    const double* pp = phidp.data();
    {
        py::gil_scoped_release release;
        parallel_rows(rows, n_threads, [&](int64_t r) {
            double sc = 0.0, ss = 0.0;
            int64_t k = 0;
            for (int64_t i = std::max<int64_t>(first_gate, 0); i < n && k < n_use; ++i) {
                const double p = pp[r * n + i];
                if (!std::isfinite(p)) continue;
                if (R.used() && !(R.at(r, i) >= rhohv_min)) continue;
                if (S.used() && !(S.at(r, i) >= snr_min)) continue;
                if (M.used() && !(M.at(r, i) != 0.0)) continue;
                const double a = p * kPi / 180.0;
                sc += std::cos(a);
                ss += std::sin(a);
                ++k;
            }
            double offset = kNaN, length = kNaN;
            if (k >= std::max<int64_t>(min_count, 1)) {
                offset = std::atan2(ss, sc) * 180.0 / kPi;
                length = std::sqrt(sc * sc + ss * ss) / static_cast<double>(k);
            }
            o[r * 3 + 0] = offset;
            o[r * 3 + 1] = static_cast<double>(k);
            o[r * 3 + 2] = length;
        });
    }
    return out;
}

PYBIND11_MODULE(_calibration, m) {
    m.doc() = "Calibration statistics: ZDR offset, Z self-consistency, system PhiDP";
    m.def("offset_stats", &offset_stats_py, py::arg("zdr"), py::arg("dbz"), py::arg("rhohv"),
          py::arg("snr"), py::arg("mask"), py::arg("elev"), py::arg("ref_z"), py::arg("ref_v"),
          py::arg("z_min"), py::arg("z_max"), py::arg("rhohv_min"), py::arg("snr_min"),
          py::arg("n_threads") = 0);
    m.def("consistency_stats", &consistency_stats_py, py::arg("z"), py::arg("zdr"),
          py::arg("kdp"), py::arg("rhohv"), py::arg("snr"), py::arg("mask"), py::arg("gz"),
          py::arg("glg"), py::arg("zdr_offset"), py::arg("z_min"), py::arg("z_max"),
          py::arg("bin_width"), py::arg("rhohv_min"), py::arg("snr_min"), py::arg("kdp_min"),
          py::arg("min_bin_count"), py::arg("n_threads") = 0);
    m.def("phidp_rays", &phidp_rays_py, py::arg("phidp"), py::arg("rhohv"), py::arg("snr"),
          py::arg("mask"), py::arg("first_gate"), py::arg("n_use"), py::arg("rhohv_min"),
          py::arg("snr_min"), py::arg("min_count"), py::arg("n_threads") = 0);
}
