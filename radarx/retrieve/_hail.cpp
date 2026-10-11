// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Hail products of a reflectivity profile: the severe hail index (SHI), the
// maximum expected size of hail (MESH), the probability of severe hail (POSH),
// the warning threshold of POSH and the hail kinetic energy flux.
//
// Every column is independent. A column is a list of (height, dBZ) samples
// (beams of a polar volume at one ground range, levels of a grid, or the bins of
// a quasi-vertical profile); missing samples are NaN and are left out, a sample
// of -inf is an observed absence of echo. The columns form one pool of work that
// threads take in blocks from an atomic counter.
//
// The equations are those of Witt et al. (1998, Wea. Forecasting 13, 286-303):
//   E  = 5e-6 * 10^(0.084 Z) * W(Z)                         (Eq. 1)
//   W_T(H) = clip((H - H0) / (Hm20 - H0), 0, 1)             (Eq. 2)
//   SHI = 0.1 * sum_i W_T(H_i) E_i dH_i                     (Eq. 3, discrete form)
//   WT = max(57.5 H0 - 121, 20)                             (Eq. 4)
//   POSH = clip(29 ln(SHI / WT) + 50, 0, 100)               (Eq. 5)
//   MESH = 2.54 SHI^0.5                                     (Eq. 6)
// with the layer depths dH_i of the paper (see column_shi). Details, references
// and the radarx choices are in the docstring of radarx/retrieve/hail.py. Every
// step follows the NumPy reference in that module, in the same order of
// operations.

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
constexpr double kEnergyCoefficient = 5.0e-6;   // J m-2 s-1, Witt et al. (1998) Eq. 1
constexpr double kEnergyExponent = 0.084;       // per dBZ
constexpr double kShiFactor = 0.1;              // Eq. 3
constexpr double kWarningSlope = 57.5;          // J m-1 s-1 per km, Eq. 4
constexpr double kWarningMinimum = 20.0;        // J m-1 s-1, below Eq. 4
constexpr double kPoshCoefficient = 29.0;       // Eq. 5
constexpr double kMeshCoefficient = 2.54;       // mm, Eq. 6
constexpr double kMeshExponent = 0.5;           // Eq. 6
constexpr int64_t kBlock = 256;                 // columns per scheduling block
constexpr int kRows = 6;

template <class F>
void parallel_blocks(int64_t total, int n_threads, F&& body) {
    if (total <= 0) return;
    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    int nt = n_threads > 0 ? n_threads : static_cast<int>(hw);
    const int64_t nblocks = (total + kBlock - 1) / kBlock;
    nt = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(nt, nblocks)));
    std::atomic<int64_t> next{0};
    auto worker = [&]() {
        for (;;) {
            const int64_t g0 = next.fetch_add(kBlock);
            if (g0 >= total) break;
            const int64_t g1 = std::min(total, g0 + kBlock);
            body(g0, g1);
        }
    };
    std::vector<std::thread> pool;
    for (int i = 1; i < nt; ++i) pool.emplace_back(worker);
    worker();
    for (auto& th : pool) th.join();
}

// Samples of column c that have a finite height and a value, sorted by height with a
// stable insertion sort; returns how many there are.
int64_t gather_sorted(const double* hp, const double* vp, bool shared, int64_t nk, int64_t nc,
                      int64_t c, double* sh, double* sv) {
    int64_t n = 0;
    for (int64_t k = 0; k < nk; ++k) {
        const double hk = shared ? hp[k] : hp[k * nc + c];
        const double vk = vp[k * nc + c];
        if (!std::isfinite(hk) || std::isnan(vk)) continue;
        int64_t i = n++;
        while (i > 0 && sh[i - 1] > hk) {
            sh[i] = sh[i - 1];
            sv[i] = sv[i - 1];
            --i;
        }
        sh[i] = hk;
        sv[i] = vk;
    }
    return n;
}

// Hail kinetic energy flux (J m-2 s-1) of a reflectivity (dBZ): Eq. 1 with the
// reflectivity weighting W(Z) between zl and zu.
inline double energy_flux(double dbz, double zl, double zu) {
    if (!(dbz > zl)) return 0.0;
    const double w = dbz >= zu ? 1.0 : (dbz - zl) / (zu - zl);
    return kEnergyCoefficient * std::pow(10.0, kEnergyExponent * dbz) * w;
}

// SHI of the sorted samples (J m-1 s-1). Each sample carries the depth dH_i of
// the paper: half the distance between its neighbours for an interior sample,
// the distance to the next lower one for the top sample, and for the bottom
// sample (which only counts when it is above H0) the layer from H0 to the middle
// between it and the next one.
double column_shi(const double* sh, const double* sv, int64_t n, double h0, double hm20,
                  double zl, double zu) {
    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double e = energy_flux(sv[i], zl, zu);
        if (e <= 0.0) continue;
        const double wt = std::min(1.0, std::max(0.0, (sh[i] - h0) / (hm20 - h0)));
        if (wt <= 0.0) continue;
        double dh;
        if (i == 0)
            dh = 0.5 * (sh[0] + sh[1]) - h0;
        else if (i == n - 1)
            dh = sh[i] - sh[i - 1];
        else
            dh = 0.5 * (sh[i + 1] - sh[i - 1]);
        sum += wt * e * dh;
    }
    return kShiFactor * sum;
}

void check_shapes(const DArray& h, const DArray& v, const DArray& h0, const DArray& hm20,
                  bool shared) {
    if (v.ndim() != 2) throw std::invalid_argument("v must be 2-D (levels, columns)");
    const int64_t nk = v.shape(0);
    const int64_t nc = v.shape(1);
    if (shared) {
        if (h.ndim() != 1 || h.shape(0) != nk)
            throw std::invalid_argument("shared heights must be 1-D with one value per level");
    } else if (h.ndim() != 2 || h.shape(0) != nk || h.shape(1) != nc) {
        throw std::invalid_argument("heights must have the shape of v");
    }
    if (h0.ndim() != 1 || h0.shape(0) != nc || hm20.ndim() != 1 || hm20.shape(0) != nc)
        throw std::invalid_argument("the 0 and -20 degC heights must be 1-D with one value per column");
}

}  // namespace

// h: (nk,) when shared, else (nk, nc); v: (nk, nc) dBZ; h0, hm20: (nc,) heights of the 0 and
// -20 degC levels in the same frame as h. radar_height: height of the radar in that frame.
// Returns (6, nc): SHI, MESH, POSH, WT, the hail kinetic energy flux of the lowest sample and
// the reflectivity (dBZ) of the highest sample.
py::array_t<double> columns_py(const DArray& h, const DArray& v, const DArray& h0,
                               const DArray& hm20, bool shared, double radar_height, double zl,
                               double zu, double warning_offset, double posh_offset,
                               int n_threads) {
    check_shapes(h, v, h0, hm20, shared);
    if (!(zu > zl)) throw std::invalid_argument("zu must be above zl");
    const int64_t nk = v.shape(0);
    const int64_t nc = v.shape(1);

    py::array_t<double> out({static_cast<py::ssize_t>(kRows), static_cast<py::ssize_t>(nc)});
    double* o = out.mutable_data();
    const double* hp = h.data();
    const double* vp = v.data();
    const double* z0 = h0.data();
    const double* zm = hm20.data();

    {
        py::gil_scoped_release release;
        parallel_blocks(nc, n_threads, [&](int64_t c0, int64_t c1) {
            std::vector<double> sh(nk), sv(nk);
            for (int64_t c = c0; c < c1; ++c) {
                const int64_t n = gather_sorted(hp, vp, shared, nk, nc, c, sh.data(), sv.data());
                double shi = kNaN, mesh = kNaN, posh = kNaN, wt = kNaN, hke = kNaN, last = kNaN;
                if (n > 0) {
                    hke = energy_flux(sv[0], zl, zu);
                    last = sv[n - 1];
                }
                const bool levels = std::isfinite(z0[c]) && std::isfinite(zm[c]) && zm[c] > z0[c];
                if (n > 1 && levels) {
                    shi = column_shi(sh.data(), sv.data(), n, z0[c], zm[c], zl, zu);
                    mesh = kMeshCoefficient * std::pow(shi, kMeshExponent);
                    wt = std::max(kWarningSlope * (z0[c] - radar_height) / 1000.0 - warning_offset,
                                  kWarningMinimum);
                    if (shi > 0.0)
                        posh = std::min(100.0, std::max(0.0, kPoshCoefficient * std::log(shi / wt) +
                                                                  posh_offset));
                    else
                        posh = 0.0;
                }
                o[0 * nc + c] = shi;
                o[1 * nc + c] = mesh;
                o[2 * nc + c] = posh;
                o[3 * nc + c] = wt;
                o[4 * nc + c] = hke;
                o[5 * nc + c] = last;
            }
        });
    }
    return out;
}

PYBIND11_MODULE(_hail, m) {
    m.doc() = "SHI, MESH, POSH and hail kinetic energy kernel";
    m.def("columns", &columns_py, py::arg("h"), py::arg("v"), py::arg("h0"), py::arg("hm20"),
          py::arg("shared"), py::arg("radar_height"), py::arg("zl"), py::arg("zu"),
          py::arg("warning_offset"), py::arg("posh_offset"), py::arg("n_threads") = 0);
}
