// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Two-moment rain shaft with self-collection and breakup (radarx/retrieve/rain_bulk.py), after
// Seifert (2008, J. Atmos. Sci. 65, 3608-3619, Appendix and Eqs. 19, 20): the rain is a gamma
// distribution with the shape parameter mu(Dm) of Eq. (20), the number- and mass-weighted fall
// speeds are Eq. (A10), self-collection is Eq. (A12) and breakup the relaxation of Eq. (A13) toward
// the equilibrium mean volume diameter of 1.1 mm. In the steady shaft the water flux s V_L L is
// constant and the flux of the number of drops F_N = s V_N N obeys dF_N/dz = s dN/dt|coll. The
// state is F_N; N, L and Dm follow from the two fluxes by a bisection in log Dm. Columns are
// independent and run on all cores (dynamic chunking with an atomic counter). Every step follows
// radarx/retrieve/_rain_bulk_numpy.py in the same order.

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

const double kPi = 3.14159265358979323846;
const double kA = 9.65, kB = 9.8, kC = 600.0;
const double kRR = 4.33, kBR = 1.0e3, kDeq = 1.1e-3, kDbrMin = 0.3e-3;
const double kC1 = 4.0e3, kC2 = 1.0e3, kRhoW = 1000.0;
const double kDmLow = 5.0e-5, kDmHigh = 2.5e-2;
const int kBisections = 60;

template <class F>
void parallel_for(int64_t n, int n_threads, F&& f) {
    int nt = n_threads > 0 ? n_threads : static_cast<int>(std::thread::hardware_concurrency());
    nt = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(nt, n)));
    if (nt == 1) {
        for (int64_t i = 0; i < n; ++i) f(i);
        return;
    }
    std::atomic<int64_t> next{0};
    std::vector<std::thread> pool;
    for (int t = 0; t < nt; ++t)
        pool.emplace_back([&] {
            for (int64_t i = next++; i < n; i = next++) f(i);
        });
    for (auto& t : pool) t.join();
}

double shape_parameter(double dm) {
    const double t = dm - kDeq;
    if (t <= 0.0) {
        const double th = std::tanh(kC1 * t);
        return 6.0 * th * th + 1.0;
    }
    const double th = std::tanh(kC2 * t);
    return 30.0 * th * th + 1.0;
}

void sedimentation(double dm, double& vn, double& vl) {
    const double mu = shape_parameter(dm);
    const double lam = std::cbrt((mu + 3.0) * (mu + 2.0) * (mu + 1.0)) / dm;
    const double q = 1.0 + kC / lam;
    vn = kA - kB * std::pow(q, -(mu + 1.0));
    vl = kA - kB * std::pow(q, -(mu + 4.0));
}

double mass_of_mean_drop(double dm) { return kPi / 6.0 * kRhoW * dm * dm * dm; }

double flux_ratio(double dm) {
    double vn, vl;
    sedimentation(dm, vn, vl);
    return vn / vl / mass_of_mean_drop(dm);
}

double mean_diameter_of_flux_ratio(double ratio) {
    double lo = std::log(kDmLow), hi = std::log(kDmHigh);
    for (int i = 0; i < kBisections; ++i) {
        const double mid = 0.5 * (lo + hi);
        if (flux_ratio(std::exp(mid)) > ratio)
            lo = mid;
        else
            hi = mid;
    }
    return std::exp(0.5 * (lo + hi));
}

struct State {
    double n, water, dm;
};

State concentrations(double flux_n, double flux_l, double s) {
    State st;
    st.dm = mean_diameter_of_flux_ratio(flux_n / flux_l);
    double vn, vl;
    sedimentation(st.dm, vn, vl);
    st.water = flux_l / (s * vl);
    st.n = st.water / mass_of_mean_drop(st.dm);
    return st;
}

double source(const State& st) {
    const double sc = -kRR * st.n * st.water;
    if (st.dm > kDbrMin) return sc - (kBR * (st.dm - kDeq) + 1.0) * sc;
    return sc;
}

struct Profile {
    const double* zs;
    const double* ss;
    int64_t nz;
};

double factor_at(const Profile& p, double z) {
    if (z <= p.zs[0]) return p.ss[0];
    if (z >= p.zs[p.nz - 1]) return p.ss[p.nz - 1];
    const int64_t i = std::upper_bound(p.zs, p.zs + p.nz, z) - p.zs - 1;
    const double w = (z - p.zs[i]) / (p.zs[i + 1] - p.zs[i]);
    return p.ss[i] + w * (p.ss[i + 1] - p.ss[i]);
}

double rhs(const Profile& p, double flux_n, double flux_l, double z) {
    const double s = factor_at(p, z);
    return s * source(concentrations(flux_n, flux_l, s));
}

void evolve_column(const Profile& p, double n, double w, const double* z_out, int64_t nout, double dz_max,
                   double cfl, double* res) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (int64_t q = 0; q < nout; ++q) res[3 * q] = res[3 * q + 1] = res[3 * q + 2] = nan;
    if (!(std::isfinite(n) && std::isfinite(w) && n > 0.0 && w > 0.0)) return;
    const double dm0 = std::cbrt(w / n / (kPi / 6.0 * kRhoW));
    if (!(dm0 >= kDmLow && dm0 <= kDmHigh)) return;
    double vn, vl;
    sedimentation(dm0, vn, vl);
    const double s0 = factor_at(p, 0.0);
    double flux_n = s0 * vn * n;
    const double flux_l = s0 * vl * w;
    double z = 0.0;
    for (int64_t q = 0; q < nout; ++q) {
        const double target = z_out[q];
        while (z < target) {
            const double s = factor_at(p, z);
            const State st = concentrations(flux_n, flux_l, s);
            double vn_now, vl_now;
            sedimentation(st.dm, vn_now, vl_now);
            double dz = std::min(dz_max, target - z);
            const double rate = kRR * st.water;
            if (rate > 0.0) dz = std::min(dz, cfl * s * vn_now / rate);
            const double k1 = rhs(p, flux_n, flux_l, z);
            const double k2 = rhs(p, flux_n + 0.5 * dz * k1, flux_l, z + 0.5 * dz);
            const double k3 = rhs(p, flux_n + 0.5 * dz * k2, flux_l, z + 0.5 * dz);
            const double k4 = rhs(p, flux_n + dz * k3, flux_l, z + dz);
            flux_n = flux_n + dz / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
            z = z + dz < target ? z + dz : target;
        }
        const State st = concentrations(flux_n, flux_l, factor_at(p, z));
        res[3 * q] = st.n;
        res[3 * q + 1] = st.water;
        res[3 * q + 2] = st.dm;
    }
}

py::array_t<double> evolve(const DArray& n0, const DArray& water0, const DArray& z_out, double dz_max,
                           double cfl, const DArray& zs, const DArray& ss, int n_threads) {
    if (n0.ndim() != 1 || water0.ndim() != 1 || z_out.ndim() != 1 || zs.ndim() != 1 || ss.ndim() != 1)
        throw std::invalid_argument("bad array dimensions");
    if (n0.size() != water0.size()) throw std::invalid_argument("n0 and water0 must have the same size");
    if (zs.size() != ss.size() || zs.size() < 1) throw std::invalid_argument("zs and ss must match");
    if (!(dz_max > 0.0) || !(cfl > 0.0)) throw std::invalid_argument("dz_max and cfl must be positive");
    for (int64_t q = 0; q < z_out.size(); ++q)
        if (!(z_out.data()[q] >= 0.0) || (q > 0 && !(z_out.data()[q] >= z_out.data()[q - 1])))
            throw std::invalid_argument("z_out must be ascending and not negative");
    for (int64_t q = 1; q < zs.size(); ++q)
        if (!(zs.data()[q] > zs.data()[q - 1])) throw std::invalid_argument("zs must ascend");
    for (int64_t q = 0; q < ss.size(); ++q)
        if (!(ss.data()[q] > 0.0)) throw std::invalid_argument("ss must be positive");
    const int64_t ncol = n0.size(), nout = z_out.size();
    py::array_t<double> out({nout, ncol, static_cast<int64_t>(3)});
    double* po = out.mutable_data();
    const Profile prof{zs.data(), ss.data(), zs.size()};
    const double *pn = n0.data(), *pw = water0.data(), *pz = z_out.data();
    {
        py::gil_scoped_release release;
        parallel_for(ncol, n_threads, [&](int64_t c) {
            std::vector<double> res(3 * nout);
            evolve_column(prof, pn[c], pw[c], pz, nout, dz_max, cfl, res.data());
            for (int64_t q = 0; q < nout; ++q)
                for (int k = 0; k < 3; ++k) po[(q * ncol + c) * 3 + k] = res[3 * q + k];
        });
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(_rain_bulk, m) {
    m.doc() = "Two-moment rain shaft with self-collection and breakup";
    m.def("evolve", &evolve, py::arg("n0"), py::arg("water0"), py::arg("z_out"), py::arg("dz_max"),
          py::arg("cfl"), py::arg("zs"), py::arg("ss"), py::arg("n_threads") = 0);
    m.def("shape_parameter", [](double dm) { return shape_parameter(dm); }, py::arg("dm"));
    m.def("sedimentation", [](double dm) {
        double vn, vl;
        sedimentation(dm, vn, vl);
        return py::make_tuple(vn, vl);
    }, py::arg("dm"));
}
