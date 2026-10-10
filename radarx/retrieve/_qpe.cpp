// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Rain rate from polarimetric radar variables: the power laws R(Z), R(KDP),
// R(Z, ZDR) and R(A), and the hydrometeor-class blend that picks one of them
// for every gate.
//
// Every gate is independent. The gates form one pool of work that threads take
// in blocks from an atomic counter. The relations, their coefficients and the
// choices that are radarx's own are described in the docstring of
// radarx/retrieve/qpe.py. Every step follows the NumPy reference in that
// module, in the same order of operations.
//
// Relation codes: 0 R(Z), 1 R(KDP), 2 R(Z, ZDR), 3 R(A).

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
using IArray = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr int64_t kBlock = 16384;  // gates per scheduling block

enum Relation { kZ = 0, kKdp = 1, kZZdr = 2, kA = 3 };
enum Flag { kEstimated = 0, kNoRelation = 1, kMissingInput = 2, kNotClassified = 3 };

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
            body(g0, std::min(total, g0 + kBlock));
        }
    };
    std::vector<std::thread> pool;
    for (int i = 1; i < nt; ++i) pool.emplace_back(worker);
    worker();
    for (auto& th : pool) th.join();
}

// Coefficients and options shared by all gates.
struct Params {
    double z0, z1;          // R(Z) = z0 Z^z1
    double k0, k1;          // R(KDP) = k0 |KDP|^k1 sign(KDP)
    double zz0, zz1, zz2;   // R(Z, ZDR) = zz0 Z^zz1 Zdr^zz2
    const double* a_t;      // R(A) nodes: temperature, prefactor, exponent
    const double* a_p0;
    const double* a_p1;
    int64_t a_n;
    double z_cap;           // dBZ, +inf for none
    double zdr_min;         // dB, -inf for none
    int negative;           // 0 keep the sign, 1 zero
};

// Linear interpolation in temperature, constant outside the nodes (np.interp).
inline void a_coefficients(const Params& p, double t, double& p0, double& p1) {
    const int64_t n = p.a_n;
    if (n == 1 || t <= p.a_t[0]) {
        p0 = p.a_p0[0];
        p1 = p.a_p1[0];
        return;
    }
    if (t >= p.a_t[n - 1]) {
        p0 = p.a_p0[n - 1];
        p1 = p.a_p1[n - 1];
        return;
    }
    int64_t j = 1;
    while (p.a_t[j] < t) ++j;
    const double f = (t - p.a_t[j - 1]) / (p.a_t[j] - p.a_t[j - 1]);
    p0 = p.a_p0[j - 1] + f * (p.a_p0[j] - p.a_p0[j - 1]);
    p1 = p.a_p1[j - 1] + f * (p.a_p1[j] - p.a_p1[j - 1]);
}

inline double signed_power(double c, double x, double e, int negative) {
    if (std::isnan(x)) return kNaN;
    if (x > 0.0) return c * std::pow(x, e);
    if (x == 0.0 || negative == 1) return 0.0;
    return -c * std::pow(-x, e);
}

// Rain rate (mm h-1) of one gate by one relation; NaN if an input is missing.
inline double rate(int rel, double dbz, double zdr, double kdp, double a, double t,
                   const Params& p) {
    switch (rel) {
        case kZ: {
            if (std::isnan(dbz)) return kNaN;
            return p.z0 * std::pow(std::pow(10.0, std::min(dbz, p.z_cap) / 10.0), p.z1);
        }
        case kKdp:
            return signed_power(p.k0, kdp, p.k1, p.negative);
        case kZZdr: {
            if (std::isnan(dbz) || std::isnan(zdr)) return kNaN;
            const double z = std::pow(10.0, std::min(dbz, p.z_cap) / 10.0);
            const double zd = std::pow(10.0, std::max(zdr, p.zdr_min) / 10.0);
            return p.zz0 * std::pow(z, p.zz1) * std::pow(zd, p.zz2);
        }
        case kA: {
            if (p.a_n > 1 && std::isnan(t)) return kNaN;
            double p0, p1;
            a_coefficients(p, t, p0, p1);
            return signed_power(p0, a, p1, p.negative);
        }
        default:
            return kNaN;
    }
}

// An input is either one value for all gates (size 1) or one value per gate.
struct Field {
    const double* v;
    bool single;
    double at(int64_t i) const { return single ? v[0] : v[i]; }
};

Field field(const DArray& a, int64_t n, const char* name) {
    if (a.size() == 1) return {a.data(), true};
    if (a.size() != n) throw std::invalid_argument(std::string(name) + " must have 1 or n values");
    return {a.data(), false};
}

Params make_params(const std::vector<double>& coef, const DArray& a_t, const DArray& a_p0,
                   const DArray& a_p1, double z_cap, double zdr_min, int negative) {
    if (coef.size() != 7) throw std::invalid_argument("coef must have 7 values");
    if (a_t.size() < 1 || a_p0.size() != a_t.size() || a_p1.size() != a_t.size())
        throw std::invalid_argument("R(A) nodes must have the same non-zero length");
    Params p;
    p.z0 = coef[0];
    p.z1 = coef[1];
    p.k0 = coef[2];
    p.k1 = coef[3];
    p.zz0 = coef[4];
    p.zz1 = coef[5];
    p.zz2 = coef[6];
    p.a_t = a_t.data();
    p.a_p0 = a_p0.data();
    p.a_p1 = a_p1.data();
    p.a_n = a_t.size();
    p.z_cap = std::isnan(z_cap) ? std::numeric_limits<double>::infinity() : z_cap;
    p.zdr_min = std::isnan(zdr_min) ? -std::numeric_limits<double>::infinity() : zdr_min;
    p.negative = negative;
    return p;
}

}  // namespace

// One relation on all gates. dbz, zdr, kdp, a and t have n values or one.
py::array_t<double> relation_py(int rel, const DArray& dbz, const DArray& zdr, const DArray& kdp,
                                const DArray& a, const DArray& t, int64_t n,
                                const std::vector<double>& coef, const DArray& a_t,
                                const DArray& a_p0, const DArray& a_p1, double z_cap,
                                double zdr_min, int negative, int n_threads) {
    if (rel < 0 || rel > 3) throw std::invalid_argument("rel must be 0, 1, 2 or 3");
    const Params p = make_params(coef, a_t, a_p0, a_p1, z_cap, zdr_min, negative);
    const Field fz = field(dbz, n, "dbz"), fd = field(zdr, n, "zdr"), fk = field(kdp, n, "kdp"),
                fa = field(a, n, "a"), ft = field(t, n, "t");
    py::array_t<double> out(n);
    double* o = out.mutable_data();
    {
        py::gil_scoped_release release;
        parallel_blocks(n, n_threads, [&](int64_t g0, int64_t g1) {
            for (int64_t i = g0; i < g1; ++i)
                o[i] = rate(rel, fz.at(i), fd.at(i), fk.at(i), fa.at(i), ft.at(i), p);
        });
    }
    return out;
}

// The class blend. cls holds the class code of every gate; lut_rel maps a code to a
// relation (-1: none) and lut_fac to a factor on the rain rate. Returns the rain rate,
// the flag and the relation used (1 + relation code, 0 for none).
py::tuple blend_py(const IArray& cls, const DArray& dbz, const DArray& zdr, const DArray& kdp,
                   const DArray& a, const DArray& t, const IArray& lut_rel, const DArray& lut_fac,
                   const std::vector<double>& coef, const DArray& a_t, const DArray& a_p0,
                   const DArray& a_p1, double z_cap, double zdr_min, int negative, int n_threads) {
    const int64_t n = cls.size();
    if (lut_rel.size() != lut_fac.size() || lut_rel.size() < 1)
        throw std::invalid_argument("lut_rel and lut_fac must have the same non-zero length");
    const Params p = make_params(coef, a_t, a_p0, a_p1, z_cap, zdr_min, negative);
    const Field fz = field(dbz, n, "dbz"), fd = field(zdr, n, "zdr"), fk = field(kdp, n, "kdp"),
                fa = field(a, n, "a"), ft = field(t, n, "t");
    const int64_t nl = lut_rel.size();
    const int32_t* c = cls.data();
    const int32_t* lr = lut_rel.data();
    const double* lf = lut_fac.data();
    py::array_t<double> out(n);
    py::array_t<int8_t> flag(n);
    py::array_t<int8_t> used(n);
    double* o = out.mutable_data();
    int8_t* fl = flag.mutable_data();
    int8_t* us = used.mutable_data();
    {
        py::gil_scoped_release release;
        parallel_blocks(n, n_threads, [&](int64_t g0, int64_t g1) {
            for (int64_t i = g0; i < g1; ++i) {
                const int64_t k = c[i];
                double r = kNaN;
                int8_t f = kNotClassified, u = 0;
                if (k > 0) {
                    const int rel = k < nl ? lr[k] : -1;
                    if (rel < 0) {
                        f = kNoRelation;
                    } else {
                        r = rate(rel, fz.at(i), fd.at(i), fk.at(i), fa.at(i), ft.at(i), p);
                        if (std::isnan(r)) {
                            f = kMissingInput;
                        } else {
                            r *= lf[k];
                            f = kEstimated;
                            u = static_cast<int8_t>(rel + 1);
                        }
                    }
                }
                o[i] = r;
                fl[i] = f;
                us[i] = u;
            }
        });
    }
    return py::make_tuple(out, flag, used);
}

PYBIND11_MODULE(_qpe, m) {
    m.doc() = "Rain rate relations and the hydrometeor-class blend";
    m.def("relation", &relation_py, py::arg("rel"), py::arg("dbz"), py::arg("zdr"),
          py::arg("kdp"), py::arg("a"), py::arg("t"), py::arg("n"), py::arg("coef"),
          py::arg("a_t"), py::arg("a_p0"), py::arg("a_p1"), py::arg("z_cap"),
          py::arg("zdr_min"), py::arg("negative"), py::arg("n_threads") = 0);
    m.def("blend", &blend_py, py::arg("cls"), py::arg("dbz"), py::arg("zdr"), py::arg("kdp"),
          py::arg("a"), py::arg("t"), py::arg("lut_rel"), py::arg("lut_fac"), py::arg("coef"),
          py::arg("a_t"), py::arg("a_p0"), py::arg("a_p1"), py::arg("z_cap"),
          py::arg("zdr_min"), py::arg("negative"), py::arg("n_threads") = 0);
}
