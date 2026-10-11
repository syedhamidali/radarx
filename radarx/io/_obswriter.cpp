// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Threaded text formatter of the observation writers (WRFDA ob.radar and DART
// obs_seq). The records are cut into fixed blocks that threads take from an
// atomic counter; every block formats into its own buffer and the buffers are
// joined in block order, so the text is the same for any number of threads.
// The format strings are the ones of the Python formatters in
// radarx/io/wrfda.py and radarx/io/dart.py, which give the same bytes.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace py = pybind11;
using DArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using IArray = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;

namespace {

constexpr int64_t kBlock = 2048;  // records per scheduling block

template <class F>
std::string run_blocks(int64_t total, int n_threads, F&& body) {
    const int64_t nblocks = (total + kBlock - 1) / kBlock;
    std::vector<std::string> out(std::max<int64_t>(nblocks, 0));
    {
        py::gil_scoped_release release;
        unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        int nt = n_threads > 0 ? n_threads : static_cast<int>(hw);
        nt = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(nt, nblocks)));
        std::atomic<int64_t> next{0};
        auto worker = [&]() {
            for (;;) {
                const int64_t b = next.fetch_add(1);
                if (b >= nblocks) break;
                const int64_t i0 = b * kBlock;
                const int64_t i1 = std::min(total, i0 + kBlock);
                body(i0, i1, out[b]);
            }
        };
        std::vector<std::thread> pool;
        for (int i = 1; i < nt; ++i) pool.emplace_back(worker);
        worker();
        for (auto& th : pool) th.join();
    }
    size_t size = 0;
    for (const auto& s : out) size += s.size();
    std::string all;
    all.reserve(size);
    for (const auto& s : out) all += s;
    return all;
}

// Number formatting. Floating-point values go through snprintf, which gives the same digits as
// Python's % operator; std::to_chars for double is not available at the oldest macOS
// deployment target the wheels use. Integers use std::to_chars. Width means right-aligned in
// at least that many characters, like the printf field width.
inline void pad(std::string& s, const char* b, size_t len, int width) {
    if (static_cast<int>(len) < width) s.append(static_cast<size_t>(width) - len, ' ');
    s.append(b, len);
}

// %<width>.16E
inline void put_e(std::string& s, double v, int width) {
    char b[64];
    int n = std::snprintf(b, sizeof b, "%.16E", v);
    if (n < 0 || n >= static_cast<int>(sizeof b)) throw std::runtime_error("formatting failed");
    pad(s, b, static_cast<size_t>(n), width);
}

// %<width>.<prec>f
inline void put_f(std::string& s, double v, int prec, int width) {
    char b[400];
    int n = std::snprintf(b, sizeof b, "%.*f", prec, v);
    if (n < 0 || n >= static_cast<int>(sizeof b)) throw std::runtime_error("formatting failed");
    pad(s, b, static_cast<size_t>(n), width);
}

// %<width>lld
inline void put_i(std::string& s, long long v, int width) {
    char b[32];
    auto r = std::to_chars(b, b + sizeof b, v);
    pad(s, b, static_cast<size_t>(r.ptr - b), width);
}

}  // namespace


// One text block per record of a DART obs_seq file. kind: 0 velocity, 1 reflectivity,
// 2 clear-air reflectivity; the key of record i is i + 1.
py::bytes format_dart(const IArray& kind, const DArray& value, const DArray& variance,
                      const DArray& lon, const DArray& lat, const DArray& height,
                      const IArray& days, const IArray& secs, const DArray& radar_lon,
                      const DArray& radar_lat, const DArray& radar_height, const DArray& beam,
                      const DArray& nyquist, const IArray& velkey, int vertical, int n_threads) {
    const int64_t n = kind.shape(0);
    for (auto s : {value.shape(0), variance.shape(0), lon.shape(0), lat.shape(0),
                   height.shape(0), days.shape(0), secs.shape(0), radar_lon.shape(0),
                   radar_lat.shape(0), radar_height.shape(0), beam.shape(0), nyquist.shape(0),
                   velkey.shape(0)})
        if (s != n) throw std::invalid_argument("all record arrays must have the same length");
    if (beam.ndim() != 2 || beam.shape(1) != 3) throw std::invalid_argument("beam must be (n, 3)");
    const int64_t* kd = kind.data();
    const double *va = value.data(), *vr = variance.data(), *lo = lon.data(), *la = lat.data(),
                 *h = height.data(), *rlo = radar_lon.data(), *rla = radar_lat.data(),
                 *rh = radar_height.data(), *b = beam.data(), *ny = nyquist.data();
    const int64_t *dd = days.data(), *ss = secs.data(), *vk = velkey.data();
    std::string all = run_blocks(n, n_threads, [&](int64_t i0, int64_t i1, std::string& s) {
        s.reserve(static_cast<size_t>(i1 - i0) * 700);
        for (int64_t i = i0; i < i1; ++i) {
            s += " OBS ";
            put_i(s, i + 1, 12);
            s += '\n';
            put_e(s, va[i], 25);
            s += '\n';
            put_i(s, 0, 12);
            s += '\n';
            put_i(s, i == 0 ? -1 : i, 12);
            put_i(s, i == n - 1 ? -1 : i + 2, 12);
            put_i(s, -1, 12);
            s += "\nobdef\nloc3d\n ";
            put_e(s, lo[i], 25);
            s += ' ';
            put_e(s, la[i], 25);
            s += ' ';
            put_e(s, h[i], 25);
            s += ' ';
            put_i(s, vertical, 2);
            s += "\nkind\n";
            put_i(s, kd[i] + 1, 12);
            s += '\n';
            if (kd[i] == 0) {
                s += "platform\nloc3d\n ";
                put_e(s, rlo[i], 25);
                s += ' ';
                put_e(s, rla[i], 25);
                s += ' ';
                put_e(s, rh[i], 25);
                s += ' ';
                put_i(s, vertical, 2);
                s += "\ndir3d\n ";
                put_e(s, b[3 * i], 25);
                s += ' ';
                put_e(s, b[3 * i + 1], 25);
                s += ' ';
                put_e(s, b[3 * i + 2], 25);
                s += "\n ";
                put_e(s, ny[i], 25);
                s += '\n';
                put_i(s, vk[i], 12);
                s += '\n';
            }
            put_i(s, ss[i], 6);
            s += ' ';
            put_i(s, dd[i], 10);
            s += '\n';
            put_e(s, vr[i], 25);
            s += '\n';
        }
    });
    return py::bytes(all);
}

// The columns of one radar of a WRFDA ob.radar file: a column line and one line per layer.
py::bytes format_wrfda(const DArray& centres, const DArray& col_lat, const DArray& col_lon,
                       const DArray& col_elv, const DArray& rv, const IArray& rv_qc,
                       const DArray& rv_err, const DArray& rf, const IArray& rf_qc,
                       const DArray& rf_err, const std::string& stamp, int n_threads) {
    const int64_t nlev = centres.shape(0);
    const int64_t ncol = col_lat.shape(0);
    for (const DArray* a : {&rv, &rv_err, &rf, &rf_err})
        if (a->ndim() != 2 || a->shape(0) != ncol || a->shape(1) != nlev)
            throw std::invalid_argument("level arrays must be (columns, levels)");
    for (const IArray* a : {&rv_qc, &rf_qc})
        if (a->ndim() != 2 || a->shape(0) != ncol || a->shape(1) != nlev)
            throw std::invalid_argument("level arrays must be (columns, levels)");
    if (col_lon.shape(0) != ncol || col_elv.shape(0) != ncol)
        throw std::invalid_argument("column arrays must have the same length");
    const double *c = centres.data(), *cla = col_lat.data(), *clo = col_lon.data(),
                 *cel = col_elv.data(), *v = rv.data(), *ve = rv_err.data(), *z = rf.data(),
                 *ze = rf_err.data();
    const int64_t *vq = rv_qc.data(), *zq = rf_qc.data();
    std::string all = run_blocks(ncol, n_threads, [&](int64_t i0, int64_t i1, std::string& s) {
        s.reserve(static_cast<size_t>(i1 - i0) * static_cast<size_t>(nlev + 1) * 80);
        for (int64_t i = i0; i < i1; ++i) {
            s += "FM-128 RADAR   ";
            s += stamp;
            if (stamp.size() < 19) s.append(19 - stamp.size(), ' ');
            s += "  ";
            put_f(s, cla[i], 3, 12);
            s += "  ";
            put_f(s, clo[i], 3, 12);
            s += "  ";
            put_f(s, cel[i], 1, 8);
            s += "  ";
            put_i(s, nlev, 6);
            s += '\n';
            for (int64_t k = 0; k < nlev; ++k) {
                const int64_t j = i * nlev + k;
                s += "   ";
                put_f(s, c[k], 1, 12);
                put_f(s, v[j], 3, 12);
                put_i(s, vq[j], 4);
                put_f(s, ve[j], 3, 12);
                s += "  ";
                put_f(s, z[j], 3, 12);
                put_i(s, zq[j], 4);
                put_f(s, ze[j], 3, 12);
                s += "  \n";
            }
        }
    });
    return py::bytes(all);
}

PYBIND11_MODULE(_obswriter, m) {
    m.doc() = "Threaded text formatter of the WRFDA and DART observation writers";
    m.def("format_dart", &format_dart, py::arg("kind"), py::arg("value"), py::arg("variance"),
          py::arg("lon"), py::arg("lat"), py::arg("height"), py::arg("days"), py::arg("secs"),
          py::arg("radar_lon"), py::arg("radar_lat"), py::arg("radar_height"), py::arg("beam"),
          py::arg("nyquist"), py::arg("velkey"), py::arg("vertical"), py::arg("n_threads") = 0);
    m.def("format_wrfda", &format_wrfda, py::arg("centres"), py::arg("col_lat"),
          py::arg("col_lon"), py::arg("col_elv"), py::arg("rv"), py::arg("rv_qc"),
          py::arg("rv_err"), py::arg("rf"), py::arg("rf_qc"), py::arg("rf_err"), py::arg("stamp"),
          py::arg("n_threads") = 0);
}
