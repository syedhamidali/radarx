// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// One time step of the spectral cascade model of radarx.retrieve.nowcast.
//
// Every cascade level j is a field with a Fourier transform Z_j. The level
// follows an autoregressive process in time that is the same at every point,
// so it is linear and acts on each spectral coefficient alone:
//
//     Z_j(t+1) = phi1_j Z_j(t) + phi2_j Z_j(t-1) + WH_j * E
//
// where E is the transform of a white noise field and WH_j holds the band
// filter of the level, the amplitude spectrum that shapes the noise and the
// scale of the innovation. The sum over the levels, weighted by the standard
// deviation sigma_j of each level, is the spectrum of the forecast field.
// The states of all members are updated in one call; threads take
// (member, row) pairs from an atomic counter. No allocation happens in the
// loops.

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <atomic>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

namespace py = pybind11;
using Complex = std::complex<double>;
using CArray = py::array_t<Complex, py::array::c_style | py::array::forcecast>;
using CState = py::array_t<Complex, py::array::c_style>;  // updated in place: no copy
using DArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

namespace {

template <typename Fn>
void parallel_items(int64_t n, int n_threads, Fn&& fn) {
    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    int nt = n_threads > 0 ? n_threads : static_cast<int>(hw);
    nt = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(nt, n)));
    const int64_t block = std::max<int64_t>(1, std::min<int64_t>(8, n / (8 * nt)));
    std::atomic<int64_t> next{0};
    auto worker = [&]() {
        for (;;) {
            const int64_t w0 = next.fetch_add(block);
            if (w0 >= n) break;
            fn(w0, std::min(n, w0 + block));
        }
    };
    std::vector<std::thread> pool;
    for (int t = 1; t < nt; ++t) pool.emplace_back(worker);
    worker();
    for (auto& th : pool) th.join();
}

// z1 (the latest state), z2 (the one before): (member, level, row, column),
// both updated in place. wh: (level, row, column). noise: (member, row,
// column), or empty for no innovation. phi1, phi2, sigma: (level). Returns the
// spectrum (member, row, column) of the sum over the levels of sigma_j Z_j.
CArray ar_step(CState z1, CState z2, const DArray& phi1, const DArray& phi2,
               const DArray& sigma, const DArray& wh, const CArray& noise, int n_threads) {
    if (z1.ndim() != 4 || z2.ndim() != 4) throw std::invalid_argument("states must be 4-D");
    const int64_t nm = z1.shape(0), nl = z1.shape(1), ny = z1.shape(2), nx = z1.shape(3);
    for (int k = 0; k < 4; ++k)
        if (z2.shape(k) != z1.shape(k)) throw std::invalid_argument("states differ in shape");
    if (phi1.size() != nl || phi2.size() != nl || sigma.size() != nl)
        throw std::invalid_argument("phi1, phi2 and sigma need one value per level");
    if (wh.ndim() != 3 || wh.shape(0) != nl || wh.shape(1) != ny || wh.shape(2) != nx)
        throw std::invalid_argument("wh must have the shape (level, row, column)");
    const bool has_noise = noise.size() > 0;
    if (has_noise && (noise.ndim() != 3 || noise.shape(0) != nm || noise.shape(1) != ny ||
                      noise.shape(2) != nx))
        throw std::invalid_argument("noise must have the shape (member, row, column)");
    CArray out(std::vector<py::ssize_t>{static_cast<py::ssize_t>(nm), static_cast<py::ssize_t>(ny),
                                        static_cast<py::ssize_t>(nx)});
    Complex* z1p = z1.mutable_data();
    Complex* z2p = z2.mutable_data();
    Complex* po = out.mutable_data();
    const double* a1 = phi1.data();
    const double* a2 = phi2.data();
    const double* sg = sigma.data();
    const double* whp = wh.data();
    const Complex* np_ = has_noise ? noise.data() : nullptr;
    const int64_t plane = ny * nx;
    {
        py::gil_scoped_release release;
        parallel_items(nm * ny, n_threads, [&](int64_t w0, int64_t w1) {
            for (int64_t w = w0; w < w1; ++w) {
                const int64_t m = w / ny, i = w % ny;
                const int64_t row = i * nx;
                for (int64_t j = 0; j < nx; ++j) po[m * plane + row + j] = Complex(0.0, 0.0);
                for (int64_t l = 0; l < nl; ++l) {
                    Complex* s1 = z1p + (m * nl + l) * plane + row;
                    Complex* s2 = z2p + (m * nl + l) * plane + row;
                    const double* wl = whp + l * plane + row;
                    const Complex* nz = has_noise ? np_ + m * plane + row : nullptr;
                    Complex* acc = po + m * plane + row;
                    for (int64_t j = 0; j < nx; ++j) {
                        Complex v = a1[l] * s1[j] + a2[l] * s2[j];
                        if (has_noise) v += wl[j] * nz[j];
                        s2[j] = s1[j];
                        s1[j] = v;
                        acc[j] += sg[l] * v;
                    }
                }
            }
        });
    }
    return out;
}

// Probability matching of many fields. fields: (n, ny * nx) contiguous, updated
// in place; cells: flat indices of the cells that take part (the same for every
// field); ref: the sorted reference values, one per cell. In each field the values
// at the cells are ranked (ties keep the order of the cells) and replaced by the
// reference value of the same rank. Threads take fields from an atomic counter.
template <typename T>
void rank_match(py::array_t<T, py::array::c_style> fields,
                const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& cells,
                const DArray& ref, int n_threads) {
    if (fields.ndim() != 2) throw std::invalid_argument("fields must be 2-D (field, cell)");
    if (ref.size() != cells.size())
        throw std::invalid_argument("ref needs one value per cell");
    const int64_t n = fields.shape(0), size = fields.shape(1), m = cells.size();
    const int64_t* cp = cells.data();
    for (int64_t i = 0; i < m; ++i)
        if (cp[i] < 0 || cp[i] >= size) throw std::invalid_argument("cell index out of range");
    T* fp = fields.mutable_data();
    const double* rp = ref.data();
    {
        py::gil_scoped_release release;
        parallel_items(n, n_threads, [&](int64_t f0, int64_t f1) {
            std::vector<int32_t> order(static_cast<size_t>(m));
            std::vector<T> vals(static_cast<size_t>(m));
            for (int64_t f = f0; f < f1; ++f) {
                T* field = fp + f * size;
                for (int64_t i = 0; i < m; ++i) {
                    vals[i] = field[cp[i]];
                    order[i] = static_cast<int32_t>(i);
                }
                std::stable_sort(order.begin(), order.end(),
                                 [&](int32_t a, int32_t b) { return vals[a] < vals[b]; });
                for (int64_t r = 0; r < m; ++r) field[cp[order[r]]] = static_cast<T>(rp[r]);
            }
        });
    }
}

}  // namespace

PYBIND11_MODULE(_nowcast, m) {
    m.doc() = "Compiled spectral cascade step of the radarx nowcast.";
    m.def("ar_step", &ar_step, py::arg("z1").noconvert(), py::arg("z2").noconvert(), py::arg("phi1"), py::arg("phi2"),
          py::arg("sigma"), py::arg("wh"), py::arg("noise"), py::arg("n_threads") = 0);
    m.def("rank_match", &rank_match<float>, py::arg("fields").noconvert(), py::arg("cells"),
          py::arg("ref"), py::arg("n_threads") = 0);
    m.def("rank_match", &rank_match<double>, py::arg("fields").noconvert(), py::arg("cells"),
          py::arg("ref"), py::arg("n_threads") = 0);
}
