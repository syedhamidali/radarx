// Copyright (c) 2024-2026, Radarx developers.
// Distributed under the MIT License. See LICENSE for more info.
//
// Parser of the data section of Lightning Mapping Array ASCII files.
//
// The data section is one source per line, whitespace-separated columns, one
// of which may be a hexadecimal station mask (0x...). The text is cut into
// chunks at line ends; threads count the lines of every chunk, a prefix sum
// gives each chunk its first output row, and threads then parse the chunks
// into the rows. Decimal numbers are read as an integer mantissa and a power
// of ten, which is exact (correctly rounded) up to 15 significant digits; longer
// numbers fall back to std::strtod. The NumPy reference is in radarx/io/lma.py.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace py = pybind11;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr int64_t kChunk = 1 << 20;  // bytes per chunk

const double kPow10[] = {1e0,  1e1,  1e2,  1e3,  1e4,  1e5,  1e6,  1e7,  1e8,
                         1e9,  1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17,
                         1e18, 1e19, 1e20, 1e21, 1e22};

inline bool is_space(char c) { return c == ' ' || c == '\t' || c == '\r'; }

// Parse a decimal number in [p, end); returns NaN if it is not one.
double parse_double(const char* p, const char* end) {
    const char* s = p;
    bool neg = false;
    if (p < end && (*p == '-' || *p == '+')) neg = *p++ == '-';
    uint64_t mant = 0;
    int digits = 0, frac = 0;
    bool any = false;
    while (p < end && *p >= '0' && *p <= '9') {
        if (digits < 18) {
            mant = mant * 10 + static_cast<uint64_t>(*p - '0');
            if (mant) ++digits;
        } else {
            ++digits;
        }
        ++p;
        any = true;
    }
    if (p < end && *p == '.') {
        ++p;
        while (p < end && *p >= '0' && *p <= '9') {
            if (digits < 18) {
                mant = mant * 10 + static_cast<uint64_t>(*p - '0');
                if (mant) ++digits;
                ++frac;
            }
            ++p;
            any = true;
        }
    }
    if (!any) return kNaN;
    if (p == end && digits <= 15 && frac <= 22) {
        const double v = static_cast<double>(mant) / kPow10[frac];
        return neg ? -v : v;
    }
    // exponents, long mantissas, nan/inf: the C library
    std::string tmp(s, end);
    char* stop = nullptr;
    const double v = std::strtod(tmp.c_str(), &stop);
    return stop == tmp.c_str() + tmp.size() ? v : kNaN;
}

// Parse a hexadecimal number (with or without 0x) in [p, end); -1 if invalid.
int64_t parse_hex(const char* p, const char* end) {
    if (end - p > 2 && p[0] == '0' && (p[1] == 'x' || p[1] == 'X')) p += 2;
    if (p == end || end - p > 15) return -1;
    int64_t v = 0;
    for (; p < end; ++p) {
        const char c = *p;
        int d;
        if (c >= '0' && c <= '9')
            d = c - '0';
        else if (c >= 'a' && c <= 'f')
            d = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F')
            d = c - 'A' + 10;
        else
            return -1;
        v = v * 16 + d;
    }
    return v;
}

// Is [p, end) a line with at least one non-space character?
inline bool has_data(const char* p, const char* end) {
    for (; p < end; ++p)
        if (!is_space(*p)) return true;
    return false;
}

}  // namespace

// Parse the data section `text` into a (nrow, ncol) float64 array of the
// decimal columns and a (nrow,) int64 array of the hexadecimal column
// `hex_column` (-1 for none; then it is all -1). Blank lines are skipped.
py::tuple parse_py(const py::bytes& text, int ncol, int hex_column, int n_threads) {
    if (ncol < 1) throw std::invalid_argument("ncol must be positive");
    if (hex_column >= ncol) throw std::invalid_argument("hex_column must be < ncol");
    char* buffer = nullptr;
    py::ssize_t length = 0;
    if (PYBIND11_BYTES_AS_STRING_AND_SIZE(text.ptr(), &buffer, &length))
        throw std::invalid_argument("text must be bytes");
    const char* data = buffer;
    const int64_t size = static_cast<int64_t>(length);
    const int nfloat = hex_column >= 0 ? ncol - 1 : ncol;

    // chunk boundaries just after a newline
    std::vector<int64_t> starts{0};
    for (int64_t pos = kChunk; pos < size;) {
        const char* nl = static_cast<const char*>(std::memchr(data + pos, '\n', size - pos));
        if (!nl) break;
        pos = (nl - data) + 1;
        if (pos < size) starts.push_back(pos);
        pos += kChunk;
    }
    starts.push_back(size);
    const int64_t nchunk = static_cast<int64_t>(starts.size()) - 1;
    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const int nt = static_cast<int>(
        std::max<int64_t>(1, std::min<int64_t>(n_threads > 0 ? n_threads : hw, nchunk)));

    auto run = [&](auto&& body) {
        std::atomic<int64_t> next{0};
        auto worker = [&]() {
            for (;;) {
                const int64_t c = next.fetch_add(1);
                if (c >= nchunk) break;
                body(c);
            }
        };
        std::vector<std::thread> pool;
        for (int i = 1; i < nt; ++i) pool.emplace_back(worker);
        worker();
        for (auto& th : pool) th.join();
    };
    auto for_lines = [&](int64_t c, auto&& line) {
        const char* p = data + starts[c];
        const char* stop = data + starts[c + 1];
        while (p < stop) {
            const char* nl = static_cast<const char*>(std::memchr(p, '\n', stop - p));
            const char* e = nl ? nl : stop;
            if (has_data(p, e)) line(p, e);
            p = e + 1;
        }
    };

    std::vector<int64_t> rows(nchunk + 1, 0);
    std::atomic<bool> bad{false};
    std::vector<double> values;
    std::vector<int64_t> hexes;
    {
        py::gil_scoped_release release;
        run([&](int64_t c) {
            int64_t k = 0;
            for_lines(c, [&](const char*, const char*) { ++k; });
            rows[c + 1] = k;
        });
        for (int64_t c = 0; c < nchunk; ++c) rows[c + 1] += rows[c];
        values.assign(static_cast<size_t>(rows[nchunk]) * nfloat, kNaN);
        hexes.assign(static_cast<size_t>(rows[nchunk]), -1);
        run([&](int64_t c) {
            int64_t r = rows[c];
            for_lines(c, [&](const char* p, const char* e) {
                int col = 0, fcol = 0;
                double* out = values.data() + r * nfloat;
                while (p < e && col < ncol) {
                    while (p < e && is_space(*p)) ++p;
                    if (p == e) break;
                    const char* q = p;
                    while (q < e && !is_space(*q)) ++q;
                    if (col == hex_column)
                        hexes[r] = parse_hex(p, q);
                    else
                        out[fcol++] = parse_double(p, q);
                    ++col;
                    p = q;
                }
                if (col < ncol) bad.store(true, std::memory_order_relaxed);
                ++r;
            });
        });
    }
    if (bad.load()) throw std::invalid_argument("a data line has fewer columns than expected");
    const int64_t nrow = rows[nchunk];
    py::array_t<double> v(std::vector<py::ssize_t>{nrow, nfloat});
    std::copy(values.begin(), values.end(), v.mutable_data());
    py::array_t<int64_t> h(nrow);
    std::copy(hexes.begin(), hexes.end(), h.mutable_data());
    return py::make_tuple(v, h);
}

PYBIND11_MODULE(_lma, m) {
    m.doc() = "Compiled Lightning Mapping Array ASCII parser for radarx.";
    m.def("parse", &parse_py, py::arg("text"), py::arg("ncol"), py::arg("hex_column") = -1,
          py::arg("n_threads") = 0);
}
