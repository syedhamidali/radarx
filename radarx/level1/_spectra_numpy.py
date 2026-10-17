#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
NumPy reference implementation of the IQ kernels.

Same functions, arguments and results as the compiled
``radarx.level1._spectra`` module; used when the extension is not built and as
the test oracle. The formulas are in the docstrings of
:mod:`radarx.level1.spectra` and :mod:`radarx.level1.clutter`.
"""

import numpy as np


def power_spectra(x, window, n_seg, n_threads=0):
    """Power spectra of the rows of ``x``, averaged over ``n_seg`` segments.

    Bin ``j`` of a row is the velocity ``(j - L/2) dv`` (the FFT bin
    ``(L/2 - j) mod L``); the bins add up to the mean power of the row.
    """
    x = np.asarray(x, dtype=np.complex128)
    if x.ndim != 2:
        raise ValueError("x must be 2-D (series, pulses)")
    window = np.asarray(window, dtype=np.float64)
    if n_seg < 1:
        raise ValueError("n_seg must be at least 1")
    length = x.shape[1] // n_seg
    if length < 2:
        raise ValueError("a segment needs at least 2 pulses")
    if window.ndim != 1 or window.shape[0] != length:
        raise ValueError("window must have one value per pulse of a segment")
    sw2 = float(np.sum(window * window))
    if not sw2 > 0.0:
        raise ValueError("the window is zero")
    acc = np.zeros((x.shape[0], length))
    for s in range(n_seg):
        seg = x[:, s * length : (s + 1) * length] * window
        acc += np.abs(np.fft.fft(seg, axis=1)) ** 2
    src = (length // 2 - np.arange(length)) % length
    return acc[:, src] / (length * sw2 * n_seg)


def lag_sums(x, n_threads=0):
    """Mean power and the autocorrelation at lags 1 and 2 of the rows of ``x``.

    Columns: power, Re R1, Im R1, Re R2, Im R2.
    """
    x = np.asarray(x, dtype=np.complex128)
    if x.ndim != 2:
        raise ValueError("x must be 2-D (series, pulses)")
    if x.shape[1] < 3:
        raise ValueError("at least 3 pulses are needed")
    r1 = np.mean(np.conj(x[:, :-1]) * x[:, 1:], axis=1)
    r2 = np.mean(np.conj(x[:, :-2]) * x[:, 2:], axis=1)
    p0 = np.mean(np.abs(x) ** 2, axis=1)
    return np.stack([p0, r1.real, r1.imag, r2.real, r2.imag], axis=1)


def regression_filter(x, basis, n_threads=0):
    """``x - Q (Q^T x)`` and the mean power that was removed, per row."""
    x = np.asarray(x, dtype=np.complex128)
    basis = np.asarray(basis, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("x must be 2-D (series, pulses)")
    if basis.ndim != 2 or basis.shape[0] != x.shape[1]:
        raise ValueError("basis must have one row per pulse")
    coef = x @ basis
    removed = np.sum(np.abs(coef) ** 2, axis=1) / x.shape[1]
    return x - coef @ basis.T, removed
