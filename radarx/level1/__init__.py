#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Radarx Level I Processing
=========================

Processing of radar IQ time series (Level I): reading or simulating the complex
samples, removing ground clutter, forming Doppler power spectra and estimating
the radar moments (power, mean velocity, spectrum width, skewness, kurtosis)
from the spectra or from the autocorrelation, and dealiasing the velocity with
the spectra. The heavy loops run in compiled, multithreaded C++ kernels; if
they are not available an equivalent NumPy implementation is used.

.. toctree::
    :maxdepth: 4

.. automodule:: radarx.level1.iq
.. automodule:: radarx.level1.clutter
.. automodule:: radarx.level1.spectra
"""

from .iq import *  # noqa
from .clutter import *  # noqa
from .spectra import *  # noqa

__all__ = [
    s for s in dir() if not s.startswith("_") and s not in ("iq", "clutter", "spectra")
]
