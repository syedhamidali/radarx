#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
Radarx Machine Learning
=======================

Infrastructure for machine-learning methods: trained networks are run with
ONNX Runtime (``pip install radarx[ml]``), listed in a model registry with
their licences and citations, downloaded and verified on first use, and fed
with polar patches cut from sweeps and volumes.

The methods themselves live next to their physical counterparts (for
example a retrieval in :mod:`radarx.retrieve`) and take and return xarray
objects; this package holds what they share. Training code (PyTorch, export
to ONNX) is kept outside the package, in the repository folder ``ml/``.

Sources and licences of the models. Each model of the registry names its
licence and the paper to cite (with DOI); radarx does not ship weights. The
models used so far are the CNN baseline of the TorNet benchmark (Veillette et
al. 2025, https://doi.org/10.1175/AIES-D-24-0006.1; weights MIT, as stated on
the Hugging Face model card) in :func:`radarx.retrieve.tornado_probability`
and MistNet (Lin et al. 2019, https://doi.org/10.1111/2041-210X.13280;
weights MIT, GitHub repository ``adokter/MistNet``) in
:func:`radarx.retrieve.biological_echo`. The checks against the papers
and the upstream code are documented in :mod:`radarx.retrieve.tornado` and
:mod:`radarx.retrieve.biology`. The patch extraction, blending windows and
normalisation of :mod:`radarx.ml.patches` are radarx tools without a
published source (see there).

.. autosummary::
   :nosignatures:
   :toctree: generated/

   {}
"""

from .model import Model, list_models, load_model, register_model
from .patches import PatchIndex, denormalize, normalize, polar_patches, reassemble

__all__ = [
    "Model",
    "PatchIndex",
    "denormalize",
    "list_models",
    "load_model",
    "normalize",
    "polar_patches",
    "reassemble",
    "register_model",
]

__doc__ = __doc__.format("\n   ".join(__all__))
