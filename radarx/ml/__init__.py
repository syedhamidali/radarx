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
