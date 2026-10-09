#!/usr/bin/env python
# Copyright (c) 2024-2026, Radarx developers.
# Distributed under the MIT License. See LICENSE for more info.

"""
ONNX models and the model registry.

Trained networks are run with ONNX Runtime, a small inference engine with
wheels for every platform, so radarx needs no deep-learning framework at run
time (PyTorch is only used to train models, outside the package). The
optional dependency is installed with ``pip install radarx[ml]``.

Models are listed in a registry, ``radarx/ml/models.toml``, with the URL of
the ONNX file, its SHA-256 hash, licence, citation, version and a description
of the inputs and outputs. The weights are not part of radarx: they are
downloaded from their authors' release on first use, checked against the
hash and cached under ``pooch.os_cache("radarx")/models`` (or
``$RADARX_CACHE_DIR/models``). Further registry files can be listed in the
``RADARX_MODEL_REGISTRY`` environment variable (separated by ``os.pathsep``)
and models can be registered for the session with :func:`register_model`.

Every model carries its licence and citation: they are shown by
``repr(model)`` and :func:`list_models`, and :attr:`Model.attrs` adds them to
the outputs of radarx functions that use the model.
"""

from __future__ import annotations

__all__ = ["Model", "list_models", "load_model", "register_model"]

import copy
import hashlib
import os
import tempfile
import tomllib
import warnings
from functools import lru_cache
from importlib import resources
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import numpy as np

REQUIRED = ("url", "sha256", "licence", "citation")

# accelerators used by default when ONNX Runtime has them; CoreML is opt-in
# because it converts the model on every session and may compute in float16
_DEFAULT_ACCELERATORS = (
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    "DmlExecutionProvider",
)

_ONNX_TYPES = {
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    "tensor(float16)": np.float16,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(int16)": np.int16,
    "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8,
    "tensor(bool)": np.bool_,
}

_registered = {}


def _onnxruntime():
    """Import ONNX Runtime or explain how to install it."""
    try:
        import onnxruntime
    except ImportError as err:
        raise ImportError(
            "radarx.ml runs models with ONNX Runtime, which is not installed. "
            "Install it with `pip install radarx[ml]` "
            "(or `conda install -c conda-forge onnxruntime`)."
        ) from err
    return onnxruntime


def _cache_dir():
    root = os.environ.get("RADARX_CACHE_DIR")
    if root is None:
        import pooch

        root = pooch.os_cache("radarx")
    path = Path(root, "models")
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------


def _read_registry(text, source):
    models = tomllib.loads(text).get("models", {})
    for name, entry in models.items():
        missing = [k for k in REQUIRED if k not in entry]
        if missing:
            raise ValueError(f"model {name!r} in {source} lacks {', '.join(missing)}")
    return models


@lru_cache(maxsize=1)
def _shipped():
    text = resources.files("radarx.ml").joinpath("models.toml").read_text("utf-8")
    return _read_registry(text, "radarx/ml/models.toml")


def _registry():
    """All models: shipped, from ``RADARX_MODEL_REGISTRY`` files, registered."""
    models = dict(_shipped())
    for path in filter(
        None, os.environ.get("RADARX_MODEL_REGISTRY", "").split(os.pathsep)
    ):
        models.update(_read_registry(Path(path).read_text("utf-8"), path))
    models.update(_registered)
    return {name: dict(entry, name=name) for name, entry in models.items()}


def register_model(
    name,
    path_or_url,
    sha256,
    licence,
    citation,
    *,
    version="1",
    task="",
    overwrite=False,
    **spec,
):
    """
    Register a model (a local ONNX file or a URL) for this session.

    Use it for your own or third-party models that are not in the radarx
    registry. To keep them across sessions, write the same fields to a TOML
    file (see ``radarx/ml/models.toml``) and list it in the
    ``RADARX_MODEL_REGISTRY`` environment variable.

    Parameters
    ----------
    name : str
        Name used with :func:`load_model`.
    path_or_url : str or path
        Local ``.onnx`` file or ``https://`` URL of one.
    sha256 : str or None
        SHA-256 hash of the file. Required for URLs; for a local file
        ``None`` skips the check.
    licence : str
        Licence of the weights (SPDX identifier, e.g. ``"MIT"``).
    citation : str
        Reference to cite when using the model.
    version : str, default "1"
        Model version.
    task : str, optional
        What the model does (e.g. ``"tornado-detection"``).
    overwrite : bool, default False
        Replace a model of the same name.
    **spec
        Further registry fields, e.g. ``inputs`` and ``outputs`` (free-text
        descriptions of the tensors) or ``description``.

    Returns
    -------
    dict
        The registry entry.
    """
    path_or_url = os.fspath(path_or_url)
    is_url = urlparse(path_or_url).scheme in ("http", "https", "ftp", "doi")
    if is_url and not sha256:
        raise ValueError("a model downloaded from a URL needs its sha256 hash")
    if name in _registry() and not overwrite:
        raise ValueError(f"model {name!r} is already registered (use overwrite=True)")
    if not is_url:
        path_or_url = str(Path(path_or_url).expanduser().resolve())
    _registered[name] = dict(
        spec,
        url=path_or_url,
        sha256=sha256 or "",
        licence=licence,
        citation=citation,
        version=str(version),
        task=task,
    )
    return dict(_registered[name], name=name)


def list_models():
    """
    Models in the registry, with their licences and citations.

    Returns
    -------
    list of dict
        One dict per model with ``name``, ``task``, ``version``, ``licence``,
        ``citation`` and ``url``.
    """
    keys = ("name", "task", "version", "licence", "citation", "url")
    return [{k: str(entry.get(k, "")) for k in keys} for entry in _registry().values()]


# --------------------------------------------------------------------------
# download and verification
# --------------------------------------------------------------------------


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _local_path(url):
    """Path of a local registry entry, or None for a download URL."""
    parsed = urlparse(url)
    if parsed.scheme == "file":
        return Path(url2pathname(parsed.path))
    if parsed.scheme in ("",) or (len(parsed.scheme) == 1 and os.name == "nt"):
        return Path(url).expanduser()
    return None


def _fetch(entry, cache):
    """Local path of the model file (downloaded and verified if needed)."""
    known = str(entry.get("sha256", "")).lower().removeprefix("sha256:")
    path = _local_path(entry["url"])
    if path is not None:
        if not path.is_file():
            raise FileNotFoundError(f"model file {path} does not exist")
        if known and _sha256(path) != known:
            raise ValueError(
                f"SHA256 hash of {path} does not match the registry ({known}); "
                "the file is corrupt or not the registered model"
            )
        return path, None
    import pooch

    tmp = None
    folder = _cache_dir() if cache else Path(tmp := tempfile.mkdtemp(prefix="radarx-"))
    fname = f"{entry['name']}-v{entry.get('version', '1')}.onnx"
    path = pooch.retrieve(
        entry["url"], known_hash=f"sha256:{known}", fname=fname, path=folder
    )
    return Path(path), tmp


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------


def _providers(ort, providers):
    available = ort.get_available_providers()
    if providers is None or providers == "auto":
        return [p for p in _DEFAULT_ACCELERATORS if p in available] + [
            "CPUExecutionProvider"
        ]
    if providers == "cpu":
        return ["CPUExecutionProvider"]
    if isinstance(providers, str):
        providers = [providers]
    names = [p[0] if isinstance(p, tuple) else p for p in providers]
    missing = [p for p in names if p not in available]
    if missing:
        raise ValueError(
            f"ONNX Runtime providers {missing} are not available; available: {available}"
        )
    return list(providers)


def _tensor_spec(arg):
    shape = ",".join("?" if d is None else str(d) for d in arg.shape)
    dtype = _ONNX_TYPES.get(arg.type)
    name = np.dtype(dtype).name if dtype is not None else arg.type
    return f"{name}[{shape}]"


class Model:
    """
    An ONNX model with its registry entry.

    Created by :func:`load_model`. :meth:`run` takes and returns NumPy arrays;
    the radarx functions built on a model wrap them in xarray objects and add
    :attr:`attrs` to their outputs.

    Attributes
    ----------
    name : str
        Registry name.
    info : dict
        Registry entry (``licence``, ``citation``, ``version``, ``task``,
        ``inputs``, ``outputs``, ...).
    path : pathlib.Path or None
        Local ONNX file (``None`` when not kept in the cache).
    session : onnxruntime.InferenceSession
        The ONNX Runtime session.
    """

    def __init__(self, name, info, session, path=None):
        self.name = name
        self.info = info
        self.session = session
        self.path = path

    @property
    def inputs(self):
        """Input tensors of the network: ``{name: "float32[N,C,H,W]"}``."""
        return {a.name: _tensor_spec(a) for a in self.session.get_inputs()}

    @property
    def outputs(self):
        """Output tensors of the network: ``{name: "float32[N]"}``."""
        return {a.name: _tensor_spec(a) for a in self.session.get_outputs()}

    @property
    def providers(self):
        """ONNX Runtime execution providers in use."""
        return self.session.get_providers()

    @property
    def version(self):
        return str(self.info.get("version", ""))

    @property
    def licence(self):
        return str(self.info.get("licence", ""))

    @property
    def citation(self):
        return str(self.info.get("citation", ""))

    @property
    def attrs(self):
        """Attributes recording the model on the outputs it produced."""
        return {
            "ml_model": self.name,
            "ml_model_version": self.version,
            "ml_model_licence": self.licence,
            "ml_model_citation": self.citation,
        }

    def run(self, inputs, outputs=None, *, batch_size=None):
        """
        Run the network.

        Parameters
        ----------
        inputs : dict of numpy.ndarray or numpy.ndarray
            Input tensors by name; a single array for a network with one
            input. Arrays are cast to the input's type (e.g. float32).
        outputs : list of str, optional
            Outputs to compute (default: all).
        batch_size : int, optional
            Run the first axis (the batch, e.g. patches) in chunks of this
            size to bound memory; the results are concatenated.

        Returns
        -------
        dict of numpy.ndarray
            Output tensors by name.
        """
        args = self.session.get_inputs()
        if not isinstance(inputs, dict):
            if len(args) != 1:
                raise ValueError(
                    f"the model has inputs {[a.name for a in args]}; pass a dict"
                )
            inputs = {args[0].name: inputs}
        expected = {a.name: a for a in args}
        unknown = set(inputs) - set(expected)
        missing = set(expected) - set(inputs)
        if unknown or missing:
            raise ValueError(
                f"the model expects the inputs {sorted(expected)}; "
                f"missing {sorted(missing)}, unknown {sorted(unknown)}"
            )
        feed = {
            k: np.ascontiguousarray(v, dtype=_ONNX_TYPES.get(expected[k].type))
            for k, v in inputs.items()
        }
        names = list(outputs) if outputs is not None else list(self.outputs)
        if batch_size is None:
            return dict(zip(names, self.session.run(names, feed)))
        n = {v.shape[0] for v in feed.values()}
        if len(n) != 1:
            raise ValueError("batched inputs need the same first dimension")
        (n,) = n
        parts = [
            self.session.run(names, {k: v[i : i + batch_size] for k, v in feed.items()})
            for i in range(0, n, int(batch_size))
        ]
        return {
            name: np.concatenate([p[j] for p in parts]) for j, name in enumerate(names)
        }

    def __call__(self, inputs, outputs=None, **kwargs):
        return self.run(inputs, outputs, **kwargs)

    def __repr__(self):
        lines = [f"<radarx.ml.Model {self.name!r} version {self.version}>"]
        if self.info.get("task"):
            lines.append(f"  task:      {self.info['task']}")
        lines.append(f"  licence:   {self.licence or 'unknown'}")
        lines.append(f"  citation:  {self.citation or 'unknown'}")
        lines.append(
            "  inputs:    " + ", ".join(f"{k} {v}" for k, v in self.inputs.items())
        )
        lines.append(
            "  outputs:   " + ", ".join(f"{k} {v}" for k, v in self.outputs.items())
        )
        lines.append("  providers: " + ", ".join(self.providers))
        return "\n".join(lines)


def load_model(name, *, providers=None, cache=True, session_options=None):
    """
    Load a model from the registry (downloading it on first use).

    Parameters
    ----------
    name : str or path
        Registry name (see :func:`list_models`), or the path of a local
        ``.onnx`` file that is not registered (its licence is then unknown).
    providers : str or list, optional
        ONNX Runtime execution providers. By default a CUDA, ROCm or DirectML
        GPU is used when ONNX Runtime has it, otherwise the CPU; ``"cpu"``
        forces the CPU. A list is passed to ONNX Runtime as is, e.g.
        ``["CoreMLExecutionProvider", "CPUExecutionProvider"]`` on a Mac.
    cache : bool, default True
        Keep the downloaded file in the radarx cache. With ``False`` the file
        is downloaded to a temporary folder that is removed after loading.
    session_options : onnxruntime.SessionOptions, optional
        Options of the ONNX Runtime session (threads, optimisation level).

    Returns
    -------
    Model

    Raises
    ------
    ImportError
        If ONNX Runtime is not installed (``pip install radarx[ml]``).
    KeyError
        If the model is not in the registry.
    ValueError
        If the file does not match its SHA-256 hash.
    """
    ort = _onnxruntime()
    registry = _registry()
    key = os.fspath(name)
    if key in registry:
        entry = copy.deepcopy(registry[key])
    elif Path(key).suffix == ".onnx" and Path(key).is_file():
        warnings.warn(
            f"{key} is not in the model registry; its licence and citation are "
            "unknown (use register_model to record them)",
            stacklevel=2,
        )
        entry = {
            "name": Path(key).stem,
            "url": str(Path(key).resolve()),
            "sha256": "",
            "licence": "",
            "citation": "",
            "version": "",
        }
    else:
        known = ", ".join(sorted(registry)) or "none"
        raise KeyError(f"unknown model {key!r}; registered models: {known}")

    path, tmp = _fetch(entry, cache)
    try:
        session = ort.InferenceSession(
            str(path),
            sess_options=session_options,
            providers=_providers(ort, providers),
        )
    finally:
        if tmp is not None:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)
            path = None
    return Model(entry["name"], entry, session, path)
