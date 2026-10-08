"""
Training-data generation for radar machine learning with radarx labels.

This package lives outside the ``radarx`` package (it is never imported by
radarx). It builds chunked Zarr training sets from NEXRAD Level II volumes,
using radarx's physical retrievals as labels. See ``ml/README.md`` for the
dataset schemas and ``ml/data/build_dataset.py`` for the command line.
"""

from .schema import SCHEMAS, TASKS, validate

__all__ = ["SCHEMAS", "TASKS", "validate", "open_dataset"]


def open_dataset(root, task, split):
    """
    Open one task/split of a built dataset lazily.

    Parameters
    ----------
    root : str or path-like
        Output folder of ``build_dataset.py``.
    task : str
        One of :data:`TASKS`.
    split : {"train", "val", "test"}
        Data split.

    Returns
    -------
    xarray.Dataset
        The samples on a ``sample`` dimension (dask-backed if dask is
        installed).
    """
    from pathlib import Path

    import xarray as xr

    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, not {task!r}")
    return xr.open_zarr(Path(root) / task / f"{split}.zarr")
