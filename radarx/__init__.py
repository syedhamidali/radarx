"""Top-level package for radarx."""

__author__ = """Hamid Ali Syed"""
__email__ = "hamidsyed37@gmail.com"

# versioning
try:
    from .version import version as __version__
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"

from . import io  # noqa
from . import testing  # noqa

__all__ = [s for s in dir() if not s.startswith("_")]
