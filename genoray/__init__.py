"""genoray package — lazy public API.

Heavy modules (``_pgen``, ``_svar``, ``_vcf``, ``exprs``) are loaded on first
access via PEP 562 ``__getattr__``. This keeps ``import genoray`` (and therefore
``genoray --help``) sub-second.
"""

from __future__ import annotations

import importlib
from importlib.metadata import version
from typing import TYPE_CHECKING

__version__ = version("genoray")

__all__ = ["PGEN", "VCF", "Reader", "SparseVar", "exprs"]

# Public name -> (module path, attribute name | None for the module itself).
_LAZY: dict[str, tuple[str, str | None]] = {
    "PGEN": ("genoray._pgen", "PGEN"),
    "VCF": ("genoray._vcf", "VCF"),
    "SparseVar": ("genoray._svar", "SparseVar"),
    "exprs": ("genoray.exprs", None),
}


def __getattr__(name: str):
    if name == "Reader":
        from ._pgen import PGEN
        from ._svar import SparseVar
        from ._vcf import VCF

        result = VCF | PGEN | SparseVar
        globals()[name] = result
        return result
    if name in _LAZY:
        mod_path, attr = _LAZY[name]
        mod = importlib.import_module(mod_path)
        result = mod if attr is None else getattr(mod, attr)
        globals()[name] = result
        return result
    raise AttributeError(f"module 'genoray' has no attribute {name!r}")


if TYPE_CHECKING:  # pragma: no cover
    from . import exprs as exprs
    from ._pgen import PGEN as PGEN
    from ._svar import SparseVar as SparseVar
    from ._vcf import VCF as VCF

    Reader = VCF | PGEN | SparseVar
