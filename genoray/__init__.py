from __future__ import annotations

import logging
from importlib.metadata import version

from . import exprs
from ._pgen import PGEN
from ._svar import SparseVar
from ._vcf import VCF

Reader = VCF | PGEN | SparseVar

__version__ = version("genoray")

__all__ = ["PGEN", "VCF", "Reader", "SparseVar", "exprs"]

logger = logging.getLogger("polars_bio")
logger.setLevel(logging.ERROR)
