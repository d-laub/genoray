from __future__ import annotations

import logging
from importlib.metadata import version
from typing import Union

from . import exprs
from ._pgen import PGEN
from ._svar import SparseGenotypes, SparseVar
from ._vcf import VCF

Reader = Union[VCF, PGEN, SparseVar]

__version__ = version("genoray")

__all__ = ["Reader", "VCF", "PGEN", "SparseVar", "SparseGenotypes", "exprs"]

logger = logging.getLogger("polars_bio")
logger.setLevel(logging.ERROR)
