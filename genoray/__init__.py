from __future__ import annotations

from importlib.metadata import version
from typing import Union

from ._pgen import PGEN
from ._vcf import VCF

Reader = Union[VCF, PGEN]

__version__ = version("genoray")

__all__ = ["Reader", "VCF", "PGEN"]
