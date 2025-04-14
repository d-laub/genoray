from __future__ import annotations

from typing import Union

from ._pgen import PGEN
from ._vcf import VCF

Reader = Union[VCF, PGEN]

__all__ = ["Reader", "VCF", "PGEN"]
