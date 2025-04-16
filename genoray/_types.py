from __future__ import annotations

import numpy as np

VCF_R_DTYPE = np.uint64
"""Dtype for VCF range indices. This determines the maximum size of a contig in genoray."""
PGEN_R_DTYPE = np.int64
"""Dtype for PGEN range indices. This determines the maximum size of a contig in genoray. We have to use int64 because this is what PyRanges uses!"""

UINT64_MAX = np.iinfo(VCF_R_DTYPE).max
"""Maximum value for a 64-bit unsigned integer."""

INT64_MAX = np.iinfo(PGEN_R_DTYPE).max
