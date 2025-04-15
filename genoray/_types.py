from __future__ import annotations

import numpy as np

R_DTYPE = np.uint64
"""Dtype for range indices. This determines the maximum size of a contig in genoray."""

UINT64_MAX = np.iinfo(R_DTYPE).max
"""Maximum value for a 64-bit unsigned integer."""