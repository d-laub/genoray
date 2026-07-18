from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def hap_ilens(
    genotypes: NDArray[np.integer], ilens: NDArray[np.int32]
) -> NDArray[np.int32]:
    """Get the indel lengths of haplotypes from genotypes i.e. the difference in their lengths compared to the reference sequence.

    Assumes phased genotypes.

    Args:
        genotypes: Genotypes array. Shape: (samples, ploidy, variants).
        ilens: Lengths of the segments. Shape: (variants).

    Returns:
        hap_lengths: Lengths of the haplotypes. Shape: (samples, ploidy).
    """
    # (s p v)
    ilens = np.broadcast_to(ilens, genotypes.shape)  # zero-copy, read only
    # (s p v) -> (s p)
    return ilens.sum(-1, dtype=np.int32, where=genotypes == 1)
