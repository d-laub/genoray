import numpy as np
import polars as pl

from ._ragged import Ragged

IDX_TYPE = np.uint32


class SparseVar:
    genos: Ragged[IDX_TYPE]
    records: pl.DataFrame
