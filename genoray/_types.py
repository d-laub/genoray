from typing import TypeVar

import numpy as np
import polars as pl

POS_TYPE = np.int64
V_IDX_TYPE = np.int32
POLARS_V_IDX_TYPE = pl.Int32
DOSAGE_TYPE = np.float32
INT64_MAX = np.iinfo(POS_TYPE).max
DTYPE = TypeVar("DTYPE", bound=np.generic)
