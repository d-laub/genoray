from typing import TypeVar

import numpy as np

POS_TYPE = np.int32
POS_MAX = np.iinfo(POS_TYPE).max
V_IDX_TYPE = np.int32
DOSAGE_TYPE = np.float32
DTYPE = TypeVar("DTYPE", bound=np.generic)
