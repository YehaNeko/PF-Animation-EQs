from __future__ import annotations

__all__ = ('HermiteParams', 'HermitePresets', 'F64NDArray')

from typing import TypeAlias, TypedDict

import numpy as np
from numpy.typing import NDArray


class HermiteParams(TypedDict):
    p0: float
    v0: float
    p1: float
    v1: float


F64NDArray: TypeAlias = NDArray[np.float64]
HermitePresets: TypeAlias = dict[str, HermiteParams]
