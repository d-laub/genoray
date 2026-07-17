from __future__ import annotations

import types
from typing import Any

import numpy as np
from numpy.typing import NDArray
from phantom import Phantom


def make_array_mode(
    name: str,
    dtype: type[np.generic],
    ndim: int,
    *,
    genos: bool = False,
) -> type:
    """Build a phantom ``NDArray`` mode class.

    Args:
        name: The generated class ``__name__``.
        dtype: NumPy scalar type the array must have (e.g. ``np.int8``).
        ndim: Required number of dimensions (3 for genotype arrays, 2 for
            dosage/phasing-style arrays).
        genos: If True, the ploidy axis (``shape[1]``) must be in ``(2, 3)`` and
            ``empty`` allocates a 3D ``(n_samples, ploidy, n_variants)`` array.
    """

    def predicate(obj: Any) -> bool:
        if not (
            isinstance(obj, np.ndarray) and obj.dtype.type == dtype and obj.ndim == ndim
        ):
            return False
        if genos:
            return obj.shape[1] in (2, 3)
        return True

    def empty(cls: type[Any], n_samples: int, ploidy: int, n_variants: int):
        shape = (
            (n_samples, ploidy, n_variants) if ndim == 3 else (n_samples, n_variants)
        )
        return cls.parse(np.empty(shape, dtype=dtype))

    def nbytes_per_variant(cls: type[Any], n_samples: int, ploidy: int) -> int:
        axis = ploidy if ndim == 3 else 1
        return n_samples * axis * np.dtype(dtype).itemsize

    def exec_body(ns: dict[str, Any]) -> dict[str, Any]:
        ns["_dtype"] = dtype
        ns["_gdtype"] = dtype
        ns["empty"] = classmethod(empty)
        ns["nbytes_per_variant"] = classmethod(nbytes_per_variant)
        return ns

    return types.new_class(
        name, (NDArray[dtype], Phantom), {"predicate": predicate}, exec_body
    )


def make_tuple_mode(
    name: str,
    components: tuple[type, ...],
    *,
    genos_dtype: type[np.generic],
) -> type:
    """Build a phantom tuple-of-modes class (e.g. ``(Genos, Dosages)``)."""

    def predicate(obj: Any) -> bool:
        return (
            isinstance(obj, tuple)
            and len(obj) == len(components)
            and all(isinstance(o, c) for o, c in zip(obj, components))
        )

    def empty(cls: type[Any], n_samples: int, ploidy: int, n_variants: int):
        return cls.parse(
            tuple(c.empty(n_samples, ploidy, n_variants) for c in components)
        )

    def nbytes_per_variant(cls: type[Any], n_samples: int, ploidy: int) -> int:
        return sum(c.nbytes_per_variant(n_samples, ploidy) for c in components)

    def exec_body(ns: dict[str, Any]) -> dict[str, Any]:
        ns["_dtype"] = genos_dtype
        ns["_gdtype"] = genos_dtype
        ns["_dtypes"] = tuple(c._dtype for c in components)
        ns["empty"] = classmethod(empty)
        ns["nbytes_per_variant"] = classmethod(nbytes_per_variant)
        return ns

    return types.new_class(
        name,
        (tuple[components], Phantom),  # pyrefly: ignore [not-a-type]
        {"predicate": predicate},
        exec_body,
    )
