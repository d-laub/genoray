from __future__ import annotations

from typing import Any, Generator, Generic, Protocol, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing_extensions import Self

T = TypeVar("T")
R_DTYPE = np.uint64
"""Dtype for range indices. This determines the maximum size of a contig in genoray."""


class Reader(Protocol, Generic[T]):
    available_samples: list[str]
    """All samples in the file, in the order they exist on-disk."""
    ploidy: int
    filter: Any | None
    contigs: list[str]

    @property
    def current_samples(self) -> list[str]:
        """The samples this reader will return, in order along the sample axis."""
        ...

    def set_samples(self, samples: list[str]) -> Self:
        """Set the samples this reader will return, in order along the sample axis."""
        ...

    @property
    def n_samples(self) -> int:
        return len(self.current_samples)

    def n_vars_in_ranges(
        self,
        contig: str,
        starts: ArrayLike = 0,
        ends: ArrayLike = np.iinfo(R_DTYPE).max,
    ) -> NDArray[np.uint32]:
        """Return the start and end indices of the variants in the given ranges.

        Parameters
        ----------
        contig
            Contig name.
        starts
            0-based start positions of the ranges.
        ends
            0-based, exclusive end positions of the ranges.

        Returns
        -------
        n_variants
            Shape: :code:`(ranges)`. Number of variants in the given ranges.
        """
        ...

    def read(
        self,
        contig: str,
        start: int | np.integer = 0,
        end: int | np.integer = np.iinfo(R_DTYPE).max,
        out: T | None = None,
    ) -> T | None:
        """Read genotypes and/or dosages for a range.

        Parameters
        ----------
        contig
            Contig name.
        start
            0-based start position.
        end
            0-based, exclusive end position.

        Returns
        -------
            Genotypes and/or dosages. Genotypes have shape :code:`(samples ploidy variants)` and
            dosages have shape :code:`(samples variants)`. Missing genotypes have value -1 and missing dosages
            have value np.nan. If just using genotypes or dosages, will be a single array, otherwise
            will be a tuple of arrays.
        """
        ...

    def read_chunks(
        self,
        contig: str,
        start: int | np.integer = 0,
        end: int | np.integer = np.iinfo(R_DTYPE).max,
        max_mem: int | str = "4g",
    ) -> Generator[T]:
        """Iterate over genotypes and/or dosages for a range in chunks limited by :code:`max_mem`.

        Parameters
        ----------
        contig
            Contig name.
        start
            0-based start position.
        end
            0-based, exclusive end position.
        max_mem
            Maximum memory to use for each chunk. Can be an integer or a string with a suffix
            (e.g. "4g", "2 MB").

        Returns
        -------
            Generator of genotypes and/or dosages. Genotypes have shape :code:`(samples ploidy variants)` and
            dosages have shape :code:`(samples variants)`. Missing genotypes have value -1 and missing dosages
            have value np.nan. If just using genotypes or dosages, will be a single array, otherwise
            will be a tuple of arrays.
        """
        ...

    def read_ranges(
        self,
        contig: str,
        starts: ArrayLike = 0,
        ends: ArrayLike = np.iinfo(R_DTYPE).max,
    ) -> tuple[T, NDArray[np.uint64]] | None:
        """Read genotypes and/or dosages for multiple ranges.

        Parameters
        ----------
        contig
            Contig name.
        starts
            0-based start positions.
        ends
            0-based, exclusive end positions.

        Returns
        -------
        data
            Genotypes and/or dosages. Genotypes have shape :code:`(samples ploidy variants)` and
            dosages have shape :code:`(samples variants)`. Missing genotypes have value -1 and missing dosages
            have value np.nan. If just using genotypes or dosages, will be a single array, otherwise
            will be a tuple of arrays.
        offsets
            Shape: (ranges+1). Offsets to slice out data for each range from the variants axis like so:

        Examples
        --------
        .. code-block:: python

            data, offsets = reader.read_ranges(...)
            data[..., offsets[i] : offsets[i + 1]]  # data for range i

        Note that the number of variants for range :code:`i` is :code:`np.diff(offsets)[i]`.
        """
        ...

    def read_ranges_chunks(
        self,
        contig: str,
        starts: ArrayLike = 0,
        ends: ArrayLike = np.iinfo(R_DTYPE).max,
        max_mem: int | str = "4g",
    ) -> Generator[Generator[T]]:
        """Read genotypes and/or dosages for multiple ranges in chunks limited by :code:`max_mem`.

        Parameters
        ----------
        contig
            Contig name.
        starts
            0-based start positions.
        ends
            0-based, exclusive end positions.

        Returns
        -------
            Generator of generators of genotypes and/or dosages of each ranges' data. Genotypes have shape :code:`(samples ploidy variants)` and
            dosages have shape :code:`(samples variants)`. Missing genotypes have value -1 and missing dosages
            have value np.nan. If just using genotypes or dosages, will be a single array, otherwise
            will be a tuple of arrays.

        Examples
        --------
        .. code-block:: python

            gen = reader.read_ranges_chunks(...)
            for range_ in gen:
                for chunk in range_:
                    # do something with chunk
                    pass
        """
        ...
