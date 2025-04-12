from __future__ import annotations

from typing import Any, Generator, Generic, Protocol, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing_extensions import Self

T = TypeVar("T")


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
        self, contig: str, starts: ArrayLike = 0, ends: ArrayLike | None = None
    ) -> NDArray[np.uint32]:
        """Return the start and end indices of the variants in the given ranges.

        Parameters
        ----------
        contig
            Contig name.
        starts
            0-based start positions of the regions.
        ends
            0-based, exclusive end positions of the regions.

        Returns
        -------
        n_variants
            Shape: (regions). Number of variants in the given ranges.
        """
        ...

    def read(
        self,
        contig: str,
        start: int = 0,
        end: int | None = None,
        out: T | None = None,
    ) -> T | None:
        """Read genotypes and/or dosages for a region.

        Parameters
        ----------
        contig
            Contig name.
        start
            0-based start position of the region.
        end
            0-based, exclusive end position of the region.
        samples
            Samples to read. If None, all samples are read.
        ploids
            Ploids to read. If None, all ploids are read.
        dosage_field
            Dosage field to read. If True, use the default dosage field for the format.

        Returns
        -------
        data
            Genotypes and/or dosages. Genotypes have shape (samples ploidy variants) and
            dosages have shape (samples variants). Missing genotypes have value -1 and missing dosages
            have value np.nan. If just using genotypes or dosages, will be a single array, otherwise
            will be a tuple of arrays.
        dosage
            Shape: (samples variants)
        """
        ...

    def read_chunks(
        self,
        contig: str,
        start: int = 0,
        end: int | None = None,
        max_mem: int | str = "4g",
    ) -> Generator[T]:
        """Iterate over genotypes and/or dosages for a region in chunks limited by max_mem.

        Parameters
        ----------
        contig
            Contig name.
        start
            0-based start position.
        end
            0-based, exclusive end position of the region.
        samples
            Samples to read. If None, all samples are read.
        ploids
            Ploids to read. If None, all ploids are read.

        Returns
        -------
        data
            Generator of genotypes and/or dosages. Genotypes have shape (samples ploidy variants) and
            dosages have shape (samples variants). Missing genotypes have value -1 and missing dosages
            have value np.nan. If just using genotypes or dosages, will be a single array, otherwise
            will be a tuple of arrays.
        """
        ...

    def read_ranges(
        self,
        contig: str,
        starts: ArrayLike = 0,
        ends: ArrayLike | None = None,
    ) -> tuple[T, NDArray[np.uint32]] | None:
        """Read genotypes and/or dosages for multiple regions.

        Parameters
        ----------
        contig
            Contig name.
        start
            0-based start position of the region.
        end
            0-based, exclusive end position of the region.
        samples
            Samples to read. If None, all samples are read.
        ploids
            Ploids to read. If None, all ploids are read.
        dosage_field
            Dosage field to read. If True, use the default dosage field for the format.

        Returns
        -------
        data
            Genotypes and/or dosages. Genotypes have shape (samples ploidy variants) and
            dosages have shape (samples variants). Missing genotypes have value -1 and missing dosages
            have value np.nan. If just using genotypes or dosages, will be a single array, otherwise
            will be a tuple of arrays.
        n_variants_per_region
            Shape: (regions). Number of variants in the given ranges.
        """
        ...
