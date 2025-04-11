from __future__ import annotations

from typing import Callable, Generator, Literal, Protocol, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing_extensions import Self


class GenoReader(Protocol):
    available_samples: list[str]
    """All samples in the file, in the order they exist on-disk."""
    ploidy: int
    filter: Callable | None

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
    ) -> NDArray[np.int64]:
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

    @overload
    def read(
        self,
        contig: str,
        start: int = 0,
        end: int | None = None,
        *,
        genotypes: Literal[True] = ...,
        dosages: Literal[False] = ...,
        out: NDArray[np.int8] | None = ...,
    ) -> NDArray[np.int8]: ...
    @overload
    def read(
        self,
        contig: str,
        start: int = 0,
        end: int | None = None,
        *,
        genotypes: Literal[False] = ...,
        dosages: Literal[True] = ...,
        out: NDArray[np.float32] | None = ...,
    ) -> NDArray[np.float32]: ...
    @overload
    def read(
        self,
        contig: str,
        start: int = 0,
        end: int | None = None,
        *,
        genotypes: Literal[True] = ...,
        dosages: Literal[True] = ...,
        out: tuple[NDArray[np.int8], NDArray[np.float32]] | None = ...,
    ) -> tuple[NDArray[np.int8], NDArray[np.float32]]: ...
    def read(
        self,
        contig: str,
        start: int = 0,
        end: int | None = None,
        *,
        genotypes: bool = True,
        dosages: bool = False,
        out: NDArray[np.int8 | np.float32]
        | tuple[NDArray[np.int8], NDArray[np.float32]]
        | None = None,
    ) -> NDArray[np.int8 | np.float32] | tuple[NDArray[np.int8], NDArray[np.float32]]:
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
        genotypes
            Shape: (samples ploidy variants)
        dosage
            Shape: (samples variants)
        """
        ...

    @overload
    def read_chunks(
        self,
        contig: str,
        start: int = 0,
        end: int | None = None,
        *,
        genotypes: Literal[True] = ...,
        dosages: Literal[False] = ...,
    ) -> Generator[NDArray[np.int8]]: ...
    @overload
    def read_chunks(
        self,
        contig: str,
        start: int = 0,
        end: int | None = None,
        *,
        genotypes: Literal[False] = ...,
        dosages: Literal[True] = ...,
    ) -> Generator[NDArray[np.float32]]: ...
    @overload
    def read_chunks(
        self,
        contig: str,
        start: int = 0,
        end: int | None = None,
        *,
        genotypes: Literal[True] = ...,
        dosages: Literal[True] = ...,
    ) -> Generator[tuple[NDArray[np.int8], NDArray[np.float32]]]: ...
    def read_chunks(
        self,
        contig: str,
        start: int = 0,
        end: int | None = None,
        max_mem: int | str = "4g",
        *,
        genotypes: bool = True,
        dosages: bool = False,
    ) -> (
        Generator[NDArray[np.int8 | np.float32]]
        | Generator[tuple[NDArray[np.int8], NDArray[np.float32]]]
    ):
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
            dosages have shape (samples variants).
        """
        ...
