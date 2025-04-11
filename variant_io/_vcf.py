from __future__ import annotations

from typing import Callable, Generator, Literal, cast, overload

import cyvcf2
import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing_extensions import Self

from ._types import GenoReader
from ._utils import ContigNormalizer, parse_memory


class DosageFieldError(RuntimeError): ...


class VCFReader(GenoReader):
    ploidy = 2
    filter: Callable[[cyvcf2.Variant], bool] | None
    dosage_field: str | None = None
    _vcf: cyvcf2.VCF
    _s_idx: NDArray[np.intp]
    _samples: list[str]
    _c_norm: ContigNormalizer

    def __init__(
        self,
        path: str,
        filter: Callable[[cyvcf2.Variant], bool] | None,
        dosage_field: str | None = None,
    ):
        self.path = path
        self.available_samples = self._vcf.samples
        self.contigs = self._vcf.seqnames
        self.filter = filter
        if dosage_field is True:
            dosage_field = "DS"
        self.dosage_field = dosage_field
        self._c_norm = ContigNormalizer(self.contigs)
        self.set_samples(self._vcf.samples)

    @property
    def current_samples(self) -> list[str]:
        return self._samples

    def set_samples(self, samples: list[str]) -> Self:
        if missing := set(samples).difference(self.available_samples):
            raise ValueError(
                f"Samples {missing} not found in the VCF file. "
                f"Available samples: {self.available_samples}"
            )
        self._vcf = self._open(samples)
        _, s_idx, _ = np.intersect1d(self._vcf.samples, samples, return_indices=True)
        self._samples = samples
        self._s_idx = s_idx
        return self

    def _open(self, samples: list[str] | None = None) -> cyvcf2.VCF:
        return cyvcf2.VCF(self.path, samples=samples, lazy=True)

    def __del__(self):
        self._vcf.close()

    def n_vars_in_ranges(
        self,
        contig: str,
        starts: ArrayLike = 0,
        ends: ArrayLike | None = None,
    ) -> NDArray[np.int64]:
        starts = np.atleast_1d(starts)
        ends = (
            np.full(len(starts), np.iinfo(np.int64).max)
            if ends is None
            else np.atleast_1d(ends)
        )

        c = self._c_norm.norm(contig)
        if c is None:
            return np.zeros(len(starts), dtype=np.int64)

        out = np.empty(len(starts), dtype=np.int64)
        for i, (s, e) in enumerate(zip(starts, ends)):
            coord = f"{c}:{s + 1}-{e}"
            if self.filter is None:
                out[i] = sum(1 for _ in self._vcf(coord))
            else:
                out[i] = sum(self.filter(v) for v in self._vcf(coord))

        return out

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
        if not genotypes and not dosages:
            raise ValueError("Either genotypes or dosage_field must be specified.")
        if dosages and self.dosage_field is None:
            raise ValueError(
                "Dosage field not specified. Set the VCF reader's `dosage_field` parameter."
            )

        c = self._c_norm.norm(contig)
        if c is None:
            if genotypes and not dosages:
                return np.empty((self.n_samples, self.ploidy, 0), dtype=np.int8)
            elif not genotypes and dosages:
                return np.empty((self.n_samples, 0), dtype=np.float32)
            else:
                return (
                    np.empty((self.n_samples, self.ploidy, 0), dtype=np.int8),
                    np.empty((self.n_samples, 0), dtype=np.float32),
                )

        if end is None:
            end = np.iinfo(np.int64).max

        itr = self._vcf(f"{c}:{start + 1}-{end}")  # region string is 1-based
        if out is None:
            n_variants: int = self.n_vars_in_ranges(c, start, end)[0]
            if n_variants == 0:
                if genotypes and not dosages:
                    return np.empty((self.n_samples, self.ploidy, 0), dtype=np.int8)
                elif not genotypes and dosages:
                    return np.empty((self.n_samples, 0), dtype=np.float32)
                else:
                    return (
                        np.empty((self.n_samples, self.ploidy, 0), dtype=np.int8),
                        np.empty((self.n_samples, 0), dtype=np.float32),
                    )

            if genotypes and not dosages:
                out = np.empty((self.n_samples, self.ploidy, n_variants), dtype=np.int8)
                self._fill_genos(itr, out)
            elif not genotypes and dosages:
                out = np.empty((self.n_samples, n_variants), dtype=np.float32)
                self._fill_dosages(
                    itr,
                    out,
                    self.dosage_field,  # type: ignore | guaranteed to be str by guard clause
                )
            else:
                out = (
                    np.empty((self.n_samples, self.ploidy, n_variants), dtype=np.int8),
                    np.empty((self.n_samples, n_variants), dtype=np.float32),
                )
                self._fill_genos_and_dosages(
                    itr,
                    out,
                    self.dosage_field,  # type: ignore | guaranteed to be str by guard clause
                )
        else:
            if genotypes and not dosages:
                out = cast(
                    NDArray[np.int8], out
                )  # type checker errors on call, so let it error out
                self._fill_genos(itr, out)
            elif not genotypes and dosages:
                out = cast(
                    NDArray[np.float32], out
                )  # type checker errors on call, so let it error out
                self._fill_dosages(
                    itr,
                    out,
                    self.dosage_field,  # type: ignore | guaranteed to be str by guard clause
                )
            else:
                out = cast(
                    tuple[NDArray[np.int8], NDArray[np.float32]], out
                )  # type checker errors on call, so let it error out
                self._fill_genos_and_dosages(
                    itr,
                    out,
                    self.dosage_field,  # type: ignore | guaranteed to be str by guard clause
                )

        return out

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
        if not genotypes and not dosages:
            raise ValueError("Either genotypes or dosage_field must be specified.")
        if dosages and self.dosage_field is None:
            raise ValueError(
                "Dosage field not specified. Set the VCF reader's `dosage_field` parameter."
            )

        max_mem = parse_memory(max_mem)

        c = self._c_norm.norm(contig)
        if c is None:
            yield np.empty((self.n_samples, self.ploidy, 0), dtype=np.int8)
            return

        if end is None:
            end = np.iinfo(np.int64).max

        n_variants: int = self.n_vars_in_ranges(c, start, end)[0]
        if n_variants == 0:
            yield np.empty((self.n_samples, self.ploidy, 0), dtype=np.int8)
            return

        mem_per_v = self._mem_per_variant(genotypes, dosages)
        vars_per_chunk = min(max_mem // mem_per_v, n_variants)
        if vars_per_chunk == 0:
            raise ValueError(
                f"Maximum memory {max_mem / 1e9:.2f} GB insufficient to read a single variant."
                f" Memory per variant: {mem_per_v / 1e9:.2f} GB."
            )

        n_chunks, final_chunk = divmod(n_variants, vars_per_chunk)
        if final_chunk == 0:
            # perfectly divisible so there is no final chunk
            chunk_sizes = np.full(n_chunks, vars_per_chunk)
        elif n_chunks == 0:
            # n_vars < vars_per_chunk, so we just use the remainder
            chunk_sizes = np.array([final_chunk])
        else:
            # have a final chunk that is smaller than the rest
            chunk_sizes = np.full(n_chunks + 1, vars_per_chunk)
            chunk_sizes[-1] = final_chunk

        itr = self._vcf(f"{c}:{start + 1}-{end}")  # region string is 1-based
        for chunk_size in chunk_sizes:
            if genotypes and not dosages:
                out = np.empty((self.n_samples, self.ploidy, chunk_size), dtype=np.int8)
                self._fill_genos(itr, out)
            elif not genotypes and dosages:
                out = np.empty((self.n_samples, chunk_size), dtype=np.float32)
                self._fill_dosages(
                    itr,
                    out,
                    self.dosage_field,  # type: ignore | guaranteed to be str by guard clause
                )
            else:
                out = (
                    np.empty((self.n_samples, self.ploidy, chunk_size), dtype=np.int8),
                    np.empty((self.n_samples, chunk_size), dtype=np.float32),
                )
                self._fill_genos_and_dosages(
                    itr,
                    out,
                    self.dosage_field,  # type: ignore | guaranteed to be str by guard clause
                )

    def _fill_genos(self, vcf: cyvcf2.VCF, out: NDArray[np.int8]):
        for i, v in enumerate(vcf):
            if self.filter is not None and not self.filter(v):
                continue

            out[..., i] = v.genotype.array()[:, : self.ploidy]

            if i == out.shape[-1] - 1:
                break
        else:
            raise ValueError("Not enough variants found in the given range.")

    def _fill_dosages(
        self, vcf: cyvcf2.VCF, out: NDArray[np.float32], dosage_field: str
    ):
        for i, v in enumerate(vcf):
            if self.filter is not None and not self.filter(v):
                continue
            d = v.format(dosage_field)
            if d is None:
                raise DosageFieldError(
                    f"Dosage field '{dosage_field}' not found for record {repr(v)}"
                )
            # (s, 1, 1) or (s, 1)? -> (s)
            out[..., i] = d.squeeze()

            if i == out.shape[-1] - 1:
                break
        else:
            raise ValueError("Not enough variants found in the given range.")

    def _fill_genos_and_dosages(
        self,
        vcf: cyvcf2.VCF,
        out: tuple[NDArray[np.int8], NDArray[np.float32]],
        dosage_field: str,
    ):
        for i, v in enumerate(vcf):
            if self.filter is not None and not self.filter(v):
                continue

            out[0][..., i] = v.genotype.array()[:, : self.ploidy]

            d = v.format(dosage_field)
            if d is None:
                raise DosageFieldError(
                    f"Dosage field '{dosage_field}' not found for record {repr(v)}"
                )
            # (s, 1, 1) or (s, 1)? -> (s)
            out[1][..., i] = d.squeeze()

            if i == out[0].shape[-1] - 1:
                break
        else:
            raise ValueError("Not enough variants found in the given range.")

    def _mem_per_variant(self, genotypes: bool, dosages: bool) -> int:
        """Calculate the memory required per variant for the given genotypes and dosages.

        Parameters
        ----------
        genotypes
            Whether to include genotypes.
        dosages
            Whether to include dosages.

        Returns
        -------
        int
            Memory required per variant in bytes.
        """
        return (self.n_samples * self.ploidy if genotypes else 0) + (
            self.n_samples * 4 if dosages else 0
        )
