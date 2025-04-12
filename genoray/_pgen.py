from __future__ import annotations

from pathlib import Path
from typing import Generator, Literal, cast, overload

import numpy as np
import pgenlib
import polars as pl
import pyranges as pr
from hirola import HashTable
from numpy.typing import ArrayLike, NDArray
from typing_extensions import Self

from ._types import Reader
from ._utils import ContigNormalizer, format_memory, lengths_to_offsets, parse_memory

# TODO: the index could likely be implemented using the NCLS lib underlying PyRanges and then we can
# pass np.memmap arrays directly instead of having to futz with DataFrames. This will likely make
# filtering less ergonomic/harder to make ergonomic though, but a memmap approach will be much more
# scalable.


class PGEN(Reader[np.int32]):
    available_samples: list[str]
    filter: pl.Expr | None
    ploidy = 2
    _index: pr.PyRanges
    _pgen: pgenlib.PgenReader
    _s_idx: NDArray[np.uint32]

    def __init__(self, path: str | Path, filter: pl.Expr | None = None):
        path = Path(path)
        samples = _read_psam(path.with_suffix(".psam"))
        self.filter = filter
        self._s2i = HashTable(
            max=len(self.available_samples) * 2,  # type: ignore
            dtype=samples.dtype,
        )
        self._s2i.add(samples)
        self._s_idx = np.arange(len(samples), dtype=np.uint32)
        self.available_samples = samples.tolist()
        self._pgen = pgenlib.PgenReader(bytes(path))
        if not path.with_suffix(".gvi").exists():
            _write_index(path.with_suffix(".pvar"))
        self._index = _read_index(path.with_suffix(".gvi"), self.filter)
        self._c_norm = ContigNormalizer(self._index.chromosomes)

    @property
    def current_samples(self) -> list[str]:
        return self._s2i.keys[self._s_idx].tolist()

    def set_samples(self, samples: list[str]) -> Self:
        _samples = np.atleast_1d(samples)
        s_idx = self._s2i.get(_samples).astype(np.uint32)
        if (missing := _samples[s_idx == -1]).any():
            raise ValueError(f"Samples {missing} not found in the file.")
        self._s_idx = s_idx
        self._pgen.change_sample_subset(np.sort(s_idx))
        return self

    def __del__(self):
        self._pgen.close()

    def n_vars_in_ranges(
        self,
        contig: str,
        starts: ArrayLike = 0,
        ends: ArrayLike | None = None,
    ) -> NDArray[np.int64]:
        c = self._c_norm.norm(contig)
        if c is None:
            return np.zeros_like(np.atleast_1d(starts), dtype=np.int64)

        starts = np.atleast_1d(starts)
        if ends is None:
            ends = np.full_like(starts, np.iinfo(np.int32).max)
        queries = pr.PyRanges(
            pl.DataFrame(
                {
                    "Chromosome": np.full_like(starts, contig),
                    "Start": starts,
                    "End": ends,
                }
            )
            .with_row_index("query")
            .to_pandas(use_pyarrow_extension_array=True)
        )
        return queries.count_overlaps(self._index).df["NumberOverlaps"].to_numpy()

    def _var_idxs(
        self, contig: str, starts: ArrayLike = 0, ends: ArrayLike | None = None
    ) -> tuple[NDArray[np.uint32], NDArray[np.uint64]]:
        """Get variant indices and the number of indices per region.

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
        idxs
            Shape: (tot_variants). Variant indices for the given ranges.
        offsets
            Shape: (regions+1). Offsets to get variant indices for each region.
        """
        starts = np.atleast_1d(starts)

        c = self._c_norm.norm(contig)
        if c is None:
            return np.empty(0, np.uint32), np.zeros_like(
                np.atleast_1d(starts), np.uint64
            )

        starts = np.atleast_1d(starts)
        if ends is None:
            ends = np.full_like(starts, np.iinfo(np.int32).max)
        queries = pr.PyRanges(
            pl.DataFrame(
                {
                    "Chromosome": np.full_like(starts, contig),
                    "Start": starts,
                    "End": ends,
                }
            )
            .with_row_index("query")
            .to_pandas(use_pyarrow_extension_array=True)
        )
        join = pl.from_pandas(queries.join(self._index).df).sort("query", "index")
        idxs = join["index"].to_numpy()
        lens = (
            join.group_by("query", maintain_order=True).agg(pl.len())["len"].to_numpy()
        )
        offsets = lengths_to_offsets(lens)
        return idxs, offsets

    @overload
    def read(
        self,
        contig: str,
        start: int = 0,
        end: int | None = None,
        *,
        genotypes: Literal[True] = ...,
        dosages: Literal[False] = ...,
        out: NDArray[np.int32] | None,
    ) -> NDArray[np.int32]: ...
    @overload
    def read(
        self,
        contig: str,
        start: int = 0,
        end: int | None = None,
        *,
        genotypes: Literal[False],
        dosages: Literal[True],
        out: NDArray[np.float32] | None,
    ) -> NDArray[np.float32]: ...
    @overload
    def read(
        self,
        contig: str,
        start: int = 0,
        end: int | None = None,
        *,
        genotypes: Literal[True] = ...,
        dosages: Literal[True],
        out: tuple[NDArray[np.int32], NDArray[np.float32]] | None,
    ) -> tuple[NDArray[np.int32], NDArray[np.float32]]: ...
    @overload
    def read(
        self,
        contig: str,
        start: int = 0,
        end: int | None = None,
        *,
        genotypes: bool = True,
        dosages: bool = False,
        out: NDArray[np.int32 | np.float32]
        | tuple[NDArray[np.int32], NDArray[np.float32]]
        | None = None,
    ) -> (
        NDArray[np.int32 | np.float32] | tuple[NDArray[np.int32], NDArray[np.float32]]
    ): ...
    def read(
        self,
        contig: str,
        start: int = 0,
        end: int | None = None,
        *,
        genotypes: bool = True,
        dosages: bool = False,
        out: NDArray[np.int32 | np.float32]
        | tuple[NDArray[np.int32], NDArray[np.float32]]
        | None = None,
    ) -> NDArray[np.int32 | np.float32] | tuple[NDArray[np.int32], NDArray[np.float32]]:
        if not genotypes and not dosages:
            raise ValueError("Either genotypes or dosage_field must be specified.")

        if dosages:
            raise NotImplementedError("Dosages are not yet supported.")

        c = self._c_norm.norm(contig)
        if c is None:
            if genotypes and not dosages:
                return np.empty((self.n_samples, self.ploidy, 0), dtype=np.int32)
            elif not genotypes and dosages:
                return np.empty((self.n_samples, 0), dtype=np.float32)
            else:
                return (
                    np.empty((self.n_samples, self.ploidy, 0), dtype=np.int32),
                    np.empty((self.n_samples, 0), dtype=np.float32),
                )

        if end is None:
            end = np.iinfo(np.int64).max

        var_idxs, _ = self._var_idxs(c, start, end)
        n_variants = len(var_idxs)

        if out is None:
            out = np.empty((n_variants, self.n_samples * self.ploidy), dtype=np.int32)

        out = cast(NDArray[np.int32], out)  # not implementing dosages yet

        self._pgen.read_alleles_list(var_idxs, out)
        out = out.reshape(n_variants, self.n_samples, self.ploidy).transpose(1, 2, 0)[
            self._s_idx
        ]

        return out

    @overload
    def read_chunks(
        self,
        contig: str,
        start: int = 0,
        end: int | None = None,
        max_mem: int | str = "4g",
        *,
        genotypes: Literal[True] = ...,
        dosages: Literal[False] = ...,
    ) -> Generator[NDArray[np.int32]]: ...
    @overload
    def read_chunks(
        self,
        contig: str,
        start: int = 0,
        end: int | None = None,
        max_mem: int | str = "4g",
        *,
        genotypes: Literal[False],
        dosages: Literal[True],
    ) -> Generator[NDArray[np.float32]]: ...
    @overload
    def read_chunks(
        self,
        contig: str,
        start: int = 0,
        end: int | None = None,
        max_mem: int | str = "4g",
        *,
        genotypes: Literal[True] = ...,
        dosages: Literal[True],
    ) -> Generator[tuple[NDArray[np.int32], NDArray[np.float32]]]: ...
    @overload
    def read_chunks(
        self,
        contig: str,
        start: int = 0,
        end: int | None = None,
        max_mem: int | str = "4g",
        *,
        genotypes: bool = True,
        dosages: bool = False,
    ) -> Generator[
        NDArray[np.int32 | np.float32] | tuple[NDArray[np.int32], NDArray[np.float32]]
    ]: ...
    def read_chunks(
        self,
        contig: str,
        start: int = 0,
        end: int | None = None,
        max_mem: int | str = "4g",
        *,
        genotypes: bool = True,
        dosages: bool = False,
    ) -> Generator[
        NDArray[np.int32 | np.float32] | tuple[NDArray[np.int32], NDArray[np.float32]]
    ]:
        if dosages:
            raise NotImplementedError("Dosages are not yet supported.")

        max_mem = parse_memory(max_mem)

        c = self._c_norm.norm(contig)
        if c is None:
            yield np.empty((self.n_samples, self.ploidy, 0), dtype=np.int32)
            return

        if end is None:
            end = np.iinfo(np.int64).max

        var_idxs, _ = self._var_idxs(c, start, end)
        n_variants = len(var_idxs)
        if n_variants == 0:
            yield np.empty((self.n_samples, self.ploidy, 0), dtype=np.int32)
            return

        mem_per_v = self._mem_per_variant(genotypes, dosages)
        vars_per_chunk = min(max_mem // mem_per_v, n_variants)
        if vars_per_chunk == 0:
            raise ValueError(
                f"Maximum memory {format_memory(max_mem)} insufficient to read a single variant."
                f" Memory per variant: {format_memory(mem_per_v)}."
            )

        n_chunks = -(-n_variants // vars_per_chunk)
        v_chunks = np.array_split(var_idxs, n_chunks)
        for var_idx in v_chunks:
            chunk_size = len(var_idx)
            out = np.empty((self.n_samples, self.ploidy, chunk_size), dtype=np.int32)
            self._pgen.read_alleles_list(var_idx, out)
            out = out.reshape(chunk_size, self.n_samples, self.ploidy).transpose(
                1, 2, 0
            )[self._s_idx]
            yield out

    @overload
    def read_ranges(
        self,
        contig: str,
        starts: ArrayLike = 0,
        ends: ArrayLike | None = None,
        *,
        genotypes: Literal[True] = ...,
        dosages: Literal[False] = ...,
    ) -> tuple[NDArray[np.int32], NDArray[np.uint32]]: ...
    @overload
    def read_ranges(
        self,
        contig: str,
        starts: ArrayLike = 0,
        ends: ArrayLike | None = None,
        *,
        genotypes: Literal[False],
        dosages: Literal[True],
    ) -> tuple[NDArray[np.float32], NDArray[np.uint32]]: ...
    @overload
    def read_ranges(
        self,
        contig: str,
        starts: ArrayLike = 0,
        ends: ArrayLike | None = None,
        *,
        genotypes: Literal[True] = ...,
        dosages: Literal[True],
    ) -> tuple[NDArray[np.int32], NDArray[np.float32], NDArray[np.uint32]]: ...
    @overload
    def read_ranges(
        self,
        contig: str,
        starts: ArrayLike = 0,
        ends: ArrayLike | None = None,
        *,
        genotypes: bool = True,
        dosages: bool = False,
    ) -> (
        tuple[NDArray[np.int32 | np.float32], NDArray[np.uint32]]
        | tuple[NDArray[np.int32], NDArray[np.float32], NDArray[np.uint32]]
    ): ...
    def read_ranges(
        self,
        contig: str,
        starts: ArrayLike = 0,
        ends: ArrayLike | None = None,
        *,
        genotypes: bool = True,
        dosages: bool = False,
    ) -> (
        tuple[NDArray[np.int32 | np.float32], NDArray[np.uint32]]
        | tuple[NDArray[np.int32], NDArray[np.float32], NDArray[np.uint32]]
    ):
        if dosages:
            raise NotImplementedError("Dosages are not yet supported.")

        starts = np.atleast_1d(starts)

        c = self._c_norm.norm(contig)
        if c is None:
            return np.empty(
                (self.n_samples, self.ploidy, 0), dtype=np.int32
            ), np.zeros_like(starts, dtype=np.uint32)

        var_idxs, offsets = self._var_idxs(c, starts, ends)
        n_variants = len(var_idxs)
        if n_variants == 0:
            return np.empty(
                (self.n_samples, self.ploidy, 0), dtype=np.int32
            ), np.zeros_like(starts, dtype=np.uint32)

        out = np.empty((n_variants, self.n_samples * self.ploidy), dtype=np.int32)

        self._pgen.read_alleles_list(var_idxs, out)
        out = out.reshape(n_variants, self.n_samples, self.ploidy).transpose(1, 2, 0)[
            self._s_idx
        ]

        return out, np.diff(offsets).astype(np.uint32)

    def _mem_per_variant(self, genotypes: bool, dosages: bool) -> int:
        if dosages:
            raise NotImplementedError("Dosages are not yet supported.")
        return self.n_samples * self.ploidy * np.int32().itemsize


def _read_psam(path: Path) -> NDArray[np.str_]:
    with open(path.with_suffix(".psam")) as f:
        cols = [c.strip("#") for c in f.readline().strip().split()]

    psam = pl.read_csv(
        path.with_suffix(".psam"),
        separator="\t",
        has_header=False,
        skip_rows=1,
        new_columns=cols,
        schema_overrides={
            "FID": pl.Utf8,
            "IID": pl.Utf8,
            "SID": pl.Utf8,
            "PAT": pl.Utf8,
            "MAT": pl.Utf8,
            "SEX": pl.Utf8,
        },
    )
    samples = psam["IID"].to_numpy()
    return samples


RLEN = pl.col("REF").str.len_bytes()
ALEN = pl.col("ALT").str.len_bytes()
ILEN = ALEN - RLEN
KIND = (
    pl.when(ILEN != 0)
    .then(pl.lit("INDEL"))
    .when(RLEN == 1)
    .then(pl.lit("SNP"))
    .otherwise(pl.lit("MNP"))
    .cast(pl.Categorical)
)


def _write_index(path: Path):
    (
        pl.scan_csv(
            path.with_suffix(".pvar"),
            separator="\t",
            comment_prefix="##",
            schema_overrides={"#CHROM": pl.Utf8, "POS": pl.Int32},
        )
        .select(
            Chromosome="#CHROM",
            Start=pl.col("POS") - 1,
            End=pl.col("POS"),
            kind=KIND,
        )
        .sink_ipc(path.with_suffix(".gvi"))
    )


def _read_index(path: Path, filter: pl.Expr | None) -> pr.PyRanges:
    index = pl.read_ipc(path, row_index_name="index")
    if filter is not None:
        index = index.filter(filter)
    pyr = pr.PyRanges(index.drop("kind").to_pandas(use_pyarrow_extension_array=True))
    return pyr
