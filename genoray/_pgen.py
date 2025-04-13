from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Generator, TypeVar, cast

import numpy as np
import pgenlib
import polars as pl
import pyranges as pr
from hirola import HashTable
from numpy.typing import ArrayLike, NDArray
from phantom import Phantom
from typing_extensions import Self, TypeGuard, assert_never

from ._types import R_DTYPE, Reader
from ._utils import (
    ContigNormalizer,
    format_memory,
    is_dtype,
    lengths_to_offsets,
    parse_memory,
)


def _is_genos_dosages(obj) -> TypeGuard[tuple[Genos, Dosages]]:
    """Check if the object is a tuple of genotypes and dosages.

    Parameters
    ----------
    obj
        Object to check.

    Returns
    -------
    bool
        True if the object is a tuple of genotypes and dosages, False otherwise.
    """
    return (
        isinstance(obj, tuple)
        and len(obj) == 2
        and isinstance(obj[0], Genos)
        and isinstance(obj[1], Dosages)
    )


class Genos(
    NDArray[np.int32], Phantom, predicate=partial(is_dtype, dtype=np.int32)
): ...


class Dosages(
    NDArray[np.float32], Phantom, predicate=partial(is_dtype, dtype=np.float32)
): ...


class GenosDosages(tuple[Genos, Dosages], Phantom, predicate=_is_genos_dosages): ...


T = TypeVar(
    "T",
    Genos,
    Dosages,
    GenosDosages,
)


class PGEN(Reader[T]):
    available_samples: list[str]
    filter: pl.Expr | None
    ploidy = 2
    contigs: list[str]
    _index: pr.PyRanges
    _geno_pgen: pgenlib.PgenReader
    _dose_pgen: pgenlib.PgenReader
    _s_idx: NDArray[np.uint32]
    _read_as: type[T]

    Genos = Genos
    Dosages = Dosages
    GenosDosages = GenosDosages

    def __init__(
        self,
        geno_path: str | Path,
        filter: pl.Expr | None = None,
        read_as: type[T] = Genos,
        dosage_path: str | Path | None = None,
    ):
        """Create a PGEN reader.

        Parameters
        ----------
        path
            Path to the PGEN file. Only used for genotypes if a dosage path is provided as well.
        filter
            Polars expression to filter variants. Should return True for variants to keep.
        read_as
            Type of data to read from the PGEN file. Can be PGEN.Genos, PGEN.Dosages, or PGEN.GenosDosages.
        dosage_path
            Path to a dosage PGEN file. If None, the genotype PGEN file will be used for both genotypes and dosages.
        """

        geno_path = Path(geno_path)
        samples = _read_psam(geno_path.with_suffix(".psam"))

        self.filter = filter
        self.available_samples = cast(list[str], samples.tolist())
        self._s2i = HashTable(
            max=len(samples) * 2,  # type: ignore
            dtype=samples.dtype,
        )
        self._s2i.add(samples)
        self._s_idx = np.arange(len(samples), dtype=np.uint32)
        self._geno_pgen = pgenlib.PgenReader(bytes(geno_path), len(samples))

        if dosage_path is not None:
            dosage_path = Path(dosage_path)
            dose_samples = _read_psam(dosage_path.with_suffix(".psam"))
            if (samples != dose_samples).any():
                raise ValueError(
                    "Samples in dosage file do not match those in genotype file."
                )
            self._dose_pgen = pgenlib.PgenReader(bytes(Path(dosage_path)))
        else:
            self._dose_pgen = self._geno_pgen

        if not geno_path.with_suffix(".gvi").exists():
            _write_index(geno_path.with_suffix(".pvar"))
        self._index = _read_index(geno_path.with_suffix(".gvi"), self.filter)
        self.contigs = self._index.chromosomes
        self._c_norm = ContigNormalizer(self._index.chromosomes)
        self._read_as = cast(type[T], read_as)

    @property
    def current_samples(self) -> list[str]:
        return cast(list[str], self._s2i.keys[self._s_idx].tolist())

    def set_samples(self, samples: list[str]) -> Self:
        _samples = np.atleast_1d(samples)
        s_idx = self._s2i.get(_samples).astype(np.uint32)
        if (missing := _samples[s_idx == -1]).any():
            raise ValueError(f"Samples {missing} not found in the file.")
        self._s_idx = s_idx
        self._geno_pgen.change_sample_subset(np.sort(s_idx))
        return self

    def __del__(self):
        self._geno_pgen.close()
        if self._dose_pgen is not None:
            self._dose_pgen.close()

    def n_vars_in_ranges(
        self,
        contig: str,
        starts: ArrayLike = 0,
        ends: ArrayLike = np.iinfo(R_DTYPE).max,
    ) -> NDArray[np.uint32]:
        starts = np.atleast_1d(np.asarray(starts, R_DTYPE))

        c = self._c_norm.norm(contig)
        if c is None:
            return np.zeros_like(starts, dtype=np.uint32)

        ends = np.atleast_1d(np.asarray(ends, R_DTYPE))
        queries = pr.PyRanges(
            pl.DataFrame(
                {
                    "Chromosome": np.full_like(starts, contig),
                    "Start": starts,
                    "End": ends,
                }
            ).to_pandas(use_pyarrow_extension_array=True)
        )
        return (
            queries.count_overlaps(self._index)
            .df["NumberOverlaps"]
            .to_numpy()
            .astype(np.uint32)
        )

    def _var_idxs(
        self,
        contig: str,
        starts: ArrayLike = 0,
        ends: ArrayLike = np.iinfo(R_DTYPE).max,
    ) -> tuple[NDArray[np.uint32], NDArray[np.uint64]]:
        """Get variant indices and the number of indices per range.

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
        idxs
            Shape: (tot_variants). Variant indices for the given ranges.
        offsets
            Shape: (ranges+1). Offsets to get variant indices for each range.
        """
        starts = np.atleast_1d(np.asarray(starts, R_DTYPE))

        c = self._c_norm.norm(contig)
        if c is None:
            return np.empty(0, np.uint32), np.zeros_like(starts, np.uint64)

        ends = np.atleast_1d(np.asarray(ends, R_DTYPE))
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
        join = pl.from_pandas(queries.join(self._index).df)
        if join.height == 0:
            return np.empty(0, np.uint32), np.zeros_like(
                np.atleast_1d(starts), np.uint64
            )
        join = join.sort("query", "index")
        idxs = join["index"].to_numpy()
        lens = (
            join.group_by("query", maintain_order=True).agg(pl.len())["len"].to_numpy()
        )
        offsets = lengths_to_offsets(lens)
        return idxs, offsets

    def read(
        self,
        contig: str,
        start: int | np.integer = 0,
        end: int | np.integer = np.iinfo(R_DTYPE).max,
        out: T | None = None,
    ) -> T | None:
        c = self._c_norm.norm(contig)
        if c is None:
            return

        var_idxs, _ = self._var_idxs(c, start, end)
        n_variants = len(var_idxs)
        if n_variants == 0:
            return

        if issubclass(self._read_as, Genos):
            _out = self._read_genos(var_idxs, Genos.parse(out))
        elif issubclass(self._read_as, Dosages):
            _out = self._read_dosages(var_idxs, Dosages.parse(out))
        elif issubclass(self._read_as, GenosDosages):
            _out = self._read_genos_and_dosages(var_idxs, GenosDosages.parse(out))
        else:
            assert_never(self._read_as)

        return cast(T, _out)

    def read_chunks(
        self,
        contig: str,
        start: int | np.integer = 0,
        end: int | np.integer = np.iinfo(R_DTYPE).max,
        max_mem: int | str = "4g",
    ) -> Generator[T]:
        max_mem = parse_memory(max_mem)

        c = self._c_norm.norm(contig)
        if c is None:
            return

        var_idxs, _ = self._var_idxs(c, start, end)
        n_variants = len(var_idxs)
        if n_variants == 0:
            return

        mem_per_v = self._mem_per_variant()
        vars_per_chunk = min(max_mem // mem_per_v, n_variants)
        if vars_per_chunk == 0:
            raise ValueError(
                f"Maximum memory {format_memory(max_mem)} insufficient to read a single variant."
                f" Memory per variant: {format_memory(mem_per_v)}."
            )

        n_chunks = -(-n_variants // vars_per_chunk)
        v_chunks = np.array_split(var_idxs, n_chunks)
        for var_idx in v_chunks:
            if issubclass(self._read_as, Genos):
                out = self._read_genos(var_idx)
            elif issubclass(self._read_as, Dosages):
                out = self._read_dosages(var_idx)
            elif issubclass(self._read_as, GenosDosages):
                out = self._read_genos_and_dosages(var_idx)
            else:
                assert_never(self._read_as)

            yield cast(T, out)

    def read_ranges(
        self,
        contig: str,
        starts: ArrayLike = 0,
        ends: ArrayLike = np.iinfo(R_DTYPE).max,
    ) -> tuple[T, NDArray[np.uint64]] | None:
        c = self._c_norm.norm(contig)
        if c is None:
            return

        var_idxs, offsets = self._var_idxs(c, starts, ends)
        n_variants = len(var_idxs)
        if n_variants == 0:
            return

        if issubclass(self._read_as, Genos):
            out = self._read_genos(var_idxs)
        elif issubclass(self._read_as, Dosages):
            out = self._read_dosages(var_idxs)
        elif issubclass(self._read_as, GenosDosages):
            out = self._read_genos_and_dosages(var_idxs)
        else:
            assert_never(self._read_as)

        return cast(T, out), offsets

    def read_ranges_chunks(
        self,
        contig: str,
        starts: ArrayLike = 0,
        ends: ArrayLike = np.iinfo(R_DTYPE).max,
        max_mem: int | str = "4g",
    ) -> Generator[Generator[T]]:
        # TODO: support dosages

        max_mem = parse_memory(max_mem)

        c = self._c_norm.norm(contig)
        if c is None:
            return

        starts = np.atleast_1d(np.asarray(starts, R_DTYPE))
        ends = np.atleast_1d(np.asarray(ends, R_DTYPE))

        var_idxs, offsets = self._var_idxs(c, starts, ends)
        n_variants = len(var_idxs)
        if n_variants == 0:
            return

        mem_per_v = self._mem_per_variant()
        vars_per_chunk = min(max_mem // mem_per_v, n_variants)
        if vars_per_chunk == 0:
            raise ValueError(
                f"Maximum memory {format_memory(max_mem)} insufficient to read a single variant."
                f" Memory per variant: {format_memory(mem_per_v)}."
            )

        for i in range(len(offsets) - 1):
            o_s, o_e = offsets[i], offsets[i + 1]
            range_idxs = var_idxs[o_s:o_e]
            n_variants = len(range_idxs)
            if n_variants == 0:
                continue
            n_chunks = -(-n_variants // vars_per_chunk)
            v_chunks = np.array_split(range_idxs, n_chunks)

            if issubclass(self._read_as, Genos):
                r = self._read_genos
            elif issubclass(self._read_as, Dosages):
                r = self._read_dosages
            elif issubclass(self._read_as, GenosDosages):
                r = self._read_genos_and_dosages
            else:
                assert_never(self._read_as)

            yield (cast(T, r(var_idx)) for var_idx in v_chunks)

    def _mem_per_variant(self) -> int:
        if issubclass(self._read_as, Genos):
            return self.n_samples * self.ploidy * np.int32().itemsize
        elif issubclass(self._read_as, (Dosages, GenosDosages)):
            raise NotImplementedError("Dosages are not yet supported.")
        else:
            assert_never(self._read_as)

    def _read_genos(
        self, var_idxs: NDArray[np.uint32], out: Genos | None = None
    ) -> Genos:
        if out is None:
            _out = np.empty(
                (len(var_idxs), self.n_samples * self.ploidy), dtype=np.int32
            )
        else:
            _out = out
        self._geno_pgen.read_alleles_list(var_idxs, _out)
        _out = _out.reshape(len(var_idxs), self.n_samples, self.ploidy).transpose(
            1, 2, 0
        )[self._s_idx]
        _out[_out == -9] = -1
        return Genos(_out)

    def _read_dosages(
        self, var_idxs: NDArray[np.uint32], out: Dosages | None = None
    ) -> Dosages:
        if out is None:
            _out = np.empty((len(var_idxs), self.n_samples), dtype=np.float32)
        else:
            _out = out
        self._dose_pgen.read_dosages_list(var_idxs, _out)
        _out = _out.transpose(1, 0)[self._s_idx]
        _out[_out == -9] = np.nan
        return Dosages.parse(_out)

    def _read_genos_and_dosages(
        self, var_idxs: NDArray[np.uint32], out: GenosDosages | None = None
    ) -> GenosDosages:
        if out is None:
            _out = (None, None)
        else:
            _out = out

        genos = self._read_genos(var_idxs, _out[0])
        dosages = self._read_dosages(var_idxs, _out[1])
        return GenosDosages((genos, dosages))


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
    samples = psam["IID"].to_numpy().astype(str)
    return samples


RLEN = pl.col("REF").str.len_bytes()
ALEN = pl.col("ALT").str.len_bytes()
ILEN = ALEN - RLEN
KIND = (
    pl.when(ILEN != 0)
    .then(pl.lit("INDEL"))
    .when(RLEN == 1)  # ILEN == 0 and RLEN == 1
    .then(pl.lit("SNP"))
    .otherwise(pl.lit("MNP"))  # ILEN == 0 and RLEN > 1
    .cast(pl.Categorical)
)


# TODO: can index be implemented using the NCLS lib underlying PyRanges? Then we can
# pass np.memmap arrays directly instead of having to futz with DataFrames. This will likely make
# filtering less ergonomic/harder to make ergonomic though, but a memmap approach should be scalable
# to datasets with billions+ unique variants (reduce memory), reduce instantion time, but increase query time.
# Unless, NCLS creates a bunch of data structures in memory anyway.
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
            End=pl.col("POS") + RLEN - 1,
            kind=KIND,
        )
        .sink_ipc(path.with_suffix(".gvi"))
    )


def _read_index(path: Path, filter: pl.Expr | None) -> pr.PyRanges:
    index = pl.read_ipc(path, row_index_name="index", memory_map=False)
    if filter is not None:
        index = index.filter(filter)
    pyr = pr.PyRanges(index.drop("kind").to_pandas(use_pyarrow_extension_array=True))
    return pyr
