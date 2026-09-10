# Sparse Variant Format (SVAR)

## Motivation

Typical genomic data formats such as VCF/BCF and PLINK encode genotypes in a dense matrix. However, these matrices are typically extremely sparse (< 1% density), especially with whole genome sequencing or cancer data. To avoid consuming excessive amounts of disk space, these formats use block-wise compression. However, block compressed data can easily create data processing bottlenecks in machine learning applications, where random sampling is required to during training. This was a huge problem during the development of [GenVarLoader](https://github.com/mcvickerlab/GenVarLoader), for example. By instead using a sparse format, we were able to circumvent compression while keeping the file size on par with an equivalent compressed BCF. As a result, we can memory map the genotypes to work with larger-than-RAM data with random access that is much, much faster than compressed formats. For example, GenVarLoader computes direclty on the SVAR format and this is a major factor in its 1000x speedup over alternative methods.

## Creating SVAR files

```python
from genoray import SparseVar, VCF, PGEN

SparseVar.from_vcf("out.svar", "file.vcf.gz", max_mem="4g")
SparseVar.from_pgen("out.svar", "file.pgen", max_mem="4g")

svar = SparseVar("out.svar")
```

### Region/sample-restricted SVAR2 conversion

`SparseVar2.from_vcf`, `from_pgen`, and `from_svar1` can all convert directly
into a subset of regions and samples. `from_vcf_list` supports the same
`regions=`/`merge_overlapping=`/`regions_overlap=` but has **no `samples=`**
— each input file is single-sample, so the cohort is defined by the file set
itself:

```python
from genoray import SparseVar2

SparseVar2.from_vcf(
    "subset.svar2",
    "cohort.vcf.gz",
    "reference.fa",
    regions=["chr1:1-1000000", ("chr2", 0, 500_000)],
    samples=["HG00096", "HG00097"],
    merge_overlapping=True,
    threads=8,
)
```

Region strings use bcftools-style 1-based inclusive coordinates and are
converted to 0-based half-open intervals. Tuple, BED, and frame inputs are
already interpreted as 0-based half-open. `samples` selects and reorders by
name, preserving caller order and deduplicating repeated names by first
occurrence.

The equivalent CLI flags are available on the default SVAR2 writer, for every
source kind (VCF/BCF, PGEN, an SVAR1 store, or a vcf-list directory/manifest):

```bash
genoray write cohort.vcf.gz subset.svar2 \
  --reference reference.fa \
  --regions chr1:1-1000000,chr2:1-500000 \
  --samples HG00096,HG00097 \
  --threads 8
```

Use `--regions-file/-R` for BED files and `--samples-file/-S` for
one-sample-per-line sample lists. `--samples`/`-s`/`--samples-file`/`-S` are
rejected for the multi-file (vcf-list) directory/manifest form — each input
file already contributes exactly one sample, so there's no cohort left to
subset.

`regions_overlap` selects one of three overlap modes, matching bcftools
`--regions-overlap`: `"pos"` (default; POS inside `[start,end)`), `"record"`
(POS in `[start,end+1)`, so an indel at the region's last base is kept), or
`"variant"` (the anchor-trimmed variant extent overlaps the region). In
`variant` mode a multiallelic record is kept whole if ANY of its alleles
truly overlaps the region; individual non-overlapping alleles are not
dropped. `variant` currently requires at most one region per contig; multiple
regions per contig raise — use `pos`/`record`, or convert separately.

### Tuning (`genoray.Tuning`) and logging

Every `SparseVar2.from_*` write method (`from_vcf`, `from_pgen`,
`from_vcf_list`, `from_svar1`) accepts an explicit `tuning=` argument — a
`genoray.Tuning` object holding six scheduling knobs, all `int | None`,
keyword-only:

| field | meaning |
|---|---|
| `concurrent_chroms` | contigs converted concurrently |
| `reader_workers` | independent indexed shard readers per concurrent contig |
| `overshard` | work units per reader (used only when a contig has no exact record count) |
| `dense_cap` | depth of the dense-chunk channel between reader and executor |
| `merge_threads` | gather threads for the per-contig var_key merge tail |
| `sample_interval` | monitor sampling cadence in seconds; `0` disables it |

`None` (every field's default) means "let the planner choose", not "off" —
the same choice the planner makes with no `Tuning` at all. A field you set
explicitly is **honoured or refused, never silently shrunk**: for example, an
explicit `reader_workers` that cannot fit `max_mem` raises
`InsufficientMemory` instead of quietly falling back to something smaller.

Not every knob applies to every backend — setting one a backend can't use
raises `ValueError` rather than being silently ignored:

| field | `from_vcf` | `from_pgen` | `from_vcf_list` | `from_svar1` |
|---|---|---|---|---|
| `concurrent_chroms` | yes | yes | no | yes |
| `reader_workers` | yes | no | no | no |
| `overshard` | yes | no | no | no |
| `dense_cap` | yes | yes | yes | yes |
| `merge_threads` | yes | yes | yes | yes |
| `sample_interval` | yes | yes | yes | yes |

`reader_workers`/`overshard` are sharded-VCF only: `from_pgen` pins a single
reader per contig because `pgenlib` holds the GIL through genotype decode, so
sub-contig sharding there is pure overhead, and neither `from_vcf_list` nor
`from_svar1` shards within a contig at all. `concurrent_chroms` is
unavailable on `from_vcf_list` because that pipeline walks contigs
sequentially by design.

```python
from genoray import SparseVar2, Tuning

SparseVar2.from_vcf(
    "out.svar2", "file.vcf.gz", "ref.fa",
    tuning=Tuning(reader_workers=4, dense_cap=64),
)
```

Logging and progress reporting are separate from `tuning=`:

- `log_level` — minimum severity for structured write-time log lines.
  Accepted (case-insensitive): `"off"`, `"critical"`, `"error"`, `"warning"`,
  `"info"` (default), `"debug"`, or a `logging` module integer constant.
  `"critical"` is an alias for `"error"` (there is no distinct CRITICAL
  rank). **`"warn"` is rejected** — `logging.warn()` was removed in Python
  3.13, so genoray does not accept that spelling either; use `"warning"`.
- `log_filter` — an optional `tracing`-style `EnvFilter` directive string
  applied to genoray's own stderr diagnostic output, independent of
  `log_level`. For example, `log_filter="genoray::monitor=trace"` turns on
  trace-level output for just the conversion monitor.

**No environment variable configures genoray.** Every knob above is a
constructor argument or a CLI flag; there is no runtime environment-variable
override for any of them.

#### Migrating from environment variables

An earlier version of genoray read eight `GENORAY_*` environment variables.
All eight are gone — pass the equivalent explicitly instead:

| removed | replacement |
|---|---|
| `GENORAY_CONCURRENT_CHROMS` | `Tuning(concurrent_chroms=)` / `--concurrent-chroms` |
| `GENORAY_READER_WORKERS` | `Tuning(reader_workers=)` / `--reader-workers` |
| `GENORAY_OVERSHARD` | `Tuning(overshard=)` / `--overshard` |
| `GENORAY_DENSE_CAP` | `Tuning(dense_cap=)` / `--dense-cap` |
| `GENORAY_MERGE_THREADS` | `Tuning(merge_threads=)` / `--merge-threads` |
| `GENORAY_SAMPLE_INTERVAL` | `Tuning(sample_interval=)` / `--sample-interval` |
| `GENORAY_TRACE` | `log_filter="genoray=trace"` |
| `GENORAY_LOG` | `log_level=` (levels) or `log_filter=` (directives) |

### Parallel conversion

Single-file `SparseVar2.from_vcf` shards **within a contig**. `threads=` sets
the overall budget shown above; `tuning.reader_workers` (via
`tuning=Tuning(reader_workers=N)`, see "Tuning" above) sets how many indexed
shard readers each concurrent contig gets, which is the knob that controls
sub-contig read parallelism. Leaving it `None` derives it from the core
budget — a quarter of usable cores is reserved for the merge tail, contig
concurrency is chosen preferring depth, and the remainder goes to readers. An
explicit value is honoured or refused, never silently reduced: one that
cannot fit `max_mem` raises rather than quietly planning something slower.
Sub-contig sharding only
kicks in for the default whole-contig (`regions_overlap="pos"`) path. The planner uses
a backend-specific reader budget: indexed shard readers decompress inline and
replace, rather than run alongside, the monolithic reader's HTSlib pool. This
lets medium-sized single-contig runs use their available cores without
oversubscribing multi-contig runs. Output is **byte-identical** to serial
conversion at every thread count — sharding is gated by a store-hash oracle and
does not reintroduce missingness (a `./.` haplotype and a hom-ref haplotype
remain indistinguishable in SVAR2 either way).

Sub-contig sharding is restricted to `regions_overlap="pos"` (which the
whole-contig default uses). `"record"` and `"variant"` conversions run on a
single reader per contig, because their kept-record sets do not coincide with
the POS-ownership partition sharding dedups by.

The two backends behave differently:

- **VCF** scales well: ~3.9× wall-clock speedup at 32 cores on a chr21 germline
  BCF (1176s → 300s), byte-identical. The VCF path over-decomposes shards
  (factor 4) for work-stealing load balance.
- **PGEN** sub-contig sharding is **disabled** (`from_pgen` pins a single reader
  per contig). It is byte-identical but not faster: `pgenlib`'s genotype decode
  holds the CPython GIL, so shard readers serialize on it, and a reproducible
  chr21c benchmark measured sharding as net slower than serial (44.9s vs 32.6s)
  from added coordination overhead. The machinery is retained for re-enablement
  if a future reader/executor change shifts the bottleneck onto decode.

`SparseVar2.from_vcf_list` (merging N single-sample VCFs) does not shard within
a contig — it already opens one file descriptor per input file per contig.

See `docs/roadmap/svar2-conversion-baseline-2026-07-15.md` for the full scaling
results.

## Haploid (ploidy=1) write option

For unphased somatic cohorts where phasing information is unavailable or irrelevant,
`from_vcf` and `from_pgen` accept `haploid=True` to OR-collapse both haplotypes into a
single haploid call per sample. A variant is recorded for a sample if it is present on
*either* haplotype. The resulting SVAR stores `ploidy=1` in its metadata, and all
downstream read operations (including `write_view` and `annotate_mutations`) work
transparently at ploidy=1.

```python
from genoray import SparseVar, VCF

# Collapse diploid genotypes to haploid (ploidy=1 in output)
SparseVar.from_vcf("out.svar", VCF("file.vcf.gz"), max_mem="4g", haploid=True)

svar = SparseVar("out.svar")
assert svar.ploidy == 1
# shape: (ranges, samples, ploidy=1, ~variants)
rag = svar.read_ranges("chr1", starts=[0], ends=[1_000_000])
```

The equivalent CLI command:

```bash
genoray write file.vcf.gz out.svar --max-mem 4g --haploid
```

## Reading SVAR

```python
# shape: (ranges, samples, ploidy, ~variants)
sp_genos = svar.read_ranges("1", starts=0, ends=365, samples="Aang")
```

When using an SVAR file, `read_ranges` returns a `Ragged[V_IDX_TYPE]` — a ragged array where
the number of ALT calls per sample and ploid varies. For a brief visual description of Ragged
arrays, see [this section of the GenVarLoader FAQ](https://genvarloader.readthedocs.io/en/latest/faq.html#why-does-a-dataset-return-ragged-objects-and-what-are-they).
The returned array can be arbitrarily large because its data is backed by a
[`numpy.memmap`](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html) object
(only the offsets reside in RAM).

Each value in the ragged array is a variant index: the row number in `svar.index` for the
variant that is present in each range, sample, and ploid.

```python
v_idxs = sp_genos.to_awkward()[0, 0, 0].to_numpy()
```

## Loading additional fields

Custom numeric fields stored as `.npy` files in the SVAR directory can be loaded alongside
genotype indices. Only VCF FORMAT fields with `Number=G` are currently supported.

```python
# Load at construction time
svar = SparseVar("out.svar", fields={"dosages": np.float32})

# Or derive from an existing SparseVar (shallow copy, re-opens the memmaps)
svar_with = svar.with_fields({"dosages": np.float32})

# read_ranges now returns an awkward record array
result = svar_with.read_ranges("1", starts=0, ends=365)
result.genos.data    # flat array of variant indices (uint32)
result.dosages.data  # flat array of dosage values (float32)

# Drop all fields to get back a plain Ragged[V_IDX_TYPE]
svar_plain = svar_with.with_fields(False)
```

There's a lot more that can be done with `SparseVar`; this documentation will be expanded as time permits.

## Mutational signatures (SBS-96 / DBS-78 / ID-83)

`SparseVar` and its next-gen successor `SparseVar2` both support COSMIC-style
mutation catalogue annotation and signature refitting via
`annotate_mutations`, `mutation_matrix`, and `assign_signatures`
(`SparseVar2` can also classify during conversion with
`SparseVar2.from_vcf(..., signatures=True)`). This isn't narrated here yet —
see the `genoray-api` skill (`skills/genoray-api/SKILL.md`, "Mutation
catalogues" and "SparseVar2 — quick reference → Mutational signatures"
sections) or the method docstrings in `genoray/_svar/_annotate.py` (v1) and
`genoray/_svar2_mutcat.py` (SVAR2) for the full workflow.
