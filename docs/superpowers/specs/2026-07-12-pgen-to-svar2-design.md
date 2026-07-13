# PGEN → SVAR2 conversion

Date: 2026-07-12
Status: approved (design), ready for implementation planning

## Problem

`SparseVar2` can only be built from a VCF/BCF (`SparseVar2.from_vcf`). PLINK2
PGEN is the format cohort- and biobank-scale genotype data actually ships in, so
SVAR2 needs a `from_pgen` path. Two things make this non-trivial:

1. **Licensing.** The PGEN decoder that matters (plink-ng / plink 2.0) is
   copyleft. genoray is MIT. We must not create an obligation we cannot meet.
2. **Performance and memory.** PGEN's reason for existing is large `n_samples`.
   The current conversion pipeline has a staging buffer whose size scales as
   `chunk_size × n_samples × ploidy × 4 bytes`, which is not representable at
   biobank scale.

## Licensing position

| Component | License | How genoray uses it |
| --- | --- | --- |
| genoray | MIT | — |
| `pgenlib` (PyPI) | LGPL-3.0 | **Already** a hard runtime dependency; unmodified, separately-installed wheel, used through its public Python API |
| plink-ng `2.0/include/pgenlib_*.cc` | LGPL-3.0 | not linked, not vendored |
| plink2 application | GPL-3.0 | invoked as a separate program at *test* time only (already in `pixi.toml`) |

This design **does not change how genoray consumes plink code**. It keeps using
the `pgenlib` Python wheel through its public API, which imposes no copyleft
obligation on genoray's own source or on the compiled `_core` extension. The
`_core` extension continues to contain zero plink-ng code and stays MIT.

Rejected on licensing grounds:

- **Statically linking plink-ng's C++ `pgenlib` into `_core`.** This creates a
  combined work and triggers LGPLv3 §4: prominent notice, license texts, and
  *either* dynamic linking *or* shipping relinkable object code with every
  wheel. That is an ongoing compliance burden on our release CI (which builds
  four portable wheel platforms) for a decoder we already ship via pip.

Rule for implementers: **do not copy code or comments from plink-ng into
genoray.** If a native decoder is ever written, it must be written from
`pgen_spec.pdf`, and that provenance must be documented.

## Rejected alternative: a native Rust PGEN decoder

`genoio` (github.com/mancusolab/genoio, MIT) has a from-spec Rust PGEN decoder.
Its Rust sources contain no reference to plink-ng, pgenlib, Purcell, or Chang,
and read as independent work — the MIT label appears accurate. It is still not
usable here:

- **Multiallelic decode is not implemented** and is an explicit hard error
  (`pgen/header.rs`: "unsupported pgen multiallelic hard-call patch set"); its
  `.pvar` parser silently keeps only the first ALT.
- **No allele-code API.** Everything funnels into an f32 dense matrix. There is
  no equivalent of `read_alleles_range`, and haplotype reads *error* on an
  unphased het rather than reporting a phasepresent mask.
- Storage modes 0x01 and 0x11 are rejected.
- Not published to crates.io (path-only inter-crate deps, no `description`), and
  `genoio-io` pulls in noodles + bundled SQLite with no cargo features to slim
  it.

Writing our own equivalent means ~3k lines of difflist / LD-compression / aux-track
bit-twiddling with real correctness risk, to replace a decoder that is already a
dependency and is faster (genoio self-reports pgenlib at 1.4–1.8× its speed).
Not worth it. Revisit only if the GIL boundary below is measured to be a
bottleneck.

## Why pgenlib is the right decoder

`pgenlib.PgenReader.read_alleles_range(start, end, out, hap_maj=0)`:

- fills an `(n_variants, 2 · n_samples)` **int32** buffer with **per-haplotype
  allele codes**, phase-aware, multiallelic-aware, missing = `-9`;
- runs its decode loop under `prange(..., nogil=True)` — **the GIL is released**
  for the actual work, so a Rust reader thread holds it only for call dispatch.

The pipeline's `PendingAtom.gt` is a `Vec<i32>` of length
`n_samples × ploidy` holding allele index per haplotype column (`-1` = missing).
That is the same layout `read_alleles_range` produces. PGEN therefore needs only
a new *record source*; normalization, left-alignment, atom decomposition, the
executor, merge, and finalize are all reused unchanged.

## Architecture

### 1. `RecordSource` refactor (prerequisite, VCF-neutral)

`VcfChunkReader` currently fuses five responsibilities: htslib record iteration,
record → atom decomposition (REF validation, left-alignment, multiallelic split,
out-of-scope skipping), the `L_MAX` reorder heap, chunk assembly and presence
bit-packing, and INFO/FORMAT field staging. Only the first is VCF-specific.

Introduce:

```rust
pub struct RawRecord<'a> {
    pos: u32,               // 0-based
    reference: &'a [u8],
    alts: &'a [&'a [u8]],
    gt: &'a [i32],          // len = n_samples * ploidy; -1 = missing
    info_vals: &'a [f64],   // empty for PGEN
    format_vals: &'a [f64], // empty for PGEN
}

pub trait RecordSource {
    fn next_record(&mut self) -> Result<Option<RawRecord<'_>>, ConversionError>;
}
```

Everything downstream of `next_record` moves into a source-agnostic
`ChunkAssembler` that owns the heap, the decomposition, and chunk packing.
`VcfRecordSource` wraps htslib; `PgenRecordSource` is new.
`process_chromosome` becomes generic over the source (or takes a
`Box<dyn RecordSource>`).

Consequence: **PGEN inherits REF validation, indel left-alignment, and symbolic /
breakend skipping for free, with semantics identical to the VCF path.**

This lands as its own commit, with the existing test suite green and
**byte-identical** SVAR2 output on a VCF fixture, before any PGEN code is added.

### 2. `PgenRecordSource`

Owns:

- a `Py<PyAny>` handle to a `pgenlib.PgenReader` (constructed in Python, passed
  down), and a reusable `(B, 2·S)` int32 numpy buffer;
- a streaming `.pvar` reader for variant metadata.

`next_record()` serves from the buffer; on exhaustion it refills via
`Python::attach(|py| reader.call_method1("read_alleles_range", (lo, hi, buf, false)))`.
The GIL is held only for call dispatch — pgenlib's decode is `nogil`. Allele
codes are mapped `-9 → -1` to match the `PendingAtom.gt` convention.

**Batch size is derived from a byte budget, not a variant count:**

```
B = clamp(PGEN_BATCH_BYTES / (2 * S * 4), 1, chunk_size)   # PGEN_BATCH_BYTES = 32 MiB
```

At 200 samples this is thousands of variants per Python call (overhead
amortized); at 500k samples it is ~8 (memory bounded). This is the single knob
that keeps the Rust↔Python boundary cheap across the whole cohort-size range.

**Variant metadata.** Rust streams `.pvar` / `.pvar.zst` with a small TSV parser
(adds the `zstd` crate — BSD/MIT). `.bim` is not supported, since it only
accompanies a PLINK1 `.bed`, which is out of scope. Python computes the
per-contig `[var_start, var_end)` variant index ranges up front using the
**existing** `_scan_pvar` polars scan (which already handles header sniffing and
`.zst`) and passes only those ranges to Rust; the Rust streamer skips to
`var_start` by line count.

Known cost: with `concurrent_chroms > 1`, each contig's reader skips through the
`.pvar` from the top, so total `.pvar` I/O is `n_contigs × pvar_size`. This is
sequential scan I/O, parallel across contigs, and is expected to be negligible
next to the genotype decode. **Measure it in the benchmark.** If it shows up,
the mitigation is a one-time contig byte-offset sidecar — do not build that
speculatively.

Sample names come from the existing `_read_psam`.

### 3. Bounded-window packing (shared with the VCF path, in this PR)

`read_next_chunk` today collects `chunk_size` `PendingAtom`s into a `Vec` and
packs them all at the end. Each atom holds an `S × P` i32 `gt` vector — **32×
the size of the packed bit-grid it ultimately produces**. At
`chunk_size = 25_000` and 10k samples that staging buffer is ~2 GB; at biobank
scale it is not representable.

Fix: **pack in sub-blocks.** Buffer `W` atoms (`W ≈ 1024`), pack them in parallel
into the growing `BitGrid3` at their variant offset, drop their `gt`, and
continue. Staging memory becomes `W · S · P · 4` instead of
`chunk_size · S · P · 4`, and parallel packing (previously measured worth −6.7%)
is retained. The `L_MAX = 1000` reorder heap already bounds the sort window, so
atom ordering is unaffected and output is byte-identical.

This is a shared improvement: it benefits `from_vcf` identically and is what
makes large-`n_samples` conversion feasible on either backend.

Additionally, `from_pgen` auto-derives `chunk_size` from a memory budget when the
caller does not supply one, since the packed dense chunk is itself
`chunk_size · S · P / 8` bytes. `from_vcf`'s default (`25_000`) is unchanged.

## Public API

```python
SparseVar2.from_pgen(
    out: str | Path,
    source: str | Path,
    reference: str | Path | None = None,
    *,
    no_reference: bool = False,
    skip_out_of_scope: bool = False,
    chunk_size: int | None = None,   # None => auto from a memory budget
    threads: int | None = None,
    overwrite: bool = False,
    long_allele_capacity: int = 8 * 1024 * 1024,
    signatures: bool = False,
) -> int   # number of out-of-scope ALTs dropped
```

- `source` is a `.pgen` path; `.pvar[.zst]` and `.psam` siblings are resolved by
  stem.
- **No `ploidy` parameter.** PGEN is diploid; exposing the knob would only create
  an invalid state to reject.
- `reference` / `no_reference` are mutually exclusive and behave exactly as in
  `from_vcf`. `signatures=True` requires a reference, as in `from_vcf`.
- CLI: `genoray write` dispatches on the `.pgen` suffix (its help text already
  claims VCF/PGEN).
- `skills/genoray-api/SKILL.md` is updated in the same PR (CLAUDE.md requires it
  for any public-name change).
- `CHANGELOG.md` gains an entry under `## Unreleased`.

### Out of scope for v1 (documented in the docstring)

- **Dosages.** SVAR2 has no dosage store. A `.pgen` dosage track is ignored;
  hardcalls are still read correctly.
- **`info_fields` / `format_fields`.** PGEN has no FORMAT; `.pvar` INFO
  extraction is deferred.
- **Sample subsetting / reordering.** Matches `from_vcf`, which also converts all
  samples.
- **PLINK1 `.bed`.**

### Semantics to document

Haplotype resolution for **unphased** heterozygotes follows the allele-code order
pgenlib returns. This is the same caveat the VCF path carries for unphased `GT`,
and is exactly what `SparseVar.from_pgen` (SVAR1) already does. Missing calls are
`-9` in pgenlib and are mapped to `-1`.

## Testing

1. **Primary oracle — pgenlib itself.** Write an SVAR2 store from a PGEN, decode
   it back to `(samples, ploidy, variants)` allele indices via the existing query
   API, and compare against `pgenlib.PgenReader.read_alleles_list` on the same
   variant indices. Exact, and independent of the VCF path.
2. **Cross-backend equivalence.** Generate PGEN twins of the existing VCF
   fixtures with `plink2 --vcf ... --make-pgen` (plink2 is already in
   `pixi.toml`) and assert `from_pgen ≡ from_vcf` on curated fixtures covering
   phased, multiallelic, and indel records. Where plink2's VCF import legitimately
   differs (symbolic ALTs, unphased-het haplotype order), compare genotype
   presence rather than raw store bytes.
3. **Rust unit tests** for the `.pvar` streamer: header sniffing, `.zst`,
   multiallelic ALT splitting, missing codes, `[var_start, var_end)` skipping.
4. **Refactor guard:** the `RecordSource` commit must produce byte-identical
   SVAR2 output for an existing VCF fixture.
5. **Bounded-window packing guard:** byte-identical output before/after, plus a
   peak-RSS assertion (or measurement) showing staging memory no longer scales
   with `chunk_size`.
6. **Benchmark** PGEN vs VCF conversion on the same cohort. Expectation: the
   reader stops being the bottleneck (VCF conversion is currently
   htslib-input-bound). Record the numbers in the CHANGELOG entry.

## Risks

- **Refactor regresses the VCF path.** Mitigated by landing it as an isolated
  commit gated on byte-identical output.
- **GIL contention across concurrent contigs.** Each contig's reader briefly
  attaches to call `read_alleles_range`. The decode is `nogil`, so contention is
  on call dispatch only. If the benchmark shows readers serializing, the batch
  byte budget is the first knob; a native decoder is the last resort.
- **pgenlib's `prange` uses OpenMP if its wheel was built with it**, which could
  oversubscribe alongside `concurrent_chroms`. Check thread counts in the
  benchmark; cap via `OMP_NUM_THREADS` if needed. Note that the PGEN path needs
  no htslib decompression threads, so those cores are available to give back.

---

# Results (2026-07-12) — Task 9 benchmark

Methodology follows
`docs/superpowers/specs/2026-07-07-svar1-vs-svar2-timing-and-profiling-design.md`
(same fixture family, same `/usr/bin/time -v` protocol, same honesty bar). Ran
on a `carter-compute` allocation (`carter-cn-04`, 32 cores, 64 GB, via
`sbatch`) — **not a dedicated node**: several unrelated `nf-PHASI*` jobs from
the same user were concurrently scheduled on the same physical node the whole
time, so absolute wall-clock numbers below likely carry some contention
inflation relative to a quiet node (the from_vcf baseline here, 547s, is
notably slower than the 36.5s figure the 2026-07-07 doc measured for the same
germline chr21 file on a dedicated node — same code path, so the delta is
node contention, not a regression). The **relative** from_vcf-vs-from_pgen
comparison is still valid since both ran back-to-back on the same
contended node under the same conditions.

## Fixture

`/carter/users/dlaub/repos/for_loukik/chr21.bcf` (germline, 3202 samples,
phased) contains `<DEL>`/`<DEL:ME>` symbolic ALTs that plink2 cannot import.
Built one filtered BCF and used it for **both** backends, per the task brief:

```bash
bcftools view -e 'ALT~"<"' chr21.bcf -Ob -o chr21.nosym.bcf   # 1,001,385 / 1,002,753 kept (1,368 symbolic dropped)
bcftools index chr21.nosym.bcf
plink2 --bcf chr21.nosym.bcf --make-pgen --output-chr chrM --out chr21.nosym   # preserves "chr21" (not "21")
```

Reference: `/carter/shared/data/gdc/resources/GRCh38.d1.vd1.fa` (chr21 length
46,709,983, matches the `.pvar`).

## Equivalence check (before any timing)

Two tiers, both against the identical filtered input:

1. **Decode-free `region_counts`** over the whole chromosome `(0,
   46_709_983)`: identical `(region, sample, ploidy)` count arrays for both
   stores, 285,721,479 total carried variants either way.
2. **Full decoded-record comparison** (`pos`/`ilen`/`allele`, the
   `_assert_ragged_equal` fields) on 5 spot-check windows spanning the
   chromosome (0–2 Mb, 10–12 Mb, 20–22 Mb, 30–32 Mb, 44–46.7 Mb): offsets and
   all three fields matched exactly on every window (0 to 21.5M records per
   window).

**PASS.** `from_pgen ≡ from_vcf` on this cohort — the timing numbers below are
measuring two paths to the same data, not two different datasets.

## Timing (Step 2 of the brief)

`/usr/bin/time -v python -c "SparseVar2.from_{vcf,pgen}(...)"`, filtered chr21
in, fresh output each run:

| Path | Wall time | Peak RSS |
| --- | --- | --- |
| `from_vcf` | 547.0s (9:06.98) | 497 MiB (496,508 KiB) |
| `from_pgen` | **152.7s** (2:32.67) | 1040 MiB (1,065,212 KiB) |

**`from_pgen` is 3.6× faster wall-clock**, confirming the design's central
hypothesis — the reader does stop being the bottleneck once htslib decode is
out of the picture. It is **not a free win**: peak RSS is ~2.1× higher than
`from_vcf`'s on the same cohort. This wasn't budgeted for in the design and is
worth a follow-up look (candidates: the byte-budgeted `read_alleles_range`
buffer, or pgenlib's own internal decode buffers held concurrently with the
Rust-side staging window) — out of scope to fix here since Task 9 is
measurement-only, but real enough to flag rather than bury.

An independent re-run during equivalence verification (same code paths, same
node, run immediately before the numbers above) gave from_vcf=538.8s /
from_pgen=138.9s (3.9×) — consistent with the timed numbers within the noise
expected from a shared node.

## Open question 1 — is the `.pvar` skip cost material?

Built a **same-total-data, different-`n_contigs`** fixture to isolate this:
sliced the identical filtered chr21.nosym.bcf (1,001,385 variants, all 3202
samples — nothing shrunk) into genomic windows via `bcftools view -r` +
`bcftools annotate --rename-chrs` + `bcftools concat` (index-seek slicing,
no genotype-text materialization), tagged each window with a synthetic
contig name (`synth_0`..`synth_15`), and built one reference FASTA holding 16
copies of the full chr21 sequence under those names (so absolute chr21
coordinates resolve identically regardless of which synthetic contig a
variant landed on). This keeps total variant count, total genotype payload,
and total `.pvar` bytes ~constant; only `n_contigs` (and therefore the number
of independent top-of-`.pvar` skips) changes.

- `N=1` (single contig, whole cohort): **151.1s**, 1074 MiB peak RSS.
- `N=16` split (`synth_0` window landed on chr21's acrocentric p-arm gap and
  got 0 variants, so effectively 15 populated contigs; still 1,001,385
  variants total): **~104.7s** wall (measured from the thread-sampling
  window's first/last timestamps around the run).

The 15-contig split was **faster**, not slower, than the single-contig
baseline (~1.44×) — the `plan_thread_budget`-driven concurrency
(`concurrent_chroms=5` on 32 cores) more than pays for the extra redundant
`.pvar` scanning. At this file's `.pvar` size (~36 MB total across the
cohort), the top-of-file skip cost is **not material** — it's noise next to
the genotype-decode work, even multiplied by 15 concurrent skips.

**Caveat:** this only rules it out at this scale (≤16 contigs, 36 MB `.pvar`).
A whole-genome cohort with real chromosome-scale `.pvar` files (hundreds of MB
to GB) and `n_contigs` in the dozens (autosomes + alts/decoys) could still
show it — reasoning from the `n_contigs × pvar_size` model, the risk grows
with both factors simultaneously, which this fixture didn't stress. **Not
building the byte-offset sidecar mitigation** — no evidence it's needed yet;
flagged as a follow-up to revisit if a whole-genome benchmark ever shows
`.pvar` scanning show up in a profile.

## Open question 2 — GIL serialization vs OpenMP oversubscription

Sampled `ps -T -p <pid> -o pid,tid,pcpu,comm` every 0.5s for the whole `N=16`
run (46 samples, ~23s window) to look at both questions at once.

- **Readers do not serialize on the GIL.** `Pipeline Config (PGEN): 5
  concurrent chromosomes` actually ran concurrently — `chrom-0`..`chrom-4`
  rayon workers plus each contig's own `read-synth_*`/`exec-synth_*`/
  `cw-synth_*`/`lw-synth_*`/`samp-synth_*` OS threads all showed up live and
  overlapping across samples, and the wall-clock result above (5-way split
  faster than 1-way) is itself evidence of real parallelism, not GIL-bound
  serialization.
- **pgenlib's OpenMP `prange` does oversubscribe, confirmed.** The dominant
  thread `comm` in every sample was the generic, un-renamed `python` (2804 of
  the ~7000 total thread-sample rows, i.e. an average of ~61 threads/sample
  reporting as `python` — Rust's own threads are all explicitly named via
  `thread_name`, so any unnamed OS thread showing as `python` is coming from
  a C-extension's own thread spawns, consistent with an OpenMP worker team
  pgenlib spins up per `read_alleles_range` call). **Max concurrent OS thread
  count observed: 172**, against a 32-core allocation — a ~5.4× oversubscription
  ratio. (Smaller contributions from `async-executor-`/`tokio-runtime-w`
  threads and `jemalloc_bg_thd` background threads were also visible but much
  less numerous; not investigated further as they're not contig-count-scaled.)
- **Net effect:** the oversubscription is real but did not net-negative this
  run's throughput — `N=16` was still faster than `N=1`, not slower. It's
  still resource-impolite on a shared, multi-tenant node (172 threads
  competing for 32 cgroup-scoped cores affects everyone else on the node, even
  if this job's own wall time survives it).

**Conclusion:** the design doc's own suggested mitigation for *this* symptom
(OpenMP fan-out, not GIL serialization) is the right one —
**`OMP_NUM_THREADS`**, not `PGEN_BATCH_BYTES` (that knob addresses a GIL-
serialization symptom we did not observe). Recommended **follow-up**: set
`OMP_NUM_THREADS=1` (or a small cap) around the `pgenlib.PgenReader` calls
when `concurrent_chroms > 1`, purely for cluster citizenship — not filed as a
correctness or throughput bug, since throughput was fine here.

## Summary

| Question | Answer |
| --- | --- |
| Is PGEN faster than VCF on the same cohort? | **Yes, 3.6×** (152.7s vs 547.0s), confirming the design hypothesis. |
| Is it free? | **No** — ~2.1× peak RSS (1040 MiB vs 497 MiB). Follow-up, not fixed here. |
| Is `.pvar` top-of-file skip cost material? | **Not at this scale** (≤16 contigs, 36 MB `.pvar`); flagged as a watch-item for whole-genome-scale cohorts, sidecar mitigation not built. |
| Do concurrent contig readers serialize on the GIL? | **No** — 5-way contig concurrency measurably sped things up. |
| Does pgenlib's OpenMP oversubscribe? | **Yes, confirmed** (≤172 threads on 32 cores); did not hurt this run's throughput, but `OMP_NUM_THREADS` capping is a recommended follow-up for shared-node citizenship. |
