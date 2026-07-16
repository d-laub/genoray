# `from_vcf_list` memory: allocation churn, not live data

Design for issue #120 (`from_vcf_list` OOMs on large cohorts) and the deferred Phase 3
work from PR #121.

**This document supersedes `2026-07-15-svar2-from-vcf-list-scaling-baseline.md`.** That
doc's central claims — the `RAM(N) ≈ 348 MB + 0.383 MB × N` fit, "R2 (open readers) is
the driver", "`chunk_size` is ~1% of peak" — are artifacts of a bench that passed **zero
fields** and ran **one contig**. Every one of them is corrected below.

## 1. What actually happens

`SparseVar2.from_vcf_list` merges N single-sample VCFs. For a **somatic** cohort, nearly
every union variant has exactly **one carrier**. The pipeline nevertheless materializes
FORMAT values for **all N samples at every union variant**, three times over. Those
materializations are short-lived, so they barely register in peak *live* heap — but they
are allocated and freed once per record, producing tens of millions of small O(N)
allocations that shred the glibc arenas. Peak RSS is then dominated by memory glibc has
freed but cannot return to the OS.

The failure is **allocation churn**, not a leak and not a large live set.

### Measured on the real cohort

The 256 GB re-run of the exact #120 cohort (SLURM 13271678, 7089 Hartwig somatic
single-sample VCFs, hg19, F=7 FORMAT fields) was profiled live on 2026-07-16:

| | |
|---|---|
| MaxRSS / VmHWM | **132 GB** |
| RSS while on **contig 5**, contigs 1-4 already written | **129 GB** |
| glibc 64 MB arena heaps | **2,593** → 161.7 GB virtual, **111.5 GB resident (90% of RSS)** |
| Thread stacks (3 MB × 7,099) | 0.72 GB |
| Threads | **7,148** (≈1 htslib thread per open file) |
| Open fds | 7,098 |
| VmPeak / VmData | 221 GB / 206 GB |
| Node cores / glibc | **96** → default `arena_max` = 8×96 = **768**; glibc 2.28, `MALLOC_ARENA_MAX` unset |

**Memory is not released between contigs.** Issue #120 assumed it was. The job sits at
129 GB on contig 5 — a smaller contig than chr1, with contigs 1-4 flushed to disk and
nothing yet gathered. Growth is monotone across contigs, which is why the original 64 GB
run died mid-**contig 2** rather than on chr1.

### Measured under control

1000 real Hartwig VCFs subset to specific contigs (so the code path is exactly the
genoray 3.0.0 that OOMed), F = the production 7 FORMAT fields:

| exp | N | contigs | F | `MALLOC_ARENA_MAX` | MaxRSS | wall |
|---|---:|---:|---:|---|---:|---:|
| B | 1000 | 1 | 0 | default | 2.41 GB | 135 s |
| A | 1000 | 1 | 7 | default | 4.36 GB | 399 s |
| C | 1000 | 1 | 7 | 2 | 4.43 GB | 442 s |
| D | 4000 | 1 | 7 | default | 15.44 GB | 4,729 s |
| E | 4000 | 1 | 7 | 2 | 17.36 GB | 8,179 s |
| T | 1000 | **3** | 7 | default | **9.20 GB** | 1,106 s |
| T2 | 1000 | **3** | 7 | 2 | **6.89 GB** | 1,093 s |

Within one contig, RSS is linear in N (**RSS ≈ 0.66 GB + 3.7 MB × N**, → ~27 GB for chr1
at N=7089), but the **arena share grows superlinearly**: 1.78 GB (41% of peak) at N=1000
→ 11.54 GB (**75%**) at N=4000; heaps 88 → 396.

Across contigs, RSS **ratchets and never falls**:

| 3 contigs, N=1000, F=7 | contig 1 | contig 2 | contig 3 | after return |
|---|---:|---:|---:|---:|
| arena heaps | 94 | 186 | 268 | 252 |
| arena RSS | 2.42 GB | 4.82 GB | **6.99 GB** | **6.43 GB** |
| big-mmap RSS | 1.66 GB | 1.40 GB | 1.32 GB | **0.19 GB** |

Each contig permanently adds ~+90 heaps / **+2.3 GB**. The large mmap'd buffers
(`big_gb`) are released correctly every contig — **they were never the problem.**

The per-contig ratchet plus the per-contig floor reproduce both real observations:
chr1 ≈ 27 GB at N=7089, ratcheting → ~64 GB by contig 2 (**the original OOM**) and
~113 GB by contig 5 (**observed: 132 GB**).

### dhat: the decisive measurement

Native bench (`bench_from_vcf_list`, no Python), N=250, contigs 1,2,3, F=7:

```
Total:     140,910,349,865 bytes in 158,811,094 blocks
At t-gmax:     861,967,945 bytes in 10,946 blocks
At t-end:           73,296 bytes in 93 blocks
```

- **At end: 73 KB of 140.9 GB churned. There is no leak** — Rust frees everything.
- **Peak live heap is 862 MB**, while the same shape at N=1000 shows 9.20 GB RSS. Live
  heap is a small fraction of RSS; the rest is fragmentation.
- **140.9 GB churned through 158.8 million blocks — at N=250.**

Top churn sites:

| churned | blocks | peak live | site |
|---:|---:|---:|---|
| **70.66 GB** | **11,776,688** | 43 MB | `VcfListRecordSource::next_record` → `vcf_list_reader.rs:485` |
| **23.55 GB** | **1,682,384** | 364 MB | `decompose_raw_record` → `chunk_assembler.rs:338` |
| 10.09 GB | 408 | 150 MB | `StagedColumn::with_capacity` ← `read_next_chunk` |
| 8.41 GB ×2 | 340 | 125 MB | `StagedColumn::with_capacity` ← `rvk::dense2sparse_vk` |

The arithmetic pins the attribution: site #2 averages **14,000 B/block = F×N×8 =
7×250×8** exactly, over 1,682,384 blocks = the atom count across three contigs — one
F×N `Vec<f64>` allocated and freed **per atom**. Sites #1+#2 are **67% of all churn**.

Note the contrast: the *big* `StagedColumn` buffers are 10 GB across **408 blocks** —
few, large, mmap'd, returned cleanly. The damage comes from ~13.5M **small** O(N)
allocations that glibc scatters across hundreds of arenas and cannot coalesce.

Churn scales as `V(N) × N`. For private somatic variants `V ∝ N`, so **churn ∝ N²**: at
N=7089 it is ~800× the N=250 run.

## 2. The offending code

**Site #1 — `src/vcf_list_reader.rs:484-492`.** An N-wide vector-of-vectors built per
field per record to carry (typically) one value:

```rust
let format_raw: Vec<Option<Vec<Vec<f64>>>> = (0..self.format_specs.len())
    .map(|j| {
        let mut per_sample: Vec<Vec<f64>> = vec![Vec::new(); self.num_samples];
        for (col, atom) in &group {
            per_sample[*col] = vec![atom.format_vals[j]];
        }
        Some(per_sample)
    })
    .collect();
```

`vec![Vec::new(); num_samples]` allocates N `Vec`s; each carrier then heap-allocates a
1-element `vec![...]`. 70.66 GB churned, never more than 43 MB live.

**Site #2 — `src/chunk_assembler.rs:337-343`.** Widens each atom to F×N:

```rust
let mut format_vals = Vec::with_capacity(format_fields.len() * num_samples);
for (spec, raw) in format_fields.iter().zip(rec.format_raw.iter()) {
    for s in 0..num_samples {
        let sample_vals = raw.as_ref().map(|v| v[s].as_slice());
        format_vals.push(resolve_scalar(sample_vals, atom.source_alt_index, spec));
    }
}
```

Stored on `AtomMeta.format_vals` (`chunk_assembler.rs:207`) at **f64** width for
`chunk_size` atoms. The comment at `chunk_assembler.rs:200-201` claims per-chunk staging
"no longer scales with `chunk_size * num_samples * ploidy`" — true of the presence bits,
false of the fields beside them, and misleading. Fix the comment.

**Site #3 — `src/rvk.rs:218-222`.** Reserves the whole chunk width for **both** dense
classes before any variant is classified dense:

```rust
sub.field_format = fields
    .iter()
    .filter(|f| f.category == FieldCategory::Format)
    .map(|f| StagedColumn::with_capacity(f.stage_is_float(), v_variants * num_samples))
    .collect();
```

`v_variants` is the whole chunk, not `n_dense_variants`; the loop runs for `Snp` **and**
`Indel`. At N=7089: 2 × 7 × 25000 × 7089 × 4 = **9.92 GB reserved** for a somatic cohort
in which ~nothing is dense. Untouched pages cost address space, not RSS — this is why
VmData is 206 GB against 132 GB RSS. Not an OOM cause; real under `ulimit -v`.

**Site #4 — `src/orchestrator.rs:978` + `src/vcf_reader.rs:299`.**

```rust
const VCF_LIST_HTSLIB_THREADS: usize = 1;   // orchestrator.rs:978
reader.set_threads(htslib_threads)          // vcf_reader.rs:299 — called per file
```

One htslib decompression thread **per file, per contig** → 7,089 threads created and
destroyed on each of 24 contigs. Each wave gets fresh glibc arenas whose 64 MB heaps are
never unmapped. The threads buy nothing: the read phase is pinned at `read≈99%` (one
core). rust-htslib's `set_threads` asserts `n_threads > 0` (`bcf/mod.rs:171`), so the
call must be **skipped**, not passed 0.

**Site #5 — `python/genoray/_svar2.py:1690-1702`.** `_auto_chunk_size` budgets only the
bit grid:

```python
_DENSE_CHUNK_TARGET_BYTES = 256 * 1024 * 1024
bits_per_variant = n_samples * ploidy
by_budget = (_DENSE_CHUNK_TARGET_BYTES * 8) // max(bits_per_variant, 1)
return max(1024, min(25_000, int(by_budget)))
```

At N=7089 the grid is 42.3 MiB while `format_staged` is 4.96 GB — the budgeted term is
**112× too small** (ratio = 32·F/P), so the function returns the unchanged 25,000. Its
256 MiB "budget" is meaningless whenever fields are requested.

## 3. Design

### 3.1 Carry FORMAT carrier-sparsely (the fix)

Replace the N-wide representation across the reader→assembler seam with a
carrier-indexed one. Invalid states — "a value for a sample that has no call" — become
unrepresentable rather than defaulted.

```rust
/// FORMAT values for one merged record, one entry per carrying column.
/// Length is the carrier count (≈1 for somatic), not `num_samples`.
pub struct FormatCalls {
    /// Merged column index per carrier, ascending.
    cols: Vec<u32>,
    /// `vals[i * n_fields + j]` = field j for carrier i.
    vals: Vec<f64>,
    n_fields: usize,
}
```

- `next_record` fills `FormatCalls` directly from the frontier group it already walks —
  no N-wide intermediate, no per-carrier 1-element `Vec`.
- `AtomMeta` carries `FormatCalls` instead of an F×N `Vec<f64>`.
- Non-carriers resolve to the field default **at materialization** (when a dense column
  is actually written), not at staging.
- Both structs are reused across records via a cleared buffer, so steady-state
  allocation per record is ~zero rather than O(N).

Expected: sites #1 and #2 (**67% of churn**) drop by ~N×, and churn stops scaling as N².

### 3.2 Stop the per-contig thread storm

Guard the call and set the constant to 0:

```rust
if htslib_threads > 0 {
    reader.set_threads(htslib_threads)...?;
}
```

`VCF_LIST_HTSLIB_THREADS: usize = 0`. Removes ~7,089 threads per contig. Decompression
becomes inline on the reading thread, which is where the work already happens.

### 3.3 Reserve dense staging on the dense count

`rvk.rs:218-222`: reserve `n_dense_variants * num_samples`, and only for classes that
have variants. Reclaims ~10 GB of address space per chunk at N=7089.

### 3.4 Field-aware `chunk_size`, then `max_mem` (revived Task 8)

Make `_auto_chunk_size` budget the real per-chunk cost:

```
bytes_per_variant ≈ n_samples * ploidy / 8            # bit grid
                  + n_format_fields * n_samples * 4    # staged FORMAT
```

Then `max_mem` sizes `chunk_size` against that. The baseline doc called this "a knob that
does nothing" — that verdict was an artifact of `fields = []`. With F=7 it is the
dominant per-chunk lever.

### 3.5 Fix the bench (non-negotiable)

The current harness cannot observe any of the above:

1. **Zero fields.** `run_bench.py:17-21` never passes `info_fields`/`format_fields`;
   `bench_from_vcf_list.rs` hardcoded `Vec::new()` for both. ⇒ every F×N term is exactly
   zero.
2. **One contig.** ⇒ the cross-contig ratchet — the thing that actually OOMs — is
   invisible.
3. **Saturating union.** `generate_cohort.py:36,45,75` draws private positions from a
   fixed `contig_len=1e6` pool, so V asymptotes at ~1e6 **regardless of N**, while real
   private somatic variants make V grow linearly in N.

Changes: generator emits FORMAT fields and multiple contigs, with a position space that
scales so V ∝ N; `run_bench.py` and the Rust bench accept a field list and a contig list
(the Rust side is already patched for the dhat run above); the sweep records **live heap
(dhat) alongside RSS**, since their divergence *is* the bug.

### 3.6 `MALLOC_ARENA_MAX`: document, do not default

It is not a safe default:

| | MaxRSS | wall |
|---|---:|---:|
| N=4000, 1 contig, default | **15.44 GB** | **4,729 s** |
| N=4000, 1 contig, `=2` | 17.36 GB | **8,179 s (+73%)** |
| N=1000, 3 contigs, default | 9.20 GB | 1,106 s |
| N=1000, 3 contigs, `=2` | **6.89 GB** | 1,093 s |

With thousands of threads contending on two arena locks it is **12% worse on memory and
73% slower**. It only helps once §3.2 has collapsed thread count. Document it in the
`from_vcf_list` docstring as a tuning knob for large multi-contig merges, with the
contention caveat. Do **not** call `mallopt` from the library — process-wide allocator
policy is the caller's to set.

## 4. Explicitly dropped

- **Task 9 (staged / batched merge).** The whole point was capping concurrent open
  readers. Readers cost ~2 GB of 132 GB (~250-350 KB/file: BGZF blocks 2×64 KiB, resident
  index, duplicated header). It is expensive work aimed at 1.5% of the problem. The
  `ulimit -n` ceiling it would also retire is better addressed on its own if it bites.
- **Task 10 (ledger trim).** `var_key_ledgers`/`dense_ledgers` (`executor.rs:29-31`) are
  `n_streams × ceil(V/chunk_size) × N × P × 4` = **13.6 MB** at N=7089, V=3e6. dhat
  confirms they are nowhere near the top. Its gate is now run; the answer is no.
- **S2 (parallel leaf decompression).** Orthogonal to memory, and §3.2 moves in the
  opposite direction on threads. If revisited, measure thread utilisation first.

## 5. Verification

- **Correctness: byte-identical output.** The parity fixture from PR #121 must stay green
  through every change — carrier-sparse staging must produce the same store as N-wide
  staging. Extend it to cover a **fields + multi-contig** cohort, which it currently
  does not.
- **Unit:** carrier-sparse resolution for a non-carrier column returns the field default,
  identical to today's `resolve_scalar` contract on an empty buffer; multi-allelic and
  record-absent-field cases included.
- **Regression gate (the number that matters):** dhat on the fixed bench at N≥250,
  3 contigs, F=7. Today: **140.9 GB churned / 158.8M blocks**. Target: churn no longer
  scales with N — assert total blocks fall by ≥10× and that the two named sites leave the
  top-5.
- **End-to-end:** 3-contig N=1000 trace. Today: **9.20 GB peak, +2.3 GB/contig ratchet**.
  Target: flat across contigs.
- **The real thing:** re-run the 7089-file cohort and compare against the 132 GB
  baseline. This is the only test that exercises N=7089 × 24 contigs.

## 6. Risks

- **Behaviour change on defaults.** §3.2 changes threading and §3.4 changes `chunk_size`
  for any caller passing fields. `chunk_size` is a public kwarg and appears in stored
  layout decisions — confirm a smaller default cannot change store bytes, or gate it.
- **`htslib_threads = 0` is unmeasured.** The claim that the 7,089 threads are idle rests
  on `read≈99%` (one core busy) in the production log. Cohorts with large BGZF blocks
  (the `MAX_HTSLIB_THREADS` bump at `budget.rs:13` cites gdc's 16007-sample records) may
  genuinely use them. Scope the constant to the **vcf_list** path only — it already is —
  and measure wall-clock before/after on the bench.
- **Public API.** `max_mem` on `from_vcf_list` is a new public kwarg ⇒
  `skills/genoray-api/SKILL.md` must be updated in the same PR (repo rule).
- **N² is still N².** Carrier-sparse removes the O(N) *per record* factor, but total work
  remains O(V×N) = O(N²) in records-touched for a fully private cohort. This design makes
  the constant tiny and stops the allocator pathology; it does not change the asymptote.
  A cohort 10× larger again will need the merge itself rethought.

## Reproducing

```bash
# real-data arms (chr1-only subsets of the Hartwig cohort)
bcftools view -r 1 -Oz -o <out>/<s>.vcf.gz <src>; bcftools index -t <out>/<s>.vcf.gz
python run_arm.py <manifest> <out>.svar2 on|off <label>       # RSS + arena-heap split
python run_trace.py <manifest> <out>.svar2 <label>            # RSS time series (ratchet)

# dhat (the measurement that settles live-vs-fragmentation)
export CARGO_TARGET_DIR=/tmp/genoray_dhat     # NFS target/ bus-errors: memory map must have non-zero length
cargo build --release --no-default-features --features conversion,dhat-heap \
    --bin bench_from_vcf_list
./target/release/bench_from_vcf_list <manifest> <out> "1,2,3" <ref.fa> \
    "VAF:float,DP:int,PURPLE_AF:float,PURPLE_CN:float,PURPLE_VCN:float,PURPLE_MACN:float,SUBCL:float"

# live inspection of a running merge
srun --overlap --jobid=<id> bash -c 'grep -E "VmRSS|VmHWM|Threads" /proc/<pid>/status'
# 64MB-aligned mappings are glibc arena heaps; sum their Rss from /proc/<pid>/smaps
```
