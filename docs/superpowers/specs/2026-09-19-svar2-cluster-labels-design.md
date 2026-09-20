# Per-mutation cluster labels for SVAR2 (`annotate_clusters`)

Design for porting SigProfilerClusters' per-mutation subclassification
(doublet / MBS / omikli / kataegis / other) onto SVAR2 stores.

Upstream reference: `AlexandrovLab/SigProfilerClusters` v1.2.2, master
@ `81bfe5bd83832812e24c595bc2338fed42e47b63` (2026-05-08), reviewed from a
shallow clone. Line citations below are to that commit:
`SPC` = `SigProfilerClusters/SigProfilerClusters.py`,
`CF` = `.../classifyFunctions.py`, `HS` = `.../hotspot.py`.

## Problem

`SparseVar2` can already carry COSMIC mutation-catalogue codes per variant
(`annotate_mutations` → SBS96/DBS78/ID83/384) and refit signature
activities, but it has no way to express *clusteredness*: whether a
mutation is a doublet, part of a multi-base substitution, an omikli, or a
kataegis event. GenVarLoader consumers want those labels per mutation,
per sample, as a first-class field they can select, decode, and filter on.

SigProfilerClusters computes exactly these labels, but as a
file-orchestration pipeline: VCF in, hard-coded `output/` trees out,
`multiprocessing`, `pickle` interchanges, and a mandatory cohort of
simulated backgrounds. The classification decision itself is
deterministic, simple, and independent of the machinery that derives its
thresholds. This spec ports the decision, not the orchestration.

## Scope

### In scope (v1)

- Faithful port of `CF.findClustersOfClusters` (VAF/CCF mode) and
  `CF.findClustersOfClusters_noVAF` (no-VAF mode) with
  `correction=False`: inter-mutational distance (IMD) computation, the
  per-sample cutoff pre-filter, 10 kb event grouping, the
  ClassIA/IB/IC/II/III decision tree, and the ClassIII greedy re-split.
- Post-hoc annotation on `SparseVar2` (`annotate_clusters`), storing one
  `u8` label per carrier sample per SNP call in the new FORMAT field
  `cluster_class`.
- Read-back through the existing field machinery
  (`with_fields(["cluster_class"])` / `decode`), no new read API in v1.
- Rust kernel + Python mixin, mirroring the `mutcat` split.
- Upstream parity tests in the existing `sigprofiler` pixi environment.

### Out of scope (v1), deliberately

- **Simulation-derived cutoffs** (`HS.first_run`/`refineIMD`'s z-tests,
  BH FDR, 90/95 % cumulative-fraction rule, 10 000 cap) and the
  SigProfilerSimulator dependency. The caller supplies the cutoff(s);
  feeding upstream's `imds.pickle` values reproduces upstream labels
  exactly.
- **Density correction** (`HS.densityCorrection`, region-specific
  `imds_corrected`). The faithful `correction=False` branch is ported.
- **Processivity subclasses** 2Y/2K/2S/2N (`CF.processivitySubclassification`).
- **Indel subclassification.** Upstream subclassifies SNVs only; the
  SNV-context pipeline never sees indels in its clustered file.
- **Rainfall/IMD plots, matrices per subclass, event-probability KDEs,
  `eventProbability`.**
- **SVAR1** (`SparseVar`), and **dense sub-streams** (see Storage).
- **Write-time classification** (`from_vcf(..., clusters=True)`).
  Clustering needs the complete per-contig per-sample mutation list;
  post-hoc is the honest place for it.

## Upstream algorithm (the part we port)

### Definitions

Positions are compared as differences, so 0-based (genoray) and 1-based
(upstream) coordinates are equivalent everywhere; the store's 0-based
positions are used directly. All mutation lists are per **(sample,
contig)**.

For comparison with upstream's ClassIA/IB/IC/II/III names, the label map
is: `IA → doublet`, `IB → MBS`, `IC → omikli`, `II → kataegis`,
`III → other` (`convertToVCF.py:63-69` documents the same mapping into
`DBS/MBS/omikli/kataegis/other` directories).

### Step 1 — per-sample mutation list

Upstream input is a biallelic-split VCF, one row per call. A `1/2`
genotype therefore yields two rows at the same position. Reproduce this
by taking the sample's SNP calls (all ploidy columns), forming `(pos,
alt)` pairs, deduplicating exactly, and sorting by `(pos, alt_code)`.
A heterozygous single-alt call contributes one mutation; a homozygous-alt
call contributes one (two calls, one distinct pair); a `1/2` genotype
contributes two mutations at distance 0.

The order among equal positions only affects how ≥3 same-position alts
pair. Upstream inherits its file order; the port fixes `alt` code
ascending for determinism. Two same-position mutations always form the
single pair between them either way.

### Step 2 — inter-mutational distance

Upstream: for adjacent same-sample mutations `x`, `y`,

```
gap(x→y) = y.pos - (x.pos - 2 + len(x.ref) + len(x.alt))
```

(`SPC:374-403`). For SNVs this is exactly `y.pos - x.pos`. Each
mutation's recorded IMD is the minimum of its upstream and downstream
gaps (`SPC:404-431`, `final_distances`); the first mutation of a
segment has only the downstream gap, the last only the upstream gap. A
sample with exactly one mutation on the contig gets IMD `1_000_000`
(`SPC:470-485`), i.e. never clustered.

Port: `IMD[i] = min(pos[i]-pos[i-1], pos[i+1]-pos[i])` with one-sided
ends; list length 1 → `1e6`.

### Step 3 — cutoff pre-filter

A mutation is *clustered* iff `IMD <= cutoff[sample]`
(`HS:590-643`; `int(x[0]) <= distance_cut`). Upstream's cutoff is
derived per sample from simulation cohorts and hard-capped at 10 000
(`HS:587-588`). The port takes `cutoff[sample]` from the caller.

### Step 4 — event grouping

Within a sample's clustered mutations (already sorted by position), a
new event starts when the raw position difference to the previous
clustered mutation is `>= 10001`; consecutive clustered mutations with
`pos[j] - pos[j-1] < 10001` chain transitively into one event
(`CF:761`, `CF:823-941`). The threshold is a raw position difference,
not an IMD.

### Step 5 — per-event counts

For each adjacent pair `(a, b)` inside an event, with
`d = pos_b - pos_a` (`CF:1053-1186`, `correction=False` branch):

```
distancesLine   = #{ d > 1 and d <= cutoff[sample] }
distancesFailed = #{ d > cutoff[sample] }
zeroDistances   = #{ d > 1 }
vafs            = #{ round(|vaf_b - vaf_a|, 4) > vaf_cut }     # VAF/CCF path only
```

`vaf_cut` is `0.1` by default, `0.25` when CCFs are used
(`CF:762-764`). Upstream's `unknownVafs` counter is dead code
(`abs(diff) < 0`, `CF:1173-1179`) and is not ported.

### Step 6 — decision tree

VAF path (`CF:1188-1207`), with `n = event size`. An event of size 1
(a clustered mutation whose nearest clustered neighbour is ≥ 10001 bp
away) is `other`; upstream skips such events entirely (`CF:1049-1051`).

```
if vafs == 0 and distancesFailed == 0:
    if distancesLine > 1:
        kataegis  if n >= 4
        omikli    otherwise                       # n == 3 in practice (two gaps need n >= 3)
    else:
        if zeroDistances == 0:
            doublet   if n == 2
            MBS       otherwise                   # n >= 3
        else:
            omikli
else:
    ClassIII (greedy re-split, Step 7)
```

No-VAF path (`CF:2343-2362`) is identical minus the `vafs` term. In
upstream, failing no-VAF events are computed and then silently never
written (there is no ClassIII writer in
`findClustersOfClusters_noVAF`); see Deviations.

### Step 7 — ClassIII greedy re-split (VAF path only)

`CF:1270-1692`, `correction=False`:

```
while len(remaining) > 1:
    event = [remaining[0]]; last = remaining[0]
    for m in remaining[1:]:
        if |vaf_m - vaf_last| < vaf_cut and pos_m - pos_last <= cutoff[sample]:
            event.append(m); last = m
    if len(event) > 1:
        classify(event) with Step 6's tree, where within this sub-event
        distancesLine == 0 is equivalent to zeroDistances == 0
        (all consecutive gaps passed the <= cutoff test)
    else:
        the single mutation is `other`
    remove event from remaining
any last remaining singleton is `other`
```

Upstream records a `reason` (`vaf` / `unknown`) on ClassIII rows; the
`unknown` branch is unreachable (Step 5), so the port stores only the
label.

### Step 8 — label emission

Every mutation of a classified event receives that event's label.
Upstream writes only clustered mutations to subclass files; the port
additionally labels clustered mutations that upstream drops (singletons,
no-VAF failures) as `other`, and labels every non-clustered SNP call as
`nonclustered`. Labels are total over in-scope SNP calls.

## Design

### Public API

```python
def annotate_clusters(
    self,
    *,
    imd_cutoff: float | Mapping[str, float],
    vaf_field: str | None = None,
    vaf_cut: float = 0.1,
    contigs: Sequence[str] | None = None,
) -> None:
    """Label every SNP call with its SigProfilerClusters class."""
```

- No `reference=` argument: the classifier uses no sequence context.
- `imd_cutoff`: scalar applied to every sample, or a mapping from sample
  name to cutoff. A mapping must name every sample in
  `available_samples` and no others; anything else raises `ValueError`
  (fail fast — a silently defaulted cutoff changes labels).
- `vaf_field`: name of a stored FORMAT field (canonical key or bare
  name as in `available_fields`) holding per-call VAF or CCF. Must
  resolve to `category == "format"` with a floating dtype
  (`f16`/`f32`); anything else raises `ValueError`. `None` selects the
  no-VAF path.
- Missing VAF values in that field are mapped to upstream's `-1.5`
  sentinel before comparison: `NaN` for float storage, and the store's
  reserved sentinel when the manifest declares no `default`. A declared
  `default` is an ordinary value (documented).
- `vaf_cut`: passed through; `0.25` reproduces upstream's
  `includedCCFs=True`. Validated `> 0`.
- `contigs`: same resolution semantics as
  `_MutcatMixin.annotate_mutations` (`ContigNormalizer`, intersection
  with `self.contigs`, `ValueError` if nothing resolves). Out-of-scope
  contigs are filled with the `not_annotated` code so that the field is
  safe to read everywhere (see Storage). Passing explicit `contigs`
  still costs one fill write per excluded contig; annotating everything
  is the normal case.

After the call, the same `SparseVar2` instance is stale for the new
field by construction (`available_fields` is read at `__init__`);
`sv2.with_fields(["cluster_class"])` opens the updated manifest. This is
the existing post-hoc pattern and needs no new machinery.

### Label codebook

| code | name | upstream |
|---|---|---|
| 0 | `nonclustered` | not in the clustered file |
| 1 | `doublet` | ClassIA (DBS) |
| 2 | `mbs` | ClassIB |
| 3 | `omikli` | ClassIC |
| 4 | `kataegis` | ClassII |
| 5 | `other` | ClassIII |
| 255 | `not_annotated` | — (out-of-scope contig / non-SNP call) |

Constants live in `src/cluster/mod.rs` and
`python/genoray/_svar2_clusters.py`; a test pins them equal, mirroring
the `mutcat` codebook tests.

### Storage

The label is per **(carrier sample, variant)**, not per variant: a
variant can be a doublet in one sample and isolated in another. That is
exactly FORMAT-field semantics in SVAR2, so the labels are written as a
normal FORMAT field rather than a bespoke per-variant sidecar:

```
{store}/{contig}/fields/format/cluster_class/var_key_snp/values.bin    # u8 per SNP call
{store}/{contig}/fields/format/cluster_class/var_key_indel/values.bin  # 255 per indel call
```

- `values.bin` is 1:1 with the var_key call stream, in the same
  column-major `(sample, ploid)` order the merge produces. Confirmed
  against `merge::merge_var_key_field_values` ("field values are staged
  1:1 with calls", `src/merge.rs:553-556`) and
  `FieldView::value_at` ("element `i` is a var_key **call** index",
  `src/query/field.rs:98-102`). Both haplotypes of a carrier sample get
  the same label; a `1/2` sample has a label per recorded alt.
- Indel calls get 255. Upstream does not subclassify indels, but the
  field reader opens all four sub-streams (`py_query_decode.rs`
  `views: [FieldView; 4]`), so a missing `var_key_indel/values.bin`
  would misindex — the fill is a correctness requirement, not a
  courtesy.
- Out-of-scope contigs (when `contigs=` is passed) get 255-filled
  `values.bin` files of the right length. Without this, selecting the
  field and decoding an unannotated contig would index an empty view
  and panic across the FFI boundary.
- v1 refuses stores with any non-empty dense sub-stream
  (`dense/snp` or `dense/indel` records): their field values are also
  opened by the reader and would need labels of their own. `ValueError`
  naming the contig. Every `from_vcf`/`from_vcf_list` cohort store is
  var_key, which is the target.

`meta.json` additions (stamped last, same atomic-write pattern as
`_MutcatMixin`):

```json
{
  "cluster_version": 1,
  "cluster_contigs": ["chr1", "..."],
  "cluster_cutoff": 1000.0,            // or {"S1": 1000.0, ...}
  "cluster_vaf_field": "VAF",          // or null
  "cluster_vaf_cut": 0.1,
  "fields": [ ..., {"name": "cluster_class", "category": "format",
                    "dtype": "u8", "default": null} ]
}
```

`_load_field_manifest` (`python/genoray/_svar2_fields.py:210-242`) picks
the entry up unchanged; `u8` is already a supported storage dtype.
Re-running is unconditional and overwrites in-scope contigs; a killed
run leaves no manifest entry (meta is stamped only after every contig's
files are renamed into place), so partial output is never advertised.

### Rust

- `src/cluster/mod.rs` — label constants, `CLUSTER_VERSION`,
  `N_LABELS`.
- `src/cluster/classify.rs` — pure functions over a single sample:
  - `fn imds(positions: &[u32]) -> Vec<u32>` (Step 2),
  - `fn cluster_sample(positions: &[u32], vafs: Option<&[f64]>,
    cutoff: f64, vaf_cut: f64) -> Vec<u8>` (Steps 3–8).
  Unit-testable with no I/O; the parity oracle can also drive these
  directly through a test-only binding if useful.
- `src/cluster/write.rs` — build `values.bin` per sub-stream
  (temp + rename, `fsync` before rename, mirroring
  `src/mutcat/sidecar.rs`), deriving per-column byte ranges from
  `vk_snp.column(col)` so peak buffering is one column, not the whole
  contig; each written `values.bin` length must equal that sub-stream's
  total call count (asserted before rename).
- `src/py_cluster.rs` — `PyContigReader::annotate_clusters(base_out_dir,
  chrom, cutoffs: PyReadonlyArray1<f64>, vaf: Option<(name, dtype)>,
  vaf_cut)`; per-sample classification is independent and runs under
  rayon when the cohort is wide.

### Python

- `python/genoray/_svar2_clusters.py` — `_ClustersMixin` with
  `annotate_clusters`, mirroring `_MutcatMixin`'s scope resolution,
  validation, and meta stamping. Wired into `SparseVar2`'s bases in
  `_svar2.py`. Light imports only at module scope (the mutcat lesson:
  keep `import genoray` cheap); `numpy`/`polars` are already imported by
  the mutcat mixin.
- VAF dtype resolution reuses `available_fields` /
  `_META_DTYPE` from `_svar2_fields.py`.

### Errors

| condition | result |
|---|---|
| missing/unknown sample in a `imd_cutoff` mapping | `ValueError` listing them |
| non-positive cutoff or `vaf_cut` | `ValueError` |
| `vaf_field` unknown / not FORMAT / non-float dtype | `ValueError` |
| any non-empty dense sub-stream | `ValueError` (v1 limitation) |
| no contigs resolve from `contigs=` | `ValueError` (mutcat semantics) |

## Deviations from upstream (all deliberate, all documented)

| upstream | port | why |
|---|---|---|
| per-sample cutoff from ≥100 simulated cohorts | caller-supplied `imd_cutoff` | no simulator in genoray; feeding `imds.pickle` reproduces upstream exactly |
| density-corrected regional cutoffs (`correction=True`, the code's default) | not implemented (`correction=False` branch) | needs sims; branch is self-contained and faithful |
| centromere/chromosome-arm split (`SPC:733-881`) | per-contig IMDs, no arm split | upstream only replaces a cross-centromere pair's distance with the adjacent in-arm gap (`SPC:408-431`); the cross gap is always the larger min term, so recorded IMDs and events coincide unless mutations sit inside a centromere |
| no-VAF failed events and clustered singletons silently dropped | labeled `other` | silent data loss; labels must be total |
| `unknownVafs` guard | dropped | dead code (`abs(diff) < 0`) |
| row-level TSV/VCF outputs, `clust_group`, processivity subclasses | one FORMAT field per call | SVAR2-native storage and read path; group ids can be added later |

## Testing

1. **Rust unit tests** (`src/cluster/classify.rs`): IMD one-sided ends and
   the 1-mutation `1e6` rule; dedup by `(pos, alt)`; grouping chains
   across 10 kb; every leaf of the Step 6 tree; the VAF re-split
   (including leftover singletons); boundary values `d == 1`, `d == cutoff`,
   `d == 10000/10001`; multiallelic distance-0 doublet.
2. **Python end-to-end** (`tests/test_svar2_clusters.py`): tiny VCF →
   `from_vcf` → `annotate_clusters` → `with_fields(["cluster_class"])` →
   `decode`; per-call label order verified against a hand-built
   expectation; `contigs=` fill; validation errors; atomicity (block a
   temp path, assert old files survive and meta is unstamped); re-run
   idempotence; a killed-run simulation leaves no manifest entry.
3. **Upstream parity** (`tests/test_clusters_calibration.py`, guarded by
   `pytest.importorskip("SigProfilerClusters")`, run in the existing
   `sigprofiler` pixi env — add `sigprofilerclusters` to
   `[feature.sigprofiler.pypi-dependencies]`):
   - VAF path: write a fixture `output/vcf_files/T_clustered/
     T_clustered_vaf.txt` plus `output/simulations/data/imds.pickle`,
     call the real `CF.findClustersOfClusters(..., correction=False)`,
     parse `subclasses/class{1a,1b,1c,2,3}` files, and compare per-mutation
     labels against the SVAR2 store built from the same mutations and
     VAFs.
   - No-VAF path: same with `SNV/T_clustered.txt` and
     `findClustersOfClusters_noVAF`; because upstream drops failing
     events, assert our `other` set equals upstream's dropped set and all
     non-`other` labels match.
   - The shipped `examples/BRCA_example` row set is used as one realistic
     fixture (rows copied into the test fixture, not vendored wholesale).
4. **Manifest/read invariant**: after annotation on a multi-sample,
   multi-contig fixture, every `(sample, hap)` call's decoded
   `cluster_class` equals the writer's intended value — the test that
   pins the `values.bin` ordering contract.

## Performance

Per contig: one pass per sample column pair (gather positions, alt
codes, optional VAFs), O(calls) for the label vector and O(m log m) for
sorting the sample's mutations. `values.bin` is written directly at
per-column offsets via mmap; no store-wide buffer. Classification is
embarrassingly parallel over samples (rayon). Storage cost: one byte
per SNP call plus one byte per indel call (the fill), i.e. ~25 % of the
`positions.bin` stream.

## Future work

- Cutoff estimation: port `HS.first_run`'s binning + cumulative z-test +
  BH FDR + `refineIMD` binary search on top of a SigProfilerSimulator
  equivalent (or a genoray-native background model) so `imd_cutoff`
  becomes optional.
- Density-corrected regional cutoffs (`correction=True`), including
  upstream's known `cutoffCatch` bug — replicate or fix, but decide
  explicitly.
- Processivity subclasses (`processivitySubclassification`) as label
  values 6–9 or a second field.
- Event group ids (`cluster_group`, u32, per-sample event ordinal) for
  aggregating events.
- Dense sub-stream support (255-fill or computed labels per dense
  element) and write-time `from_vcf(..., clusters=True)`.
- Indel clusteredness (`clustered`/`nonclustered` only, per upstream
  docs).

## Post-implementation requirements

- `skills/genoray-api/SKILL.md` gains `annotate_clusters` and the
  `cluster_class` field/codebook (repo rule: public API changes ship
  with their skill update).
- `CHANGELOG.md` is untouched (commitizen owns it); commits use
  Conventional Commits (`feat(svar2): ...`, `test(clusters): ...`).
