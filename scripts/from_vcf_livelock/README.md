# `from_vcf` livelock repro fixture (#135) — generation notes

Diagnostic-spike Task 1: how to synthesize a multi-contig BCF carrying a
`##FORMAT=<ID=VAF,Number=A,Type=Float,...>` field via the `vcfixture 0.3.0
bulk` CLI, for reproducing the `SparseVar2.from_vcf` concurrent-chromosome
livelock.

## Install (off-NFS target dir)

```bash
pixi run bash -lc 'CARGO_TARGET_DIR=/tmp/genoray-target-svar2 cargo install vcfixture --version 0.3.0 --features cli --root /tmp/vcfixture-cli --locked'
export PATH=/tmp/vcfixture-cli/bin:$PATH
vcfixture bulk --help
```

Installed clean (`vcfixture v0.3.0`, executables `vcfixture` and
`validate-profile`). There is no `--version` flag on the top-level binary
(`vcfixture --version` errors with "unexpected argument"); `vcfixture bulk`
is the only generation subcommand (`vcfixture <COMMAND>` → `bulk` | `help`).

## `vcfixture bulk --help` structure (as installed)

```
Usage: vcfixture bulk [OPTIONS] --output <OUTPUT>

  --profile <PROFILE>                Builtin profile name (germline-1kgp,
                                      germline-1kgp-unphased, somatic-gdc),
                                      or a path to a profile JSON
                                      [default: germline-1kgp]
  --samples <SAMPLES>                [default: 1]
  --contigs <CONTIGS>                [default: chr1 chr2 chr3]
  --target-size <TARGET_SIZE>        e.g. 100MB
  --records <RECORDS>                exact total, split across contigs
  --records-per-contig <RECORDS_PER_CONTIG>
  --payload <PAYLOAD>                Override the profile's payload preset
                                      [possible values: gt-only, gt-vaf,
                                      gatk, mutect2]
  --format <FORMAT>                  [default: bcf] [bcf, vcf-gz, vcf]
  --seed <SEED>                      [default: 0]
  --compression-level <COMPRESSION_LEVEL>
  --threads <THREADS>
  -o, --output <OUTPUT>              also writes a sibling `.csi` and
                                      `.summary.json`
```

There is only one somatic-flavored builtin profile: `somatic-gdc` (alongside
`germline-1kgp` / `germline-1kgp-unphased`). There is no `gt-vaf`-labeled
profile as such — `gt-vaf` is a **payload preset**, selectable independently
of profile via `--payload`, that emits `GT` + `VAF` per sample.

## Can `bulk` natively emit `VAF Number=A Float`? No — confirmed empirically and in source.

Probed three combinations and inspected the crate source
(`~/.cargo/registry/src/*/vcfixture-0.3.0/src/bulk/{mod.rs,profile.rs,generate.rs}`):

| Invocation | VAF-related FORMAT header line |
|---|---|
| `--payload gt-vaf` (any profile) | `##FORMAT=<ID=VAF,Number=1,Type=Float,...>` |
| `--profile somatic-gdc` (default payload) | same — `somatic-gdc`'s dialed payload is `gt-vaf`, still `Number=1` |
| `--payload mutect2` | emits `AF` (not `VAF`), also `Number=1` |

Source confirms this is structural, not a flag we missed:
- `Dialed.payload` (`src/bulk/profile.rs`) is a closed enum `{GtOnly, GtVaf,
  Gatk, Mutect2}` — no per-field Number/Type override reachable from the
  profile JSON or the CLI.
- `src/bulk/mod.rs` hard-codes the FORMAT header definition per payload key:
  `"VAF" | "AF" => Map::<HeaderFormatMap>::new(FormatNumber::Count(1), ...)`.

So **no CLI flag, profile, or profile-JSON knob produces `VAF Number=A`** —
the `bulk` generator's `VAF`/`AF` fields are always `Number=1`. (The crate
does have a separate hand-rolled `VcfBuilder`/`Field::typed(..., Number::A,
...)` API for small precision-crafted fixtures — see the `vcfixture` skill —
but that's a different, non-`bulk` code path and isn't suited to generating
a multi-MB benchmark-scale file.)

## Chosen mechanism: native `gt-vaf` payload + a pure header rewrite (no data rewrite needed)

Both bundled profiles (`germline-1kgp`, `somatic-gdc`) fit
`"multiallelic_rate": 0.0` — `bulk` never emits a multiallelic record from
either. Verified on a 272,501-record probe: every `ALT` column has exactly
one allele. Since `Number=A`'s cardinality is `n_alt` per record, and every
record here has `n_alt == 1`, the *on-disk VAF values* that `gt-vaf` already
writes (one float per sample) are already valid for a `Number=A` field —
only the **header declaration** (`Number=1` → `Number=A`) needs to change.
No per-record value synthesis/injection is needed for this profile family.

This means the fallback is a plain `bcftools reheader`, not a record-level
rewrite:

All generated output (the raw `bulk` BCF, the patched header, the
reheadered BCF, and its index) MUST live under the ephemeral
`$CLAUDE_JOB_DIR/tmp` scratch area, never the repo checkout's working
directory (which is NFS-backed) — every output path below is prefixed
accordingly:

```bash
export PATH=/tmp/vcfixture-cli/bin:$PATH
OUT="$CLAUDE_JOB_DIR/tmp/probe"
mkdir -p "$OUT"

# 1. Generate with the gt-vaf payload (VAF present, but Number=1 in the header).
vcfixture bulk --samples 50 --contigs chr1,chr2 \
    --target-size 5MB --seed 0 --payload gt-vaf \
    -o "$OUT/probe.bcf"

# 2. Dump the header, patch the VAF FORMAT line's Number from 1 to A.
pixi run bash -lc "bcftools view -h \"$OUT/probe.bcf\" > \"$OUT/header.txt\""
sed -i 's/##FORMAT=<ID=VAF,Number=1,Type=Float,Description="Variant allele frequency">/##FORMAT=<ID=VAF,Number=A,Type=Float,Description="Variant allele frequency">/' "$OUT/header.txt"

# 3. Swap the header in place (record bytes are untouched — bcftools reheader
#    does not re-encode data, so this is cheap and exact).
pixi run bash -lc "bcftools reheader -h \"$OUT/header.txt\" -o \"$OUT/probe.vaf-a.bcf\" \"$OUT/probe.bcf\""
pixi run bash -lc "bcftools index -f \"$OUT/probe.vaf-a.bcf\""
```

Verified the result is well-formed and round-trips through the reader
genoray actually uses:
- `bcftools view` / `bcftools index -s` parse `probe.vaf-a.bcf` without
  error (272,501 records across `chr1`/`chr2`, matching the pre-reheader
  count).
- `cyvcf2.VCF(...)` opens it and `rec.format("VAF")` returns a
  `(n_samples, 1)` float array per record, as expected for a biallelic
  `Number=A` field.

### Caveat for Task 2 / future profiles

This reheader-only trick is only correct because `n_alt == 1` for every
record under the two bundled profiles. If Task 2 (or a future profile JSON)
ever turns on `multiallelic_rate > 0`, a pure header patch would silently
produce a header/data mismatch (declared `Number=A` cardinality `> 1` per
multiallelic record, but only 1 stored value) — that would need a real
per-record rewrite (e.g. via `pysam`/`cyvcf2`, duplicating or perturbing the
single generated VAF across each ALT). Not needed for the repro fixture as
currently scoped; flagging so a future contributor doesn't reheader a
multiallelic-enabled file and assume it's still valid.

## Step 5 verification (acceptance check)

```bash
pixi run bash -lc 'bcftools view -h "$CLAUDE_JOB_DIR/tmp/probe/probe.vaf-a.bcf" | grep -E "ID=VAF.*Number=A.*Type=Float"'
```

Output (one matching line, as expected):

```
##FORMAT=<ID=VAF,Number=A,Type=Float,Description="Variant allele frequency">
```

## Full header (contigs + FORMAT lines) of the final probe fixture

```
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=VAF,Number=A,Type=Float,Description="Variant allele frequency">
##contig=<ID=chr1,length=5828670>
##contig=<ID=chr2,length=5954436>
```

(50 samples `s0`..`s49`, 272,501 records total, generated in ~44s at
`--target-size 5MB`.)

## Generator usage

Task 2 wraps the mechanism above into a reusable script,
`scripts/from_vcf_livelock/generate_repro.py`:

```bash
python scripts/from_vcf_livelock/generate_repro.py \
    --out <dir> --samples N --contigs chr1,chr2,... \
    --target-size 100MB --seed K
```

| Flag | Meaning |
|---|---|
| `--out <dir>` | Output directory; writes `<dir>/cohort.bcf` + `<dir>/cohort.bcf.csi` |
| `--samples N` | Number of samples |
| `--contigs C` | Comma-separated contig names, e.g. `chr1,chr2,chr3,chr4,chr5,chr6` |
| `--target-size S` | Approximate output size, e.g. `8MB`, `500MB` |
| `--seed K` | RNG seed passed to `vcfixture bulk` |

The script resolves the `vcfixture` binary from `PATH` first, then falls
back to `/tmp/vcfixture-cli/bin/vcfixture` (the Task-1 install location). If
neither is found, it raises with the install command from this README's
"Install" section rather than silently reinstalling. It always emits a
`cohort.bcf` with the `VAF` FORMAT header patched to `Number=A` (applying the
reheader fallback documented above) and ensures a CSI index exists.

As with all commands here, keep output under `$CLAUDE_JOB_DIR/tmp` (or an
equivalent non-NFS scratch dir) — never the repo checkout.

Covered by `tests/test_from_vcf_livelock_fixture.py`, which generates a small
(`--target-size 8MB`, 40 samples, 6 contigs) cohort and asserts the BCF/CSI
exist, the header carries `VAF Number=A Type=Float` and ≥5 `##contig=`
lines, and `bcftools index -s` reports ≥5 contigs.
