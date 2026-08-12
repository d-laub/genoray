"""Generate bgzf-VCF corpora for the RAM-law sweep via `vcfixture bulk`.

The same generator and profile `pgen_corpus.py` uses, stopped one step before
plink2. Sharing it retires the caveat on `RamLaw::PGEN` that the two backends'
laws are not comparable because their corpora came from different generators.

NOT a replacement for `scale_corpus.py`, which still owns the hold-out, both
V-linearity ladders, the FORMAT-field corpora and `size_corpus`'s chunk-size
derivation.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from scripts.bench_svar2.records import CorpusManifest, from_json, to_json
from scripts.bench_svar2.vcfixture_cli import PROFILE, bulk, cli_version

# Versions THIS module's generation logic (layout, indexing, manifest shape) --
# not the CLI's output bytes, which `generator_cli_version` records separately.
GENERATOR_VERSION = 1


@dataclass(frozen=True)
class VcfCorpusSpec:
    samples: int
    variants: int
    contigs: tuple[str, ...]
    seed: int


def corpus_stem(spec: VcfCorpusSpec) -> str:
    """Filename stem for a corpus of this shape.

    Corpora land FLAT in one directory because `model._load_manifests` globs
    `*.manifest.json` non-recursively and keys by FILENAME -- a per-corpus
    subdirectory would make every manifest `corpus.manifest.json` and they
    would collide. The `vcfx_` prefix keeps these distinct from
    `scale_corpus.py`'s `s{N}` stems in that same flat namespace.
    """
    return f"vcfx_s{spec.samples}_v{spec.variants}"


def generate(spec: VcfCorpusSpec, outdir: Path) -> CorpusManifest:
    """Generate (or reuse) a bgzf-VCF corpus in `outdir`.

    Cached on the full spec plus GENERATOR_VERSION plus the profile's content
    hash plus the CLI version: `vcfixture --seed` is byte-reproducible
    regardless of thread count WITHIN a CLI major version, so a corpus is
    reproducible and there is no reason to pay for it twice -- but v0.5.0
    changed those bytes, so the version belongs in the key.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    stem = corpus_stem(spec)
    manifest_path = outdir / f"{stem}.manifest.json"
    version = cli_version()
    profile_hash = hashlib.sha256(PROFILE.read_bytes()).hexdigest()
    # Round-tripped through JSON before comparing: `dataclasses.asdict(spec)`
    # holds `contigs` as a tuple while the cached copy loaded off disk holds a
    # list, and comparing those directly never matches -- the corpus would be
    # silently regenerated on every call.
    key = json.loads(
        json.dumps(
            {
                **dataclasses.asdict(spec),
                "generator_version": GENERATOR_VERSION,
                "profile_hash": profile_hash,
                "cli_version": version,
            }
        )
    )

    if manifest_path.exists():
        cached = json.loads(manifest_path.read_text())
        if cached.get("_key") == key and Path(cached["path"]).exists():
            payload = {k: v for k, v in cached.items() if k != "_key"}
            return from_json(CorpusManifest, json.dumps(payload))

    vcf = outdir / f"{stem}.vcf.gz"
    bulk(
        samples=spec.samples,
        variants=spec.variants,
        contigs=spec.contigs,
        seed=spec.seed,
        fmt="vcf-gz",
        out=vcf,
    )
    # The sharded reader seeks per shard, so an index is not optional.
    subprocess.run(["tabix", "-f", "-p", "vcf", str(vcf)], check=True)

    manifest = CorpusManifest(
        path=str(vcf),
        samples=spec.samples,
        variants=spec.variants,
        contigs=spec.contigs,
        format_fields=(),
        ploidy=2,
        cells=spec.samples * spec.variants,
        compressed_bytes=vcf.stat().st_size,
        seed=spec.seed,
        generator_version=GENERATOR_VERSION,
        generator_cli_version=version,
    )
    payload = json.loads(to_json(manifest))
    payload["_key"] = key
    manifest_path.write_text(json.dumps(payload, indent=1) + "\n")
    return manifest
