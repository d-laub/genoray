import numpy as np

from genoray import SparseVar2


def test_decode_record_shape_and_counts(svar2_store):
    sv = SparseVar2(svar2_store)
    rag = sv.decode("chr1", [(0, 40)])

    # Record layout: one shared variant-axis offsets object, shape (R,S,P,None).
    assert rag.shape[:3] == (1, sv.n_samples, sv.ploidy)
    assert rag.shape[3] is None
    # Fields present.
    assert set(rag.dtype.names) == {"pos", "ilen", "allele"}

    # Per-hap lengths (region 0): SNP@3→S0h0; INS@7→S0h1,S1h0,S1h1;
    # DEL@12→S0h0,S0h1,S1h1  ⇒ [2, 2, 1, 2].
    lengths = rag["pos"].lengths.reshape(-1)
    assert lengths.tolist() == [2, 2, 1, 2]

    # region_counts is the decode-free image of those lengths, shaped (R,S,P).
    counts = sv.region_counts("chr1", [(0, 40)])
    assert counts.shape == (1, sv.n_samples, sv.ploidy)
    assert counts.reshape(-1).tolist() == [2, 2, 1, 2]

    # SNP field spot-check (unambiguous under normalization): ilen 0, ALT 'G'.
    pos0 = np.asarray(rag["pos"].data)
    ilen0 = np.asarray(rag["ilen"].data)
    snp = np.where(pos0 == 2)[0]  # 0-based POS of VCF POS 3
    assert snp.size >= 1
    assert ilen0[snp[0]] == 0


def test_decode_empty_region(svar2_store):
    sv = SparseVar2(svar2_store)
    rag = sv.decode("chr1", [(0, 1)])  # before the first variant
    assert rag["pos"].lengths.reshape(-1).tolist() == [0, 0, 0, 0]
