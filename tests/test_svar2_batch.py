import numpy as np

from genoray import SparseVar2


def _carried_counts(res: dict, n_samples: int, ploidy: int) -> list[int]:
    """Per-hap carried-variant count for region 0 = var_key slice length +
    dense-present popcount. Invariant to left-alignment position shifts."""
    vk_off = res["vk_off"]
    dp = np.unpackbits(res["dense_present"], bitorder="little")
    dpo = res["dense_present_off"]
    counts = []
    for h in range(n_samples * ploidy):  # region 0 → h = (0*S + s)*P + p = s*P + p
        vk_n = int(vk_off[h + 1] - vk_off[h])
        dn = int(dp[dpo[h] : dpo[h + 1]].sum())
        counts.append(vk_n + dn)
    return counts


def test_overlap_batch_counts_and_dtypes(svar2_store):
    sv = SparseVar2(svar2_store)
    res = sv._overlap_batch("chr1", [(0, 40)])

    # Frozen-contract dtypes.
    for k in ("vk_pos", "vk_key", "dense_pos", "dense_key"):
        assert res[k].dtype == np.int32
    for k in ("vk_off", "dense_present_off", "lut_off"):
        assert res[k].dtype == np.int64
    assert res["dense_present"].dtype == np.uint8
    assert res["dense_range"].shape == (1, 2)

    # H + 1 offsets for 2 samples, ploidy 2, 1 region.
    assert len(res["vk_off"]) == 2 * 2 * 1 + 1
    assert int(res["vk_off"][-1]) == len(res["vk_pos"])

    # Known carriers: SNP@POS3 (S0h0), INS@POS7 (S0h1,S1h0,S1h1),
    # DEL@POS12 (S0h0,S0h1,S1h1) → per-hap [2, 2, 1, 2], total 7.
    assert _carried_counts(res, sv.n_samples, sv.ploidy) == [2, 2, 1, 2]

    # All positions within the reference.
    assert np.all((res["vk_pos"] >= 0) & (res["vk_pos"] < 40))
