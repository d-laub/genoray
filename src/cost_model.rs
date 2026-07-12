//! Per-variant dense-vs-sparse routing. Pure leaf: no I/O, all integer bit
//! arithmetic (no floats, so the crossover is exact and reproducible).
//! Costs are the *actual on-disk bits* one variant occupies in each
//! representation (see docs/roadmap/data-model.md#dense-vs-sparse-cost-model).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    Snp,
    Indel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Representation {
    VarKey,
    Dense,
}

/// Per-call u32 position, in bits. Paid once per call in `var_key`, once per
/// variant in `dense`.
pub const POS_BITS: u64 = 32;

/// Key width in bits by class: SNP is a 2-bit ALT code, indel a 32-bit key.
#[inline]
pub fn key_bits(class: Class) -> u64 {
    match class {
        Class::Snp => 2,
        Class::Indel => 32,
    }
}

/// Mutational-signature sidecar cost, in bits, when signatures are enabled:
/// an 8-bit mutation-category code plus a 2-bit ref-base code for SNPs.
pub const SIDECAR_BITS_SNP: u64 = 10; // 8-bit code + 2-bit ref
/// Mutational-signature sidecar cost, in bits, for indels: 8-bit code only
/// (no ref-base disambiguation needed).
pub const SIDECAR_BITS_INDEL: u64 = 8; // 8-bit code

/// Choose the cheaper representation for one variant with `x_calls` carriers.
///
/// var_key = x · (POS_BITS + key_bits + sidecar_bits + info_bits + format_bits)
///           (position + key + sidecar + INFO + FORMAT inlined per call)
/// dense   = POS_BITS + key_bits + np + sidecar_bits + info_bits + format_bits · n_samples
///           (table row once + 1-bit-per-hap mask + sidecar once + INFO once +
///           FORMAT per sample)
///
/// `sidecar_bits` is the per-record mutational-signature sidecar cost (0 when
/// signatures are disabled; see `SIDECAR_BITS_SNP`/`SIDECAR_BITS_INDEL`). It is
/// paid once per carrier call in `var_key` (the sidecar rides along with each
/// inlined call) but only once per variant in `dense` (the sidecar lives in
/// the table row, not the per-hap mask).
///
/// `info_bits`/`format_bits` are the summed per-record storage widths (in
/// bits) of the active INFO/FORMAT fields (0 when no fields are configured).
/// INFO is amortized once per variant in `dense` but paid per carrier call in
/// `var_key`; FORMAT is paid once per sample in `dense` (every sample gets a
/// column entry) but per carrier call in `var_key`.
///
/// Route to `Dense` iff strictly cheaper; ties break to `VarKey`.
#[inline]
pub fn choose_representation(
    class: Class,
    n_samples: usize,
    ploidy: usize,
    x_calls: usize,
    sidecar_bits: u64,
    info_bits: u64,
    format_bits: u64,
) -> Representation {
    let np = (n_samples as u64) * (ploidy as u64);
    let x = x_calls as u64;
    let per_call = POS_BITS + key_bits(class) + sidecar_bits;
    let var_key_bits = x * (per_call + info_bits + format_bits);
    let dense_bits = POS_BITS
        + key_bits(class)
        + np
        + sidecar_bits
        + info_bits
        + format_bits * (n_samples as u64);
    if dense_bits < var_key_bits {
        Representation::Dense
    } else {
        Representation::VarKey
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_x_zero_is_var_key() {
        // No carriers: var_key costs 0, dense costs a full row — var_key wins.
        assert_eq!(
            choose_representation(Class::Snp, 1000, 2, 0, 0, 0, 0),
            Representation::VarKey
        );
    }

    #[test]
    fn test_snp_crossover_np2000() {
        // np=2000: dense=32+2+2000=2034; var_key=34x. Dense wins when 34x > 2034
        // → x >= 60 (34*59=2006 < 2034 → var_key; 34*60=2040 > 2034 → dense).
        assert_eq!(
            choose_representation(Class::Snp, 1000, 2, 59, 0, 0, 0),
            Representation::VarKey
        );
        assert_eq!(
            choose_representation(Class::Snp, 1000, 2, 60, 0, 0, 0),
            Representation::Dense
        );
    }

    #[test]
    fn test_indel_crossover_np2000() {
        // np=2000: dense=32+32+2000=2064; var_key=64x. Dense when 64x > 2064
        // → x >= 33 (64*32=2048 < 2064; 64*33=2112 > 2064).
        assert_eq!(
            choose_representation(Class::Indel, 1000, 2, 32, 0, 0, 0),
            Representation::VarKey
        );
        assert_eq!(
            choose_representation(Class::Indel, 1000, 2, 33, 0, 0, 0),
            Representation::Dense
        );
    }

    #[test]
    fn test_tie_breaks_to_var_key() {
        // Construct an exact tie: dense_bits == var_key_bits → VarKey.
        // Snp np=32: dense=32+2+32=66; per_call=34. No integer x makes 34x==66,
        // so use a case where dense==var_key: pick class/np so equality holds.
        // Indel np=32: dense=32+32+32=96; per_call=64; 64*x==96 has no int soln.
        // Use np such that dense is a multiple of per_call:
        // Snp per_call=34, choose np so 32+2+np = 34*k. np=34*2-34=34 → dense=34+34=...
        // Simpler: assert the boundary just-below stays VarKey (covered above) and
        // that a hand-built equal case resolves to VarKey via direct formula check.
        let np = 34u64 * 3 - 34; // = 68 → dense_bits = 34 + 68 = 102 = 34*3
        let n = np as usize; // ploidy 1
        assert_eq!(
            choose_representation(Class::Snp, n, 1, 3, 0, 0, 0), // var_key = 34*3 = 102 == dense
            Representation::VarKey,
            "exact tie must resolve to VarKey"
        );
    }

    #[test]
    fn sidecar_bits_shift_snp_crossover_toward_dense() {
        // Without sidecar: np=2000 → dense wins at x>=60 (see test_snp_crossover_np2000).
        // With +10 sidecar/call on var_key and +10 once on dense:
        // dense = 32+2+2000+10 = 2044; per_call = 34+10 = 44.
        // dense < 44x → x > 46.45 → dense at x>=47 (vs 60 without).
        assert_eq!(
            choose_representation(Class::Snp, 1000, 2, 46, SIDECAR_BITS_SNP, 0, 0),
            Representation::VarKey
        );
        assert_eq!(
            choose_representation(Class::Snp, 1000, 2, 47, SIDECAR_BITS_SNP, 0, 0),
            Representation::Dense
        );
    }

    #[test]
    fn zero_sidecar_matches_legacy() {
        assert_eq!(
            choose_representation(Class::Snp, 1000, 2, 60, 0, 0, 0),
            Representation::Dense
        );
    }

    #[test]
    fn info_bits_shift_toward_dense() {
        // np=2000, x=59 sits just below the (0-field) crossover (VarKey; see
        // test_snp_crossover_np2000). INFO is amortized once per variant in
        // dense (2034+32=2066) but paid on every one of the 59 carrier calls
        // in var_key (59*(34+32)=3894), so a single 32-bit INFO field alone
        // flips this variant from VarKey to Dense.
        let base = choose_representation(Class::Snp, 1000, 2, 59, 0, 0, 0);
        let with_info = choose_representation(Class::Snp, 1000, 2, 59, 0, 32, 0);
        assert_eq!(base, Representation::VarKey);
        assert_eq!(with_info, Representation::Dense);
    }

    #[test]
    fn format_bits_do_not_cancel() {
        // n_samples=10, ploidy=35 (np=350), x=11: baseline (0-field) dense=384
        // vs var_key=374 -> VarKey. FORMAT costs format_bits*n_samples once in
        // dense (paid per sample, genotype-aligned Option A) vs
        // format_bits*x_calls in var_key (paid per carrier call); since
        // x_calls(11) > n_samples(10) here, a 16-bit FORMAT field grows the
        // var_key side faster than the dense side (550 vs 544), flipping this
        // variant to Dense. (High ploidy is chosen purely to keep x_calls >
        // n_samples while the 0-field baseline is still VarKey; see report
        // for the algebra.)
        let r = choose_representation(Class::Snp, 10, 35, 11, 0, 0, 16);
        assert_eq!(r, Representation::Dense);

        let base = choose_representation(Class::Snp, 10, 35, 11, 0, 0, 0);
        assert_eq!(base, Representation::VarKey);
    }

    proptest::proptest! {
        // Monotonic: once Dense wins at x, it wins for every larger x.
        #[test]
        fn test_monotonic_in_x(
            n in 1usize..5000,
            ploidy in 1usize..3,
            x in 0usize..20000,
        ) {
            let here = choose_representation(Class::Snp, n, ploidy, x, 0, 0, 0);
            if here == Representation::Dense {
                let more = choose_representation(Class::Snp, n, ploidy, x + 1, 0, 0, 0);
                prop_assert_eq!(more, Representation::Dense);
            }
        }
    }
}
