//! Optional runtime calibration of the per-contig reader count.
//!
//! The scale bench fitted a knee at w ~ 3-7 that moves with cohort size. That
//! fit was taken on synthetic corpora on one machine, and node speed on this
//! cluster varies by 2.08x; `t_read` and `t_exec` also move with compression
//! ratio, field count, and ploidy, none of which a fitted knee sees. This
//! measures the ratio on the actual input, on the actual machine.

/// Per-chunk timings from the probe.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rates {
    /// Seconds for ONE shard worker to produce one dense chunk.
    pub t_read_s: f64,
    /// Seconds for `dense2sparse_vk` to consume one dense chunk.
    pub t_exec_s: f64,
}

/// Safety clamp, not a tuning parameter: it bounds the damage from a probe
/// that measured a pathological prefix. The harness never observed a knee
/// above 7, so this sits well clear of any real workload -- reaching it means
/// the probe, not the workload, is what to look at.
pub const W_MAX: usize = 16;

/// Chunks timed per probe. Two is enough to get past first-chunk warmup while
/// staying negligible against a real conversion.
pub const PROBE_CHUNKS: usize = 2;

/// Readers needed to keep one executor fed: `w/t_read >= 1/t_exec`.
///
/// Rounds UP -- rounding down starves the executor, which is the serial stage
/// the whole probe exists to keep busy.
pub fn workers_from_rates(rates: &Rates) -> usize {
    let exec_is_positive = rates.t_exec_s > 0.0;
    if !exec_is_positive || !rates.t_read_s.is_finite() || rates.t_read_s <= 0.0 {
        // A non-positive or non-finite measurement is a broken clock reading
        // on a small chunk, not an infinitely fast stage. Do not divide by it.
        return 1;
    }
    let w = (rates.t_read_s / rates.t_exec_s).ceil();
    if !w.is_finite() {
        return 1;
    }
    (w as usize).clamp(1, W_MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// To keep the executor fed, w readers must supply at least as fast as one
    /// executor drains: w/t_read >= 1/t_exec, so w = ceil(t_read/t_exec).
    #[test]
    fn workers_are_the_read_to_exec_ratio_rounded_up() {
        assert_eq!(
            workers_from_rates(&Rates {
                t_read_s: 0.30,
                t_exec_s: 0.10
            }),
            3
        );
        // 0.35/0.10 = 3.5 -> 4: rounding DOWN starves the executor, which is
        // the bottleneck this whole probe exists to keep busy.
        assert_eq!(
            workers_from_rates(&Rates {
                t_read_s: 0.35,
                t_exec_s: 0.10
            }),
            4
        );
    }

    /// A reader faster than the executor still needs one worker, not zero.
    #[test]
    fn floor_is_one_worker() {
        assert_eq!(
            workers_from_rates(&Rates {
                t_read_s: 0.01,
                t_exec_s: 0.50
            }),
            1
        );
    }

    /// W_MAX bounds the damage from a probe that hit a pathological prefix --
    /// an all-reference stretch reads far faster than it converts. The harness
    /// never observed a knee above 7.
    #[test]
    fn clamped_at_w_max() {
        assert_eq!(
            workers_from_rates(&Rates {
                t_read_s: 100.0,
                t_exec_s: 0.001
            }),
            W_MAX
        );
    }

    /// A zero or negative t_exec is a broken measurement (clock granularity on
    /// a tiny chunk), not an infinitely fast executor. Fall back rather than
    /// dividing by it.
    #[test]
    fn degenerate_exec_time_falls_back_to_one() {
        assert_eq!(
            workers_from_rates(&Rates {
                t_read_s: 1.0,
                t_exec_s: 0.0
            }),
            1
        );
        assert_eq!(
            workers_from_rates(&Rates {
                t_read_s: 1.0,
                t_exec_s: -1.0
            }),
            1
        );
    }
}
