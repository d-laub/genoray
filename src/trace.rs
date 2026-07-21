//! Env-gated diagnostic heartbeats for the #135 `from_vcf` concurrent-chromosome
//! livelock spike.
//!
//! Set `GENORAY_TRACE=1` to enable stderr heartbeats at a handful of pipeline
//! seams (reader chunk assembly / forward-to-`tx_dense` in `shard_exec.rs`,
//! `dense2sparse_vk` enter/exit and `tx_sparse.send` in `executor.rs`, and a
//! one-time concurrency-plan line in `orchestrator.rs`). Unset (the default),
//! `traced()` is a single cached `bool` read and no `eprintln!` ever
//! executes -- default pipeline behavior and output are unchanged.

/// Whether `GENORAY_TRACE` is set. Read from the environment once per process
/// (cached in a `OnceLock`) -- cheap enough to call at every `trace_ll!` site
/// without callers caching it themselves.
#[inline]
pub(crate) fn traced() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("GENORAY_TRACE").is_some())
}

/// `eprintln!`-alike gated on `traced()`. When `GENORAY_TRACE` is unset the
/// format arguments are never evaluated -- only the cached `bool` check runs,
/// so this is zero-cost on the default (untraced) path.
macro_rules! trace_ll {
    ($($a:tt)*) => {
        if $crate::trace::traced() {
            eprintln!($($a)*);
        }
    };
}
pub(crate) use trace_ll;
