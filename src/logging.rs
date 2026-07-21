//! SVAR2 write-path logging & progress events bridged to Python.
use crossbeam_channel::Sender;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Mutex, Once};
use tracing::field::{Field, Visit};
use tracing::level_filters::LevelFilter;
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogLevel {
    Warning,
    Info,
    Debug,
}

impl LogLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Warning => "warning",
            LogLevel::Info => "info",
            LogLevel::Debug => "debug",
        }
    }
}

#[derive(Debug)]
pub enum Event {
    ContigStart {
        chrom: String,
        total: Option<u64>,
    },
    Progress {
        chrom: String,
        delta: u64,
    },
    ContigDone {
        chrom: String,
        kept: u64,
        excluded: u64,
        elapsed_ms: u64,
    },
    Log {
        level: LogLevel,
        chrom: Option<String>,
        message: String,
        target: String,
    },
}

#[derive(Clone)]
pub struct EventSink {
    inner: Option<Arc<SinkInner>>,
}

struct SinkInner {
    tx: Sender<Event>,
    // Keyed by chrom so ticks from concurrently-dispatched contigs (rayon
    // fan-out over chroms) never accumulate into a shared counter and get
    // flushed under the wrong chrom's label.
    pending: Mutex<HashMap<String, u64>>,
    flush_every: u64,
}

impl EventSink {
    pub fn new(tx: Sender<Event>, flush_every: u64) -> Self {
        EventSink {
            inner: Some(Arc::new(SinkInner {
                tx,
                pending: Mutex::new(HashMap::new()),
                flush_every: flush_every.max(1),
            })),
        }
    }
    pub fn disabled() -> Self {
        EventSink { inner: None }
    }

    pub fn contig_start(&self, chrom: &str, total: Option<u64>) {
        if let Some(i) = &self.inner {
            let _ = i.tx.send(Event::ContigStart {
                chrom: chrom.to_string(),
                total,
            });
        }
    }

    pub fn tick(&self, chrom: &str, n: u64) {
        if let Some(i) = &self.inner {
            let take = {
                let mut map = i.pending.lock().unwrap();
                let entry = map.entry(chrom.to_string()).or_insert(0);
                *entry += n;
                if *entry >= i.flush_every {
                    std::mem::replace(entry, 0)
                } else {
                    0
                }
            };
            if take > 0 {
                let _ = i.tx.send(Event::Progress {
                    chrom: chrom.to_string(),
                    delta: take,
                });
            }
        }
    }

    pub fn flush(&self, chrom: &str) {
        if let Some(i) = &self.inner {
            let take = {
                let mut map = i.pending.lock().unwrap();
                map.remove(chrom).unwrap_or(0)
            };
            if take > 0 {
                let _ = i.tx.send(Event::Progress {
                    chrom: chrom.to_string(),
                    delta: take,
                });
            }
        }
    }

    pub fn contig_done(&self, chrom: &str, kept: u64, excluded: u64, elapsed_ms: u64) {
        if let Some(i) = &self.inner {
            let _ = i.tx.send(Event::ContigDone {
                chrom: chrom.to_string(),
                kept,
                excluded,
                elapsed_ms,
            });
        }
    }

    pub fn send_log(&self, level: LogLevel, chrom: Option<&str>, target: &str, message: String) {
        if let Some(i) = &self.inner {
            let _ = i.tx.send(Event::Log {
                level,
                chrom: chrom.map(|s| s.to_string()),
                message,
                target: target.to_string(),
            });
        }
    }
}

pub fn level_from_str(s: &str) -> Option<LevelFilter> {
    match s {
        "off" => Some(LevelFilter::OFF),
        "warning" => Some(LevelFilter::WARN),
        "info" => Some(LevelFilter::INFO),
        "debug" => Some(LevelFilter::DEBUG),
        _ => None,
    }
}

fn to_log_level(l: &tracing::Level) -> LogLevel {
    match *l {
        tracing::Level::ERROR | tracing::Level::WARN => LogLevel::Warning,
        tracing::Level::INFO => LogLevel::Info,
        _ => LogLevel::Debug,
    }
}

#[derive(Default)]
struct FieldGrab {
    message: String,
    chrom: Option<String>,
}

impl Visit for FieldGrab {
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "chrom" {
            self.chrom = Some(value.to_string());
        } else if field.name() == "message" {
            self.message = value.to_string();
        }
    }
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        let v = format!("{value:?}");
        if field.name() == "chrom" {
            self.chrom = Some(v.trim_matches('"').to_string());
        } else if field.name() == "message" {
            self.message = v;
        }
    }
}

// --- Process-global routing state --------------------------------------
//
// `tracing::subscriber::with_default` is THREAD-LOCAL: it does not propagate
// to OS threads spawned below a rayon pool (reader/executor/writer threads
// inside `process_chromosome`), so events emitted there were silently
// dropped. `tracing`'s cross-thread propagation only works through the
// process-global default subscriber (`set_global_default`), so instead of
// scoping the *subscriber*, we install ONE global subscriber for the whole
// process and scope only the *destination* (which `EventSink` is currently
// "live", and at what verbosity) via these globals.
//
// Which sink (if any) `ChannelLayer` routes to right now. Set for the
// duration of a write via `with_channel_subscriber`; `None` otherwise (e.g.
// pure-Rust bench runs, or between writes).
static CURRENT_SINK: Mutex<Option<EventSink>> = Mutex::new(None);
// Current channel log level as a rank: off=0, warning=1, info=2, debug=3.
static CURRENT_LEVEL: AtomicU8 = AtomicU8::new(2); // info

fn level_rank(s: &str) -> u8 {
    match s {
        "off" => 0,
        "warning" => 1,
        "info" => 2,
        "debug" => 3,
        _ => 2,
    }
}

fn event_rank(l: &tracing::Level) -> u8 {
    match *l {
        tracing::Level::ERROR | tracing::Level::WARN => 1,
        tracing::Level::INFO => 2,
        // TRACE is treated as debug for channel gating, but TRACE events
        // never reach `on_event` in the first place: the channel layer is
        // filtered to max DEBUG at install time (see `ensure_global_subscriber`)
        // so `genoray::monitor`'s trace-level sampler stays GENORAY_LOG-only.
        tracing::Level::DEBUG | tracing::Level::TRACE => 3,
    }
}

pub struct ChannelLayer;
impl ChannelLayer {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ChannelLayer {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: tracing::Subscriber> Layer<S> for ChannelLayer {
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        // Channel-level gate: drop events more verbose than the active level,
        // and drop everything when level is "off" (rank 0).
        let active = CURRENT_LEVEL.load(Ordering::Relaxed);
        let lvl = event.metadata().level();
        if active == 0 || event_rank(lvl) > active {
            return;
        }
        // Clone the sink (Arc-backed, cheap) out of the guard and drop the
        // lock before sending, so the critical section is just the clone.
        let sink = { CURRENT_SINK.lock().unwrap().clone() };
        if let Some(sink) = sink {
            let mut g = FieldGrab::default();
            event.record(&mut g);
            sink.send_log(
                to_log_level(lvl),
                g.chrom.as_deref(),
                event.metadata().target(),
                g.message,
            );
        }
    }
}

static INSTALL: Once = Once::new();

/// Install the single process-global tracing subscriber, at most once per
/// process. Combines the channel layer (routes to `CURRENT_SINK`, gated by
/// `CURRENT_LEVEL`) with an optional stderr fmt layer driven by `GENORAY_LOG`.
/// A library installing a global default is impolite but REQUIRED for
/// cross-thread routing (see module-level comment above); if the host
/// process already installed a global default, `set_global_default` fails
/// and we ignore the error — channel logging degrades gracefully (the
/// progress bar is unaffected: it uses direct `EventSink` sends, not
/// `tracing`).
fn ensure_global_subscriber() {
    INSTALL.call_once(|| {
        // Channel layer is filtered to max DEBUG so `on_event` ever sees
        // debug events at all; the real per-write gate is `CURRENT_LEVEL`
        // inside `on_event`. TRACE stays out of the channel entirely.
        let channel = ChannelLayer::new().with_filter(LevelFilter::DEBUG);
        // Optional stderr fmt layer, only active when GENORAY_LOG is set.
        let fmt_filter = tracing_subscriber::EnvFilter::try_from_env("GENORAY_LOG").ok();
        let fmt = fmt_filter.map(|f| {
            tracing_subscriber::fmt::layer()
                .with_target(true)
                .compact()
                .with_writer(std::io::stderr)
                .with_filter(f)
        });
        let subscriber = tracing_subscriber::registry().with(channel).with(fmt);
        let _ = tracing::subscriber::set_global_default(subscriber);
    });
}

/// Route `tracing::` events emitted during `f` (from ANY thread, including
/// OS threads spawned below a rayon pool) to `sink` at `level`.
///
/// NOTE on concurrency: `CURRENT_SINK`/`CURRENT_LEVEL` are process-global
/// slots, not thread-local or call-scoped. Nested/sequential calls on one
/// thread compose correctly (each restores the caller's previous
/// sink/level on exit, RAII-style, even on panic). But two *concurrent*
/// `with_channel_subscriber` calls in the same process (e.g. two Python
/// threads each calling `from_vcf` at once) share the one global slot —
/// last writer wins. This is an accepted limitation: the progress bar is
/// unaffected (it sends directly to its `EventSink`, bypassing `tracing`),
/// and concurrent in-process writes are not a supported logging scenario.
pub fn with_channel_subscriber<R>(sink: EventSink, level: &str, f: impl FnOnce() -> R) -> R {
    ensure_global_subscriber();
    let prev_level = CURRENT_LEVEL.swap(level_rank(level), Ordering::Relaxed);
    let prev_sink = {
        let mut g = CURRENT_SINK.lock().unwrap();
        g.replace(sink)
    };

    struct Restore {
        prev_level: u8,
        prev_sink: Option<EventSink>,
    }
    impl Drop for Restore {
        fn drop(&mut self) {
            CURRENT_LEVEL.store(self.prev_level, Ordering::Relaxed);
            *CURRENT_SINK.lock().unwrap() = self.prev_sink.take();
        }
    }
    let _restore = Restore {
        prev_level,
        prev_sink,
    };

    f()
}

/// Install the global tracing subscriber, at most once per process, so a
/// pure-Rust entry point (bench bin) gets the `GENORAY_LOG` stderr fmt layer.
/// Called by `src/bin/bench_from_vcf_list.rs` — NOT by the Python pipeline,
/// which uses `with_channel_subscriber` instead. Since `CURRENT_SINK` stays
/// `None` outside of a `with_channel_subscriber` scope, only the fmt layer
/// fires here (when `GENORAY_LOG` is set) — same observable behavior as the
/// old thread-local-only fallback.
pub fn install_fmt_fallback() {
    ensure_global_subscriber();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossbeam_channel::unbounded;

    #[test]
    fn tick_flushes_on_threshold_and_remainder() {
        let (tx, rx) = unbounded();
        let sink = EventSink::new(tx, 100);
        sink.contig_start("chr1", Some(250));
        sink.tick("chr1", 60);
        sink.tick("chr1", 60); // crosses 100 -> swap drains buffer, emits Progress{delta:120}
        sink.tick("chr1", 20); // below threshold -> stays buffered as remainder
        sink.flush("chr1"); // emits the buffered remainder: Progress{delta:20}
        sink.contig_done("chr1", 230, 20, 1234);

        let evs: Vec<Event> = rx.try_iter().collect();
        assert!(matches!(
            evs[0],
            Event::ContigStart {
                total: Some(250),
                ..
            }
        ));
        // one Progress of 120 (threshold), then the flushed remainder of 20
        let deltas: Vec<u64> = evs
            .iter()
            .filter_map(|e| match e {
                Event::Progress { delta, .. } => Some(*delta),
                _ => None,
            })
            .collect();
        assert_eq!(deltas, vec![120, 20]);
        assert_eq!(deltas.iter().sum::<u64>(), 140);
        assert!(matches!(
            evs.last().unwrap(),
            Event::ContigDone {
                kept: 230,
                excluded: 20,
                elapsed_ms: 1234,
                ..
            }
        ));
    }

    #[test]
    fn tick_attributes_per_chrom_under_interleaving() {
        let (tx, rx) = unbounded();
        let sink = EventSink::new(tx, 100);

        sink.tick("chrA", 60);
        sink.tick("chrB", 60);
        sink.tick("chrA", 60); // chrA now 120 >= 100 -> emits Progress{chrom:"chrA", delta:120}
        sink.flush("chrA"); // chrA remainder 0 -> nothing
        sink.flush("chrB"); // chrB 60 -> emits Progress{chrom:"chrB", delta:60}

        let progress: Vec<(String, u64)> = rx
            .try_iter()
            .filter_map(|e| match e {
                Event::Progress { chrom, delta } => Some((chrom, delta)),
                _ => None,
            })
            .collect();

        assert_eq!(progress.len(), 2, "expected exactly two Progress events");
        assert!(progress.contains(&("chrA".to_string(), 120)));
        assert!(progress.contains(&("chrB".to_string(), 60)));

        // No Progress event may mix counts across chroms: totals per chrom
        // must equal exactly what was ticked for that chrom.
        let chr_a_total: u64 = progress
            .iter()
            .filter(|(c, _)| c == "chrA")
            .map(|(_, d)| d)
            .sum();
        let chr_b_total: u64 = progress
            .iter()
            .filter(|(c, _)| c == "chrB")
            .map(|(_, d)| d)
            .sum();
        assert_eq!(chr_a_total, 120);
        assert_eq!(chr_b_total, 60);
    }

    #[test]
    fn disabled_sink_is_silent() {
        let sink = EventSink::disabled();
        sink.tick("chr1", 10);
        sink.contig_done("chr1", 1, 0, 1); // must not panic
    }

    // `CURRENT_SINK`/`CURRENT_LEVEL` are process-global (that's the whole point
    // of the fix — cross-thread propagation), which means the tests below that
    // call `with_channel_subscriber` share that global state. `cargo test` runs
    // tests in parallel threads by default, so without serializing them here
    // they could interleave and cross-talk (exactly the documented "concurrent
    // in-process writes: last writer wins" limitation) — flaky, not a real bug.
    // A test-only lock keeps this file's tests deterministic.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn channel_layer_routes_events_at_level() {
        use crossbeam_channel::unbounded;
        let _guard = TEST_LOCK.lock().unwrap();
        let (tx, rx) = unbounded();
        let sink = EventSink::new(tx, 1);
        with_channel_subscriber(sink, "info", || {
            tracing::info!(chrom = "chr1", "excluded 12 records");
            tracing::debug!(chrom = "chr1", "chr1:100 REF mismatch"); // filtered out at info
        });
        let logs: Vec<(String, String)> = rx
            .try_iter()
            .filter_map(|e| match e {
                Event::Log { chrom, message, .. } => Some((chrom.unwrap_or_default(), message)),
                _ => None,
            })
            .collect();
        assert_eq!(logs.len(), 1);
        assert_eq!(logs[0].0, "chr1");
        assert!(logs[0].1.contains("excluded 12"));
    }

    /// Regression guard for the global-subscriber fix: events emitted from an
    /// OS thread spawned *inside* `with_channel_subscriber` (standing in for
    /// the reader/executor/writer threads `process_chromosome` spawns below a
    /// rayon pool) must still reach the channel. Under the old thread-local
    /// `tracing::subscriber::with_default` mechanism this FAILS (the spawned
    /// thread has no subscriber at all, so the event is dropped); under the
    /// process-global mechanism it PASSES.
    #[test]
    fn global_subscriber_routes_from_spawned_thread() {
        use crossbeam_channel::unbounded;
        let _guard = TEST_LOCK.lock().unwrap();
        let (tx, rx) = unbounded();
        let sink = EventSink::new(tx, 1);
        with_channel_subscriber(sink, "debug", || {
            std::thread::spawn(|| {
                tracing::info!(chrom = "chrX", "from worker");
            })
            .join()
            .unwrap();
        });
        let logs: Vec<(String, String)> = rx
            .try_iter()
            .filter_map(|e| match e {
                Event::Log { chrom, message, .. } => Some((chrom.unwrap_or_default(), message)),
                _ => None,
            })
            .collect();
        assert_eq!(
            logs.len(),
            1,
            "expected exactly one Log event from the spawned thread"
        );
        assert_eq!(logs[0].0, "chrX");
        assert!(logs[0].1.contains("from worker"));
    }
}
