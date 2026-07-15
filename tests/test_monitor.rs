use genoray_core::monitor::StopSignal;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

#[test]
fn stop_signal_interrupts_a_long_sampler_wait() {
    let stop = Arc::new(StopSignal::new());
    let worker_stop = Arc::clone(&stop);
    let started = Instant::now();
    let worker = thread::spawn(move || worker_stop.wait_timeout(Duration::from_secs(30)));

    thread::sleep(Duration::from_millis(20));
    stop.stop();

    assert!(worker.join().unwrap());
    assert!(started.elapsed() < Duration::from_millis(500));
}
