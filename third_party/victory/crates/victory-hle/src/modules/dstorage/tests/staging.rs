use std::cell::RefCell;
use std::sync::{Arc, Condvar, Mutex};

use super::*;

thread_local! {
    static BOUNCE_WAVE: RefCell<Option<Arc<Wave>>> = const { RefCell::new(None) };
}

struct Wave {
    workers: usize,
    observed: Mutex<(usize, usize)>,
    ready: Condvar,
}

/// Hold the allocations at the same point so the observed peak does not
/// depend on which worker finishes its file read first.
pub(in crate::modules::dstorage) fn observe_bounce(bytes: usize) {
    BOUNCE_WAVE.with_borrow(|wave| {
        let Some(wave) = wave else { return };
        let mut observed = wave.observed.lock().unwrap();
        observed.0 += 1;
        observed.1 += bytes;
        if observed.0 == wave.workers {
            wave.ready.notify_all();
        }
        let (observed, _) = wave
            .ready
            .wait_timeout_while(observed, Duration::from_secs(10), |seen| {
                seen.0 < wave.workers
            })
            .unwrap();
        assert_eq!(
            observed.0, wave.workers,
            "a worker did not reach its bounce allocation"
        );
    });
}

fn execute_wave(lanes: &mut Lanes, jobs: Vec<Job>) -> usize {
    let wave = Arc::new(Wave {
        workers: jobs.len(),
        observed: Mutex::new((0, 0)),
        ready: Condvar::new(),
    });
    std::thread::scope(|scope| {
        let workers = jobs
            .into_iter()
            .map(|job| {
                let wave = Arc::clone(&wave);
                scope.spawn(move || {
                    BOUNCE_WAVE.set(Some(wave));
                    let result = execute(&job.request);
                    (job, result)
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            let (job, result) = worker.join().unwrap();
            lanes.finish(&job, result);
            assert_eq!(
                result,
                Ok(()),
                "raw read at file offset {}",
                job.request.offset
            );
        }
    });
    assert_eq!(lanes.staged, 0);
    wave.observed.lock().unwrap().1
}

struct SourceFile(*mut c_void);

impl Drop for SourceFile {
    fn drop(&mut self) {
        release::<File>(self.0.cast());
    }
}

#[test]
fn raw_file_bounces_respect_staging_budget() {
    const WORKERS: usize = 8;
    const READ_BYTES: usize = 8 << 20;
    const OFFSET_STEP: usize = 16 << 10;
    const ROUNDS: usize = 2;

    let contents = pattern(READ_BYTES + WORKERS * OFFSET_STEP);
    let (_dir, file) = file_with(&contents);
    let file = SourceFile(file);
    let mut destinations = vec![vec![0xa5; READ_BYTES]; WORKERS];
    let mut lanes = lanes(STAGING);
    lanes.lanes.insert(0, Lane::new(0, 64));
    let mut peak_bounce = 0;
    let mut peak_reserved = 0;
    let mut peak_active = 0;
    let mut waves = 0;

    for round in 0..ROUNDS {
        let offset = |index| ((index + round) % WORKERS) * OFFSET_STEP;
        for (index, destination) in destinations.iter_mut().enumerate() {
            let read = request(
                0,
                destination.as_mut_ptr().cast(),
                destination.len(),
                file.0,
                offset(index) as u64,
                READ_BYTES,
            );
            lanes
                .lanes
                .get_mut(&0)
                .unwrap()
                .unsubmitted
                .push(Command::Read(read));
        }
        lanes.submit(0);
        while !lanes.lanes[&0].reads.is_empty() {
            let jobs = std::iter::from_fn(|| lanes.start())
                .take(WORKERS)
                .collect::<Vec<_>>();
            assert!(
                !jobs.is_empty(),
                "the scheduler stopped with pending reads and no active jobs"
            );
            peak_active = peak_active.max(jobs.len());
            peak_reserved = peak_reserved.max(lanes.staged);
            peak_bounce = peak_bounce.max(execute_wave(&mut lanes, jobs));
            waves += 1;
        }
        for (index, destination) in destinations.iter().enumerate() {
            let start = offset(index);
            assert_eq!(destination.len(), READ_BYTES);
            assert!(
                destination.as_slice() == &contents[start..start + READ_BYTES],
                "raw bytes differ in round {round}, destination {index}",
            );
        }
    }

    eprintln!(
        "staging_budget_bytes={STAGING}, peak_bounce_bytes={peak_bounce}, peak_reserved_bytes={peak_reserved}, active_reads={peak_active}, waves={waves}, verified_bytes={}",
        ROUNDS * WORKERS * READ_BYTES
    );
    assert_eq!(
        peak_bounce, STAGING as usize,
        "raw file bounce allocations exceeded the staging budget"
    );
    assert_eq!(peak_reserved, STAGING);
}
