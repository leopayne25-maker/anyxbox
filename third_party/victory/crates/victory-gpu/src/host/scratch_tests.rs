use std::process::Command;
use std::sync::atomic::Ordering::Relaxed;

use windows::Win32::Graphics::Direct3D12::D3D12_RANGE;
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1, IDXGIFactory4,
};

use super::{
    ALIVE, MADE, SCRATCH_CHUNK, SCRATCH_FIRST, SPARE, SPARE_MOST, give_chunks, host, take_chunk,
};

const TEST_NAME: &str = "host::scratch_tests::scratch_pool_reuses_small_chunks_after_large_burst";
const CHILD: &str = "VICTORY_TEST_SCRATCH_POOL";
const LISTS: usize = 8;
const ROUNDS: usize = 6;

#[test]
fn scratch_pool_reuses_small_chunks_after_large_burst() {
    if std::env::var_os(CHILD).is_none() {
        // Give this workload its own device, idle pool, and counters.
        let output = Command::new(std::env::current_exe().unwrap())
            .args([TEST_NAME, "--exact", "--nocapture", "--test-threads=1"])
            .env(CHILD, "1")
            .env("VICTORY_SCRATCH_POOL", "1")
            .output()
            .unwrap();
        print!("{}", String::from_utf8_lossy(&output.stdout));
        eprint!("{}", String::from_utf8_lossy(&output.stderr));
        assert!(
            output.status.success(),
            "scratch pool probe: {}",
            output.status
        );
        return;
    }
    let Some(host) = host() else {
        eprintln!("scratch pool probe skipped: D3D12 device unavailable");
        return;
    };
    // SAFETY: query the adapter that owns the test's live D3D12 device.
    unsafe {
        let factory = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
        let adapter = factory
            .EnumAdapterByLuid::<IDXGIAdapter1>(host.device.GetAdapterLuid())
            .unwrap();
        let desc = adapter.GetDesc1().unwrap();
        let name = String::from_utf16_lossy(&desc.Description);
        eprintln!(
            "adapter={} vendor={:#06x}",
            name.trim_end_matches('\0'),
            desc.VendorId
        );
    }

    // Keep all the large chunks checked out until the initial burst ends.
    let warm = (0..SPARE_MOST / SCRATCH_CHUNK)
        .map(|_| (take_chunk(host, SCRATCH_CHUNK).unwrap(), SCRATCH_CHUNK))
        .collect();
    give_chunks(warm);
    assert_eq!(SPARE.lock().unwrap().bytes, SPARE_MOST);

    let bytes_per_round = SCRATCH_FIRST * LISTS as u64;
    let readback = host.read_back_buffer(bytes_per_round).unwrap();
    let lists = (0..LISTS)
        .map(|_| host.new_list().unwrap())
        .collect::<Vec<_>>();
    let refs = lists.iter().collect::<Vec<_>>();
    let [made_before, bytes_before, reused_before] = [
        MADE[0].load(Relaxed),
        MADE[1].load(Relaxed),
        ALIVE[5].load(Relaxed),
    ];
    let marker = |round: usize, index: usize| (round * LISTS + index + 1) as u8;

    for round in 0..ROUNDS {
        for (index, list) in lists.iter().enumerate() {
            let (address, data) = list.scratch(SCRATCH_FIRST).unwrap();
            let scratch = list.scratch.lock().unwrap();
            let (chunk, capacity) = &scratch.chunks[0];
            let offset = address - chunk.gpu;
            assert!(offset + SCRATCH_FIRST <= *capacity);
            // SAFETY: the mapped scratch range was reserved by the list.
            // Its upload resource stays live through the waited submission.
            unsafe {
                data.write_bytes(marker(round, index), SCRATCH_FIRST as usize);
                list.list.CopyBufferRegion(
                    &readback,
                    index as u64 * SCRATCH_FIRST,
                    &chunk._buffer,
                    offset,
                    SCRATCH_FIRST,
                );
            }
            drop(scratch);
            list.close();
        }
        host.execute(&refs);
        // SAFETY: the submission has finished writing the read-back buffer.
        unsafe {
            let mut data = std::ptr::null_mut();
            let range = D3D12_RANGE {
                Begin: 0,
                End: bytes_per_round as usize,
            };
            readback.Map(0, Some(&range), Some(&mut data)).unwrap();
            assert!(!data.is_null());
            let bytes = std::slice::from_raw_parts(data.cast::<u8>(), range.End);
            for (index, slice) in bytes.chunks_exact(SCRATCH_FIRST as usize).enumerate() {
                assert!(
                    slice.iter().all(|&byte| byte == marker(round, index)),
                    "scratch bytes differ in round {round}, list {index}",
                );
            }
            readback.Unmap(0, Some(&D3D12_RANGE::default()));
        }
        for list in &lists {
            list.reset();
        }
    }

    let created = MADE[0].load(Relaxed) - made_before;
    let created_bytes = MADE[1].load(Relaxed) - bytes_before;
    let reused = ALIVE[5].load(Relaxed) - reused_before;
    let spare = SPARE.lock().unwrap();
    let idle_small = spare
        .kept
        .iter()
        .filter(|(_, size, _)| *size == SCRATCH_FIRST)
        .count();
    eprintln!(
        "cap_bytes={SPARE_MOST}, small_buffers_created={created}, created_bytes={created_bytes}, reuse_hits={reused}, idle_small={idle_small}, idle_bytes={}, verified_bytes={}",
        spare.bytes,
        bytes_per_round * ROUNDS as u64,
    );
    assert!(spare.bytes <= SPARE_MOST);
    assert_eq!(ALIVE[3].load(Relaxed), spare.kept.len() as u64);
    assert_eq!(ALIVE[4].load(Relaxed), spare.bytes);
    assert_eq!(ALIVE[2].load(Relaxed), spare.bytes);
    assert_eq!(
        created, LISTS as u64,
        "later rounds should reuse their small scratch chunks"
    );
    assert_eq!(idle_small, LISTS);
}
