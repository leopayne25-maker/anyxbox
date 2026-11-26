//! (Neptune) The D3D12 runtime's background shader work at idle priority, where it cannot be
//! switched off.
//!
//! `VICTORY_D3D12_BACKGROUND=0` (the launcher's set) asks the runtime to switch its background
//! work off (`ID3D12Device6::SetBackgroundProcessingMode`). Without Windows' Developer Mode the
//! runtime refuses (the log said `ok=false` in every run on a machine without it), and the driver
//! goes on recompiling shaders on four "D3D Background Thread"s in our process: in a profile of the
//! world they were 2.3 cores, 94 % of it in the driver's shader compiler, and in an A/B of the world
//! the runs where they ran were 16.1 fps against 17.6-18.2 for those where they did not.
//!
//! Then those threads are put at idle priority, so that they compile on cores nothing else of
//! ours wants: they are made by the runtime when it has work for them, so they are looked for
//! every few seconds by a thread of ours ([`watch`]).

use std::collections::HashSet;

use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE, LocalFree};
use windows_sys::Win32::System::Diagnostics::ToolHelp::{
    CreateToolhelp32Snapshot, TH32CS_SNAPTHREAD, THREADENTRY32, Thread32First, Thread32Next,
};
use windows_sys::Win32::System::Threading::{
    GetCurrentProcessId, GetThreadDescription, OpenThread, SetThreadPriority, THREAD_PRIORITY_IDLE,
    THREAD_QUERY_LIMITED_INFORMATION, THREAD_SET_INFORMATION,
};

/// What the runtime names its background threads (followed by a number).
const NAME: &str = "D3D Background Thread";

/// The threads of this process whose name starts with `prefix` and that are not in `done`:
/// put at idle priority, and added to `done`. How many were.
pub(crate) fn idle_threads_named(prefix: &str, done: &mut HashSet<u32>) -> usize {
    let mut idled = 0;
    // SAFETY: plain Win32 calls; the snapshot and thread handles are closed below, the entry
    // is sized as the call asks, and the description is freed as `GetThreadDescription` says.
    unsafe {
        let snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        if snapshot == INVALID_HANDLE_VALUE {
            return 0;
        }
        let pid = GetCurrentProcessId();
        let mut entry: THREADENTRY32 = std::mem::zeroed();
        entry.dwSize = size_of::<THREADENTRY32>() as u32;
        let mut more = Thread32First(snapshot, &mut entry) != 0;
        while more {
            let tid = entry.th32ThreadID;
            if entry.th32OwnerProcessID == pid && !done.contains(&tid) {
                let thread = OpenThread(THREAD_QUERY_LIMITED_INFORMATION | THREAD_SET_INFORMATION, 0, tid);
                if !thread.is_null() {
                    let mut name: *mut u16 = std::ptr::null_mut();
                    if GetThreadDescription(thread, &mut name) >= 0 && !name.is_null() {
                        let len = (0..).take_while(|&i| *name.add(i) != 0).count();
                        let text = String::from_utf16_lossy(std::slice::from_raw_parts(name, len));
                        LocalFree(name.cast());
                        if text.starts_with(prefix) && SetThreadPriority(thread, THREAD_PRIORITY_IDLE) != 0 {
                            done.insert(tid);
                            idled += 1;
                        }
                    }
                    CloseHandle(thread);
                }
            }
            more = Thread32Next(snapshot, &mut entry) != 0;
        }
        CloseHandle(snapshot);
    }
    idled
}

/// Looks for the runtime's background threads every 2 s from now on, on a thread of its own at
/// low priority, and puts each one found at idle priority (once). Called once, when the runtime
/// refused to switch its background work off.
pub(crate) fn watch() {
    let _ = std::thread::Builder::new().name("victory-background-idle".into()).spawn(|| {
        // SAFETY: the calling thread's own pseudo-handle.
        unsafe {
            use windows_sys::Win32::System::Threading::{GetCurrentThread, THREAD_PRIORITY_LOWEST};
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
        }
        let mut done = HashSet::new();
        loop {
            let idled = idle_threads_named(NAME, &mut done);
            if idled > 0 {
                tracing::info!(idled, all = done.len(), "the runtime's background shader threads put at idle priority (VICTORY_D3D12_BACKGROUND=0)");
            }
            std::thread::sleep(std::time::Duration::from_secs(2));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::idle_threads_named;
    use std::collections::HashSet;
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::System::Threading::{GetThreadPriority, THREAD_PRIORITY_IDLE};

    /// A thread named as the runtime names its background threads is put at idle priority,
    /// once; a thread of another name is left as it is.
    #[test]
    fn threads_named_so_go_to_idle_priority_once() {
        let (stop, wait) = std::sync::mpsc::channel::<()>();
        let wait = std::sync::Arc::new(std::sync::Mutex::new(wait));
        let spawn = |name: &str| {
            let wait = wait.clone();
            std::thread::Builder::new()
                .name(name.into())
                .spawn(move || {
                    let _ = wait.lock().unwrap().recv();
                })
                .unwrap()
        };
        let background = spawn("victory-test Background Thread 9");
        let other = spawn("victory-test other thread");
        // Give both time to start, so that they are in the snapshot.
        std::thread::sleep(std::time::Duration::from_millis(100));
        let mut done = HashSet::new();
        assert_eq!(idle_threads_named("victory-test Background Thread", &mut done), 1);
        // SAFETY: live thread handles of the join handles above.
        let (b, o) = unsafe {
            (GetThreadPriority(background.as_raw_handle()), GetThreadPriority(other.as_raw_handle()))
        };
        assert_eq!(b, THREAD_PRIORITY_IDLE);
        assert_ne!(o, THREAD_PRIORITY_IDLE);
        // Already done: not counted again.
        assert_eq!(idle_threads_named("victory-test Background Thread", &mut done), 0);
        drop(stop);
        let _ = (background.join(), other.join());
    }
}
