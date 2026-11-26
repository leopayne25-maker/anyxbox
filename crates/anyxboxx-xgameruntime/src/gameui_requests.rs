//! XGameUi: UI requests handed to the game's callbacks (XGameUiSetUiCallbacks). Port of
//! `gameui_requests.cpp`.
//!
//! Without Gaming Services there is no system UI. The GDK documents another way: the game gives
//! its own screens through XGameUiSetUiCallbacks, receives each request with an
//! XGameUiCallbackHandle, then gives its screen's answer through XGameUiSet*UiResponse.
//! https://learn.microsoft.com/gaming/gdk/docs/reference/system/xgameui/structs/xgameuiuicallbacks
//!
//! Each request is an XAsync call: Begin schedules it on the game's queue, DoWork calls the game's
//! callback and leaves the call pending (E_PENDING), then the game's answer or a cancellation ends
//! it. The handle given to the game is the request's address; it is accepted once, and only while
//! the request waits.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, LazyLock, Mutex, MutexGuard};

use victory_hle::modules::xasync::{self, AsyncBlock, E_ABORT, E_PENDING, ProviderData};

use crate::log::trace;
use crate::object::{E_INVALIDARG, HResult, S_OK};

/// XAsyncOp values, as VIctory's provider receives them.
const OP_BEGIN: u32 = 0;
const OP_DO_WORK: u32 = 1;
const OP_GET_RESULT: u32 = 2;
const OP_CANCEL: u32 = 3;
const OP_CLEANUP: u32 = 4;

/// The order of the callbacks in XGameUiUiCallbacks (after `context`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum UiKind {
    PlayerProfileCard = 0,
    PlayerPicker = 1,
    SendGameInvite = 2,
    Achievements = 3,
    MultiplayerActivityGameInvite = 4,
    MessageDialog = 5,
    ErrorDialog = 6,
    TextEntry = 7,
}

/// Number of callbacks in XGameUiUiCallbacks.
const UI_KINDS: usize = 8;

/// XGameUiUiCallbacks: the game's context, then one callback per screen (null: system UI).
#[repr(C)]
pub struct UiCallbacks {
    pub context: *mut c_void,
    pub callbacks: [*const c_void; UI_KINDS],
}

/// Calls the game's callback with (request handle, operation queue). Pointers it passes on are
/// kept as `usize` so the request can cross threads.
pub type UiInvoke = Box<dyn Fn(*mut c_void, *mut c_void) + Send + Sync>;

struct Request {
    kind: UiKind,
    /// Shared with DoWork: the texts given to the game stay valid during its callback, even if the
    /// answer arrives from another thread (and frees the request) before the callback returns.
    invoke: Arc<UiInvoke>,
    block: *mut AsyncBlock,
    result: Mutex<Vec<u8>>,
}

#[derive(Default)]
struct UiState {
    context: usize,
    callbacks: [usize; UI_KINDS],
    /// Requests begun, neither answered nor cancelled: address → kind.
    waiting: HashMap<usize, UiKind>,
}

fn state() -> MutexGuard<'static, UiState> {
    static STATE: LazyLock<Mutex<UiState>> = LazyLock::new(Default::default);
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Takes the request out of the waiting set; true for one caller only (answer, cancel or end).
fn claim(request: usize) -> bool {
    state().waiting.remove(&request).is_some()
}

/// XGameUiSetUiCallbacks: replaces every callback (null removes them).
///
/// # Safety
/// `callbacks` is null or points at one readable `UiCallbacks`.
pub unsafe fn set_callbacks(callbacks: *const UiCallbacks) {
    // SAFETY: per this function's contract.
    let table = unsafe { callbacks.as_ref() };
    let mut state = state();
    state.context = table.map_or(0, |t| t.context as usize);
    state.callbacks = table.map_or([0; UI_KINDS], |t| t.callbacks.map(|c| c as usize));
}

/// The game's callback for `kind` and its context; `None` (logged) when the game gave none, as
/// there is no system UI to show instead.
pub fn find_callback(kind: UiKind, name: &str) -> Option<(usize, usize)> {
    let state = state();
    let callback = state.callbacks[kind as usize];
    if callback == 0 {
        drop(state);
        trace!("[runtime] {name} : no system UI without Gaming Services -> E_NOTIMPL");
        return None;
    }
    Some((callback, state.context))
}

extern "system" fn provider(op: u32, data: *const ProviderData) -> HResult {
    // SAFETY: XAsync passes its own provider data, with the context given to `begin_request`.
    let data = unsafe { &*data };
    let request = data.context.cast::<Request>();
    match op {
        OP_BEGIN => {
            // SAFETY: the request is ours and nobody else holds it yet.
            let kind = unsafe {
                (*request).block = data.block;
                (*request).kind
            };
            state().waiting.insert(request as usize, kind);
            xasync::XAsyncSchedule(data.block, 0)
        }
        OP_DO_WORK => {
            if !state().waiting.contains_key(&(request as usize)) {
                return E_ABORT; // cancelled before it reached the game
            }
            // SAFETY: a waiting request is alive; the invocation is cloned before the callback,
            // which may answer (and so free the request) before it returns.
            let invoke = unsafe { Arc::clone(&(*request).invoke) };
            // SAFETY: the block lives while the call is pending; its queue is the game's.
            let queue = unsafe { (*data.block).queue };
            invoke(request.cast(), queue.cast_mut().cast());
            E_PENDING // ended by the game's answer
        }
        OP_GET_RESULT => {
            // SAFETY: the request lives until cleanup; XAsync checked the buffer's size.
            let result = unsafe { &(*request).result };
            let bytes = result.lock().unwrap_or_else(|e| e.into_inner());
            // SAFETY: as above.
            unsafe {
                data.buffer
                    .cast::<u8>()
                    .copy_from(bytes.as_ptr(), bytes.len())
            };
            S_OK
        }
        // XAsync then completes the call with E_ABORT, unless the game answered first.
        OP_CANCEL => {
            claim(request as usize);
            S_OK
        }
        OP_CLEANUP => {
            claim(request as usize);
            // SAFETY: leaked by `begin_request`, freed exactly once here.
            drop(unsafe { Box::from_raw(request) });
            S_OK
        }
        _ => S_OK,
    }
}

/// Starts the request: `invoke` runs on the queue of `block`, and the call stays pending until the
/// game's answer ([`complete_request`]) or its cancellation (E_ABORT).
pub fn begin_request(
    block: *mut AsyncBlock,
    identity: *const c_void,
    name: &str,
    kind: UiKind,
    invoke: UiInvoke,
) -> HResult {
    if block.is_null() {
        return E_INVALIDARG;
    }
    let request = Box::into_raw(Box::new(Request {
        kind,
        invoke: Arc::new(invoke),
        block: std::ptr::null_mut(),
        result: Mutex::new(Vec::new()),
    }));
    let begun = xasync::XAsyncBegin(
        block,
        request.cast(),
        identity,
        std::ptr::null(),
        Some(provider),
    );
    if begun < 0 {
        // SAFETY: XAsync never called the provider; the request is still ours.
        drop(unsafe { Box::from_raw(request) });
    }
    trace!("[runtime] {name} : request handed to the game's callback");
    begun
}

/// The game's answer: E_INVALIDARG when the handle is unknown, already answered, cancelled, or of
/// another screen. `result` becomes what the `*Result` functions read.
pub fn complete_request(handle: *mut c_void, kind: UiKind, result: Vec<u8>) -> HResult {
    let request = handle.cast::<Request>();
    let (block, size) = {
        let mut state = state();
        if state.waiting.get(&(request as usize)) != Some(&kind) {
            return E_INVALIDARG;
        }
        state.waiting.remove(&(request as usize));
        let size = result.len();
        // SAFETY: a request found waiting is alive, and only cleanup (after completion) frees it.
        let block = unsafe {
            *(*request).result.lock().unwrap_or_else(|e| e.into_inner()) = result;
            (*request).block
        };
        (block, size)
    };
    // Without a result, the request may be freed here.
    xasync::XAsyncComplete(block, S_OK, size);
    S_OK
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unknown_handle_is_refused() {
        let handle = std::ptr::dangling_mut::<c_void>();
        assert_eq!(
            complete_request(handle, UiKind::MessageDialog, Vec::new()),
            E_INVALIDARG
        );
    }

    #[test]
    fn a_request_needs_a_block() {
        let invoke: UiInvoke = Box::new(|_, _| {});
        assert_eq!(
            begin_request(
                std::ptr::null_mut(),
                std::ptr::null(),
                "test",
                UiKind::TextEntry,
                invoke
            ),
            E_INVALIDARG
        );
    }
}
