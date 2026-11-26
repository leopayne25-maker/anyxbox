//! Offline XGameStreaming: the game runs on the PC and no device receives it as a stream. Port of
//! `streaming_interface.cpp`.
//!
//! Methods: `gdk_api_table.inc`. Signatures, error codes and the behaviour without a client:
//! https://learn.microsoft.com/gaming/gdk/docs/reference/system/xgamestreaming/xgamestreaming_members
//! (one page per function under .../xgamestreaming/functions/<lower-case name>).
//! Under AnyXboxX no streaming client ever connects: the client count is 0, any question about a
//! given client answers E_GAMESTREAMING_CLIENT_NOT_CONNECTED, and a controller is never a streaming
//! controller. Nothing is simulated: no answer pretends there is a stream.

use std::ffi::{c_char, c_void};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::events;
use crate::log::trace;
use crate::object::{E_INVALIDARG, HResult, S_OK};

/// Documented codes (page "XGameStreaming (API contents)", section Error Codes).
const NOT_INITIALIZED: HResult = 0x8924_5400_u32 as HResult;
const CLIENT_NOT_CONNECTED: HResult = 0x8924_5401_u32 as HResult;
const NO_DATA: HResult = 0x8924_5402_u32 as HResult;
const NO_DATACENTER: HResult = 0x8924_5403_u32 as HResult;
const NOT_STREAMING_CONTROLLER: HResult = 0x8924_5404_u32 as HResult;

/// XGameStreamingClientId: an opaque 8-byte id, passed by value.
type ClientId = u64;
/// XGameStreamingConnectionState::Disconnected (enums/xgamestreamingconnectionstate).
const DISCONNECTED: u32 = 0;
/// 16/9 bounds both ratios of XGameStreamingGetDisplayDetails.
const SIXTEEN_BY_NINE: f32 = 16.0 / 9.0;

/// XGameStreamingInitialize / Uninitialize: the other APIs return
/// E_GAMESTREAMING_NOT_INITIALIZED until it is initialised.
static INITIALIZED: AtomicBool = AtomicBool::new(false);

fn initialized() -> bool {
    INITIALIZED.load(Ordering::Acquire)
}

/// `result` once initialised, E_GAMESTREAMING_NOT_INITIALIZED before.
fn when_initialized(result: HResult) -> HResult {
    if initialized() {
        result
    } else {
        NOT_INITIALIZED
    }
}

/// The common answer of the APIs that ask about a client: none is connected.
fn no_client(function: &str) -> HResult {
    if !initialized() {
        return NOT_INITIALIZED;
    }
    trace!("[runtime] {function} : no streaming client connected");
    CLIENT_NOT_CONNECTED
}

// --- Initialisation (functions/xgamestreaminginitialize, xgamestreaminguninitialize) ----------

unsafe extern "system" fn initialize(_this: *mut c_void) -> HResult {
    INITIALIZED.store(true, Ordering::Release);
    trace!("[runtime] XGameStreamingInitialize : local game, no streaming client");
    S_OK
}

unsafe extern "system" fn uninitialize(_this: *mut c_void) {
    INITIALIZED.store(false, Ordering::Release);
}

// --- Connection state (getclientcount, getclients, getconnectionstate, isstreaming) -----------

unsafe extern "system" fn is_streaming(_this: *mut c_void) -> bool {
    false
}

unsafe extern "system" fn get_client_count(_this: *mut c_void) -> u32 {
    0
}

/// # Safety
/// `clients_used` is null or points at one writable `u32`.
unsafe extern "system" fn get_clients(
    _this: *mut c_void,
    _count: u32,
    _clients: *mut ClientId,
    clients_used: *mut u32,
) -> HResult {
    if !initialized() {
        return NOT_INITIALIZED;
    }
    if clients_used.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: non-null and writable, per this function's contract.
    unsafe { clients_used.write(0) };
    S_OK
}

unsafe extern "system" fn get_connection_state(_this: *mut c_void, _client: ClientId) -> u32 {
    DISCONNECTED
}

/// On registration the callback runs for each client already connected: none here, and none
/// will connect. The token stays valid until it is unregistered.
unsafe extern "system" fn register_connection_state_changed(
    _this: *mut c_void,
    _queue: *mut c_void,
    _context: *mut c_void,
    callback: *const c_void,
    token: *mut u64,
) -> HResult {
    if !initialized() {
        return NOT_INITIALIZED;
    }
    events::add_subscription(callback, token)
}

unsafe extern "system" fn unregister_connection_state_changed(
    _this: *mut c_void,
    token: u64,
    _wait: bool,
) -> bool {
    events::remove_subscription(token)
}

// --- Properties of a client: no client, so E_GAMESTREAMING_CLIENT_NOT_CONNECTED --------------

unsafe extern "system" fn register_client_properties_changed(
    _this: *mut c_void,
    _client: ClientId,
    _queue: *mut c_void,
    _context: *mut c_void,
    _callback: *const c_void,
    _token: *mut u64,
) -> HResult {
    no_client("XGameStreamingRegisterClientPropertiesChanged")
}

/// No registration could succeed: there is nothing to remove.
unsafe extern "system" fn unregister_client_properties_changed(
    _this: *mut c_void,
    _client: ClientId,
    _token: u64,
    _wait: bool,
) -> bool {
    false
}

unsafe extern "system" fn get_stream_physical_dimensions(
    _this: *mut c_void,
    _client: ClientId,
    _width: *mut u32,
    _height: *mut u32,
) -> HResult {
    no_client("XGameStreamingGetStreamPhysicalDimensions")
}

unsafe extern "system" fn is_touch_input_enabled(
    _this: *mut c_void,
    _client: ClientId,
    _enabled: *mut bool,
) -> HResult {
    no_client("XGameStreamingIsTouchInputEnabled")
}

unsafe extern "system" fn get_last_frame_displayed(
    _this: *mut c_void,
    _client: ClientId,
    _frame: *mut c_void,
) -> HResult {
    no_client("XGameStreamingGetLastFrameDisplayed")
}

unsafe extern "system" fn get_touch_bundle_version(
    _this: *mut c_void,
    _client: ClientId,
    _version: *mut c_void,
    _size: usize,
    _name: *mut c_char,
) -> HResult {
    no_client("XGameStreamingGetTouchBundleVersion")
}

unsafe extern "system" fn get_client_ip_address(
    _this: *mut c_void,
    _client: ClientId,
    _size: usize,
    _address: *mut c_char,
) -> HResult {
    no_client("XGameStreamingGetClientIPAddress")
}

unsafe extern "system" fn get_session_id(
    _this: *mut c_void,
    _client: ClientId,
    _size: usize,
    _id: *mut c_char,
    _used: *mut usize,
) -> HResult {
    no_client("XGameStreamingGetSessionId")
}

/// "If no version name is specified in the touch adaptation bundle then the size will be 0."
unsafe extern "system" fn get_touch_bundle_version_name_size(
    _this: *mut c_void,
    _client: ClientId,
) -> usize {
    0
}

/// Parameters checked as the page documents: maxSupportedPixels > 0,
/// widestSupportedAspectRatio >= 16/9 and finite, 0 < tallestSupportedAspectRatio <= 16/9.
fn display_details_are_valid(max_pixels: u32, widest: f32, tallest: f32) -> bool {
    max_pixels > 0
        && widest.is_finite()
        && widest >= SIXTEEN_BY_NINE
        && tallest > 0.0
        && tallest <= SIXTEEN_BY_NINE
}

unsafe extern "system" fn get_display_details(
    _this: *mut c_void,
    _client: ClientId,
    max_pixels: u32,
    widest: f32,
    tallest: f32,
    _details: *mut c_void,
) -> HResult {
    if !initialized() {
        return NOT_INITIALIZED;
    }
    if !display_details_are_valid(max_pixels, widest, tallest) {
        return E_INVALIDARG;
    }
    no_client("XGameStreamingGetDisplayDetails")
}

/// The documented latency call does not know CLIENT_NOT_CONNECTED: without a stream, no measure.
unsafe extern "system" fn get_stream_added_latency(
    _this: *mut c_void,
    _client: ClientId,
    _average: *mut u32,
    _minimum: *mut u32,
    _maximum: *mut u32,
) -> HResult {
    when_initialized(NO_DATA)
}

// --- Datacenter: the game runs on the PC, not in an Azure datacenter --------------------------

unsafe extern "system" fn get_server_location_name_size(_this: *mut c_void) -> usize {
    0
}

unsafe extern "system" fn get_server_location_name(
    _this: *mut c_void,
    _size: usize,
    _name: *mut c_char,
) -> HResult {
    when_initialized(NO_DATACENTER)
}

// --- Controller: a local GameInput reading never comes from a streaming controller ------------

/// XGameStreamingGetAssociatedFrame and XGameStreamingGetGamepadPhysicality.
unsafe extern "system" fn reading_not_from_stream(
    _this: *mut c_void,
    _reading: *mut c_void,
    _out: *mut c_void,
) -> HResult {
    when_initialized(NOT_STREAMING_CONTROLLER)
}

// --- Touch controls: "If the player is not streaming the game ... this function does nothing."
// The global variants address every connected client, that is none. ----------------------------

unsafe extern "system" fn hide_touch_controls(_this: *mut c_void) {}

unsafe extern "system" fn show_touch_control_layout(_this: *mut c_void, _layout: *const c_char) {}

unsafe extern "system" fn hide_touch_controls_on_client(_this: *mut c_void, _client: ClientId) {}

unsafe extern "system" fn show_touch_control_layout_on_client(
    _this: *mut c_void,
    _client: ClientId,
    _layout: *const c_char,
) {
}

fn touch_state_update(count: usize, operations: *const c_void) -> HResult {
    if !initialized() {
        return NOT_INITIALIZED;
    }
    if count > 0 && operations.is_null() {
        E_INVALIDARG
    } else {
        S_OK
    }
}

unsafe extern "system" fn update_touch_controls_state(
    _this: *mut c_void,
    count: usize,
    operations: *const c_void,
) -> HResult {
    touch_state_update(count, operations)
}

unsafe extern "system" fn show_touch_controls_with_state_update(
    _this: *mut c_void,
    _layout: *const c_char,
    count: usize,
    operations: *const c_void,
) -> HResult {
    touch_state_update(count, operations)
}

unsafe extern "system" fn update_touch_controls_state_on_client(
    _this: *mut c_void,
    _client: ClientId,
    _count: usize,
    _operations: *const c_void,
) -> HResult {
    no_client("XGameStreamingUpdateTouchControlsStateOnClient")
}

unsafe extern "system" fn show_touch_controls_with_state_update_on_client(
    _this: *mut c_void,
    _client: ClientId,
    _layout: *const c_char,
    _count: usize,
    _operations: *const c_void,
) -> HResult {
    no_client("XGameStreamingShowTouchControlsWithStateUpdateOnClient")
}

/// The method table of the streaming object: slots 3 to 33 of interface {57786622…}. Slot 34
/// (XGameStreamingSetResolution) stays E_NOTIMPL: the stream resolution applies to connected
/// clients, and the page does not say what a console that is not streaming answers.
pub fn methods() -> Vec<(usize, *const c_void)> {
    vec![
        (3, initialize as *const c_void),
        (4, uninitialize as *const c_void),
        (5, is_streaming as *const c_void),
        (6, register_client_properties_changed as *const c_void),
        (7, unregister_client_properties_changed as *const c_void),
        (8, get_stream_physical_dimensions as *const c_void),
        (9, get_client_count as *const c_void),
        (10, get_clients as *const c_void),
        (11, get_connection_state as *const c_void),
        (12, register_connection_state_changed as *const c_void),
        (13, unregister_connection_state_changed as *const c_void),
        (14, get_stream_added_latency as *const c_void),
        (15, get_server_location_name_size as *const c_void),
        (16, get_server_location_name as *const c_void),
        (17, hide_touch_controls as *const c_void),
        (18, show_touch_control_layout as *const c_void),
        (19, hide_touch_controls_on_client as *const c_void),
        (20, show_touch_control_layout_on_client as *const c_void),
        (21, is_touch_input_enabled as *const c_void),
        (22, get_last_frame_displayed as *const c_void),
        (23, reading_not_from_stream as *const c_void),
        (24, reading_not_from_stream as *const c_void),
        (25, update_touch_controls_state as *const c_void),
        (26, update_touch_controls_state_on_client as *const c_void),
        (27, show_touch_controls_with_state_update as *const c_void),
        (
            28,
            show_touch_controls_with_state_update_on_client as *const c_void,
        ),
        (29, get_touch_bundle_version_name_size as *const c_void),
        (30, get_touch_bundle_version as *const c_void),
        (31, get_client_ip_address as *const c_void),
        (32, get_session_id as *const c_void),
        (33, get_display_details as *const c_void),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_details_follow_the_documented_bounds() {
        assert!(display_details_are_valid(
            1920 * 1080,
            21.5 / 9.0,
            16.0 / 10.0
        ));
        assert!(!display_details_are_valid(0, 21.5 / 9.0, 16.0 / 10.0));
        assert!(!display_details_are_valid(1, 4.0 / 3.0, 16.0 / 10.0));
        assert!(!display_details_are_valid(1, f32::INFINITY, 16.0 / 10.0));
        assert!(!display_details_are_valid(1, 21.5 / 9.0, 2.0));
        assert!(!display_details_are_valid(1, 21.5 / 9.0, 0.0));
    }

    #[test]
    fn the_table_leaves_only_the_resolution_unwritten() {
        let slots: Vec<usize> = methods().iter().map(|&(slot, _)| slot).collect();
        assert_eq!(slots, (3..=33).collect::<Vec<_>>());
    }
}
