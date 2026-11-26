//! XGameUi: the `*Result` functions the title reads, the game's `XGameUiSet*UiResponse` answers and
//! the settings. Port of the second half of `gameui_interface.cpp`.

use std::ffi::{CStr, c_char, c_void};

use victory_hle::modules::xasync::{self, AsyncBlock};

use crate::gameui::{
    ACHIEVEMENTS_IDENTITY, ACTIVITY_IDENTITY, ERROR_IDENTITY, INVITE_IDENTITY, MESSAGE_IDENTITY,
    PICKER_IDENTITY, PROFILE_IDENTITY, TEXT_IDENTITY, identity,
};
use crate::gameui_requests::{self as requests, UiCallbacks, UiKind, complete_request};
use crate::log::trace;
use crate::object::{E_INVALIDARG, HResult, S_OK};

/// XGameUiMessageDialogButton: First, Second, Third.
const LAST_BUTTON: u32 = 2;
const PLAYER_ID_SIZE: usize = size_of::<u64>();

// --- Results ------------------------------------------------------------------------------

/// The end of a screen without a result: the status of the call (nothing to copy).
fn no_payload_result(block: *mut AsyncBlock, of: &'static [u8; 1]) -> HResult {
    xasync::XAsyncGetResult(
        block,
        identity(of),
        0,
        std::ptr::null_mut(),
        std::ptr::null_mut(),
    )
}

pub unsafe extern "system" fn player_profile_card_result(
    _this: *mut c_void,
    block: *mut AsyncBlock,
) -> HResult {
    no_payload_result(block, &PROFILE_IDENTITY)
}
pub unsafe extern "system" fn send_game_invite_result(
    _this: *mut c_void,
    block: *mut AsyncBlock,
) -> HResult {
    no_payload_result(block, &INVITE_IDENTITY)
}
pub unsafe extern "system" fn achievements_result(
    _this: *mut c_void,
    block: *mut AsyncBlock,
) -> HResult {
    no_payload_result(block, &ACHIEVEMENTS_IDENTITY)
}
pub unsafe extern "system" fn activity_invite_result(
    _this: *mut c_void,
    block: *mut AsyncBlock,
) -> HResult {
    no_payload_result(block, &ACTIVITY_IDENTITY)
}
pub unsafe extern "system" fn error_dialog_result(
    _this: *mut c_void,
    block: *mut AsyncBlock,
) -> HResult {
    no_payload_result(block, &ERROR_IDENTITY)
}

/// XGameUiShowMessageDialogResult: the button the game's screen reports.
pub unsafe extern "system" fn message_dialog_result(
    _this: *mut c_void,
    block: *mut AsyncBlock,
    button: *mut u32,
) -> HResult {
    xasync::XAsyncGetResult(
        block,
        identity(&MESSAGE_IDENTITY),
        size_of::<u32>(),
        button.cast(),
        std::ptr::null_mut(),
    )
}

/// The result size of the call, in bytes; `out` gets 0 when the call failed.
fn result_size(block: *mut AsyncBlock) -> (HResult, usize) {
    let mut bytes = 0usize;
    let status = xasync::XAsyncGetResultSize(block, &mut bytes);
    (status, bytes)
}

/// XGameUiShowTextEntryResultSize: the text's size, terminating zero included.
///
/// # Safety
/// `size` is null or writable.
pub unsafe extern "system" fn text_entry_result_size(
    _this: *mut c_void,
    block: *mut AsyncBlock,
    size: *mut u32,
) -> HResult {
    if size.is_null() {
        return E_INVALIDARG;
    }
    let (status, bytes) = result_size(block);
    // SAFETY: non-null and writable, per this function's contract.
    unsafe { size.write(bytes as u32) };
    status
}

/// XGameUiShowTextEntryResult.
///
/// # Safety
/// `buffer` is writable for `size` bytes; `used` is null or writable.
pub unsafe extern "system" fn text_entry_result(
    _this: *mut c_void,
    block: *mut AsyncBlock,
    size: u32,
    buffer: *mut c_char,
    used: *mut u32,
) -> HResult {
    let mut bytes = 0usize;
    let result = xasync::XAsyncGetResult(
        block,
        identity(&TEXT_IDENTITY),
        size as usize,
        buffer.cast(),
        &mut bytes,
    );
    if !used.is_null() {
        // SAFETY: non-null and writable, per this function's contract.
        unsafe { used.write(bytes as u32) };
    }
    result
}

/// XGameUiShowPlayerPickerResultCount: the number of players chosen.
///
/// # Safety
/// `count` is null or writable.
pub unsafe extern "system" fn player_picker_result_count(
    _this: *mut c_void,
    block: *mut AsyncBlock,
    count: *mut u32,
) -> HResult {
    if count.is_null() {
        return E_INVALIDARG;
    }
    let (status, bytes) = result_size(block);
    // SAFETY: non-null and writable, per this function's contract.
    unsafe { count.write((bytes / PLAYER_ID_SIZE) as u32) };
    status
}

/// XGameUiShowPlayerPickerResult.
///
/// # Safety
/// `players` is writable for `count` ids; `used` is null or writable.
pub unsafe extern "system" fn player_picker_result(
    _this: *mut c_void,
    block: *mut AsyncBlock,
    count: u32,
    players: *mut u64,
    used: *mut u32,
) -> HResult {
    let mut bytes = 0usize;
    let size = count as usize * PLAYER_ID_SIZE;
    let result = xasync::XAsyncGetResult(
        block,
        identity(&PICKER_IDENTITY),
        size,
        players.cast(),
        &mut bytes,
    );
    if !used.is_null() {
        // SAFETY: non-null and writable, per this function's contract.
        unsafe { used.write((bytes / PLAYER_ID_SIZE) as u32) };
    }
    result
}

// --- The game's answers -------------------------------------------------------------------

/// XGameUiSetMessageDialogUiResponse: the button chosen on the game's screen.
pub unsafe extern "system" fn set_message_dialog_response(
    _this: *mut c_void,
    handle: *mut c_void,
    button: u32,
) -> HResult {
    if button > LAST_BUTTON {
        return E_INVALIDARG;
    }
    complete_request(handle, UiKind::MessageDialog, button.to_ne_bytes().to_vec())
}

/// XGameUiSetTextEntryUiResponse: the UTF-8 text, terminating zero included.
///
/// # Safety
/// `response` is null or a NUL-terminated string.
pub unsafe extern "system" fn set_text_entry_response(
    _this: *mut c_void,
    handle: *mut c_void,
    response: *const c_char,
) -> HResult {
    if response.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: per this function's contract.
    let text = unsafe { CStr::from_ptr(response) }
        .to_bytes_with_nul()
        .to_vec();
    complete_request(handle, UiKind::TextEntry, text)
}

/// XGameUiSetPlayerPickerUiResponse: the players chosen on the game's screen.
///
/// # Safety
/// `players` is null or readable for `count` ids.
pub unsafe extern "system" fn set_player_picker_response(
    _this: *mut c_void,
    handle: *mut c_void,
    count: u32,
    players: *const u64,
) -> HResult {
    if count > 0 && players.is_null() {
        return E_INVALIDARG;
    }
    let bytes = if count == 0 {
        Vec::new()
    } else {
        // SAFETY: per this function's contract.
        unsafe { std::slice::from_raw_parts(players.cast::<u8>(), count as usize * PLAYER_ID_SIZE) }
            .to_vec()
    };
    complete_request(handle, UiKind::PlayerPicker, bytes)
}

/// The kind of a `dismissed` answer, by its `UiKind` value.
fn kind_of(value: u32) -> Option<UiKind> {
    [
        UiKind::PlayerProfileCard,
        UiKind::SendGameInvite,
        UiKind::Achievements,
        UiKind::MultiplayerActivityGameInvite,
        UiKind::ErrorDialog,
    ]
    .into_iter()
    .find(|&kind| kind as u32 == value)
}

/// Answers without a value (profile card, invite, achievements, activity invite, error dialog):
/// the game's screen was closed.
pub unsafe extern "system" fn dismissed<const KIND: u32>(
    _this: *mut c_void,
    handle: *mut c_void,
) -> HResult {
    match kind_of(KIND) {
        Some(kind) => complete_request(handle, kind, Vec::new()),
        None => E_INVALIDARG,
    }
}

// --- Settings -----------------------------------------------------------------------------

/// XGameUiSetUiCallbacks. "useSystemUiIfAvailable": there is no system UI here, the game's
/// callbacks always serve.
///
/// # Safety
/// `callbacks` is null or points at one readable `UiCallbacks`.
pub unsafe extern "system" fn set_ui_callbacks(
    _this: *mut c_void,
    callbacks: *const UiCallbacks,
    _use_system_ui: bool,
) -> HResult {
    // SAFETY: forwarded to this function's contract.
    unsafe { requests::set_callbacks(callbacks) };
    S_OK
}

/// XGameUiSetNotificationPositionHint: "This API exists on Windows, however it currently has no
/// effect on achievement and other toast notifications."
/// (functions/xgameuisetnotificationpositionhint): the same here.
pub unsafe extern "system" fn set_notification_position_hint(
    _this: *mut c_void,
    position: u32,
) -> HResult {
    trace!("[runtime] XGameUiSetNotificationPositionHint({position}) : no effect on PC");
    S_OK
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dismissed_answers_map_to_their_screen() {
        assert_eq!(
            kind_of(UiKind::ErrorDialog as u32),
            Some(UiKind::ErrorDialog)
        );
        assert_eq!(
            kind_of(UiKind::PlayerProfileCard as u32),
            Some(UiKind::PlayerProfileCard)
        );
        assert_eq!(kind_of(UiKind::MessageDialog as u32), None);
    }

    #[test]
    fn answers_are_checked_before_the_handle() {
        let handle = std::ptr::dangling_mut::<c_void>();
        // SAFETY: refused before the handle is used; then an unknown handle.
        unsafe {
            assert_eq!(
                set_message_dialog_response(std::ptr::null_mut(), handle, LAST_BUTTON + 1),
                E_INVALIDARG
            );
            assert_eq!(
                set_text_entry_response(std::ptr::null_mut(), handle, std::ptr::null()),
                E_INVALIDARG
            );
            assert_eq!(
                set_player_picker_response(std::ptr::null_mut(), handle, 1, std::ptr::null()),
                E_INVALIDARG
            );
            assert_eq!(
                set_message_dialog_response(std::ptr::null_mut(), handle, 0),
                E_INVALIDARG
            );
        }
    }
}
