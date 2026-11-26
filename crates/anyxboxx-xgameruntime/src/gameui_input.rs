//! XGameUi screens that return a value: message box (the button), text entry (the text) and
//! player picker (the players chosen). Port of part of `gameui_interface.cpp`; see `gameui.rs`.

use std::ffi::c_char;

use victory_hle::modules::xasync::AsyncBlock;

use crate::gameui::{
    H, MESSAGE_IDENTITY, PICKER_IDENTITY, TEXT_IDENTITY, Text, as_callback, identity,
};
use crate::gameui_requests::{UiKind, begin_request, find_callback};
use crate::object::{E_INVALIDARG, E_NOTIMPL, HResult};

// --- Message box -----------------------------------------------------------------------------

type MessageDialogCallback = unsafe extern "system" fn(
    H,
    H,
    *const c_char,
    *const c_char,
    *const c_char,
    *const c_char,
    *const c_char,
    u32,
    u32,
    H,
);
pub unsafe extern "system" fn show_message_dialog_async(
    _this: H,
    block: *mut AsyncBlock,
    title: *const c_char,
    content: *const c_char,
    first: *const c_char,
    second: *const c_char,
    third: *const c_char,
    default_button: u32,
    cancel_button: u32,
) -> HResult {
    let name = "XGameUiShowMessageDialogAsync";
    let Some((callback, context)) = find_callback(UiKind::MessageDialog, name) else {
        return E_NOTIMPL;
    };
    // SAFETY: the game's callback for this screen; its strings are null or NUL-terminated.
    let (callback, texts): (MessageDialogCallback, [Text; 5]) = unsafe {
        (
            as_callback(callback),
            [title, content, first, second, third].map(|t| Text::copy(t)),
        )
    };
    let invoke = move |handle, queue| {
        let [t, c, b1, b2, b3] = &texts;
        // SAFETY: the game's own callback, with the documented arguments.
        unsafe {
            callback(
                handle,
                queue,
                t.get(),
                c.get(),
                b1.get(),
                b2.get(),
                b3.get(),
                default_button,
                cancel_button,
                context as H,
            )
        }
    };
    begin_request(
        block,
        identity(&MESSAGE_IDENTITY),
        name,
        UiKind::MessageDialog,
        Box::new(invoke),
    )
}

// --- Text entry: the result is the game's UTF-8 string, terminating zero included -------------

type TextEntryCallback =
    unsafe extern "system" fn(H, H, *const c_char, *const c_char, *const c_char, u32, u32, H);
pub unsafe extern "system" fn show_text_entry_async(
    _this: H,
    block: *mut AsyncBlock,
    title: *const c_char,
    description: *const c_char,
    initial: *const c_char,
    scope: u32,
    max_length: u32,
) -> HResult {
    let name = "XGameUiShowTextEntryAsync";
    let Some((callback, context)) = find_callback(UiKind::TextEntry, name) else {
        return E_NOTIMPL;
    };
    // SAFETY: the game's callback for this screen; its strings are null or NUL-terminated.
    let (callback, texts): (TextEntryCallback, [Text; 3]) = unsafe {
        (
            as_callback(callback),
            [title, description, initial].map(|t| Text::copy(t)),
        )
    };
    let invoke = move |handle, queue| {
        let [t, d, i] = &texts;
        // SAFETY: the game's own callback, with the documented arguments.
        unsafe {
            callback(
                handle,
                queue,
                t.get(),
                d.get(),
                i.get(),
                scope,
                max_length,
                context as H,
            )
        }
    };
    begin_request(
        block,
        identity(&TEXT_IDENTITY),
        name,
        UiKind::TextEntry,
        Box::new(invoke),
    )
}

// --- Player picker: XGameUiPlayerPickerInfo (structs/xgameuiplayerpickerinfo) ------------------

#[repr(C)]
struct PlayerPickerInfo {
    requesting_user: H,
    prompt_text: *const c_char,
    select_from_count: u32,
    select_from: *const u64,
    preselected_count: u32,
    preselected: *const u64,
    min_selection: u32,
    max_selection: u32,
}

type PlayerPickerCallback = unsafe extern "system" fn(H, H, *mut PlayerPickerInfo, H);

/// A copy of the title's `count` ids, or `None` when `count` is non-zero with a null array.
///
/// # Safety
/// `ids` is null or readable for `count` values.
unsafe fn copy_ids(count: u32, ids: *const u64) -> Option<Vec<u64>> {
    match (count, ids.is_null()) {
        (0, _) => Some(Vec::new()),
        (_, true) => None,
        // SAFETY: per this function's contract.
        _ => Some(unsafe { std::slice::from_raw_parts(ids, count as usize) }.to_vec()),
    }
}

pub unsafe extern "system" fn show_player_picker_async(
    _this: H,
    block: *mut AsyncBlock,
    user: H,
    prompt: *const c_char,
    from_count: u32,
    from: *const u64,
    preselected_count: u32,
    preselected: *const u64,
    min_selection: u32,
    max_selection: u32,
) -> HResult {
    // SAFETY: the title's arrays, readable for their counts.
    let (Some(players), Some(chosen)) = (unsafe {
        (
            copy_ids(from_count, from),
            copy_ids(preselected_count, preselected),
        )
    }) else {
        return E_INVALIDARG;
    };
    let name = "XGameUiShowPlayerPickerAsync";
    let Some((callback, context)) = find_callback(UiKind::PlayerPicker, name) else {
        return E_NOTIMPL;
    };
    // SAFETY: the game's callback for this screen; its string is null or NUL-terminated.
    let (callback, text): (PlayerPickerCallback, Text) =
        unsafe { (as_callback(callback), Text::copy(prompt)) };
    let user = user as usize;
    let invoke = move |handle, queue| {
        let mut info = PlayerPickerInfo {
            requesting_user: user as H,
            prompt_text: text.get(),
            select_from_count: from_count,
            select_from: players.as_ptr(),
            preselected_count,
            preselected: chosen.as_ptr(),
            min_selection,
            max_selection,
        };
        // SAFETY: the game's own callback; the info and its arrays live during the call.
        unsafe { callback(handle, queue, &mut info, context as H) }
    };
    begin_request(
        block,
        identity(&PICKER_IDENTITY),
        name,
        UiKind::PlayerPicker,
        Box::new(invoke),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_arrays_need_a_pointer_when_not_empty() {
        let ids = [11u64, 22];
        // SAFETY: a valid array of two values, then null arrays.
        unsafe {
            assert_eq!(copy_ids(2, ids.as_ptr()), Some(vec![11, 22]));
            assert_eq!(copy_ids(0, std::ptr::null()), Some(Vec::new()));
            assert_eq!(copy_ids(1, std::ptr::null()), None);
        }
    }
}
