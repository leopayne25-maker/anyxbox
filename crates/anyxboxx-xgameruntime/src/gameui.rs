//! XGameUi without a system UI: the screens are handed to the game's callbacks. Port of
//! `gameui_interface.cpp` (the `*Show*Async` half: screens without a result here, message box,
//! text entry and player picker in `gameui_input.rs`; results and answers in `gameui_responses.rs`).
//!
//! Methods: `gdk_api_table.inc`. Signatures: the XGameUi pages of Microsoft Learn,
//! https://learn.microsoft.com/gaming/gdk/docs/reference/system/xgameui/xgameui_members
//! (one page per function under .../xgameui/functions/<lower-case name>).
//! Without Gaming Services, AnyXboxX has no system UI to show. If the game gave its own screens
//! (XGameUiSetUiCallbacks), each XGameUiShow*Async calls the game's callback and returns the answer
//! that callback gives through XGameUiSet*UiResponse. Otherwise the call fails with E_NOTIMPL: no
//! user answer is invented.

use std::ffi::{CStr, CString, c_char, c_void};

use victory_hle::modules::xasync::AsyncBlock;

use crate::gameui_input as input;
use crate::gameui_requests::{UiKind, begin_request, find_callback};
use crate::gameui_responses as responses;
use crate::object::{E_NOTIMPL, HResult};

/// The identities of the results: XAsyncGetResult checks it reads what the call produced.
pub static MESSAGE_IDENTITY: [u8; 1] = [20];
pub static INVITE_IDENTITY: [u8; 1] = [21];
pub static PROFILE_IDENTITY: [u8; 1] = [22];
pub static ACHIEVEMENTS_IDENTITY: [u8; 1] = [23];
pub static PICKER_IDENTITY: [u8; 1] = [24];
pub static ERROR_IDENTITY: [u8; 1] = [25];
pub static TEXT_IDENTITY: [u8; 1] = [26];
pub static ACTIVITY_IDENTITY: [u8; 1] = [27];

pub fn identity(of: &'static [u8; 1]) -> *const c_void {
    of.as_ptr().cast()
}

/// An optional text, copied: the game may free its own as soon as `*Async` returns.
pub(crate) struct Text(Option<CString>);

impl Text {
    /// # Safety
    /// `text` is null or a NUL-terminated string.
    pub(crate) unsafe fn copy(text: *const c_char) -> Text {
        // SAFETY: per this function's contract.
        Text((!text.is_null()).then(|| unsafe { CStr::from_ptr(text) }.to_owned()))
    }

    pub(crate) fn get(&self) -> *const c_char {
        self.0
            .as_ref()
            .map_or(std::ptr::null(), |text| text.as_ptr())
    }
}

/// Reinterprets the game's callback address as its documented signature.
///
/// # Safety
/// `callback` is the game's non-null callback of type `F` (a function pointer).
pub(crate) unsafe fn as_callback<F: Copy>(callback: usize) -> F {
    // SAFETY: per this function's contract; `F` is a function pointer, as wide as `usize`.
    unsafe { std::mem::transmute_copy(&callback) }
}

pub(crate) type H = *mut c_void;

// --- Screens without a result: profile, invites, achievements, error ------------------------

type ProfileCardCallback = unsafe extern "system" fn(H, H, H, u64, H);
unsafe extern "system" fn show_player_profile_card_async(
    _this: H,
    block: *mut AsyncBlock,
    user: H,
    target: u64,
) -> HResult {
    let name = "XGameUiShowPlayerProfileCardAsync";
    let Some((callback, context)) = find_callback(UiKind::PlayerProfileCard, name) else {
        return E_NOTIMPL;
    };
    // SAFETY: the game's callback for this screen.
    let callback: ProfileCardCallback = unsafe { as_callback(callback) };
    let user = user as usize;
    let invoke = move |handle, queue| {
        // SAFETY: the game's own callback, with the documented arguments.
        unsafe { callback(handle, queue, user as H, target, context as H) }
    };
    begin_request(
        block,
        identity(&PROFILE_IDENTITY),
        name,
        UiKind::PlayerProfileCard,
        Box::new(invoke),
    )
}

type SendInviteCallback = unsafe extern "system" fn(
    H,
    H,
    H,
    *const c_char,
    *const c_char,
    *const c_char,
    *const c_char,
    *const c_char,
    H,
);
unsafe extern "system" fn show_send_game_invite_async(
    _this: H,
    block: *mut AsyncBlock,
    user: H,
    scid: *const c_char,
    template_name: *const c_char,
    session: *const c_char,
    text: *const c_char,
    activation: *const c_char,
) -> HResult {
    let name = "XGameUiShowSendGameInviteAsync";
    let Some((callback, context)) = find_callback(UiKind::SendGameInvite, name) else {
        return E_NOTIMPL;
    };
    // SAFETY: the game's callback for this screen; its strings are null or NUL-terminated.
    let (callback, texts): (SendInviteCallback, [Text; 5]) = unsafe {
        (
            as_callback(callback),
            [scid, template_name, session, text, activation].map(|t| Text::copy(t)),
        )
    };
    let user = user as usize;
    let invoke = move |handle, queue| {
        let [a, b, c, d, e] = &texts;
        // SAFETY: the game's own callback, with the documented arguments.
        unsafe {
            callback(
                handle,
                queue,
                user as H,
                a.get(),
                b.get(),
                c.get(),
                d.get(),
                e.get(),
                context as H,
            )
        }
    };
    begin_request(
        block,
        identity(&INVITE_IDENTITY),
        name,
        UiKind::SendGameInvite,
        Box::new(invoke),
    )
}

type AchievementsCallback = unsafe extern "system" fn(H, H, H, u32, H);
unsafe extern "system" fn show_achievements_async(
    _this: H,
    block: *mut AsyncBlock,
    user: H,
    title_id: u32,
) -> HResult {
    let name = "XGameUiShowAchievementsAsync";
    let Some((callback, context)) = find_callback(UiKind::Achievements, name) else {
        return E_NOTIMPL;
    };
    // SAFETY: the game's callback for this screen.
    let callback: AchievementsCallback = unsafe { as_callback(callback) };
    let user = user as usize;
    let invoke = move |handle, queue| {
        // SAFETY: the game's own callback, with the documented arguments.
        unsafe { callback(handle, queue, user as H, title_id, context as H) }
    };
    begin_request(
        block,
        identity(&ACHIEVEMENTS_IDENTITY),
        name,
        UiKind::Achievements,
        Box::new(invoke),
    )
}

type ActivityInviteCallback = unsafe extern "system" fn(H, H, H, H);
unsafe extern "system" fn show_multiplayer_activity_game_invite_async(
    _this: H,
    block: *mut AsyncBlock,
    user: H,
) -> HResult {
    let name = "XGameUiShowMultiplayerActivityGameInviteAsync";
    let kind = UiKind::MultiplayerActivityGameInvite;
    let Some((callback, context)) = find_callback(kind, name) else {
        return E_NOTIMPL;
    };
    // SAFETY: the game's callback for this screen.
    let callback: ActivityInviteCallback = unsafe { as_callback(callback) };
    let user = user as usize;
    let invoke = move |handle, queue| {
        // SAFETY: the game's own callback, with the documented arguments.
        unsafe { callback(handle, queue, user as H, context as H) }
    };
    begin_request(
        block,
        identity(&ACTIVITY_IDENTITY),
        name,
        kind,
        Box::new(invoke),
    )
}

type ErrorDialogCallback = unsafe extern "system" fn(H, H, HResult, *const c_char, H);
unsafe extern "system" fn show_error_dialog_async(
    _this: H,
    block: *mut AsyncBlock,
    error: HResult,
    service_context: *const c_char,
) -> HResult {
    let name = "XGameUiShowErrorDialogAsync";
    let Some((callback, context)) = find_callback(UiKind::ErrorDialog, name) else {
        return E_NOTIMPL;
    };
    // SAFETY: the game's callback for this screen; its string is null or NUL-terminated.
    let (callback, text): (ErrorDialogCallback, Text) =
        unsafe { (as_callback(callback), Text::copy(service_context)) };
    let invoke = move |handle, queue| {
        // SAFETY: the game's own callback, with the documented arguments.
        unsafe { callback(handle, queue, error, text.get(), context as H) }
    };
    begin_request(
        block,
        identity(&ERROR_IDENTITY),
        name,
        UiKind::ErrorDialog,
        Box::new(invoke),
    )
}

/// The method table of the XGameUi object: slots 3 to 50 of interface {EAF669DF…}. Without a
/// system UI nor a callback the GDK provides, web authentication, space management, state share
/// and the virtual keyboard (XGameUiTextEntry*) stay E_NOTIMPL (slots 24 to 29, 34 to 41; 20 to
/// 23, 32 and 33 are absent from `gdk_api_table.inc`).
pub fn methods() -> Vec<(usize, *const c_void)> {
    use UiKind::*;
    vec![
        (3, input::show_message_dialog_async as *const c_void),
        (4, responses::message_dialog_result as *const c_void),
        (5, show_send_game_invite_async as *const c_void),
        (6, responses::send_game_invite_result as *const c_void),
        (7, show_player_profile_card_async as *const c_void),
        (8, responses::player_profile_card_result as *const c_void),
        (9, show_achievements_async as *const c_void),
        (10, responses::achievements_result as *const c_void),
        (11, input::show_player_picker_async as *const c_void),
        (12, responses::player_picker_result_count as *const c_void),
        (13, responses::player_picker_result as *const c_void),
        (14, show_error_dialog_async as *const c_void),
        (15, responses::error_dialog_result as *const c_void),
        (
            16,
            responses::set_notification_position_hint as *const c_void,
        ),
        (17, input::show_text_entry_async as *const c_void),
        (18, responses::text_entry_result_size as *const c_void),
        (19, responses::text_entry_result as *const c_void),
        (
            30,
            show_multiplayer_activity_game_invite_async as *const c_void,
        ),
        (31, responses::activity_invite_result as *const c_void),
        (42, responses::set_ui_callbacks as *const c_void),
        (43, responses::set_message_dialog_response as *const c_void),
        (44, responses::set_player_picker_response as *const c_void),
        (45, responses::set_text_entry_response as *const c_void),
        (
            46,
            responses::dismissed::<{ PlayerProfileCard as u32 }> as *const c_void,
        ),
        (
            47,
            responses::dismissed::<{ SendGameInvite as u32 }> as *const c_void,
        ),
        (
            48,
            responses::dismissed::<{ Achievements as u32 }> as *const c_void,
        ),
        (
            49,
            responses::dismissed::<{ MultiplayerActivityGameInvite as u32 }> as *const c_void,
        ),
        (
            50,
            responses::dismissed::<{ ErrorDialog as u32 }> as *const c_void,
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn texts_are_copied_and_null_stays_null() {
        // SAFETY: a valid C string, then null.
        let (text, none) = unsafe { (Text::copy(c"Save".as_ptr()), Text::copy(std::ptr::null())) };
        // SAFETY: the copy is NUL-terminated.
        assert_eq!(unsafe { CStr::from_ptr(text.get()) }, c"Save");
        assert!(none.get().is_null());
    }

    #[test]
    fn without_a_game_callback_the_screen_is_not_implemented() {
        let mut block = std::mem::MaybeUninit::<AsyncBlock>::zeroed();
        // SAFETY: no callback is registered for the achievements screen in the unit tests.
        let result = unsafe {
            show_achievements_async(
                std::ptr::null_mut(),
                block.as_mut_ptr(),
                std::ptr::null_mut(),
                1,
            )
        };
        assert_eq!(result, E_NOTIMPL);
    }
}
