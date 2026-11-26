//! XUser and devices: the real default audio endpoint of Windows, the controller on PC. Port of
//! `user_device_interface.cpp`.
//!
//! Methods: `gdk_api_table.inc` (interface {0CC6A956…}). Signatures and behaviour on PC:
//! https://learn.microsoft.com/gaming/gdk/docs/reference/system/xuser/functions/<lower-case name>
//! - XUserGetDefaultAudioEndpointUtf16: the id of Windows' default communications device
//!   (IMMDeviceEnumerator::GetDefaultAudioEndpoint, role eCommunications), the one the page's
//!   example then passes to IMMDeviceEnumerator::GetDevice.
//! - XUserFindControllerForUserWithUiAsync: "On desktop, this function will not show UI", and the
//!   result is "always ... an all-zero device ID".
//! - XUserRegisterForDeviceAssociationChanged and XUserRegisterForDefaultAudioEndpointUtf16Changed:
//!   E_NOTIMPL, logged (rule 3). AnyXboxX tracks neither controller-user associations nor changes
//!   of Windows' default audio device, so it could never call the callback.
//!   XUserGetDefaultAudioEndpointUtf16 gives the current value on every call (docs/progress.md,
//!   "Change subscriptions").
//!
//! XUserFindForDevice stays E_NOTIMPL: the page does not say what a PC answers.

use std::ffi::c_void;

use victory_hle::modules::xasync::AsyncBlock;
use windows::Win32::Media::Audio::{
    IMMDeviceEnumerator, MMDeviceEnumerator, eCapture, eCommunications, eRender,
};
use windows::Win32::System::Com::{
    CLSCTX_ALL, COINIT_MULTITHREADED, CoCreateInstance, CoInitializeEx, CoTaskMemFree,
    CoUninitialize,
};

use crate::log::trace;
use crate::object::{
    E_GAMEUSER_USER_NOT_FOUND, E_INVALIDARG, E_NOT_SUFFICIENT_BUFFER, HResult, S_OK,
};
use crate::save_async::{self, Completion};

/// XUserDefaultAudioEndpointKind (enums/xuserdefaultaudioendpointkind).
const COMMUNICATION_RENDER: u32 = 0;
const COMMUNICATION_CAPTURE: u32 = 1;

/// APP_LOCAL_DEVICE_ID: 32 bytes (structs/app_local_device_id).
type LocalDeviceId = [u8; 32];

/// Identity of the controller search: XAsync compares its address.
static CONTROLLER_IDENTITY: [u8; 1] = [13];

/// Copies `id` (with its terminating zero) into the title's buffer of `count` UTF-16 units.
///
/// # Safety
/// `buffer` is null or writable for `count` units; `used` is null or writable.
unsafe fn copy_endpoint_id(
    id: &[u16],
    count: usize,
    buffer: *mut u16,
    used: *mut usize,
) -> HResult {
    let needed = id.len() + 1;
    if !used.is_null() {
        // SAFETY: non-null and writable, per this function's contract.
        unsafe { used.write(needed) };
    }
    if buffer.is_null() || count < needed {
        return E_NOT_SUFFICIENT_BUFFER;
    }
    // SAFETY: the buffer holds at least `needed` units, checked above.
    unsafe {
        buffer.copy_from_nonoverlapping(id.as_ptr(), id.len());
        buffer.add(id.len()).write(0);
    }
    S_OK
}

/// The id of Windows' default communications device for `capture` (render otherwise).
fn default_communications_endpoint(capture: bool) -> Result<Vec<u16>, HResult> {
    // SAFETY: COM initialisation; paired with CoUninitialize below when it succeeds.
    let init = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
    let id = (|| {
        // SAFETY: the class and interface ids come from the crate.
        let enumerator: IMMDeviceEnumerator =
            unsafe { CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL) }
                .map_err(|error| error.code().0)?;
        let flow = if capture { eCapture } else { eRender };
        // E_NOTFOUND without a device, as the page's example expects.
        // SAFETY: a valid enumerator and enumeration values.
        let device = unsafe { enumerator.GetDefaultAudioEndpoint(flow, eCommunications) }
            .map_err(|error| error.code().0)?;
        // SAFETY: a valid device; the string is ours to free with CoTaskMemFree.
        let text = unsafe { device.GetId() }.map_err(|error| error.code().0)?;
        // SAFETY: GetId returned a NUL-terminated string.
        let id = unsafe { text.as_wide() }.to_vec();
        // SAFETY: allocated by COM for us, freed once.
        unsafe { CoTaskMemFree(Some(text.0.cast_const().cast())) };
        Ok(id)
    })();
    if init.is_ok() {
        // SAFETY: paired with the successful CoInitializeEx above.
        unsafe { CoUninitialize() };
    }
    id
}

/// XUserGetDefaultAudioEndpointUtf16 (slot 6).
///
/// # Safety
/// `buffer` is null or writable for `count` UTF-16 units; `used` is null or writable.
unsafe extern "system" fn get_default_audio_endpoint(
    _this: *mut c_void,
    user: u64,
    kind: u32,
    count: usize,
    buffer: *mut u16,
    used: *mut usize,
) -> HResult {
    if !crate::user::is_local_user_local_id(user) {
        return E_GAMEUSER_USER_NOT_FOUND;
    }
    if kind != COMMUNICATION_RENDER && kind != COMMUNICATION_CAPTURE {
        return E_INVALIDARG;
    }
    let result = match default_communications_endpoint(kind == COMMUNICATION_CAPTURE) {
        // SAFETY: forwarded to this function's contract.
        Ok(id) => unsafe { copy_endpoint_id(&id, count, buffer, used) },
        Err(error) => error,
    };
    trace!(
        "[runtime] XUserGetDefaultAudioEndpointUtf16({kind}) : 0x{:08X}",
        result as u32
    );
    result
}

/// XUserFindControllerForUserWithUiAsync (slot 9): no UI on desktop, an all-zero device id.
///
/// # Safety
/// `user` is null or a handle the title got from this runtime; `block` is the title's block.
unsafe extern "system" fn find_controller_async(
    _this: *mut c_void,
    user: *mut c_void,
    block: *mut AsyncBlock,
) -> HResult {
    // SAFETY: forwarded to this function's contract.
    if unsafe { crate::user::local_user_xuid(user) }.is_none() {
        return E_INVALIDARG;
    }
    let device: LocalDeviceId = [0; 32];
    save_async::begin(
        block,
        CONTROLLER_IDENTITY.as_ptr().cast(),
        Completion::value(S_OK, device),
    )
}

/// XUserFindControllerForUserWithUiResult (slot 10).
unsafe extern "system" fn find_controller_result(
    _this: *mut c_void,
    block: *mut AsyncBlock,
    device: *mut LocalDeviceId,
) -> HResult {
    save_async::result(
        block,
        CONTROLLER_IDENTITY.as_ptr().cast(),
        size_of::<LocalDeviceId>(),
        device.cast(),
    )
}

/// Unsubscribe methods (token, wait): no subscription can succeed (E_NOTIMPL), nothing to remove.
unsafe extern "system" fn unsubscribe(_this: *mut c_void, _token: u64, _wait: bool) -> bool {
    false
}

/// The method table of the user-device object. Slots 3 (XUserFindForDevice), 4 and 7 (the
/// change subscriptions) stay E_NOTIMPL.
pub fn methods() -> Vec<(usize, *const c_void)> {
    vec![
        (5, unsubscribe as *const c_void),
        (6, get_default_audio_endpoint as *const c_void),
        (8, unsubscribe as *const c_void),
        (9, find_controller_async as *const c_void),
        (10, find_controller_result as *const c_void),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_endpoint_id_is_copied_with_its_terminating_zero() {
        let id: Vec<u16> = "{0.0.1}".encode_utf16().collect();
        let mut buffer = [0xFFFFu16; 16];
        let mut used = 0usize;
        // SAFETY: valid buffers of the given sizes.
        unsafe {
            assert_eq!(
                copy_endpoint_id(&id, 4, buffer.as_mut_ptr(), &mut used),
                E_NOT_SUFFICIENT_BUFFER
            );
            assert_eq!(used, id.len() + 1);
            assert_eq!(
                copy_endpoint_id(&id, 16, buffer.as_mut_ptr(), &mut used),
                S_OK
            );
        }
        assert_eq!(&buffer[..id.len()], id.as_slice());
        assert_eq!(buffer[id.len()], 0);
    }

    #[test]
    fn user_and_kind_are_checked_first() {
        let call = |user, kind| {
            // SAFETY: refused before any buffer is touched.
            unsafe {
                get_default_audio_endpoint(
                    std::ptr::null_mut(),
                    user,
                    kind,
                    0,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                )
            }
        };
        assert_eq!(call(2, COMMUNICATION_RENDER), E_GAMEUSER_USER_NOT_FOUND);
        assert_eq!(call(1, 2), E_INVALIDARG);
    }

    #[test]
    fn subscriptions_are_left_not_implemented() {
        let slots: Vec<usize> = methods().iter().map(|&(slot, _)| slot).collect();
        assert!(!slots.contains(&3) && !slots.contains(&4) && !slots.contains(&7));
    }
}
