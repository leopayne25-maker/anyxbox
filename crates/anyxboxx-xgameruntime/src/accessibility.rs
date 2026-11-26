//! XAccessibility: the real high contrast mode of Windows. Port of `accessibility_interface.cpp`.
//!
//! Methods: `gdk_api_table.inc`. Signature and enumeration: Microsoft Learn,
//! https://learn.microsoft.com/gaming/gdk/docs/reference/system/xaccessibility/functions/xhighcontrastgetmode
//! and .../xaccessibility/enums/xhighcontrastmode (Off = 0, Dark = 1, Light = 2, Other = 3).
//! The mode comes from SystemParametersInfo(SPI_GETHIGHCONTRAST); dark or light from the window
//! background colour of the active theme. Captions, speech synthesis and speech to text stay
//! E_NOTIMPL: neither their settings nor their voices are invented.

use std::ffi::c_void;

use windows_sys::Win32::Foundation::GetLastError;
use windows_sys::Win32::Graphics::Gdi::{COLOR_WINDOW, GetSysColor};
use windows_sys::Win32::UI::Accessibility::{HCF_HIGHCONTRASTON, HIGHCONTRASTW};
use windows_sys::Win32::UI::WindowsAndMessaging::{SPI_GETHIGHCONTRAST, SystemParametersInfoW};

use crate::log::trace;
use crate::object::{E_INVALIDARG, HResult, S_OK};

/// XHighContrastMode values.
const MODE_OFF: u32 = 0;
const MODE_DARK: u32 = 1;
const MODE_LIGHT: u32 = 2;

/// Relative luminance (Rec. 709, 0 to 255) below which a background is dark.
const DARK_BACKGROUND: f64 = 128.0;

/// HRESULT_FROM_WIN32 of a Win32 error code.
fn hresult_from_win32(error: u32) -> HResult {
    if error == 0 {
        S_OK
    } else {
        ((error & 0xFFFF) | 0x8007_0000) as HResult
    }
}

/// Whether a COLORREF (0x00BBGGRR) is dark.
fn is_dark(color: u32) -> bool {
    let [red, green, blue, _] = color.to_le_bytes();
    let luminance = 0.2126 * f64::from(red) + 0.7152 * f64::from(green) + 0.0722 * f64::from(blue);
    luminance < DARK_BACKGROUND
}

/// XHighContrastGetMode (slot 5).
///
/// # Safety
/// `mode` is null or points at one writable `u32`.
unsafe extern "system" fn get_mode(_this: *mut c_void, mode: *mut u32) -> HResult {
    if mode.is_null() {
        return E_INVALIDARG;
    }
    let mut contrast = HIGHCONTRASTW {
        cbSize: size_of::<HIGHCONTRASTW>() as u32,
        dwFlags: 0,
        lpszDefaultScheme: std::ptr::null_mut(),
    };
    // SAFETY: the structure is writable and its size is set, as SPI_GETHIGHCONTRAST requires.
    let read = unsafe {
        SystemParametersInfoW(
            SPI_GETHIGHCONTRAST,
            contrast.cbSize,
            (&raw mut contrast).cast(),
            0,
        )
    };
    if read == 0 {
        // SAFETY: no arguments; reads the calling thread's last error.
        return hresult_from_win32(unsafe { GetLastError() });
    }
    let value = if contrast.dwFlags & HCF_HIGHCONTRASTON == 0 {
        MODE_OFF
    } else if is_dark(
        // SAFETY: a valid system colour index.
        unsafe { GetSysColor(COLOR_WINDOW) },
    ) {
        MODE_DARK
    } else {
        MODE_LIGHT
    };
    // SAFETY: non-null and writable, per this function's contract.
    unsafe { mode.write(value) };
    trace!("[runtime] XHighContrastGetMode : {value}");
    S_OK
}

/// The method table of the accessibility object: slot 5; slots 3, 4 and 6 to 21 (captions,
/// speech synthesis, speech to text) stay E_NOTIMPL.
pub fn methods() -> Vec<(usize, *const c_void)> {
    vec![(5, get_mode as *const c_void)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn background_luminance_gives_dark_or_light() {
        assert!(is_dark(0x0000_0000));
        assert!(!is_dark(0x00FF_FFFF));
        assert!(is_dark(0x00FF_0000)); // pure blue: luminance 18
        assert!(!is_dark(0x0000_FF00)); // pure green: luminance 182
    }

    #[test]
    fn the_mode_follows_windows_and_needs_an_output() {
        let mut mode = 7u32;
        // SAFETY: a valid pointer to a local.
        assert_eq!(unsafe { get_mode(std::ptr::null_mut(), &mut mode) }, S_OK);
        assert!(mode <= MODE_LIGHT);
        // SAFETY: a null output is part of the contract.
        assert_eq!(
            unsafe { get_mode(std::ptr::null_mut(), std::ptr::null_mut()) },
            E_INVALIDARG
        );
        assert_eq!(hresult_from_win32(5), 0x8007_0005_u32 as HResult);
    }
}
