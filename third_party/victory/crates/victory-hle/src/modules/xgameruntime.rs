//! `xgameruntime.dll`: the console game runtime (XGame*, XUser*, XTaskQueue*...).
//!
//! Titles rarely import it. A stub linked into the title loads the DLL with
//! `LoadLibraryExA` (answered with a pseudo handle, see `kernel`), looks up
//! the `*ApiImpl` functions and `XErrorReport` with `GetProcAddress`, and
//! initialises. Each runtime API call then asks `QueryApiImpl` for a COM-style
//! object of its area (provider and interface GUIDs), calls one method and
//! releases it. Until an area is implemented its query fails, and the call
//! returns that error without touching the object. An area's interface id
//! changes when its method table does, so each object answers only the ids
//! whose slots we know. Implemented, as an offline retail console with one
//! signed-in user: task queues and async blocks (`xasync`), the user
//! (`xuser`), system information (`xsystem`), title id, feature checks,
//! invites and events (`xgame`), persistent local storage (`storage`), game
//! saves (`xgamesave`), networking (`xnetworking`), the package
//! (`xpackage`), the store (`xstore`), system dialogs (`xgameui`), speech
//! (`speech`), HDR, URIs, display deferrals and screenshots (`display`).
#![allow(non_snake_case)]
// Title code calls these through its import table, passing pointers the real
// API would trust.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::{CStr, c_char, c_void};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use super::{
    display, speech, storage, xasync, xgame, xgamesave, xgameui, xnetworking, xpackage, xstore,
    xsystem, xuser,
};
use crate::{HleModule, hle_export};

pub type HRESULT = i32;
pub const S_OK: HRESULT = 0;
/// `HRESULT_FROM_WIN32(ERROR_NOT_SUPPORTED)`.
pub const NOT_SUPPORTED: HRESULT = 0x8007_0032_u32 as i32;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Guid {
    pub data1: u32,
    pub data2: u16,
    pub data3: u16,
    pub data4: [u8; 8],
}

impl Guid {
    pub const fn new(data1: u32, data2: u16, data3: u16, data4: [u8; 8]) -> Guid {
        Guid {
            data1,
            data2,
            data3,
            data4,
        }
    }
}

impl std::fmt::Display for Guid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let d = self.data4;
        write!(
            f,
            "{:08x}-{:04x}-{:04x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
            self.data1, self.data2, self.data3, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]
        )
    }
}

/// A runtime object as title code calls it: its first word points at the
/// method table. Ours are static, so reference counting is a no-op.
#[repr(C)]
pub struct Object {
    vtable: *const *const (),
    /// Named in the log when a title calls a method we lack.
    area: &'static str,
    /// Slots in the table before its guard.
    methods: usize,
}

// SAFETY: the table is a static array of function addresses.
unsafe impl Sync for Object {}

impl Object {
    pub const fn new<const N: usize>(area: &'static str, methods: &'static Methods<N>) -> Object {
        Object {
            // From the whole table, so the guard after the slots is in reach.
            vtable: (methods as *const Methods<N>).cast(),
            area,
            methods: N,
        }
    }
}

/// Slots of `past_end` stubs after every method table. The runtime's
/// interfaces are not public, so their real method counts are unknown; a
/// title built against a newer runtime may call methods past the ones we
/// list.
pub const PAST_END_SLOTS: usize = 64;

/// An object's method table, by slot (`QueryInterface`, `AddRef`,
/// `Release`, then the area's methods from offset 0x18), followed by
/// `PAST_END_SLOTS` stubs so that a call past the last known slot fails
/// with `E_NOTIMPL` instead of jumping through whatever follows the table.
#[repr(C)]
pub struct Methods<const N: usize> {
    slots: [*const (); N],
    past_end: [*const (); PAST_END_SLOTS],
}

// SAFETY: function addresses only.
unsafe impl<const N: usize> Sync for Methods<N> {}

impl<const N: usize> Methods<N> {
    pub const fn new(slots: [*const (); N]) -> Methods<N> {
        assert!(
            N + PAST_END_SLOTS <= PAST_END.len() * 10,
            "PAST_END too short"
        );
        let mut past_end = [std::ptr::null(); PAST_END_SLOTS];
        let mut i = 0;
        while i < PAST_END_SLOTS {
            let slot = N + i;
            past_end[i] = PAST_END[slot / 10][slot % 10];
            i += 1;
        }
        Methods { slots, past_end }
    }
}

/// `past_end::<SLOT>` for every slot index, in rows of ten.
macro_rules! past_end_rows {
    ($($row:literal)*) => { [$(past_end_rows!(@row $row),)*] };
    (@row $r:literal) => {
        past_end_rows!(@slots $r; 0 1 2 3 4 5 6 7 8 9)
    };
    (@slots $r:literal; $($c:literal)*) => {
        [$(past_end::<{ $r * 10 + $c }> as *const (),)*]
    };
}

const PAST_END: [[*const (); 10]; 16] = past_end_rows!(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15);

/// A slot past the end of an object's known methods. Takes no arguments,
/// so any argument list is safe to pass (the x64 caller cleans up).
/// Logged once per area and slot, as such a call would repeat each frame.
extern "system" fn past_end<const SLOT: usize>(this: *const Object) -> HRESULT {
    // SAFETY: title code calls methods on objects we handed out.
    let area = unsafe { (*this).area };
    if first_call_past_end(area, SLOT) {
        tracing::warn!(
            area,
            slot = format_args!("{:#x}", SLOT * 8),
            "runtime method not implemented (past the end of its table)"
        );
    }
    xasync::E_NOTIMPL
}

/// Whether this is the first call to `slot` past the end of `area`'s table.
fn first_call_past_end(area: &'static str, slot: usize) -> bool {
    static CALLED: Mutex<Vec<(&str, usize)>> = Mutex::new(Vec::new());
    let mut called = CALLED.lock().unwrap_or_else(|e| e.into_inner());
    let first = !called.contains(&(area, slot));
    if first {
        called.push((area, slot));
    }
    first
}

/// A (provider, interface id) pair a title may ask for, and our object.
pub type Interface = (Guid, Guid, &'static Object);

const AREAS: &[&[Interface]] = &[
    xasync::INTERFACES,
    xuser::INTERFACES,
    xsystem::INTERFACES,
    xgame::INTERFACES,
    storage::INTERFACES,
    xnetworking::INTERFACES,
    xpackage::INTERFACES,
    display::INTERFACES,
    xgamesave::INTERFACES,
    xstore::INTERFACES,
    xgameui::INTERFACES,
    speech::INTERFACES,
];

/// `AddRef` and `Release`.
pub extern "system" fn add_ref(_: *const Object) -> u32 {
    1
}

/// A slot nobody has needed yet; logged with its area. Wrong for a slot
/// that returns `bool` or a value.
pub extern "system" fn missing<const SLOT: usize>(this: *const Object) -> HRESULT {
    // SAFETY: title code calls methods on objects we handed out.
    let area = unsafe { (*this).area };
    tracing::warn!(
        area,
        slot = format_args!("{:#x}", SLOT * 8),
        "runtime method not implemented"
    );
    xasync::E_NOTIMPL
}

/// Writes `value` through `out` if it is non-null.
pub fn put<T>(out: *mut T, value: T) -> HRESULT {
    if out.is_null() {
        return xasync::E_INVALIDARG;
    }
    // SAFETY: checked non-null; the title passes room for a `T`.
    unsafe { out.write(value) };
    S_OK
}

/// Copies `text` and a NUL into the title's `size`-byte buffer and reports
/// the bytes used, as the runtime's string getters do.
pub fn put_string(text: &str, size: usize, buffer: *mut u8, used: *mut usize) -> HRESULT {
    let len = text.len() + 1;
    if buffer.is_null() || size < len {
        return xasync::E_NOT_SUFFICIENT_BUFFER;
    }
    // SAFETY: the buffer holds `size` bytes, checked above.
    unsafe {
        buffer.copy_from_nonoverlapping(text.as_ptr(), text.len());
        buffer.add(text.len()).write(0);
    }
    if !used.is_null() {
        // SAFETY: checked non-null.
        unsafe { used.write(len) };
    }
    S_OK
}

/// The title's NUL-terminated string, if any.
pub fn text(s: *const c_char) -> Option<String> {
    // SAFETY: the title passes a NUL-terminated string or null.
    (!s.is_null()).then(|| unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned())
}

/// A new registration token for callbacks we never call (events nobody
/// raises offline).
pub fn token() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// Calls method `slot` (byte offset) of `object` as title code would.
#[cfg(test)]
pub(crate) fn method<F: Copy>(object: &Object, slot: usize) -> F {
    // SAFETY: the caller names the slot's signature; tables cover the slots
    // tests use.
    unsafe { std::mem::transmute_copy(&*object.vtable.add(slot / 8)) }
}

/// Checks that `object`'s guard answers every slot past its methods, called
/// with arguments as title code would; the object's area is not otherwise
/// touched.
#[cfg(test)]
pub(crate) fn assert_guarded(object: &Object) {
    type Any = extern "system" fn(*const Object, usize, usize, usize, usize, usize) -> HRESULT;
    for index in object.methods..object.methods + PAST_END_SLOTS {
        // The stub for this very slot, so the log names the slot called.
        let stub = method::<*const ()>(object, index * 8);
        assert_eq!(
            stub,
            PAST_END[index / 10][index % 10],
            "{} slot {index}",
            object.area
        );
        let call = method::<Any>(object, index * 8);
        assert_eq!(
            call(object, 1, 2, 3, 4, 5),
            xasync::E_NOTIMPL,
            "{} slot {index}",
            object.area
        );
    }
}

pub extern "system" fn XGameRuntimeInitialize() -> HRESULT {
    tracing::info!("XGameRuntimeInitialize");
    S_OK
}

pub extern "system" fn XGameRuntimeUninitialize() {
    tracing::info!("XGameRuntimeUninitialize");
}

/// The arguments identify the title's build of the stub; not checked.
pub extern "system" fn InitializeApiImpl(version: u64, build: u64) -> HRESULT {
    tracing::info!(version, build, "InitializeApiImpl");
    S_OK
}

pub extern "system" fn InitializeApiImplEx(version: u64, build: u64, flags: u64) -> HRESULT {
    tracing::info!(version, build, flags, "InitializeApiImplEx");
    S_OK
}

/// The stub prefers this one when present. It passes the same three values
/// and a pointer to the title's initialisation options, null unless the
/// title initialises with options (the title does not).
pub extern "system" fn InitializeApiImplEx2(
    version: u64,
    build: u64,
    flags: u64,
    options: *const c_void,
) -> HRESULT {
    tracing::info!(version, build, flags, ?options, "InitializeApiImplEx2");
    S_OK
}

pub extern "system" fn UninitializeApiImpl() {
    tracing::info!("UninitializeApiImpl");
}

/// The object for an implemented (provider, interface id); others fail,
/// each pair logged once.
pub extern "system" fn QueryApiImpl(
    provider: *const Guid,
    iid: *const Guid,
    out: *mut *mut c_void,
) -> HRESULT {
    static QUERIED: Mutex<Vec<(Guid, Guid)>> = Mutex::new(Vec::new());
    // SAFETY: the title passes pointers to GUIDs (null is tolerated).
    let (Some(&provider), Some(&iid)) = (unsafe { provider.as_ref() }, unsafe { iid.as_ref() })
    else {
        return NOT_SUPPORTED;
    };
    #[cfg(feature = "diagnostics")]
    if tracing::enabled!(tracing::Level::DEBUG) {
        // Each runtime call site once: which title code uses which area, when.
        static CALLS: Mutex<Vec<String>> = Mutex::new(Vec::new());
        let from = super::kernel::callers();
        let mut calls = CALLS.lock().unwrap_or_else(|e| e.into_inner());
        if !calls.contains(&from) {
            tracing::debug!(%provider, %iid, %from, "runtime call");
            calls.push(from);
        }
    }
    let found = AREAS
        .iter()
        .flat_map(|area| area.iter())
        .find(|(p, i, _)| (*p, *i) == (provider, iid));
    if let Some(&(_, _, object)) = found
        && !out.is_null()
    {
        // SAFETY: checked non-null; the title passes room for a pointer.
        unsafe { out.write((object as *const Object).cast_mut().cast()) };
        return S_OK;
    }
    let mut queried = QUERIED.lock().unwrap_or_else(|e| e.into_inner());
    if !queried.contains(&(provider, iid)) {
        queried.push((provider, iid));
        tracing::warn!(%provider, %iid, "runtime API area not implemented");
    }
    NOT_SUPPORTED
}

pub extern "system" fn XErrorReport(hr: HRESULT, message: *const c_char) {
    let message = if message.is_null() {
        "".into()
    } else {
        // SAFETY: a non-null message is a NUL-terminated string.
        unsafe { CStr::from_ptr(message) }.to_string_lossy()
    };
    tracing::warn!(hr = format_args!("{:#010x}", hr as u32), %message, "XErrorReport");
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "xgameruntime",
        dlls: &["xgameruntime.dll"],
        exports: vec![
            hle_export!(XGameRuntimeInitialize),
            hle_export!(XGameRuntimeUninitialize),
            hle_export!(InitializeApiImpl),
            hle_export!(InitializeApiImplEx),
            hle_export!(InitializeApiImplEx2),
            hle_export!(UninitializeApiImpl),
            hle_export!(QueryApiImpl),
            hle_export!(XErrorReport),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modules::kernel::{FreeLibrary, GetProcAddress, LoadLibraryExA};

    #[test]
    fn every_area_table_is_guarded_past_its_methods() {
        let objects: Vec<&Object> = AREAS
            .iter()
            .flat_map(|area| area.iter().map(|&(_, _, object)| object))
            .collect();
        assert!(objects.len() > 10);
        for object in objects {
            assert_guarded(object);
        }
    }

    #[test]
    fn a_call_past_the_end_returns_e_notimpl_and_is_logged_once() {
        static TABLE: Methods<4> = Methods::new([
            missing::<0> as *const (),
            add_ref as *const (),
            add_ref as *const (),
            missing::<3> as *const (),
        ]);
        static OBJECT: Object = Object::new("guard test", &TABLE);
        // Slot 4 (0x20) is the first past the end; called with arguments.
        type Call = extern "system" fn(*const Object, u32, *mut c_void, u64) -> HRESULT;
        let call = method::<Call>(&OBJECT, 0x20);
        assert_eq!(call(&OBJECT, 7, std::ptr::null_mut(), 9), xasync::E_NOTIMPL);
        assert_eq!(call(&OBJECT, 7, std::ptr::null_mut(), 9), xasync::E_NOTIMPL);
        let last = method::<Call>(&OBJECT, (4 + PAST_END_SLOTS - 1) * 8);
        assert_eq!(last(&OBJECT, 0, std::ptr::null_mut(), 0), xasync::E_NOTIMPL);
        // Both slots already logged; another slot is not.
        assert!(!first_call_past_end("guard test", 4));
        assert!(!first_call_past_end("guard test", 4 + PAST_END_SLOTS - 1));
        assert!(first_call_past_end("guard test", 5));
        // Existing slots are unchanged.
        type AddRef = extern "system" fn(*const Object) -> u32;
        assert_eq!(method::<AddRef>(&OBJECT, 0x08)(&OBJECT), 1);
    }

    #[test]
    fn title_stub_loads_the_runtime_dynamically() {
        let load = |name: &CStr| LoadLibraryExA(name.as_ptr().cast(), std::ptr::null_mut(), 0x800);
        let runtime = load(c"xgameruntime.dll");
        assert!(!runtime.is_null());
        let proc = |name: &CStr| GetProcAddress(runtime, name.as_ptr().cast());
        for name in [
            c"InitializeApiImpl",
            c"InitializeApiImplEx",
            c"InitializeApiImplEx2",
            c"QueryApiImpl",
            c"UninitializeApiImpl",
            c"XErrorReport",
        ] {
            assert!(!proc(name).is_null(), "{name:?}");
        }
        assert!(proc(c"NotAnExport").is_null());
        let ordinal = std::ptr::without_provenance::<u8>(1);
        assert!(GetProcAddress(runtime, ordinal).is_null());

        type Init = extern "system" fn(u64, u64, u64) -> HRESULT;
        type Query = extern "system" fn(*const Guid, *const Guid, *mut *mut c_void) -> HRESULT;
        // SAFETY: the addresses are our functions with these signatures.
        let (init, query) = unsafe {
            (
                std::mem::transmute::<*mut c_void, Init>(proc(c"InitializeApiImplEx")),
                std::mem::transmute::<*mut c_void, Query>(proc(c"QueryApiImpl")),
            )
        };
        assert_eq!(init(1, 2, 2), S_OK);
        let guid = Guid {
            data1: 1,
            data2: 2,
            data3: 3,
            data4: [4; 8],
        };
        let untouched = 0x1234 as *mut c_void;
        let mut out = untouched;
        assert!(query(&guid, &guid, &mut out) < 0);
        assert_eq!(out, untouched);
        assert_eq!(FreeLibrary(runtime), 1);

        #[cfg(windows)]
        {
            let kernel32 = load(c"kernel32.dll");
            assert!(!kernel32.is_null(), "host DLLs still load from the host");
            assert!(!GetProcAddress(kernel32, c"GetTickCount".as_ptr().cast()).is_null());
        }
    }
}
