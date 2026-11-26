//! Connectivity hint of the PC (XNetworking), from Windows: Network List Manager for the level, the
//! default route's interface for its IANA type, plus the console-only TCP receive buffer settings and
//! statistics, answered as Microsoft Learn documents for the PC. Port of the networking half of
//! `platform_interface.cpp`. Slots 3 to 14 stay E_NOTIMPL (the preferred UDP port is not invented,
//! URL security information comes from the Xbox Live portal), and so does slot 16: connectivity
//! changes are not tracked, so a callback could never run (rule 3); XNetworkingGetConnectivityHint
//! gives the current state on every call (docs/progress.md, "Change subscriptions").

use std::ffi::c_void;

use windows::Win32::NetworkManagement::IpHelper::{GetBestInterface, GetIfEntry, MIB_IFROW};
use windows::Win32::Networking::NetworkListManager::{
    INetworkListManager, NLM_CONNECTIVITY, NLM_CONNECTIVITY_IPV4_INTERNET,
    NLM_CONNECTIVITY_IPV4_LOCALNETWORK, NLM_CONNECTIVITY_IPV4_SUBNET,
    NLM_CONNECTIVITY_IPV6_INTERNET, NLM_CONNECTIVITY_IPV6_LOCALNETWORK,
    NLM_CONNECTIVITY_IPV6_SUBNET, NetworkListManager,
};
use windows::Win32::System::Com::{
    CLSCTX_ALL, COINIT_MULTITHREADED, CoCreateInstance, CoInitializeEx, CoUninitialize,
};

use crate::log::trace;
use crate::object::{E_INVALIDARG, E_NOTIMPL, HResult, S_OK};

/// XNetworkingConnectivityLevelHint values.
const LEVEL_UNKNOWN: u32 = 0;
const LEVEL_NONE: u32 = 1;
const LEVEL_LOCAL: u32 = 2;
const LEVEL_INTERNET: u32 = 3;

/// XNetworkingConnectivityHint (XNetworking.h): the structure the title passes by pointer.
#[repr(C)]
#[derive(Default)]
pub struct ConnectivityHint {
    pub connectivity_level: u32,
    /// 0: unknown (the cost comes from the WinRT API, not queried).
    pub connectivity_cost: u32,
    pub iana_interface_type: u32,
    pub network_initialized: bool,
    pub approaching_data_limit: bool,
    pub over_data_limit: bool,
    pub roaming: bool,
}

/// The level of connectivity Network List Manager reports; `LEVEL_UNKNOWN` if it cannot be asked.
fn level_from_windows() -> u32 {
    // SAFETY: the calling convention of COM initialisation; the result is paired below.
    let init = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
    let level = match unsafe {
        CoCreateInstance::<_, INetworkListManager>(&NetworkListManager, None, CLSCTX_ALL)
    } {
        Ok(manager) => match unsafe { manager.GetConnectivity() } {
            Ok(connectivity) => classify(connectivity),
            Err(_) => LEVEL_UNKNOWN,
        },
        Err(_) => LEVEL_UNKNOWN,
    };
    if init.is_ok() {
        // SAFETY: paired with the successful CoInitializeEx above.
        unsafe { CoUninitialize() };
    }
    level
}

/// Maps Network List Manager's connectivity flags to the title's levels.
fn classify(connectivity: NLM_CONNECTIVITY) -> u32 {
    let internet = NLM_CONNECTIVITY_IPV4_INTERNET.0 | NLM_CONNECTIVITY_IPV6_INTERNET.0;
    let local = NLM_CONNECTIVITY_IPV4_LOCALNETWORK.0
        | NLM_CONNECTIVITY_IPV6_LOCALNETWORK.0
        | NLM_CONNECTIVITY_IPV4_SUBNET.0
        | NLM_CONNECTIVITY_IPV6_SUBNET.0;
    if connectivity.0 & internet != 0 {
        LEVEL_INTERNET
    } else if connectivity.0 & local != 0 {
        LEVEL_LOCAL
    } else {
        LEVEL_NONE
    }
}

/// The IANA type (ifType) of the interface of the default route; 0 when there is none.
fn default_route_interface_type() -> u32 {
    let mut index = 0u32;
    // SAFETY: the destination 0 is the default route; `index` is writable.
    if unsafe { GetBestInterface(0, &mut index) } != 0 {
        return 0;
    }
    let mut row = MIB_IFROW {
        dwIndex: index,
        ..Default::default()
    };
    // SAFETY: the row is writable and its index is set, as GetIfEntry requires.
    if unsafe { GetIfEntry(&mut row) } == 0 {
        row.dwType
    } else {
        0
    }
}

/// XNetworkingGetConnectivityHint (slot 15).
///
/// # Safety
/// `hint` is null or points at one writable `ConnectivityHint`.
pub unsafe extern "system" fn get_connectivity_hint(
    _this: *mut c_void,
    hint: *mut ConnectivityHint,
) -> HResult {
    let Some(hint) = (unsafe { hint.as_mut() }) else {
        return E_INVALIDARG;
    };
    let mut level = level_from_windows();
    if level == LEVEL_UNKNOWN {
        level = LEVEL_NONE;
    }
    *hint = ConnectivityHint {
        connectivity_level: level,
        iana_interface_type: default_route_interface_type(),
        network_initialized: true,
        ..Default::default()
    };
    S_OK
}

/// XNetworkingUnregisterConnectivityHintChanged (slot 17): no subscription can succeed (slot 16
/// answers E_NOTIMPL), so there is nothing to remove.
unsafe extern "system" fn unregister_hint_changed(
    _this: *mut c_void,
    _token: u64,
    _wait: bool,
) -> bool {
    false
}

/// The TCP receive buffers of XNetworkingConfigurationSetting and XNetworkingStatisticsType: Title,
/// System, Tools. Any other value is E_INVALIDARG (functions/xnetworkingqueryconfigurationsetting,
/// xnetworkingquerystatistics).
const TCP_BUFFER_KINDS: u32 = 3;

/// XNetworkingTcpQueuedReceivedBufferUsageStatistics, the member of XNetworkingStatisticsBuffer for
/// each of the three types (structs/xnetworkingstatisticsbuffer).
#[repr(C)]
#[derive(Default)]
pub struct TcpQueuedReceiveStatistics {
    pub bytes_currently_queued: u64,
    pub peak_bytes_ever_queued: u64,
    pub total_bytes_queued: u64,
    pub bytes_dropped_for_exceeding_max: u64,
    pub bytes_dropped_for_any_failure: u64,
}

/// XNetworkingQueryConfigurationSetting (slot 18).
///
/// The 0 is Microsoft Learn's answer for the PC, for each of the three settings: "Retrieving this
/// setting on PC will always report the value 0." ("this value" for
/// MaxToolsTcpQueuedReceiveBufferSize; XNetworkingConfigurationSetting,
/// https://learn.microsoft.com/en-us/gaming/gdk/docs/reference/networking/xnetworking/enums/xnetworkingconfigurationsetting)
///
/// # Safety
/// `value` is null or points at one writable `u64`.
unsafe extern "system" fn query_configuration_setting(
    _this: *mut c_void,
    setting: u32,
    value: *mut u64,
) -> HResult {
    if setting >= TCP_BUFFER_KINDS || value.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: non-null and writable, per this function's contract.
    unsafe { value.write(0) };
    S_OK
}

/// XNetworkingSetConfigurationSetting (slot 19): the buffers cannot be set on PC, so E_NOTIMPL,
/// logged (same page as above).
unsafe extern "system" fn set_configuration_setting(
    _this: *mut c_void,
    setting: u32,
    value: u64,
) -> HResult {
    if setting >= TCP_BUFFER_KINDS {
        return E_INVALIDARG;
    }
    trace!(
        "[runtime] XNetworkingSetConfigurationSetting({setting}, {value}) : not settable on PC -> E_NOTIMPL"
    );
    E_NOTIMPL
}

/// XNetworkingQueryStatistics (slot 20): only the member of the requested type is written.
///
/// All zero is Microsoft Learn's answer for the PC, for each of the three types: "On PC, all
/// statistics in the structure are always reported as 0." (XNetworkingStatisticsType,
/// https://learn.microsoft.com/en-us/gaming/gdk/docs/reference/networking/xnetworking/enums/xnetworkingstatisticstype)
///
/// # Safety
/// `statistics` is null or points at one writable `TcpQueuedReceiveStatistics`.
unsafe extern "system" fn query_statistics(
    _this: *mut c_void,
    kind: u32,
    statistics: *mut TcpQueuedReceiveStatistics,
) -> HResult {
    if kind >= TCP_BUFFER_KINDS || statistics.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: non-null and writable, per this function's contract.
    unsafe { statistics.write(TcpQueuedReceiveStatistics::default()) };
    S_OK
}

/// The method table of the networking object: slots 15 and 17 to 20 (16 stays E_NOTIMPL).
pub fn methods() -> Vec<(usize, *const c_void)> {
    vec![
        (15, get_connectivity_hint as *const c_void),
        (17, unregister_hint_changed as *const c_void),
        (18, query_configuration_setting as *const c_void),
        (19, set_configuration_setting as *const c_void),
        (20, query_statistics as *const c_void),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn network_list_flags_give_the_title_levels() {
        assert_eq!(classify(NLM_CONNECTIVITY(0)), LEVEL_NONE);
        assert_eq!(classify(NLM_CONNECTIVITY_IPV4_LOCALNETWORK), LEVEL_LOCAL);
        assert_eq!(classify(NLM_CONNECTIVITY_IPV6_INTERNET), LEVEL_INTERNET);
    }

    #[test]
    fn a_null_hint_is_refused() {
        // SAFETY: a null pointer is part of the contract.
        assert_eq!(
            unsafe { get_connectivity_hint(std::ptr::null_mut(), std::ptr::null_mut()) },
            E_INVALIDARG
        );
    }

    #[test]
    fn tcp_buffers_answer_like_a_pc() {
        for kind in 0..TCP_BUFFER_KINDS {
            let mut value = 99u64;
            let mut usage = TcpQueuedReceiveStatistics {
                bytes_currently_queued: 1,
                ..Default::default()
            };
            // SAFETY: valid pointers to locals.
            unsafe {
                assert_eq!(
                    query_configuration_setting(std::ptr::null_mut(), kind, &mut value),
                    S_OK
                );
                assert_eq!(
                    set_configuration_setting(std::ptr::null_mut(), kind, 1 << 20),
                    E_NOTIMPL
                );
                assert_eq!(
                    query_statistics(std::ptr::null_mut(), kind, &mut usage),
                    S_OK
                );
            }
            assert_eq!((value, usage.bytes_currently_queued), (0, 0));
        }
        // SAFETY: as above; an unknown kind is refused before any write.
        unsafe {
            assert_eq!(
                query_configuration_setting(std::ptr::null_mut(), TCP_BUFFER_KINDS, &mut 0),
                E_INVALIDARG
            );
            assert_eq!(
                set_configuration_setting(std::ptr::null_mut(), TCP_BUFFER_KINDS, 0),
                E_INVALIDARG
            );
            assert_eq!(
                query_statistics(std::ptr::null_mut(), 0, std::ptr::null_mut()),
                E_INVALIDARG
            );
        }
    }

    #[test]
    fn connectivity_changes_are_not_subscribed() {
        let table = methods();
        assert!(table.iter().all(|&(slot, _)| slot != 16));
        // SAFETY: the method ignores its arguments.
        assert!(!unsafe { unregister_hint_changed(std::ptr::null_mut(), 1, true) });
    }
}
