"""Runtime GDK : état HDR (XDisplay), réseau (XNetworking) et contraste élevé (XAccessibility) du PC.

Structures et signatures : XDisplay.h et XNetworking.h du GDK public (version 261000), pages
XNetworking et XAccessibility de Microsoft Learn. Les valeurs viennent de Windows (DXGI, Network
List Manager, SPI_GETHIGHCONTRAST) ou de la réponse documentée du PC : rien n'est inventé.
"""
import ctypes
import math
import os
import pathlib
import subprocess
import sys

import pytest
from test_runtime_cpp import S_OK, GdkInterface, load
from test_runtime_events import E_INVALIDARG, E_NOTIMPL, EVENT_CALLBACK
from test_runtime_user_device import USER_DEVICE

HDR = ("03F0FE74-FDD9-4E5C-B630-F9339C47ACC5", "35F07670-706E-4BFB-9476-090798C5EBF3")
NETWORKING = ("37E56907-2F10-41E8-B72F-36EDB185331A", "BF2346B2-39AF-4658-B5EA-44713C7E83B3")
HDR_UNKNOWN, HDR_ENABLED, HDR_DISABLED = 0, 1, 2
PREFER_HDR = 0
FALLBACK_LUMINANCE = (0.01, 1499.0, 799.0)  # repli de DXVK (NormalizeDisplayMetadata)
LEVEL_NONE, LEVEL_CONSTRAINED = 1, 4


class HdrModeInfo(ctypes.Structure):
    _fields_ = [("min_luminance", ctypes.c_float), ("max_luminance", ctypes.c_float),
                ("max_full_frame_luminance", ctypes.c_float)]


class ConnectivityHint(ctypes.Structure):
    _fields_ = [("level", ctypes.c_uint32), ("cost", ctypes.c_uint32), ("iana_interface_type", ctypes.c_uint32),
                ("network_initialized", ctypes.c_bool), ("approaching_data_limit", ctypes.c_bool),
                ("over_data_limit", ctypes.c_bool), ("roaming", ctypes.c_bool)]


def test_hdr_mode_reports_the_pc_display(runtime_dir):
    try_enable = GdkInterface(runtime_dir, *HDR).method(4, ctypes.c_uint32, ctypes.c_uint32,
                                                        ctypes.POINTER(HdrModeInfo))
    info = HdrModeInfo()
    result = try_enable(PREFER_HDR, ctypes.byref(info))
    assert result in (HDR_UNKNOWN, HDR_ENABLED, HDR_DISABLED)
    if result == HDR_ENABLED:
        assert info.max_luminance > info.min_luminance >= 0
    assert try_enable(PREFER_HDR, None) == result                   # infos facultatives


def _normalized(runtime_dir, minimum, maximum, full_frame):
    normalize = load(runtime_dir, "xgameruntime.dll").AnyXboxX_NormalizeHdrLuminance
    normalize.restype, normalize.argtypes = None, [ctypes.POINTER(HdrModeInfo)]
    info = HdrModeInfo(minimum, maximum, full_frame)
    normalize(ctypes.byref(info))
    return info.min_luminance, info.max_luminance, info.max_full_frame_luminance


@pytest.mark.parametrize("reported", [
    (0.0, 0.0, 0.0),                 # écran sans métadonnées HDR statiques (issue #191, RTX 5070)
    (0.5, 0.5, 0.0),                 # pic égal au minimum
    (0.05, 0.02, 300.0),             # pic sous le minimum
    (-1.0, 600.0, 400.0),            # minimum négatif
    (math.nan, 600.0, 400.0),
    (0.01, math.nan, 400.0),
])
def test_hdr_luminance_without_usable_display_metadata_falls_back(runtime_dir, reported):
    assert _normalized(runtime_dir, *reported) == pytest.approx(FALLBACK_LUMINANCE)


@pytest.mark.parametrize("reported, expected", [
    ((0.0006, 993.5, 276.8), (0.0006, 993.5, 276.8)),   # valeurs réelles d'un écran HDR : inchangées
    ((0.0, 400.0, 400.0), (0.0, 400.0, 400.0)),         # minimum 0 accepté quand le pic est connu
    ((0.01, 1000.0, 0.0), (0.01, 1000.0, 799.0)),       # plein écran absent : repli
    ((0.01, 600.0, 0.0), (0.01, 600.0, 600.0)),         # repli plein écran plafonné au pic
    ((0.01, 400.0, 800.0), (0.01, 400.0, 400.0)),       # plein écran au-dessus du pic
])
def test_hdr_luminance_keeps_coherent_display_values(runtime_dir, reported, expected):
    assert _normalized(runtime_dir, *reported) == pytest.approx(expected, rel=1e-6)


def test_connectivity_hint_comes_from_windows(runtime_dir):
    get = GdkInterface(runtime_dir, *NETWORKING).method(15, ctypes.c_uint32, ctypes.POINTER(ConnectivityHint))
    hint = ConnectivityHint()
    assert get(ctypes.byref(hint)) == S_OK
    assert LEVEL_NONE <= hint.level <= LEVEL_CONSTRAINED and hint.network_initialized
    assert get(None) == E_INVALIDARG


UNTOUCHED = 0x5EED_5EED_5EED_5EED
TRACE_PROBE = """
import ctypes, pathlib, sys
sys.path.insert(0, sys.argv[2])
from test_runtime_cpp import GdkInterface
api = GdkInterface(pathlib.Path(sys.argv[1]), sys.argv[3], sys.argv[4])
subscribe = api.method(int(sys.argv[5]), ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.POINTER(ctypes.c_uint64))
print(hex(subscribe(None, None, None, ctypes.byref(ctypes.c_uint64()))))
"""


def test_connectivity_subscription_is_not_implemented(runtime_dir):
    """Règle 3 : les changements de connectivité ne sont pas suivis, l'abonnement ne réussit pas."""
    api = GdkInterface(runtime_dir, *NETWORKING)
    subscribe = api.method(16, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                           ctypes.POINTER(ctypes.c_uint64))
    unsubscribe = api.method(17, ctypes.c_bool, ctypes.c_uint64, ctypes.c_bool)
    callback, token = EVENT_CALLBACK(lambda context, hint: None), ctypes.c_uint64(UNTOUCHED)
    assert subscribe(None, None, ctypes.cast(callback, ctypes.c_void_p), ctypes.byref(token)) == E_NOTIMPL
    assert token.value == UNTOUCHED and unsubscribe(token.value, True) is False


@pytest.mark.parametrize("interface, method", [
    (NETWORKING, 16),                     # XNetworkingRegisterConnectivityHintChanged
    (USER_DEVICE, 4),                     # XUserRegisterForDeviceAssociationChanged
    (USER_DEVICE, 7),                     # XUserRegisterForDefaultAudioEndpointUtf16Changed
])
def test_refused_subscriptions_are_logged(runtime_dir, interface, method):
    """Avec ANYXBOXX_TRACE=1, l'appel refusé laisse une ligne « E_NOTIMPL » dans le journal (stderr)."""
    tests = pathlib.Path(__file__).parent
    result = subprocess.run([sys.executable, "-c", TRACE_PROBE, str(runtime_dir), str(tests), *interface, str(method)],
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
                            env={**os.environ, "ANYXBOXX_TRACE": "1"})
    assert result.stdout.strip() == hex(E_NOTIMPL)
    assert "E_NOTIMPL" in result.stderr


def test_other_networking_methods_are_not_implemented(runtime_dir):
    port = GdkInterface(runtime_dir, *NETWORKING).method(3, ctypes.c_uint32, ctypes.c_void_p)
    assert port(None) == E_NOTIMPL                     # XNetworkingQueryPreferredLocalUdpMultiplayerPort


ACCESSIBILITY = ("3E241977-6237-41E9-8559-102C2D9983F1", "D722B373-8C4D-4692-9E51-D6AD9B37AA7D")
TCP_BUFFER_KINDS = 3                                   # Title, System, Tools
CONTRAST_OFF, CONTRAST_DARK, CONTRAST_LIGHT = 0, 1, 2
SPI_GETHIGHCONTRAST, HCF_HIGHCONTRASTON = 0x0042, 0x1


class TcpQueuedReceiveStatistics(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in
                ("queued", "peak", "total", "dropped_for_max", "dropped_for_any_failure")]


def test_tcp_buffer_settings_answer_like_a_pc(runtime_dir):
    """Microsoft Learn : sur PC, lecture à 0, modification en E_NOTIMPL, statistiques à 0."""
    api = GdkInterface(runtime_dir, *NETWORKING)
    query = api.method(18, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64))
    configure = api.method(19, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint64)
    statistics = api.method(20, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(TcpQueuedReceiveStatistics))
    for kind in range(TCP_BUFFER_KINDS):
        value, usage = ctypes.c_uint64(99), TcpQueuedReceiveStatistics(1, 2, 3, 4, 5)
        assert query(kind, ctypes.byref(value)) == S_OK and value.value == 0
        assert configure(kind, 1 << 20) == E_NOTIMPL
        assert statistics(kind, ctypes.byref(usage)) == S_OK
        assert [getattr(usage, name) for name, _ in usage._fields_] == [0] * 5
    assert query(TCP_BUFFER_KINDS, ctypes.byref(ctypes.c_uint64())) == E_INVALIDARG
    assert configure(TCP_BUFFER_KINDS, 0) == E_INVALIDARG
    assert statistics(TCP_BUFFER_KINDS, ctypes.byref(TcpQueuedReceiveStatistics())) == E_INVALIDARG


class HighContrast(ctypes.Structure):
    _fields_ = [("size", ctypes.c_uint32), ("flags", ctypes.c_uint32), ("scheme", ctypes.c_wchar_p)]


def test_high_contrast_mode_follows_windows(runtime_dir):
    get = GdkInterface(runtime_dir, *ACCESSIBILITY).method(5, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32))
    windows = HighContrast(ctypes.sizeof(HighContrast))
    assert ctypes.windll.user32.SystemParametersInfoW(SPI_GETHIGHCONTRAST, windows.size, ctypes.byref(windows), 0)
    mode = ctypes.c_uint32(7)
    assert get(ctypes.byref(mode)) == S_OK
    if windows.flags & HCF_HIGHCONTRASTON:
        assert mode.value in (CONTRAST_DARK, CONTRAST_LIGHT)
    else:
        assert mode.value == CONTRAST_OFF
    assert get(None) == E_INVALIDARG


def test_captions_and_speech_are_not_implemented(runtime_dir):
    api = GdkInterface(runtime_dir, *ACCESSIBILITY)
    assert api.method(3, ctypes.c_uint32, ctypes.c_void_p)(None) == E_NOTIMPL     # XClosedCaptionGetProperties
    assert api.method(9, ctypes.c_uint32, ctypes.c_void_p)(None) == E_NOTIMPL     # XSpeechSynthesizerCreate
