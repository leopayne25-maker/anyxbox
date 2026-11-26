"""Runtime GDK hors ligne : XGameStreaming sans aucun client de streaming.

Signatures et codes d'erreur : pages XGameStreaming de Microsoft Learn
(learn.microsoft.com/gaming/gdk/docs/reference/system/xgamestreaming/xgamestreaming_members).
Le jeu tourne sur le PC : aucun appareil ne le reçoit, rien ne prétend le contraire.
"""
import ctypes

import pytest

from test_runtime_cpp import S_OK, GdkInterface
from test_runtime_events import E_INVALIDARG, E_NOTIMPL, EVENT_CALLBACK

STREAMING = ("0A2192AA-B2D5-4D58-83BE-383B6D80799E", "57786622-6605-46D0-B917-0F22BBCD9C52")
NOT_INITIALIZED, CLIENT_NOT_CONNECTED, NO_DATA, NO_DATACENTER, NOT_STREAMING_CONTROLLER = (
    0x89245400, 0x89245401, 0x89245402, 0x89245403, 0x89245404)
DISCONNECTED = 0
CLIENT = 0x1234                         # XGameStreamingClientId quelconque : jamais connecté
HRESULT, U32, U64, SIZE, PTR = ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_void_p


@pytest.fixture
def streaming(runtime_dir):
    api = GdkInterface(runtime_dir, *STREAMING)
    assert api.method(3, HRESULT)() == S_OK                         # XGameStreamingInitialize
    return api


def test_streaming_apis_require_initialization(streaming):
    streaming.method(4, None)()                                     # XGameStreamingUninitialize
    used = U32(7)
    try:
        assert streaming.method(10, HRESULT, U32, PTR, ctypes.POINTER(U32))(0, None, ctypes.byref(used)) \
            == NOT_INITIALIZED
        assert streaming.method(16, HRESULT, SIZE, PTR)(0, None) == NOT_INITIALIZED
        assert streaming.method(21, HRESULT, U64, PTR)(CLIENT, None) == NOT_INITIALIZED
        assert used.value == 7
    finally:
        assert streaming.method(3, HRESULT)() == S_OK


def test_a_local_game_has_no_streaming_client(streaming):
    assert streaming.method(5, ctypes.c_bool)() is False            # XGameStreamingIsStreaming
    assert streaming.method(9, U32)() == 0                          # XGameStreamingGetClientCount
    clients, used = (U64 * 4)(), U32(9)
    assert streaming.method(10, HRESULT, U32, PTR, ctypes.POINTER(U32))(4, clients, ctypes.byref(used)) == S_OK
    assert used.value == 0
    assert streaming.method(10, HRESULT, U32, PTR, PTR)(0, None, None) == E_INVALIDARG
    assert streaming.method(11, U32, U64)(CLIENT) == DISCONNECTED   # XGameStreamingGetConnectionState


def test_connection_changes_can_be_subscribed(streaming):
    subscribe = streaming.method(12, HRESULT, PTR, PTR, PTR, ctypes.POINTER(U64))
    unsubscribe = streaming.method(13, ctypes.c_bool, U64, ctypes.c_bool)
    callback, token = EVENT_CALLBACK(lambda context, client: None), U64()
    assert subscribe(None, None, ctypes.cast(callback, PTR), ctypes.byref(token)) == S_OK
    assert subscribe(None, None, None, ctypes.byref(U64())) == E_INVALIDARG
    assert unsubscribe(token.value, True) is True and unsubscribe(token.value, True) is False


def test_client_queries_report_the_client_as_not_connected(streaming):
    buffer, size = ctypes.create_string_buffer(64), SIZE()
    calls = [
        streaming.method(6, HRESULT, U64, PTR, PTR, PTR, PTR)(CLIENT, None, None, None, None),
        streaming.method(8, HRESULT, U64, PTR, PTR)(CLIENT, None, None),
        streaming.method(21, HRESULT, U64, PTR)(CLIENT, None),
        streaming.method(22, HRESULT, U64, PTR)(CLIENT, None),
        streaming.method(26, HRESULT, U64, SIZE, PTR)(CLIENT, 0, None),
        streaming.method(28, HRESULT, U64, ctypes.c_char_p, SIZE, PTR)(CLIENT, b"Driving", 0, None),
        streaming.method(30, HRESULT, U64, PTR, SIZE, PTR)(CLIENT, None, 64, buffer),
        streaming.method(31, HRESULT, U64, SIZE, PTR)(CLIENT, 64, buffer),
        streaming.method(32, HRESULT, U64, SIZE, PTR, PTR)(CLIENT, 64, buffer, ctypes.byref(size)),
    ]
    assert calls == [CLIENT_NOT_CONNECTED] * len(calls)
    assert streaming.method(7, ctypes.c_bool, U64, U64, ctypes.c_bool)(CLIENT, 1, True) is False
    assert streaming.method(29, SIZE, U64)(CLIENT) == 0             # GetTouchBundleVersionNameSize


def test_display_details_check_the_documented_parameters(streaming):
    details = streaming.method(33, HRESULT, U64, U32, ctypes.c_float, ctypes.c_float, PTR)
    assert details(CLIENT, 1920 * 1080, 21.5 / 9, 16 / 10, None) == CLIENT_NOT_CONNECTED
    assert details(CLIENT, 0, 21.5 / 9, 16 / 10, None) == E_INVALIDARG
    assert details(CLIENT, 1920 * 1080, 4 / 3, 16 / 10, None) == E_INVALIDARG      # plus étroit que 16/9
    assert details(CLIENT, 1920 * 1080, float("inf"), 16 / 10, None) == E_INVALIDARG
    assert details(CLIENT, 1920 * 1080, 21.5 / 9, 2.0, None) == E_INVALIDARG       # plus large que 16/9


def test_no_stream_means_no_latency_datacenter_or_streaming_controller(streaming):
    assert streaming.method(14, HRESULT, U64, PTR, PTR, PTR)(CLIENT, None, None, None) == NO_DATA
    assert streaming.method(15, SIZE)() == 0                        # GetServerLocationNameSize
    assert streaming.method(16, HRESULT, SIZE, PTR)(0, None) == NO_DATACENTER
    assert streaming.method(23, HRESULT, PTR, PTR)(None, None) == NOT_STREAMING_CONTROLLER
    assert streaming.method(24, HRESULT, PTR, PTR)(None, None) == NOT_STREAMING_CONTROLLER


def test_touch_controls_for_all_clients_reach_no_one(streaming):
    streaming.method(17, None)()                                    # HideTouchControls : sans effet
    streaming.method(18, None, ctypes.c_char_p)(b"Driving")
    streaming.method(19, None, U64)(CLIENT)
    streaming.method(20, None, U64, ctypes.c_char_p)(CLIENT, None)
    assert streaming.method(25, HRESULT, SIZE, PTR)(0, None) == S_OK
    assert streaming.method(25, HRESULT, SIZE, PTR)(2, None) == E_INVALIDARG
    assert streaming.method(27, HRESULT, ctypes.c_char_p, SIZE, PTR)(b"Driving", 0, None) == S_OK


def test_stream_resolution_is_not_implemented(streaming):
    assert streaming.method(34, HRESULT, U32, U32)(1280, 720) == E_NOTIMPL
