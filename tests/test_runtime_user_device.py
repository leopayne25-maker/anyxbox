"""Runtime GDK : XUser et périphériques (point de terminaison audio de Windows, manette sur PC).

Signatures et comportement sur PC : pages XUser de Microsoft Learn
(XUserGetDefaultAudioEndpointUtf16, XUserFindControllerForUserWithUiAsync/Result).
L'oracle audio est Windows lui-même : IMMDeviceEnumerator interrogé ici par COM.
"""
import ctypes
import uuid

from test_runtime_cpp import (E_GAMEUSER_USER_NOT_FOUND, S_OK, THREAD_POOL, AsyncBlock, GdkInterface,
                              TaskQueueApi, add_local_user)
from test_runtime_events import E_INVALIDARG, E_NOTIMPL, EVENT_CALLBACK

USER_DEVICE = ("7D824997-10DC-45AB-86B7-2737767C0BF1", "0CC6A956-E7E1-4FDF-9341-9D5DA649EBC8")
LOCAL_ID, RENDER, CAPTURE = 1, 0, 1
UNTOUCHED = 0x5EED_5EED_5EED_5EED
E_NOTFOUND, E_INSUFFICIENT_BUFFER = 0x80070490, 0x8007007A
CLSID_ENUMERATOR, IID_ENUMERATOR = "BCDE0395-E52F-467C-8E3D-C4579291692E", "A95664D2-9614-4F35-A746-DE8DB63617E6"
E_RENDER, E_CAPTURE, E_COMMUNICATIONS = 0, 1, 2
CLSCTX_ALL, COINIT_MULTITHREADED = 0x17, 0


def _call(pointer, index, *argtypes):
    vtable = ctypes.cast(ctypes.cast(pointer, ctypes.POINTER(ctypes.c_void_p))[0], ctypes.POINTER(ctypes.c_void_p))
    return ctypes.WINFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, *argtypes)(vtable[index])


def windows_default_endpoint(flow):
    """Identifiant du périphérique de communication par défaut, ou None (oracle Windows)."""
    ole32 = ctypes.oledll.ole32
    ctypes.windll.ole32.CoInitializeEx(None, COINIT_MULTITHREADED)
    enumerator, device, text = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_wchar_p()
    clsid = (ctypes.c_ubyte * 16).from_buffer_copy(uuid.UUID(CLSID_ENUMERATOR).bytes_le)
    iid = (ctypes.c_ubyte * 16).from_buffer_copy(uuid.UUID(IID_ENUMERATOR).bytes_le)
    ole32.CoCreateInstance(clsid, None, CLSCTX_ALL, iid, ctypes.byref(enumerator))
    try:
        get_default = _call(enumerator, 4, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p))
        if get_default(enumerator, flow, E_COMMUNICATIONS, ctypes.byref(device)) != S_OK:
            return None
        _call(device, 5, ctypes.POINTER(ctypes.c_wchar_p))(device, ctypes.byref(text))   # IMMDevice::GetId
        _call(device, 2)(device)
        return text.value
    finally:
        _call(enumerator, 2)(enumerator)


def endpoint(runtime_dir, local_id, kind, count=256):
    get = GdkInterface(runtime_dir, *USER_DEVICE).method(
        6, ctypes.c_uint32, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_size_t, ctypes.c_wchar_p,
        ctypes.POINTER(ctypes.c_size_t))
    buffer, used = ctypes.create_unicode_buffer(count), ctypes.c_size_t(0)
    return get(local_id, kind, count, buffer, ctypes.byref(used)), buffer.value, used.value


def test_default_audio_endpoint_is_the_windows_communications_device(runtime_dir):
    for kind, flow in ((RENDER, E_RENDER), (CAPTURE, E_CAPTURE)):
        expected = windows_default_endpoint(flow)
        result, text, used = endpoint(runtime_dir, LOCAL_ID, kind)
        if expected is None:
            assert result == E_NOTFOUND
            continue
        assert result == S_OK and text == expected and used == len(expected) + 1
        short = endpoint(runtime_dir, LOCAL_ID, kind, count=4)
        assert short[0] == E_INSUFFICIENT_BUFFER and short[2] == len(expected) + 1


def test_default_audio_endpoint_checks_user_and_kind(runtime_dir):
    assert endpoint(runtime_dir, LOCAL_ID + 1, RENDER)[0] == E_GAMEUSER_USER_NOT_FOUND
    assert endpoint(runtime_dir, LOCAL_ID, 2)[0] == E_INVALIDARG


def test_controller_search_on_pc_returns_an_all_zero_device(runtime_dir):
    _, _, _, user = add_local_user(runtime_dir)
    api, tasks = GdkInterface(runtime_dir, *USER_DEVICE), TaskQueueApi(runtime_dir)
    queue = ctypes.c_void_p()
    assert tasks.method(12, ctypes.c_uint32, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p))(
        THREAD_POOL, THREAD_POOL, ctypes.byref(queue)) == S_OK
    block = AsyncBlock(queue=queue)
    find = api.method(9, ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(AsyncBlock))
    assert find(None, ctypes.byref(AsyncBlock(queue=queue))) == E_INVALIDARG
    assert find(user, ctypes.byref(block)) == S_OK
    assert tasks.method(3, ctypes.c_uint32, ctypes.POINTER(AsyncBlock), ctypes.c_bool)(ctypes.byref(block), True) == S_OK
    device = (ctypes.c_ubyte * 32)(*([0xFF] * 32))
    assert api.method(10, ctypes.c_uint32, ctypes.POINTER(AsyncBlock), ctypes.c_void_p)(
        ctypes.byref(block), device) == S_OK
    assert bytes(device) == bytes(32)
    tasks.method(17, None, ctypes.c_void_p)(queue)


def test_association_and_endpoint_subscriptions_are_not_implemented(runtime_dir):
    """Règle 3 : AnyXboxX ne suit pas ces changements, l'abonnement ne réussit pas (pas de jeton écrit)."""
    api = GdkInterface(runtime_dir, *USER_DEVICE)
    for register, unregister in ((4, 5), (7, 8)):
        subscribe = api.method(register, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                               ctypes.POINTER(ctypes.c_uint64))
        unsubscribe = api.method(unregister, ctypes.c_bool, ctypes.c_uint64, ctypes.c_bool)
        callback, token = EVENT_CALLBACK(lambda context, value: None), ctypes.c_uint64(UNTOUCHED)
        assert subscribe(None, None, ctypes.cast(callback, ctypes.c_void_p), ctypes.byref(token)) == E_NOTIMPL
        assert token.value == UNTOUCHED and unsubscribe(token.value, True) is False


def test_finding_the_user_of_a_device_is_not_implemented(runtime_dir):
    find = GdkInterface(runtime_dir, *USER_DEVICE).method(3, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p)
    assert find(None, None) == E_NOTIMPL
