"""Runtime GDK hors ligne : XGameUi sans interface système, écrans confiés aux rappels du jeu.

Signatures : pages XGameUi de Microsoft Learn (XGameUiSetUiCallbacks, XGameUiUiCallbacks,
XGameUiShow*UiCallback, XGameUiSet*UiResponse). Sans rappel du jeu, rien n'est montré et
aucune réponse n'est inventée : E_NOTIMPL.
"""
import ctypes
import threading

import pytest

from test_runtime_cpp import S_OK, THREAD_POOL, AsyncBlock, GdkInterface, TaskQueueApi
from test_runtime_events import E_INVALIDARG, E_NOTIMPL

GAME_UI = ("DFCD4649-4FF8-4043-BA07-35D607DF98B0", "EAF669DF-5542-4590-99A3-8DC061F837CC")
E_ABORT = 0x80004004
FIRST, SECOND, THIRD = 0, 1, 2
PROFILE, PICKER, INVITE, ACHIEVEMENTS, ACTIVITY, MESSAGE, ERROR, TEXT = range(8)
TIMEOUT = 5
P, U32, U64, STR = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64, ctypes.c_char_p
MESSAGE_CALLBACK = ctypes.WINFUNCTYPE(None, P, P, STR, STR, STR, STR, STR, U32, U32, P)
TEXT_CALLBACK = ctypes.WINFUNCTYPE(None, P, P, STR, STR, STR, U32, U32, P)
ERROR_CALLBACK = ctypes.WINFUNCTYPE(None, P, P, ctypes.c_int32, STR, P)


class UiCallbacks(ctypes.Structure):
    _fields_ = [("context", P), ("callbacks", P * 8)]


class PlayerPickerInfo(ctypes.Structure):
    _fields_ = [("user", P), ("prompt", STR), ("from_count", U32), ("from_players", ctypes.POINTER(U64)),
                ("preselected_count", U32), ("preselected", ctypes.POINTER(U64)),
                ("min_selection", U32), ("max_selection", U32)]


PICKER_CALLBACK = ctypes.WINFUNCTYPE(None, P, P, ctypes.POINTER(PlayerPickerInfo), P)


class GameUi:
    """XGameUi et une file du pool de threads, rappels du jeu retirés à la fin."""

    def __init__(self, runtime_dir):
        self.api, self.tasks = GdkInterface(runtime_dir, *GAME_UI), TaskQueueApi(runtime_dir)
        self.queue = ctypes.c_void_p()
        create = self.tasks.method(12, U32, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p))
        assert create(THREAD_POOL, THREAD_POOL, ctypes.byref(self.queue)) == S_OK
        self.seen, self.called, self.keep = [], threading.Event(), []

    def set_callbacks(self, context=None, **by_index):
        table = UiCallbacks(context=context)
        for index, callback in by_index.items():
            self.keep.append(callback)
            table.callbacks[int(index[1:])] = ctypes.cast(callback, P)
        assert self.api.method(42, U32, ctypes.POINTER(UiCallbacks), ctypes.c_bool)(ctypes.byref(table), True) == S_OK

    def block(self):
        return AsyncBlock(queue=self.queue)

    def wait(self, block):
        return self.tasks.method(3, U32, ctypes.POINTER(AsyncBlock), ctypes.c_bool)(ctypes.byref(block), True)

    def close(self):
        self.api.method(42, U32, P, ctypes.c_bool)(None, False)
        self.tasks.method(17, None, ctypes.c_void_p)(self.queue)


@pytest.fixture
def game_ui(runtime_dir):
    ui = GameUi(runtime_dir)
    yield ui
    ui.close()


def show_message(ui, block):
    show = ui.api.method(3, U32, ctypes.POINTER(AsyncBlock), STR, STR, STR, STR, STR, U32, U32)
    return show(ctypes.byref(block), b"Save", b"Overwrite?", b"Yes", b"No", None, FIRST, SECOND)


def test_message_dialog_is_answered_by_the_game_ui(game_ui):
    def on_dialog(handle, queue, *texts_and_context):
        game_ui.seen.append((handle, queue, texts_and_context))
        game_ui.called.set()
    game_ui.set_callbacks(context=0x77, _5=MESSAGE_CALLBACK(on_dialog))
    block = game_ui.block()
    assert show_message(game_ui, block) == S_OK
    assert game_ui.called.wait(TIMEOUT)
    handle, queue, arguments = game_ui.seen[0]
    assert queue == game_ui.queue.value
    assert arguments == (b"Save", b"Overwrite?", b"Yes", b"No", None, FIRST, SECOND, 0x77)
    respond = game_ui.api.method(43, U32, P, U32)
    assert respond(handle, THIRD + 1) == E_INVALIDARG                  # bouton inexistant
    assert respond(handle, SECOND) == S_OK
    assert respond(handle, FIRST) == E_INVALIDARG                      # déjà répondu
    assert game_ui.wait(block) == S_OK
    button = U32(9)
    assert game_ui.api.method(4, U32, ctypes.POINTER(AsyncBlock), ctypes.POINTER(U32))(
        ctypes.byref(block), ctypes.byref(button)) == S_OK
    assert button.value == SECOND


def test_text_entry_returns_the_text_typed_in_the_game_ui(game_ui):
    respond = game_ui.api.method(45, U32, P, STR)
    answers = []
    game_ui.set_callbacks(_7=TEXT_CALLBACK(lambda handle, queue, title, description, initial, scope, length, context:
                                           answers.append(respond(handle, b"AnyXboxX"))))
    block = game_ui.block()
    show = game_ui.api.method(17, U32, ctypes.POINTER(AsyncBlock), STR, STR, STR, U32, U32)
    assert show(ctypes.byref(block), b"Name", b"Your name", b"", 0, 16) == S_OK
    assert game_ui.wait(block) == S_OK and answers == [S_OK]
    size = U32()
    assert game_ui.api.method(18, U32, ctypes.POINTER(AsyncBlock), ctypes.POINTER(U32))(
        ctypes.byref(block), ctypes.byref(size)) == S_OK
    assert size.value == len(b"AnyXboxX") + 1                          # zéro final compris
    text, used = ctypes.create_string_buffer(size.value), U32()
    assert game_ui.api.method(19, U32, ctypes.POINTER(AsyncBlock), U32, ctypes.c_char_p, ctypes.POINTER(U32))(
        ctypes.byref(block), size.value, text, ctypes.byref(used)) == S_OK
    assert text.value == b"AnyXboxX" and used.value == size.value


def test_player_picker_returns_the_chosen_players(game_ui):
    def on_picker(handle, queue, info, context):
        game_ui.seen.append((handle, info.contents.prompt, info.contents.from_count,
                             [info.contents.from_players[i] for i in range(info.contents.from_count)],
                             info.contents.max_selection))
        game_ui.called.set()
    game_ui.set_callbacks(_1=PICKER_CALLBACK(on_picker))
    players = (U64 * 3)(11, 22, 33)
    block = game_ui.block()
    show = game_ui.api.method(11, U32, ctypes.POINTER(AsyncBlock), P, STR, U32, P, U32, P, U32, U32)
    assert show(ctypes.byref(block), None, b"Pick", 3, players, 0, None, 1, 2) == S_OK
    assert game_ui.called.wait(TIMEOUT)
    handle, prompt, count, offered, most = game_ui.seen[0]
    assert (prompt, count, offered, most) == (b"Pick", 3, [11, 22, 33], 2)
    chosen = (U64 * 2)(33, 11)
    assert game_ui.api.method(44, U32, P, U32, P)(handle, 2, chosen) == S_OK
    assert game_ui.wait(block) == S_OK
    count = U32()
    assert game_ui.api.method(12, U32, ctypes.POINTER(AsyncBlock), ctypes.POINTER(U32))(
        ctypes.byref(block), ctypes.byref(count)) == S_OK and count.value == 2
    result, used = (U64 * 2)(), U32()
    assert game_ui.api.method(13, U32, ctypes.POINTER(AsyncBlock), U32, P, ctypes.POINTER(U32))(
        ctypes.byref(block), 2, result, ctypes.byref(used)) == S_OK
    assert list(result) == [33, 11] and used.value == 2


def test_error_dialog_completes_when_the_game_closes_it(game_ui):
    def on_error(handle, queue, error, service_context, context):
        game_ui.seen.append((handle, error, service_context))
        game_ui.called.set()
    game_ui.set_callbacks(_6=ERROR_CALLBACK(on_error))
    block = game_ui.block()
    show = game_ui.api.method(14, U32, ctypes.POINTER(AsyncBlock), ctypes.c_int32, STR)
    assert show(ctypes.byref(block), ctypes.c_int32(E_ABORT - (1 << 32)).value, b"service") == S_OK
    assert game_ui.called.wait(TIMEOUT)
    handle, error, service_context = game_ui.seen[0]
    assert (error & 0xFFFFFFFF, service_context) == (E_ABORT, b"service")
    assert game_ui.api.method(43, U32, P, U32)(handle, FIRST) == E_INVALIDARG    # autre écran
    assert game_ui.api.method(50, U32, P)(handle) == S_OK
    assert game_ui.wait(block) == S_OK
    assert game_ui.api.method(15, U32, ctypes.POINTER(AsyncBlock))(ctypes.byref(block)) == S_OK


def test_cancelled_request_refuses_a_late_answer(game_ui):
    def on_dialog(handle, *rest):
        game_ui.seen.append(handle)
        game_ui.called.set()
    game_ui.set_callbacks(_5=MESSAGE_CALLBACK(on_dialog))
    block = game_ui.block()
    assert show_message(game_ui, block) == S_OK
    assert game_ui.called.wait(TIMEOUT)
    game_ui.tasks.method(5, None, ctypes.POINTER(AsyncBlock))(ctypes.byref(block))   # XAsyncCancel
    assert game_ui.wait(block) == E_ABORT
    assert game_ui.api.method(43, U32, P, U32)(game_ui.seen[0], FIRST) == E_INVALIDARG


def test_without_a_game_callback_nothing_is_shown(game_ui):
    game_ui.set_callbacks(_7=TEXT_CALLBACK(lambda *arguments: None))   # seul l'écran de texte est fourni
    assert show_message(game_ui, game_ui.block()) == E_NOTIMPL
    error = game_ui.api.method(14, U32, ctypes.POINTER(AsyncBlock), ctypes.c_int32, STR)
    assert error(ctypes.byref(game_ui.block()), 0, None) == E_NOTIMPL
    achievements = game_ui.api.method(9, U32, ctypes.POINTER(AsyncBlock), P, U32)
    assert achievements(ctypes.byref(game_ui.block()), None, 0x1234) == E_NOTIMPL
    assert game_ui.api.method(43, U32, P, U32)(ctypes.c_void_p(0x1000), FIRST) == E_INVALIDARG   # poignée inconnue


def test_notification_hint_is_accepted_and_other_ui_is_not_implemented(game_ui):
    assert game_ui.api.method(16, U32, U32)(5) == S_OK                 # TopRight : sans effet sur PC
    assert game_ui.api.method(34, U32, P, U32, STR, U32, P)(None, 16, None, 0, None) == E_NOTIMPL  # TextEntryOpen
    assert game_ui.api.method(24, U32, P)(None) == E_NOTIMPL           # ShowWebAuthenticationAsync
