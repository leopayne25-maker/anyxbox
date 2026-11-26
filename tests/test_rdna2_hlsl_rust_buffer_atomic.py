"""Untyped MUBUF atomics: HLSL text and WARP parity."""
import struct

import pytest

from emu import rdna2_exec, rdna2_gpu, rdna2_hlsl
from test_rdna2_exec import LITERAL, const, program, sop1, vop1, vop2
from test_rdna2_exec_memory import mubuf
from test_rdna2_hlsl_rust_parity import rust

KINDS = ("add", "swap", "and", "or", "xor", "umin", "umax")


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("records,stride,glc,mask", [(32, 0, False, 0xffffffff), (32, 0, True, 0x55555555),
                                                     (2, 16, True, 0xaaaaaaaa), (0, 16, True, 0xffffffff)])
def test_buffer_atomic_text_and_warp(kind, records, stride, glc, mask, monkeypatch):
    if not rdna2_gpu.available():
        pytest.skip("dxc or GPU runner is unavailable")
    monkeypatch.setenv("ANYXBOXX_GPU", "warp")
    low, high = mubuf(f"buffer_atomic_{kind}", 1, 4, 20, offen=1, offset=4, soffset=24)
    data = program(sop1("s_mov_b32", 20, LITERAL), 256, sop1("s_mov_b32", 21, LITERAL), stride << 16,
                   sop1("s_mov_b32", 22, LITERAL), records, sop1("s_mov_b32", 24, LITERAL), 8,
                   vop2("v_lshlrev_b32", 1, const(2), 0), vop1("v_mov_b32", 4, LITERAL), 0x12345678,
                   sop1("s_mov_b32", 126, LITERAL), mask, low | (int(glc) << 14), high)
    result = rust(data)
    assert result.returncode == 0, result.stderr.decode()
    hlsl = result.stdout.decode("utf-8")
    assert hlsl == rdna2_hlsl.translate(data)
    initial = bytearray(1024)
    struct.pack_into("<32I", initial, 268, *range(32))
    wave = rdna2_exec.Wave()
    wave.memory.write(0, bytes(initial))
    rdna2_exec.run(data, wave)
    hlsl = hlsl.replace("  return;\n", "  g_memory.Store(512u + lane * 4u, v[4]);\n  return;\n")
    expected = bytearray(wave.memory.read(0, len(initial)))
    struct.pack_into("<32I", expected, 512, *wave.v[4])
    assert rdna2_gpu.run(hlsl, bytes(initial)) == bytes(expected)


@pytest.mark.parametrize("name", ["buffer_atomic_sub", "buffer_atomic_smin", "buffer_atomic_cmpswap",
                                   "buffer_atomic_add_x2"])
def test_unsupported_buffer_atomic_is_named(name):
    result = rust(program(*mubuf(name, 1, 4, 20, offen=1)))
    assert result.returncode != 0 and name.encode() in result.stderr


def test_atomic_raw_offset_overflow_does_not_write(monkeypatch):
    if not rdna2_gpu.available():
        pytest.skip("dxc or GPU runner is unavailable")
    monkeypatch.setenv("ANYXBOXX_GPU", "warp")
    low, high = mubuf("buffer_atomic_add", 1, 4, 20, offen=1, offset=32)
    data = program(sop1("s_mov_b32", 20, LITERAL), 256, sop1("s_mov_b32", 22, LITERAL), 32,
                   vop1("v_mov_b32", 1, LITERAL), 0xfffffff0, vop1("v_mov_b32", 4, LITERAL), 7,
                   low | (1 << 14), high)
    result = rust(data)
    assert result.returncode == 0, result.stderr.decode()
    hlsl = result.stdout.decode("utf-8")
    assert hlsl == rdna2_hlsl.translate(data)
    initial = bytes(1024)
    wave = rdna2_exec.Wave()
    wave.memory.write(0, initial)
    rdna2_exec.run(data, wave)
    hlsl = hlsl.replace("  return;\n", "  g_memory.Store(512u + lane * 4u, v[4]);\n  return;\n")
    expected = bytearray(wave.memory.read(0, len(initial)))
    struct.pack_into("<32I", expected, 512, *wave.v[4])
    assert rdna2_gpu.run(hlsl, initial) == bytes(expected)
