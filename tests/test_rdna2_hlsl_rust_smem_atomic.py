"""Scalar memory atomic HLSL text and WARP parity."""
import struct

import pytest

from emu import rdna2_exec, rdna2_gpu, rdna2_hlsl
from test_rdna2_exec import LITERAL, program, sop1
from test_rdna2_exec_memory import OPCODES, smem
from test_rdna2_hlsl_rust_parity import rust

KINDS = ("add", "sub", "swap", "and", "or", "xor", "umin", "umax", "smin", "smax", "cmpswap", "inc", "dec")


@pytest.mark.parametrize("kind,buffer", [(kind, buffer) for kind in KINDS for buffer in (False, True)
                                         if ("SMEM", f"s_{'buffer_' if buffer else ''}atomic_{kind}") in OPCODES])
@pytest.mark.parametrize("glc,old", [(False, 0), (True, 5), (True, 0xffffffff)])
def test_scalar_atomics(kind, buffer, glc, old, monkeypatch):
    if not rdna2_gpu.available():
        pytest.skip("dxc or GPU runner is unavailable")
    monkeypatch.setenv("ANYXBOXX_GPU", "warp")
    name = f"s_{'buffer_' if buffer else ''}atomic_{kind}"
    low, high = smem(name, 8, 4, 4)
    data = program(sop1("s_mov_b32", 4, LITERAL), 256, sop1("s_mov_b32", 5, LITERAL), 0,
                   sop1("s_mov_b32", 6, LITERAL), 32, sop1("s_mov_b32", 8, LITERAL), 7,
                   sop1("s_mov_b32", 9, LITERAL), 5, low | (int(glc) << 16), high)
    result = rust(data)
    assert result.returncode == 0, result.stderr.decode()
    hlsl = result.stdout.decode("utf-8")
    assert hlsl == rdna2_hlsl.translate(data)
    initial = bytearray(512)
    struct.pack_into("<I", initial, 260, old)
    wave = rdna2_exec.Wave()
    wave.memory.write(0, bytes(initial))
    rdna2_exec.run(data, wave)
    hlsl = hlsl.replace("  return;\n", "  if (lane == 0) g_memory.Store(0u, s[8]);\n  return;\n")
    expected = bytearray(wave.memory.read(0, len(initial)))
    struct.pack_into("<I", expected, 0, wave.s[8])
    assert rdna2_gpu.run(hlsl, bytes(initial)) == bytes(expected)


def test_atomic_64_is_named():
    result = rust(program(*smem("s_atomic_add_x2", 8, 4)))
    assert result.returncode != 0 and b"s_atomic_add_x2" in result.stderr


@pytest.mark.parametrize("offset", [-4, 32])
def test_buffer_atomic_outside_bounds(offset, monkeypatch):
    if not rdna2_gpu.available():
        pytest.skip("dxc or GPU runner is unavailable")
    monkeypatch.setenv("ANYXBOXX_GPU", "warp")
    low, high = smem("s_buffer_atomic_add", 8, 4, offset)
    data = program(sop1("s_mov_b32", 4, LITERAL), 256, sop1("s_mov_b32", 6, LITERAL), 32,
                   sop1("s_mov_b32", 8, LITERAL), 7, low | (1 << 16), high)
    result = rust(data)
    assert result.returncode == 0, result.stderr.decode()
    hlsl = result.stdout.decode("utf-8")
    assert hlsl == rdna2_hlsl.translate(data)
    initial = bytes(512)
    wave = rdna2_exec.Wave()
    wave.memory.write(0, initial)
    rdna2_exec.run(data, wave)
    hlsl = hlsl.replace("  return;\n", "  if (lane == 0) g_memory.Store(0u, s[8]);\n  return;\n")
    expected = bytearray(wave.memory.read(0, len(initial)))
    struct.pack_into("<I", expected, 0, wave.s[8])
    assert rdna2_gpu.run(hlsl, initial) == bytes(expected)
