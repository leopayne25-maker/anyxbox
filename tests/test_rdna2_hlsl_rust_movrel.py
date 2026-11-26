"""Indexed scalar register moves: text parity and WARP against the interpreter."""
import struct

import pytest

from emu import rdna2_exec, rdna2_gpu, rdna2_hlsl
from test_rdna2_exec import LITERAL, program, sop1
from test_rdna2_hlsl_rust_parity import rust

MOVES = ("s_movrels_b32", "s_movrels_b64", "s_movreld_b32", "s_movreld_b64", "s_movrelsd_2_b32")


@pytest.mark.parametrize("name", MOVES)
@pytest.mark.parametrize("offset", [0, 2])
def test_indexed_scalar_move(name, offset, monkeypatch):
    if not rdna2_gpu.available():
        pytest.skip("dxc or GPU runner is unavailable")
    monkeypatch.setenv("ANYXBOXX_GPU", "warp")
    m0 = (offset << 16 | offset) if name.endswith("_2_b32") else offset
    data = program(sop1("s_mov_b32", 10, LITERAL), 0x10203040,
                   sop1("s_mov_b32", 11, LITERAL), 0x50607080,
                   sop1("s_mov_b32", 12, LITERAL), 0x90a0b0c0,
                   sop1("s_mov_b32", 13, LITERAL), 0xd0e0f000,
                   sop1("s_mov_b32", 124, LITERAL), m0, sop1(name, 20, 10))
    result = rust(data)
    assert result.returncode == 0, result.stderr.decode()
    hlsl = result.stdout.decode("utf-8")
    assert hlsl == rdna2_hlsl.translate(data)
    wave = rdna2_exec.run(data)
    stores = " ".join(f"g_memory.Store({4 * i}, s[{20 + i}]);" for i in range(4))
    hlsl = hlsl.replace("  return;\n", f"  if (lane == 0) {{ {stores} }}\n  return;\n")
    assert rdna2_gpu.run(hlsl, bytes(16)) == struct.pack("<4I", *wave.s[20:24])


@pytest.mark.parametrize("name", ["s_movrels_b32", "s_movrels_b64"])
def test_null_destination_emits_no_statement(name):
    data = program(sop1(name, 125, 0))
    result = rust(data)
    assert result.returncode == 0, result.stderr.decode()
    assert result.stdout == rdna2_hlsl.translate(data).encode("utf-8")


@pytest.mark.parametrize("name,dst,src", [("s_movrels_b32", 0, 126), ("s_movrelsd_2_b32", 126, 0),
                                           ("s_movreld_b64", 0, LITERAL)])
def test_invalid_index_operand_is_named(name, dst, src):
    words = (sop1(name, dst, src), 0) if src == LITERAL else (sop1(name, dst, src),)
    result = rust(program(*words))
    assert result.returncode != 0 and name.encode() in result.stderr


def test_split_offsets_and_literal_source(monkeypatch):
    if not rdna2_gpu.available():
        pytest.skip("dxc or GPU runner is unavailable")
    monkeypatch.setenv("ANYXBOXX_GPU", "warp")
    data = program(sop1("s_mov_b32", 11, LITERAL), 0x12345678,
                   sop1("s_mov_b32", 124, LITERAL), 1 | 4 << 16,
                   sop1("s_movrelsd_2_b32", 20, 10),
                   sop1("s_mov_b32", 124, LITERAL), 2,
                   sop1("s_movreld_b32", 20, LITERAL), 0xabcdef01)
    result = rust(data)
    assert result.returncode == 0, result.stderr.decode()
    hlsl = result.stdout.decode("utf-8")
    assert hlsl == rdna2_hlsl.translate(data)
    wave = rdna2_exec.run(data)
    hlsl = hlsl.replace("  return;\n", "  if (lane == 0) { g_memory.Store(0, s[22]); "
                        "g_memory.Store(4, s[24]); }\n  return;\n")
    assert rdna2_gpu.run(hlsl, bytes(8)) == struct.pack("<II", wave.s[22], wave.s[24])
