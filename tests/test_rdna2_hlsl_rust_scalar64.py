"""64-bit scalar HLSL text and WARP/interpreter parity."""
import struct

import pytest

from emu import rdna2_exec, rdna2_gpu, rdna2_hlsl
from test_rdna2_exec import LITERAL, const, program, sop1, sop2
from test_rdna2_hlsl_rust_parity import rust

UNARY = ("s_mov_b64", "s_cmov_b64", "s_not_b64", "s_bcnt1_i32_b64",
         "s_bcnt0_i32_b64", "s_ff1_i32_b64", "s_ff0_i32_b64")
LOGIC = ("and", "or", "xor", "andn2", "orn2", "nand", "nor", "xnor")
BINARY = ("s_cselect_b64", *(f"s_{op}_b64" for op in LOGIC))
EXEC = (*(f"s_{op}_saveexec_b64" for op in (*LOGIC, "andn1", "orn1")),
        "s_andn1_wrexec_b64", "s_andn2_wrexec_b64")


@pytest.mark.parametrize("name", UNARY + BINARY + EXEC)
@pytest.mark.parametrize("a0,a1,b0,b1", [(0, 0, 0, 0), (0xffffffff, 0xffffffff, 0, 1),
                                       (0x12345678, 0x80000000, 0x87654321, 0x7fffffff)])
def test_scalar_wide_matches_python_and_warp(name, a0, a1, b0, b1, monkeypatch):
    if not rdna2_gpu.available():
        pytest.skip("dxc or GPU runner is unavailable")
    monkeypatch.setenv("ANYXBOXX_GPU", "warp")
    op = sop2(name, 4, 0, 2) if name in BINARY else sop1(name, 4, 0)
    data = program(sop1("s_mov_b32", 0, LITERAL), a0, sop1("s_mov_b32", 1, LITERAL), a1,
                   sop1("s_mov_b32", 2, LITERAL), b0, sop1("s_mov_b32", 3, LITERAL), b1,
                   sop1("s_not_b32", 6, const(0)), op)
    result = rust(data)
    assert result.returncode == 0, result.stderr.decode()
    hlsl = result.stdout.decode("utf-8")
    assert hlsl == rdna2_hlsl.translate(data)
    wave = rdna2_exec.run(data)
    hlsl = hlsl.replace("  return;\n", "  if (lane == 0) { g_memory.Store(0, s[4]); "
                        "g_memory.Store(4, s[5]); g_memory.Store(8, s[126]); "
                        "g_memory.Store(12, s[127]); g_memory.Store(16, scc ? 1u : 0u); }\n  return;\n")
    assert rdna2_gpu.run(hlsl, bytes(20)) == struct.pack("<IIIII", wave.s[4], wave.s[5],
                                                        wave.s[126], wave.s[127], int(wave.scc))


@pytest.mark.parametrize("name", ["s_mov_b64", "s_not_b64", "s_and_saveexec_b64"])
def test_scalar_wide_literal_rejected(name):
    result = rust(program(sop1(name, 4, LITERAL), 42))
    assert result.returncode != 0 and name.encode() in result.stderr


@pytest.mark.parametrize("name", ["s_mov_b64", "s_cmov_b64", "s_not_b64", "s_and_b64", "s_cselect_b64"])
@pytest.mark.parametrize("initial_scc", [False, True])
def test_scalar_wide_overlapping_registers(name, initial_scc, monkeypatch):
    if not rdna2_gpu.available():
        pytest.skip("dxc or GPU runner is unavailable")
    monkeypatch.setenv("ANYXBOXX_GPU", "warp")
    op = sop2(name, 0, 0, 2) if name in BINARY else sop1(name, 0, 0)
    init_scc = sop1("s_not_b32", 6, const(0) if initial_scc else 193)
    data = program(sop1("s_mov_b32", 0, LITERAL), 0xdeadbeef,
                   sop1("s_mov_b32", 1, LITERAL), 0x87654321,
                   sop1("s_mov_b32", 2, LITERAL), 0x12345678,
                   sop1("s_mov_b32", 3, LITERAL), 0xabcdef01, init_scc, op)
    result = rust(data)
    assert result.returncode == 0, result.stderr.decode()
    hlsl = result.stdout.decode("utf-8")
    assert hlsl == rdna2_hlsl.translate(data)
    wave = rdna2_exec.run(data)
    hlsl = hlsl.replace("  return;\n", "  if (lane == 0) { g_memory.Store(0, s[0]); "
                        "g_memory.Store(4, s[1]); g_memory.Store(8, scc ? 1u : 0u); }\n  return;\n")
    assert rdna2_gpu.run(hlsl, bytes(12)) == struct.pack("<III", wave.s[0], wave.s[1], int(wave.scc))


@pytest.mark.parametrize("code", [125, 128, 192, 193, 208])
def test_scalar_wide_inline_constant(code, monkeypatch):
    if not rdna2_gpu.available():
        pytest.skip("dxc or GPU runner is unavailable")
    monkeypatch.setenv("ANYXBOXX_GPU", "warp")
    data = program(sop1("s_mov_b64", 0, code))
    result = rust(data)
    assert result.returncode == 0, result.stderr.decode()
    hlsl = result.stdout.decode("utf-8")
    assert hlsl == rdna2_hlsl.translate(data)
    wave = rdna2_exec.run(data)
    hlsl = hlsl.replace("  return;\n", "  if (lane == 0) { g_memory.Store(0, s[0]); "
                        "g_memory.Store(4, s[1]); }\n  return;\n")
    assert rdna2_gpu.run(hlsl, bytes(8)) == struct.pack("<II", wave.s[0], wave.s[1])
