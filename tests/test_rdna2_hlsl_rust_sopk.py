"""SOPK HLSL text and GPU/interpreter state parity."""
import struct

import pytest

from emu import rdna2_exec, rdna2_gpu, rdna2_hlsl
from test_rdna2_alu_parity import sopk
from test_rdna2_exec import LITERAL, const, program, sop1
from test_rdna2_hlsl_rust_parity import rust

CMP = tuple(f"s_cmpk_{op}_{kind}" for op in ("eq", "lg", "gt", "ge", "lt", "le")
            for kind in ("i32", "u32"))
OPS = ("s_movk_i32", "s_cmovk_i32", "s_addk_i32", "s_mulk_i32", *CMP)
WAITS = ("s_version", "s_waitcnt_vscnt", "s_waitcnt_vmcnt", "s_waitcnt_expcnt", "s_waitcnt_lgkmcnt")


@pytest.mark.parametrize("name", OPS + WAITS)
@pytest.mark.parametrize("value,imm,initial_scc", [(0, 0, False), (0x7fffffff, 1, True),
                                                    (0x80000000, 0xffff, False),
                                                    (0xffffffff, 0x8000, True)])
def test_sopk_matches_python_and_warp(name, value, imm, initial_scc, monkeypatch):
    if not rdna2_gpu.available():
        pytest.skip("dxc or GPU runner is unavailable")
    monkeypatch.setenv("ANYXBOXX_GPU", "warp")
    # s_not of zero sets SCC; s_not of -1 clears it.
    init_scc = sop1("s_not_b32", 2, const(0) if initial_scc else 193)
    data = program(sop1("s_mov_b32", 1, LITERAL), value, init_scc, sopk(name, 1, imm))
    result = rust(data)
    assert result.returncode == 0, result.stderr.decode()
    hlsl = result.stdout.decode("utf-8")
    assert hlsl == rdna2_hlsl.translate(data)
    wave = rdna2_exec.run(data)
    hlsl = hlsl.replace("  return;\n", "  if (lane == 0) { g_memory.Store(0, s[1]); "
                        "g_memory.Store(4, scc ? 1u : 0u); }\n  return;\n")
    assert rdna2_gpu.run(hlsl, bytes(8)) == struct.pack("<II", wave.s[1], int(wave.scc))


@pytest.mark.parametrize("name", ["s_getreg_b32", "s_setreg_b32", "s_call_b64"])
def test_unimplemented_sopk_is_named(name):
    result = rust(program(sopk(name, 0, 0)))
    assert result.returncode != 0 and name.encode() in result.stderr
