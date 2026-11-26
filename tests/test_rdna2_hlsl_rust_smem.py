"""SMEM load/store HLSL text and WARP/interpreter parity."""
import struct

import pytest

from emu import rdna2_exec, rdna2_gpu, rdna2_hlsl
from test_rdna2_exec import LITERAL, program, sop1
from test_rdna2_exec_memory import smem
from test_rdna2_hlsl_rust_parity import rust

LOADS = tuple(f"s_{prefix}load_{width}" for prefix in ("", "buffer_")
              for width in ("dword", "dwordx2", "dwordx4", "dwordx8", "dwordx16"))
STORES = tuple(f"s_{prefix}store_{width}" for prefix in ("", "buffer_")
               for width in ("dword", "dwordx2", "dwordx4"))


def init():
    words = []
    for reg, value in [(4, 256), (5, 0), (6, 32), (7, 0), (30, 8),
                       *((8 + i, 0xa0000000 + i) for i in range(16))]:
        words.extend((sop1("s_mov_b32", reg, LITERAL), value))
    return words


@pytest.mark.parametrize("name", LOADS + STORES)
@pytest.mark.parametrize("offset,soffset", [(0, 125), (28, 125), (-4, 30)])
def test_smem_text_and_warp(name, offset, soffset, monkeypatch):
    if not rdna2_gpu.available():
        pytest.skip("dxc or GPU runner is unavailable")
    monkeypatch.setenv("ANYXBOXX_GPU", "warp")
    data = program(*init(), *smem(name, 8, 4, offset, soffset))
    result = rust(data)
    assert result.returncode == 0, result.stderr.decode()
    hlsl = result.stdout.decode("utf-8")
    assert hlsl == rdna2_hlsl.translate(data)
    initial = bytearray(512)
    struct.pack_into("<32I", initial, 256, *(0x100 + i for i in range(32)))
    wave = rdna2_exec.Wave()
    wave.memory.write(0, bytes(initial))
    rdna2_exec.run(data, wave)
    # Capture all potentially loaded SGPRs after the instruction.
    stores = " ".join(f"g_memory.Store({i * 4}u, s[{8 + i}]);" for i in range(16))
    hlsl = hlsl.replace("  return;\n", f"  if (lane == 0) {{ {stores} }}\n  return;\n")
    expected = bytearray(wave.memory.read(0, len(initial)))
    struct.pack_into("<16I", expected, 0, *wave.s[8:24])
    assert rdna2_gpu.run(hlsl, bytes(initial)) == bytes(expected)


def test_smem_atomic_remains_named():
    result = rust(program(*smem("s_atomic_add_x2", 8, 4)))
    assert result.returncode != 0 and b"s_atomic_add_x2" in result.stderr


@pytest.mark.parametrize("name,dest,offset", [("s_load_dwordx2", 4, 0),
                                               ("s_buffer_load_dwordx4", 6, 0),
                                               ("s_buffer_load_dwordx2", 8, -4)])
def test_unsafe_smem_load_is_named(name, dest, offset):
    result = rust(program(*smem(name, dest, 4, offset)))
    assert result.returncode != 0 and name.encode() in result.stderr
