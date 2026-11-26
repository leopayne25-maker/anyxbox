"""LDS 32-bit integer atomics: HLSL text and WARP parity."""
import struct

import pytest

from emu import rdna2_exec, rdna2_gpu, rdna2_hlsl
from test_rdna2_exec import LITERAL, const, program, sop1, vop1, vop2
from test_rdna2_exec_memory import ds
from test_rdna2_hlsl_rust_parity import rust

OPS = tuple(f"ds_{kind}{'_rtn' if returns else ''}_{typ}" for kind, typ in
            (("add", "u32"), ("min", "u32"), ("max", "u32"), ("and", "b32"), ("or", "b32"), ("xor", "b32"))
            for returns in (False, True))


@pytest.mark.parametrize("name", OPS)
@pytest.mark.parametrize("old,value,mask", [(0, 1, 0xffffffff), (0xffffffff, 1, 0x55555555),
                                            (0x12345678, 0xf0f0f0f0, 0xaaaaaaaa)])
def test_lds_atomic_text_and_warp(name, old, value, mask, monkeypatch):
    if not rdna2_gpu.available():
        pytest.skip("dxc or GPU runner is unavailable")
    monkeypatch.setenv("ANYXBOXX_GPU", "warp")
    data = program(vop2("v_lshlrev_b32", 1, const(2), 0), vop1("v_mov_b32", 2, LITERAL), old,
                   vop1("v_mov_b32", 3, LITERAL), value, vop1("v_mov_b32", 4, LITERAL), 0xabcdef01,
                   *ds("ds_write_b32", 1, data0=2), sop1("s_mov_b32", 126, LITERAL), mask,
                   *ds(name, 1, data0=3, vdst=4), *ds("ds_read_b32", 1, vdst=5))
    result = rust(data)
    assert result.returncode == 0, result.stderr.decode()
    hlsl = result.stdout.decode("utf-8")
    assert hlsl == rdna2_hlsl.translate(data)
    wave = rdna2_exec.run(data)
    hlsl = hlsl.replace("  return;\n", "  g_memory.Store(lane * 4u, v[4]); "
                        "g_memory.Store(128u + lane * 4u, v[5]);\n  return;\n")
    expected = struct.pack("<64I", *wave.v[4], *wave.v[5])
    assert rdna2_gpu.run(hlsl, bytes(256)) == expected


@pytest.mark.parametrize("name", ["ds_sub_u32", "ds_min_i32", "ds_add_f32", "ds_add_u64"])
def test_unimplemented_lds_atomics_are_named(name):
    result = rust(program(*ds(name, 1, data0=3, vdst=4)))
    assert result.returncode != 0 and name.encode() in result.stderr
