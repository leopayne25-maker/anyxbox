"""VOP2 floating-point HLSL text and WARP parity."""
import struct

import pytest

from emu import rdna2_exec, rdna2_gpu, rdna2_hlsl
from test_rdna2_exec import LITERAL, V, program, sop1, vop1, vop2
from test_rdna2_hlsl_rust_parity import rust

OPS = ("v_mul_legacy_f32", "v_fmac_f32", "v_fmac_legacy_f32", "v_mac_f32")


@pytest.mark.parametrize("name", OPS)
@pytest.mark.parametrize("a,b,c,mask", [(0x3fc00000, 0x40000000, 0xbf800000, 0xffffffff),
                                         (0, 0x7f800000, 0x3f800000, 0x55555555),
                                         (0x80000000, 0, 0, 0xaaaaaaaa),
                                         (0x7fc01234, 0x3f800000, 0, 0xffffffff),
                                         (1, 0x3f000000, 0x3f800000, 0xffffffff),
                                         (0x3f800000, 0x7fc01234, 0, 0xffffffff),
                                         (0x7f800000, 0xff800000, 0, 0xffffffff),
                                         (0x7f800000, 0x7f800000, 0, 0xffffffff)])
def test_vop2_float(name, a, b, c, mask, monkeypatch):
    if not rdna2_gpu.available():
        pytest.skip("dxc or GPU runner is unavailable")
    monkeypatch.setenv("ANYXBOXX_GPU", "warp")
    data = program(vop1("v_mov_b32", 1, LITERAL), a, vop1("v_mov_b32", 2, LITERAL), b,
                   vop1("v_mov_b32", 3, LITERAL), c, sop1("s_mov_b32", 126, LITERAL), mask,
                   vop2(name, 3, V + 1, 2))
    result = rust(data)
    assert result.returncode == 0, result.stderr.decode()
    hlsl = result.stdout.decode("utf-8")
    assert hlsl == rdna2_hlsl.translate(data)
    wave = rdna2_exec.run(data)
    hlsl = hlsl.replace("  return;\n", "  g_memory.Store(lane * 4u, v[3]);\n  return;\n")
    assert rdna2_gpu.run(hlsl, bytes(128)) == struct.pack("<32I", *wave.v[3])


@pytest.mark.parametrize("name", ["v_add_f32", "v_sub_f32", "v_subrev_f32", "v_mul_f32", "v_min_f32", "v_max_f32"])
def test_unverified_nan_parity_is_rejected(name):
    result = rust(program(vop2(name, 1, V, 0)))
    assert result.returncode != 0 and name.encode() in result.stderr
