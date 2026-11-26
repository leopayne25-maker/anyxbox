"""VOP2 half arithmetic: text and per-lane WARP parity."""
import struct

import pytest

from emu import rdna2_exec, rdna2_gpu, rdna2_hlsl
from test_rdna2_exec import LITERAL, V, program, sop1, vop1, vop2
from test_rdna2_hlsl_rust_parity import rust

OPS = ("v_add_f16", "v_sub_f16", "v_subrev_f16", "v_mul_f16", "v_fmac_f16", "v_fmamk_f16",
       "v_fmaak_f16", "v_max_f16", "v_min_f16", "v_ldexp_f16")
WITH_LITERAL = ("v_fmamk_f16", "v_fmaak_f16")


def check(data, monkeypatch):
    if not rdna2_gpu.available():
        pytest.skip("dxc or GPU runner is unavailable")
    monkeypatch.setenv("ANYXBOXX_GPU", "warp")
    result = rust(data)
    assert result.returncode == 0, result.stderr.decode()
    hlsl = result.stdout.decode("utf-8")
    assert hlsl == rdna2_hlsl.translate(data)
    wave = rdna2_exec.run(data)
    hlsl = hlsl.replace("  return;\n", "  g_memory.Store(lane * 4u, v[3]);\n  return;\n")
    assert rdna2_gpu.run(hlsl, bytes(128)) == struct.pack("<32I", *wave.v[3])


@pytest.mark.parametrize("name", OPS)
@pytest.mark.parametrize("a,b,c,mask", [(0x3c00, 0x4000, 0xbc00, 0xffffffff),
                                         (0, 0x7c00, 0x3c00, 0x55555555),
                                         (0x8000, 0, 0x8000, 0xaaaaaaaa),
                                         (0x7e12, 0x3c00, 0, 0xffffffff),
                                         (1, 0x3800, 0x3c00, 0xffffffff)])
def test_vop2_half(name, a, b, c, mask, monkeypatch):
    op = (vop2(name, 3, V + 1, 2), 0x00003c00) if name in WITH_LITERAL else (vop2(name, 3, V + 1, 2),)
    data = program(vop1("v_mov_b32", 1, LITERAL), a | 0xface0000, vop1("v_mov_b32", 2, LITERAL), b | 0xbeef0000,
                   vop1("v_mov_b32", 3, LITERAL), c | 0x12340000, sop1("s_mov_b32", 126, LITERAL), mask, *op)
    check(data, monkeypatch)


@pytest.mark.parametrize("name", [name for name in OPS if name not in WITH_LITERAL])
@pytest.mark.parametrize("code", [*range(240, 249), 125, 128, 193, V])
def test_vop2_half_source_encodings(name, code, monkeypatch):
    data = program(vop1("v_mov_b32", 2, LITERAL), 0x3c00, vop2(name, 3, code, 2))
    check(data, monkeypatch)


@pytest.mark.parametrize("name", WITH_LITERAL)
@pytest.mark.parametrize("literal", [0x00003c00, 0xabcd8000, 0x12347e01, 0x7c00])
def test_vop2_half_k_literal_and_aliased_destination(name, literal, monkeypatch):
    data = program(vop1("v_mov_b32", 1, LITERAL), 0x12343c00,
                   vop1("v_mov_b32", 3, LITERAL), 0xabcd4000,
                   vop2(name, 3, V + 1, 3), literal)
    check(data, monkeypatch)


@pytest.mark.parametrize("src", [0xe9, 0xea, 0xf9, 0xfa])
def test_vop2_half_extensions_refused(src):
    result = rust(program(vop2("v_add_f16", 1, src, 0), 0))
    assert result.returncode != 0 and b"v_add_f16" in result.stderr
