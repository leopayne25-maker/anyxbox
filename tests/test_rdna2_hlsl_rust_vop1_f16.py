"""VOP1 half-float conversions and rounding: text and WARP parity."""
import struct

import pytest

from emu import rdna2_exec, rdna2_gpu, rdna2_hlsl
from test_rdna2_exec import LITERAL, V, program, sop1, vop1
from test_rdna2_hlsl_rust_parity import rust

OPS = ("v_cvt_f16_f32", "v_cvt_f32_f16", "v_floor_f16", "v_ceil_f16", "v_trunc_f16", "v_rndne_f16")


def check(data, monkeypatch):
    if not rdna2_gpu.available():
        pytest.skip("dxc or GPU runner is unavailable")
    monkeypatch.setenv("ANYXBOXX_GPU", "warp")
    result = rust(data)
    assert result.returncode == 0, result.stderr.decode()
    hlsl = result.stdout.decode("utf-8")
    assert hlsl == rdna2_hlsl.translate(data)
    wave = rdna2_exec.run(data)
    hlsl = hlsl.replace("  return;\n", "  g_memory.Store(lane * 4u, v[1]);\n  return;\n")
    assert rdna2_gpu.run(hlsl, bytes(128)) == struct.pack("<32I", *wave.v[1])


@pytest.mark.parametrize("name", OPS)
@pytest.mark.parametrize("value,mask", [(0, 0xffffffff), (0x80008000, 0x55555555),
                                        (0x3fc03e00, 0xaaaaaaaa), (0xbf80bc00, 0xffffffff),
                                        (0x7f807c00, 0xffffffff), (0x7fc07e01, 0xffffffff),
                                        (0x00000001, 0xffffffff), (0x477f7bff, 0xffffffff)])
def test_vop1_half_literal(name, value, mask, monkeypatch):
    data = program(vop1("v_mov_b32", 1, LITERAL), 0x12345678,
                   sop1("s_mov_b32", 126, LITERAL), mask, vop1(name, 1, LITERAL), value)
    check(data, monkeypatch)


@pytest.mark.parametrize("name", OPS)
@pytest.mark.parametrize("code", [*range(240, 249), 125, 128, 193, V])
def test_vop1_half_source_encodings(name, code, monkeypatch):
    data = program(vop1("v_mov_b32", 1, LITERAL), 0xabcd0000, vop1(name, 1, code))
    check(data, monkeypatch)


@pytest.mark.parametrize("src", [0xe9, 0xea, 0xf9, 0xfa])
def test_half_extensions_are_named(src):
    result = rust(program(vop1("v_rndne_f16", 1, src), 0))
    assert result.returncode != 0 and b"v_rndne_f16" in result.stderr
