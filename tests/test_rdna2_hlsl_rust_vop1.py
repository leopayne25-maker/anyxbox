"""VOP1 HLSL text and per-lane WARP parity."""
import struct

import pytest

from emu import rdna2_exec, rdna2_gpu, rdna2_hlsl
from test_rdna2_exec import LITERAL, V, program, sop1, vop1
from test_rdna2_hlsl_rust_parity import rust

OPS = ("v_mov_b32", "v_not_b32", "v_bfrev_b32", "v_ffbl_b32", "v_ffbh_u32", "v_ffbh_i32",
       "v_cvt_f32_u32", "v_cvt_f32_i32", "v_cvt_u32_f32", "v_cvt_i32_f32",
       "v_cvt_f32_ubyte0", "v_cvt_f32_ubyte1", "v_cvt_f32_ubyte2", "v_cvt_f32_ubyte3",
       "v_cvt_off_f32_i4", "v_cvt_flr_i32_f32", "v_cvt_rpi_i32_f32", "v_sat_pk_u8_i16",
       "v_frexp_exp_i32_f32", "v_frexp_mant_f32", "v_nop", "v_pipeflush", "v_clrexcp")


@pytest.mark.parametrize("name", OPS)
@pytest.mark.parametrize("value,mask", [(0, 0xffffffff), (0x3fc00000, 0x55555555),
                                        (0x80000000, 0xaaaaaaaa), (0x7f800000, 0xffffffff),
                                        (0xffffffff, 0xffffffff)])
def test_vop1_literal_text_and_lanes(name, value, mask, monkeypatch):
    if not rdna2_gpu.available():
        pytest.skip("dxc or GPU runner is unavailable")
    monkeypatch.setenv("ANYXBOXX_GPU", "warp")
    data = program(vop1("v_mov_b32", 1, LITERAL), 0x12345678,
                   sop1("s_mov_b32", 126, LITERAL), mask, vop1(name, 1, LITERAL), value)
    result = rust(data)
    assert result.returncode == 0, result.stderr.decode()
    hlsl = result.stdout.decode("utf-8")
    assert hlsl == rdna2_hlsl.translate(data)
    wave = rdna2_exec.run(data)
    hlsl = hlsl.replace("  return;\n", "  g_memory.Store(lane * 4u, v[1]);\n  return;\n")
    assert rdna2_gpu.run(hlsl, bytes(128)) == struct.pack("<32I", *wave.v[1])


@pytest.mark.parametrize("name", ["v_mov_b32", "v_not_b32", "v_bfrev_b32", "v_ffbl_b32", "v_ffbh_u32", "v_ffbh_i32"])
def test_vop1_per_lane_source(name, monkeypatch):
    if not rdna2_gpu.available():
        pytest.skip("dxc or GPU runner is unavailable")
    monkeypatch.setenv("ANYXBOXX_GPU", "warp")
    data = program(vop1(name, 1, V))
    result = rust(data)
    assert result.returncode == 0, result.stderr.decode()
    hlsl = result.stdout.decode("utf-8")
    assert hlsl == rdna2_hlsl.translate(data)
    wave = rdna2_exec.run(data)
    hlsl = hlsl.replace("  return;\n", "  g_memory.Store(lane * 4u, v[1]);\n  return;\n")
    assert rdna2_gpu.run(hlsl, bytes(128)) == struct.pack("<32I", *wave.v[1])


def test_untranslated_vop1_is_named():
    result = rust(program(vop1("v_sqrt_f64", 1, V)))
    assert result.returncode != 0 and b"v_sqrt_f64" in result.stderr


@pytest.mark.parametrize("mask", [0xffffffff, 0x80000000, 0x12345678])
def test_readfirstlane_uses_first_active_lane(mask, monkeypatch):
    if not rdna2_gpu.available():
        pytest.skip("dxc or GPU runner is unavailable")
    monkeypatch.setenv("ANYXBOXX_GPU", "warp")
    data = program(sop1("s_mov_b32", 126, LITERAL), mask, vop1("v_readfirstlane_b32", 2, V))
    result = rust(data)
    assert result.returncode == 0, result.stderr.decode()
    hlsl = result.stdout.decode("utf-8")
    assert hlsl == rdna2_hlsl.translate(data)
    wave = rdna2_exec.run(data)
    hlsl = hlsl.replace("  return;\n", "  if (lane == 0) g_memory.Store(0, s[2]);\n  return;\n")
    assert rdna2_gpu.run(hlsl, bytes(4)) == struct.pack("<I", wave.s[2])


@pytest.mark.parametrize("src", [0xe9, 0xea, 0xf9, 0xfa])
def test_untranslated_extension_is_named(src):
    result = rust(program(vop1("v_mov_b32", 1, src), 0))
    assert result.returncode != 0 and b"v_mov_b32" in result.stderr
