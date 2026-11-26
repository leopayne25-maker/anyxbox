"""VOP3 cross-lane operations: HLSL text and WARP parity."""
import struct

import pytest

from emu import rdna2_exec, rdna2_gpu, rdna2_hlsl
from test_rdna2_exec import LITERAL, V, const, program, sop1, vop1, vop3
from test_rdna2_hlsl_rust_parity import rust


@pytest.mark.parametrize("name", ["v_readlane_b32", "v_readfirstlane_b32", "v_writelane_b32"])
@pytest.mark.parametrize("mask,lane", [(0xffffffff, 0), (0x55555555, 3), (0x80000000, 31), (0, 7)])
def test_vop3_lane_operations(name, mask, lane, monkeypatch):
    if not rdna2_gpu.available():
        pytest.skip("dxc or GPU runner is unavailable")
    monkeypatch.setenv("ANYXBOXX_GPU", "warp")
    dest = 1 if name == "v_writelane_b32" else 4
    source = const(7) if name == "v_writelane_b32" else V
    data = program(vop1("v_mov_b32", 1, LITERAL), 0x12345678,
                   sop1("s_mov_b32", 126, LITERAL), mask,
                   *vop3(name, dest, source, const(lane)))
    result = rust(data)
    assert result.returncode == 0, result.stderr.decode()
    hlsl = result.stdout.decode("utf-8")
    assert hlsl == rdna2_hlsl.translate(data)
    wave = rdna2_exec.run(data)
    hlsl = hlsl.replace("  return;\n", "  g_memory.Store(lane * 4u, v[1]); "
                        "if (lane == 0) g_memory.Store(128, s[4]);\n  return;\n")
    assert rdna2_gpu.run(hlsl, bytes(132)) == struct.pack("<33I", *wave.v[1], wave.s[4])


def test_writelane_vector_source_is_rejected_by_name():
    result = rust(program(*vop3("v_writelane_b32", 1, V, const(0))))
    assert result.returncode != 0 and b"v_writelane_b32" in result.stderr


def test_vop1_readfirstlane_with_empty_exec(monkeypatch):
    if not rdna2_gpu.available():
        pytest.skip("dxc or GPU runner is unavailable")
    monkeypatch.setenv("ANYXBOXX_GPU", "warp")
    data = program(sop1("s_mov_b32", 126, const(0)), vop1("v_readfirstlane_b32", 4, V))
    result = rust(data)
    assert result.returncode == 0, result.stderr.decode()
    hlsl = result.stdout.decode("utf-8")
    assert hlsl == rdna2_hlsl.translate(data)
    wave = rdna2_exec.run(data)
    hlsl = hlsl.replace("  return;\n", "  if (lane == 0) g_memory.Store(0, s[4]);\n  return;\n")
    assert rdna2_gpu.run(hlsl, bytes(4)) == struct.pack("<I", wave.s[4])
