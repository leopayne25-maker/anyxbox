"""Parity of the Rust PE loader (crates/anyxboxx-pe) with the Python one (emu/loader.read_pe).

First component ported to Rust (docs/architecture-rust.md): on the same images, built from source
by pe_builder, both must give the same base, entry, imports, exports, TLS and page protections, or
both must refuse the image. The Python loader is the oracle until it is deleted.
"""
import json
import os
import struct
import subprocess
from pathlib import Path

import pefile
import pytest
from unicorn import UC_ARCH_X86, UC_MODE_64, Uc

from emu.loader import default_allocate, read_pe
from emu.memory import THUNKS
from pe_builder import build_pe

REPO = Path(__file__).resolve().parents[1]
BINARY = REPO / "target" / "release" / ("anyxboxx-pe.exe" if os.name == "nt" else "anyxboxx-pe")
MOVED = 0x2_0000_0000
TLS_AT, CALLBACKS_AT, TEMPLATE_AT, INDEX_AT = 0x100, 0x140, 0x180, 0x1A0
PE_OFFSET = 0x80
DATA_DIRECTORIES = PE_OFFSET + 4 + 20 + 112   # signature, IMAGE_FILE_HEADER, then the optional header
DIR_DELAY, DIR_CLR = 13, 14


@pytest.fixture(scope="module")
def binary():
    if not BINARY.is_file():
        if os.environ.get("ANYXBOXX_REQUIRE_RUNTIME") == "1":
            pytest.fail(f"{BINARY} absent : cargo build --release (espace Cargo racine).")
        pytest.skip("crate Rust anyxboxx-pe non compilée (cargo build --release à la racine)")
    return BINARY


def python_view(path, base):
    """Image du loader Python, au format JSON de la crate Rust ; None s'il refuse l'image."""
    allocate = (lambda preferred, size, relocatable: base) if base is not None else default_allocate
    try:
        image, pages = read_pe(path, Uc(UC_ARCH_X86, UC_MODE_64), allocate=allocate)
    except (ValueError, pefile.PEFormatError):
        return None
    tls = image.tls and {
        "template_start": image.tls.template_start, "template_end": image.tls.template_end,
        "zero_fill": image.tls.zero_fill, "index_address": image.tls.index_address,
        "callbacks": list(image.tls.callbacks)}
    return {
        "base": image.base, "size": image.size, "entry": image.entry,
        "relocated": image.relocated, "is_dll": image.is_dll,
        "imports": [{"dll": i.dll, "name": i.name, "slot": i.slot} for i in image.imports],
        "exports": image.exports, "ordinals": {str(k): v for k, v in image.ordinals.items()},
        "forwarders": image.forwarders, "tls": tls, "pages": list(pages)}


def rust_view(binary, path, base):
    command = [str(binary), str(path), "--pages"]
    if base is not None:
        command += ["--base", hex(base)]
    done = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=30)
    if done.returncode != 0:
        assert done.stderr.startswith("anyxboxx-pe: "), done.stderr
        return None
    return json.loads(done.stdout)


def ret(iat):
    return b"\xc3"


def tls_code(iat):
    code = bytearray(CALLBACKS_AT + 24)
    struct.pack_into("<I", code, TLS_AT + 32, 0x30)   # SizeOfZeroFill
    return bytes(code)


TLS_POINTERS = {TLS_AT: TEMPLATE_AT, TLS_AT + 8: TEMPLATE_AT + 0x10, TLS_AT + 16: INDEX_AT,
                TLS_AT + 24: CALLBACKS_AT, CALLBACKS_AT: 0x40, CALLBACKS_AT + 8: 0x60}


def with_directory(path, index, rva, size):
    data = bytearray(path.read_bytes())
    struct.pack_into("<II", data, DATA_DIRECTORIES + 8 * index, rva, size)
    path.write_bytes(bytes(data))
    return path


def with_machine(path, machine):
    data = bytearray(path.read_bytes())
    struct.pack_into("<H", data, PE_OFFSET + 4, machine)
    path.write_bytes(bytes(data))
    return path


# (name, image built in tmp_path, forced base or None, expected: "ok" or "refused")
CASES = [
    ("exe-minimal", lambda p: build_pe(p, ret, dll=False), None, "ok"),
    ("dll-imports-exports", lambda p: build_pe(
        p, ret, imports=[("kernel32.dll", "ExitProcess"), ("xmem.dll", 12), ("kernel32.dll", "Sleep")],
        exports=[("Double", 0x10), ("Twice", "libB.Double"), ("Add1", 0x20)]), None, "ok"),
    ("dll-forced-base", lambda p: build_pe(p, ret, exports=[("Double", 0x10)], pointers={0x80: 0x10}),
     MOVED, "ok"),
    ("unusable-base-relocatable", lambda p: build_pe(p, ret, base=THUNKS, pointers={0x80: 0}), None, "ok"),
    ("unusable-base-fixed", lambda p: build_pe(p, ret, base=THUNKS), None, "refused"),
    ("tls-preferred-base", lambda p: build_pe(p, tls_code, dll=False, pointers=TLS_POINTERS, tls=TLS_AT),
     None, "ok"),
    ("tls-forced-base", lambda p: build_pe(p, tls_code, dll=False, pointers=TLS_POINTERS, tls=TLS_AT),
     MOVED, "ok"),
    ("dll-without-entry", lambda p: build_pe(p, ret, entry=None), None, "ok"),
    ("exe-without-entry", lambda p: build_pe(p, ret, dll=False, entry=None), None, "refused"),
    ("i386", lambda p: with_machine(build_pe(p, ret), 0x14C), None, "refused"),
    ("delay-imports", lambda p: with_directory(build_pe(p, ret), DIR_DELAY, 0x1000, 0x40), None, "refused"),
    (".net", lambda p: with_directory(build_pe(p, ret), DIR_CLR, 0x1000, 0x48), None, "refused"),
    ("not-a-pe", lambda p: (p.write_bytes(b"MZ not a PE"), p)[1], None, "refused"),
]


@pytest.mark.parametrize("make, base, expected", [c[1:] for c in CASES], ids=[c[0] for c in CASES])
def test_rust_loader_matches_python(binary, tmp_path, make, base, expected):
    path = make(tmp_path / "image.dll")
    python, rust = python_view(path, base), rust_view(binary, path, base)
    assert (python is None, rust is None) == (expected == "refused",) * 2, (python, rust)
    assert rust == python


def test_root_workspace_leaves_the_other_workspaces_alone():
    """Cargo refuses a crate claimed by two workspaces : rust/ and third_party/ keep their own."""
    import tomllib

    manifest = tomllib.loads((REPO / "Cargo.toml").read_text(encoding="utf-8"))
    workspace = manifest["workspace"]
    assert {"rust", "third_party", ".claude"} <= set(workspace["exclude"])
    assert workspace["members"] == ["crates/*"]
    assert (REPO / "crates" / "anyxboxx-pe" / "Cargo.toml").is_file()
