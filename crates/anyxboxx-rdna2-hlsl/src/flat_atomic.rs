//! GLOBAL and FLAT 32-bit integer atomic operations.

pub fn statements(name: &str, family: &str, kind: &str, word: u64) -> Result<Vec<String>, String> {
    let function = match kind {
        "add" => "InterlockedAdd",
        "swap" => "InterlockedExchange",
        "and" => "InterlockedAnd",
        "or" => "InterlockedOr",
        "xor" => "InterlockedXor",
        "umin" => "InterlockedMin",
        "umax" => "InterlockedMax",
        _ => return Err(format!("untranslated instruction: {name}")),
    };
    let raw = (word & 0xfff) as u32;
    if family == "FLAT" && raw & 0x800 != 0 {
        return Err(format!(
            "untranslated instruction: {name}: invalid FLAT offset"
        ));
    }
    let offset = if family == "FLAT" {
        raw
    } else {
        ((raw << 20) as i32 >> 20) as u32
    };
    let (addr, data, saddr, vdst) = (
        (word >> 32) & 255,
        (word >> 40) & 255,
        (word >> 48) & 127,
        (word >> 56) & 255,
    );
    let address = if family == "GLOBAL" && saddr != 125 {
        format!("(s[{saddr}] + v[{addr}] + 0x{offset:X}u)")
    } else {
        format!("(v[{addr}] + 0x{offset:X}u)")
    };
    let keep = if (word >> 16) & 1 != 0 {
        format!(" v[{vdst}] = old;")
    } else {
        String::new()
    };
    Ok(vec![
        format!(
            "if ((((s[126] >> lane) & 1u) != 0) && true) {{ uint old; g_memory.{function}({address}, v[{data}], old);{keep} }}"
        ),
        "AllMemoryBarrierWithGroupSync();".into(),
    ])
}
