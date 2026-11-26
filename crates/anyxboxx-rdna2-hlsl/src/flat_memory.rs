//! GLOBAL and FLAT full-word memory operations.

pub fn statements(name: &str, family: &str, word: u64) -> Result<Vec<String>, String> {
    let prefix = if family == "GLOBAL" {
        "global_"
    } else {
        "flat_"
    };
    let rest = name
        .strip_prefix(prefix)
        .ok_or_else(|| format!("untranslated instruction: {name}"))?;
    if let Some(kind) = rest.strip_prefix("atomic_") {
        return crate::flat_atomic::statements(name, family, kind, word);
    }
    let (store, width) = if let Some(width) = rest.strip_prefix("load_") {
        (false, width)
    } else if let Some(width) = rest.strip_prefix("store_") {
        (true, width)
    } else {
        return Err(format!("untranslated instruction: {name}"));
    };
    let addtid = family == "GLOBAL" && width == "dword_addtid";
    let count = match width {
        "dword" | "dword_addtid" => 1,
        "ubyte" | "sbyte" if !store => 1,
        "dwordx2" => 2,
        "dwordx3" => 3,
        "dwordx4" => 4,
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
    if addtid && saddr == 125 {
        return Err(format!("untranslated instruction: {name}: SADDR off"));
    }
    if !store && !addtid && count > 1 && (vdst..vdst + count).contains(&addr) {
        return Err(format!(
            "untranslated instruction: {name}: destination overlaps address"
        ));
    }
    let address = if addtid {
        format!("(s[{saddr}] + 0x{offset:X}u + lane * 4u)")
    } else if family == "GLOBAL" && saddr != 125 {
        format!("(s[{saddr}] + v[{addr}] + 0x{offset:X}u)")
    } else {
        format!("(v[{addr}] + 0x{offset:X}u)")
    };
    if store {
        let writes = (0..count)
            .map(|i| format!("g_memory.Store({address} + {}u, v[{}]);", 4 * i, data + i))
            .collect::<Vec<_>>()
            .join(" ");
        Ok(vec![
            format!("if ((((s[126] >> lane) & 1u) != 0) && true) {{ {writes} }}"),
            "AllMemoryBarrierWithGroupSync();".into(),
        ])
    } else {
        let loads = if matches!(width, "ubyte" | "sbyte") {
            let value = format!(
                "((g_memory.Load(({address}) & ~3u) >> ((({address}) & 3u) * 8u)) & 0xFFu)"
            );
            let value = if width == "sbyte" {
                format!("asuint((asint({value}) << 24) >> 24)")
            } else {
                value
            };
            format!("v[{vdst}] = {value};")
        } else {
            (0..count)
                .map(|i| format!("v[{}] = g_memory.Load(({address}) + {}u);", vdst + i, 4 * i))
                .collect::<Vec<_>>()
                .join(" ")
        };
        let zeros = (0..count)
            .map(|i| format!("v[{}] = 0u;", vdst + i))
            .collect::<Vec<_>>()
            .join(" ");
        Ok(vec![format!(
            "if (((s[126] >> lane) & 1u) != 0) {{ if (true) {{ {loads} }} else {{ {zeros} }} }}"
        )])
    }
}
