//! Untyped MUBUF word accesses without an index register.
use crate::scalar::source;

pub fn statements(name: &str, word: u64) -> Result<Vec<String>, String> {
    let rest = name
        .strip_prefix("buffer_")
        .ok_or_else(|| format!("untranslated instruction: {name}"))?;
    let atomic = match rest {
        "atomic_add" => Some("InterlockedAdd"),
        "atomic_swap" => Some("InterlockedExchange"),
        "atomic_and" => Some("InterlockedAnd"),
        "atomic_or" => Some("InterlockedOr"),
        "atomic_xor" => Some("InterlockedXor"),
        "atomic_umin" => Some("InterlockedMin"),
        "atomic_umax" => Some("InterlockedMax"),
        _ => None,
    };
    let (write, width) = if atomic.is_some() {
        (false, "dword")
    } else if let Some(width) = rest.strip_prefix("load_") {
        (false, width)
    } else if let Some(width) = rest.strip_prefix("store_") {
        (true, width)
    } else {
        return Err(format!("untranslated instruction: {name}"));
    };
    let count = match width {
        "dword" => 1,
        "ubyte" | "sbyte" if !write => 1,
        "dwordx2" => 2,
        "dwordx3" => 3,
        "dwordx4" => 4,
        _ => return Err(format!("untranslated instruction: {name}")),
    };
    if word >> 13 & 1 != 0 {
        return Err(format!("untranslated instruction: {name}: idxen"));
    }
    let offset = word & 0xfff;
    let offen = (word >> 12) & 1 != 0;
    let addr = (word >> 32) & 255;
    let data = (word >> 40) & 255;
    let resource = ((word >> 48) & 31) * 4;
    let soffset = ((word >> 56) & 255) as u32;
    if !write && offen && count > 1 && (data..data + count).contains(&addr) {
        return Err(format!(
            "untranslated instruction: {name}: destination overlaps address"
        ));
    }
    if (193..=208).contains(&soffset) {
        return Err(format!(
            "untranslated instruction: {name}: negative inline SOFFSET"
        ));
    }
    let extra = if matches!(soffset, 125 | 128) {
        String::new()
    } else {
        format!(
            " + {}",
            source(soffset, None).map_err(|e| format!("{name}: {e}"))?
        )
    };
    let byte_offset = if offen {
        format!("({offset}u + v[{addr}])")
    } else {
        format!("({offset}u)")
    };
    let no_overflow = if offen {
        format!("v[{addr}] <= 0xFFFFFFFFu - {offset}u && ")
    } else {
        String::new()
    };
    let size = if matches!(width, "ubyte" | "sbyte") {
        1
    } else {
        count * 4
    };
    let inside = format!(
        "(((s[{}] >> 16) & 0x3FFFu) != 0u ? s[{}] != 0u : ({no_overflow}{byte_offset} <= s[{}] && {}u <= s[{}] - {byte_offset}))",
        resource + 1,
        resource + 2,
        resource + 2,
        size,
        resource + 2
    );
    let address = format!("(s[{resource}]{extra} + {byte_offset})");
    if let Some(function) = atomic {
        let keep = if (word >> 14) & 1 != 0 {
            format!(" v[{data}] = old;")
        } else {
            String::new()
        };
        return Ok(vec![
            format!(
                "if ((((s[126] >> lane) & 1u) != 0) && {inside}) {{ uint old; g_memory.{function}({address}, v[{data}], old);{keep} }}"
            ),
            "AllMemoryBarrierWithGroupSync();".into(),
        ]);
    }
    if write {
        let writes = (0..count)
            .map(|i| format!("g_memory.Store({address} + {}u, v[{}]);", 4 * i, data + i))
            .collect::<Vec<_>>()
            .join(" ");
        Ok(vec![
            format!("if ((((s[126] >> lane) & 1u) != 0) && {inside}) {{ {writes} }}"),
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
            format!("v[{data}] = {value};")
        } else {
            (0..count)
                .map(|i| format!("v[{}] = g_memory.Load(({address}) + {}u);", data + i, 4 * i))
                .collect::<Vec<_>>()
                .join(" ")
        };
        let zeros = (0..count)
            .map(|i| format!("v[{}] = 0u;", data + i))
            .collect::<Vec<_>>()
            .join(" ");
        Ok(vec![format!(
            "if (((s[126] >> lane) & 1u) != 0) {{ if ({inside}) {{ {loads} }} else {{ {zeros} }} }}"
        )])
    }
}
