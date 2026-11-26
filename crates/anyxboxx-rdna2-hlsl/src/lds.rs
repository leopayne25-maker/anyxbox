//! LDS full-word reads and writes for straight-line wave32 programs.

pub fn statements(name: &str, word: u64) -> Result<Vec<String>, String> {
    if matches!(
        name,
        "ds_swizzle_b32" | "ds_permute_b32" | "ds_bpermute_b32"
    ) {
        return Ok(vec![crate::lds_lanes::statement(name, word)?]);
    }
    if let Some((function, returns)) = atomic(name) {
        let addr = (word >> 32) & 0xff;
        let data = (word >> 40) & 0xff;
        let dest = (word >> 56) & 0xff;
        let offset = word & 0xffff;
        let keep = if returns {
            format!(" v[{dest}] = old;")
        } else {
            String::new()
        };
        let index = format!("((v[{addr}] + {offset}u) >> 2) % 8128u");
        return Ok(vec![
            format!(
                "if (((s[126] >> lane) & 1u) != 0) {{ uint old; {function}(g_lds[{index}], v[{data}], old);{keep} }}"
            ),
            "GroupMemoryBarrierWithGroupSync();".into(),
        ]);
    }
    let (write, rest) = if let Some(rest) = name.strip_prefix("ds_write") {
        (true, rest)
    } else if let Some(rest) = name.strip_prefix("ds_read") {
        (false, rest)
    } else {
        return Err(format!("untranslated instruction: {name}"));
    };
    let (mode, width) = if let Some(width) = rest.strip_prefix("2st64_") {
        (2, width)
    } else if let Some(width) = rest.strip_prefix("2_") {
        (1, width)
    } else if let Some(width) = rest.strip_prefix('_') {
        (0, width)
    } else {
        return Err(format!("untranslated instruction: {name}"));
    };
    let words = match width {
        "b32" => 1,
        "b64" => 2,
        "b96" if mode == 0 => 3,
        "b128" if mode == 0 => 4,
        _ => return Err(format!("untranslated instruction: {name}")),
    };
    let offset0 = word & 0xff;
    let offset1 = (word >> 8) & 0xff;
    let addr = (word >> 32) & 0xff;
    let data0 = (word >> 40) & 0xff;
    let data1 = (word >> 48) & 0xff;
    let vdst = (word >> 56) & 0xff;
    let places = if mode == 0 {
        vec![(offset0 + (offset1 << 8), data0, vdst)]
    } else {
        let step = words * 4 * if mode == 2 { 64 } else { 1 };
        vec![
            (offset0 * step, data0, vdst),
            (offset1 * step, data1, vdst + words),
        ]
    };
    let mut lines = Vec::new();
    for (offset, src, dest) in places {
        for i in 0..words {
            let index = format!("((v[{addr}] + {}u) >> 2) % 8128u", offset + 4 * i);
            lines.push(if write {
                format!("g_lds[{index}] = v[{}];", src + i)
            } else {
                format!("v[{}] = g_lds[{index}];", dest + i)
            });
        }
    }
    let statement = format!(
        "if (((s[126] >> lane) & 1u) != 0) {{ {} }}",
        lines.join(" ")
    );
    let barrier = "GroupMemoryBarrierWithGroupSync();".to_string();
    Ok(if write {
        vec![statement, barrier]
    } else {
        vec![barrier, statement]
    })
}

fn atomic(name: &str) -> Option<(&'static str, bool)> {
    if !matches!(
        name,
        "ds_add_u32"
            | "ds_add_rtn_u32"
            | "ds_and_b32"
            | "ds_and_rtn_b32"
            | "ds_or_b32"
            | "ds_or_rtn_b32"
            | "ds_xor_b32"
            | "ds_xor_rtn_b32"
            | "ds_min_u32"
            | "ds_min_rtn_u32"
            | "ds_max_u32"
            | "ds_max_rtn_u32"
    ) {
        return None;
    }
    let (op, returns) = if let Some(op) = name.strip_suffix("_rtn_u32") {
        (op, true)
    } else if let Some(op) = name.strip_suffix("_rtn_b32") {
        (op, true)
    } else if let Some(op) = name.strip_suffix("_u32") {
        (op, false)
    } else {
        (name.strip_suffix("_b32")?, false)
    };
    let function = match op {
        "ds_add" => "InterlockedAdd",
        "ds_and" => "InterlockedAnd",
        "ds_or" => "InterlockedOr",
        "ds_xor" => "InterlockedXor",
        "ds_min" if name.ends_with("_u32") => "InterlockedMin",
        "ds_max" if name.ends_with("_u32") => "InterlockedMax",
        _ => return None,
    };
    Some((function, returns))
}
