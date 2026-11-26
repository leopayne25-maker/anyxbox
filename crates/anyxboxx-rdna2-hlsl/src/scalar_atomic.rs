//! 32-bit SMEM atomic instructions.

pub fn statements(name: &str, kind: &str, buffer: bool, word: u64) -> Result<Vec<String>, String> {
    let data = (word >> 6) & 0x7f;
    let base = (word & 0x3f) * 2;
    let raw = ((word >> 32) & 0x1f_ffff) as u32;
    let offset = ((raw << 11) as i32 >> 11) as u32;
    let soffset = (word >> 57) & 0x7f;
    let extra = if soffset == 125 {
        String::new()
    } else {
        format!(" + s[{soffset}]")
    };
    let inside = if buffer {
        format!("(asint(o) >= 0 && o + 4u <= s[{}])", base + 2)
    } else {
        "true".into()
    };
    let input = format!("s[{data}]");
    let body = match kind {
        "add" => format!("g_memory.InterlockedAdd(a, {input}, old);"),
        "sub" => format!("g_memory.InterlockedAdd(a, 0u - {input}, old);"),
        "swap" => format!("g_memory.InterlockedExchange(a, {input}, old);"),
        "and" => format!("g_memory.InterlockedAnd(a, {input}, old);"),
        "or" => format!("g_memory.InterlockedOr(a, {input}, old);"),
        "xor" => format!("g_memory.InterlockedXor(a, {input}, old);"),
        "umin" => format!("g_memory.InterlockedMin(a, {input}, old);"),
        "umax" => format!("g_memory.InterlockedMax(a, {input}, old);"),
        "smin" => format!("int r; g_memory.InterlockedMin(a, asint({input}), r); old = asuint(r);"),
        "smax" => format!("int r; g_memory.InterlockedMax(a, asint({input}), r); old = asuint(r);"),
        "cmpswap" => format!(
            "g_memory.InterlockedCompareExchange(a, s[{}], {input}, old);",
            data + 1
        ),
        "inc" | "dec" => {
            let next = if kind == "inc" {
                "o >= x ? 0u : o + 1u"
            } else {
                "(o == 0u || o > x) ? x : o - 1u"
            };
            format!(
                "uint x = {input}, o = g_memory.Load(a); [loop] for (;;) {{ uint seen; g_memory.InterlockedCompareExchange(a, o, {next}, seen); if (seen == o) break; o = seen; }} old = o;"
            )
        }
        _ => return Err(format!("untranslated instruction: {name}")),
    };
    let keep = if (word >> 16) & 1 != 0 {
        format!(" if ({inside}) s[{data}] = old;")
    } else {
        String::new()
    };
    Ok(vec![
        format!(
            "{{ uint o = 0x{offset:X}u{extra}; uint a = s[{base}] + o; uint old = 0u; if (lane == 0u && {inside}) {{ {body} }} old = ReadLane(lane, old, 0u);{keep} }}"
        ),
        "AllMemoryBarrierWithGroupSync();".into(),
    ])
}
