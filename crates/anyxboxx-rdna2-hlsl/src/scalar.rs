//! SOP1 32-bit arithmetic and bit operations.

use anyxboxx_rdna2_ref::scalar::fields;

pub(crate) fn source(code: u32, literal: Option<u32>) -> Result<String, String> {
    match code {
        125 => Ok("0u".into()),
        0..=127 => Ok(format!("s[{code}]")),
        128..=192 => Ok(format!("{}u", code - 128)),
        193..=208 => Ok(format!("0x{:X}u", 192u32.wrapping_sub(code))),
        240..=248 => {
            let bits = [
                0x3F000000u32,
                0xBF000000,
                0x3F800000,
                0xBF800000,
                0x40000000,
                0xC0000000,
                0x40800000,
                0xC0800000,
                0x3E22F983,
            ];
            Ok(format!("0x{:X}u", bits[(code - 240) as usize]))
        }
        255 => literal
            .map(|value| format!("0x{value:X}u"))
            .ok_or_else(|| "missing literal".into()),
        _ => Err(format!("untranslated operand {code}")),
    }
}

pub fn statement(name: &str, word: u32, literal: Option<u32>) -> Result<String, String> {
    let f = fields(word);
    let a = source(f.ssrc0, literal).map_err(|e| format!("{name}: {e}"))?;
    let d = if f.sdst == 125 {
        "s_null".into()
    } else {
        format!("s[{}]", f.sdst)
    };
    let line = match name {
        "s_mov_b32" => format!("{d} = {a};"),
        "s_cmov_b32" => format!("if (scc) {d} = {a};"),
        "s_not_b32" => format!("{{ uint r = ~{a}; scc = r != 0; {d} = r; }}"),
        "s_brev_b32" => format!("{d} = reversebits({a});"),
        "s_bcnt1_i32_b32" => format!("{{ uint x = {a}; scc = x != 0u; {d} = countbits(x); }}"),
        "s_ff1_i32_b32" => format!("{d} = firstbitlow({a});"),
        "s_flbit_i32_b32" => {
            format!("{{ uint x = {a}; {d} = x != 0u ? (31u - firstbithigh(x)) : 0xFFFFFFFFu; }}")
        }
        "s_sext_i32_i8" => format!("{d} = asuint((asint({a}) << 24) >> 24);"),
        "s_sext_i32_i16" => format!("{d} = asuint((asint({a}) << 16) >> 16);"),
        "s_abs_i32" => {
            format!("{{ int x = asint({a}); scc = x != 0; {d} = asuint(x < 0 ? -x : x); }}")
        }
        "s_bcnt0_i32_b32" => {
            format!("{{ uint r = 32u - countbits({a}); scc = r != 0u; {d} = r; }}")
        }
        "s_ff0_i32_b32" => format!("{d} = firstbitlow(~{a});"),
        "s_flbit_i32" => format!("{d} = LeadingSignsI32({a});"),
        "s_wqm_b32" => format!("{{ uint r = WqmB32({a}); scc = r != 0u; {d} = r; }}"),
        "s_quadmask_b32" => format!("{{ uint r = QuadMaskB32({a}); scc = r != 0u; {d} = r; }}"),
        "s_bitset0_b32" => format!("{d} = {d} & ~(1u << ({a} & 31u));"),
        "s_bitset1_b32" => format!("{d} = {d} | (1u << ({a} & 31u));"),
        _ if name.ends_with("_saveexec_b32") || name.ends_with("_wrexec_b32") => {
            return exec_statement(name, &a, &d);
        }
        _ => return Err(format!("untranslated instruction: {name}")),
    };
    Ok(line)
}

fn exec_statement(name: &str, a: &str, d: &str) -> Result<String, String> {
    let (op, write_new) = if let Some(op) = name
        .strip_prefix("s_")
        .and_then(|n| n.strip_suffix("_saveexec_b32"))
    {
        (op, false)
    } else if let Some(op) = name
        .strip_prefix("s_")
        .and_then(|n| n.strip_suffix("_wrexec_b32"))
    {
        (op, true)
    } else {
        return Err(format!("untranslated instruction: {name}"));
    };
    let combine = match op {
        "and" => format!("{a} & e"),
        "or" => format!("{a} | e"),
        "xor" => format!("{a} ^ e"),
        "andn2" => format!("{a} & ~e"),
        "orn2" => format!("{a} | ~e"),
        "nand" => format!("~({a} & e)"),
        "nor" => format!("~({a} | e)"),
        "xnor" => format!("~({a} ^ e)"),
        "andn1" => format!("~{a} & e"),
        "orn1" => format!("~{a} | e"),
        _ => return Err(format!("untranslated instruction: {name}")),
    };
    if write_new && !matches!(op, "andn1" | "andn2") {
        return Err(format!("untranslated instruction: {name}"));
    }
    let result = if write_new { "s[126]" } else { "e" };
    Ok(format!(
        "{{ uint e = s[126]; s[126] = {combine}; scc = s[126] != 0; {d} = {result}; }}"
    ))
}
