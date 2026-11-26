//! Scalar compare instructions (SOPC), matching the Python HLSL emitter.
use crate::scalar::source;
use anyxboxx_rdna2_ref::scalar::fields;

pub(crate) fn high(code: u32) -> Result<String, String> {
    match code {
        125 | 128..=192 => Ok("0u".into()),
        0..=127 => Ok(format!("s[{}]", code + 1)),
        193..=208 => Ok("0xFFFFFFFFu".into()),
        _ => Err(format!("untranslated 64-bit operand {code}")),
    }
}

pub fn statement(name: &str, word: u32, literal: Option<u32>) -> Result<String, String> {
    let f = fields(word);
    if matches!(name, "s_cmp_eq_u64" | "s_cmp_lg_u64") {
        if f.ssrc0 == 255 || f.ssrc1 == 255 {
            return Err(format!(
                "untranslated instruction: {name} with 64-bit literal"
            ));
        }
        let a0 = source(f.ssrc0, None).map_err(|e| format!("{name}: {e}"))?;
        let a1 = high(f.ssrc0).map_err(|e| format!("{name}: {e}"))?;
        let b0 = source(f.ssrc1, None).map_err(|e| format!("{name}: {e}"))?;
        let b1 = high(f.ssrc1).map_err(|e| format!("{name}: {e}"))?;
        return Ok(if name == "s_cmp_eq_u64" {
            format!("scc = {a0} == {b0} && {a1} == {b1};")
        } else {
            format!("scc = {a0} != {b0} || {a1} != {b1};")
        });
    }
    let op = match name {
        "s_cmp_eq_i32" | "s_cmp_eq_u32" => "==",
        "s_cmp_lg_i32" | "s_cmp_lg_u32" => "!=",
        "s_cmp_gt_i32" | "s_cmp_gt_u32" => ">",
        "s_cmp_ge_i32" | "s_cmp_ge_u32" => ">=",
        "s_cmp_lt_i32" | "s_cmp_lt_u32" => "<",
        "s_cmp_le_i32" | "s_cmp_le_u32" => "<=",
        "s_bitcmp0_b32" => "==",
        "s_bitcmp1_b32" => "!=",
        _ => return Err(format!("untranslated instruction: {name}")),
    };
    let a = source(f.ssrc0, literal).map_err(|e| format!("{name}: {e}"))?;
    let b = source(f.ssrc1, literal).map_err(|e| format!("{name}: {e}"))?;
    Ok(if name.starts_with("s_bitcmp") {
        format!("scc = (({a} >> ({b} & 31u)) & 1u) {op} 0u;")
    } else if name.ends_with("_i32") {
        format!("scc = asint({a}) {op} asint({b});")
    } else {
        format!("scc = {a} {op} {b};")
    })
}
