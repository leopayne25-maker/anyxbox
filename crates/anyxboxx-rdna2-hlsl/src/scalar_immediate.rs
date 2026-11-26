//! SOPK scalar operations with a 16-bit immediate.
use anyxboxx_rdna2_ref::scalar::fields;

pub fn statement(name: &str, word: u32) -> Result<Option<String>, String> {
    let f = fields(word);
    let d = if f.sdst == 125 {
        "s_null".into()
    } else {
        format!("s[{}]", f.sdst)
    };
    let signed = f.simm16 as i16 as i32 as u32;
    let k = format!("0x{signed:X}u");
    let u = format!("0x{:X}u", f.simm16);
    let line = match name {
        "s_movk_i32" => format!("{d} = {k};"),
        "s_cmovk_i32" => format!("if (scc) {d} = {k};"),
        "s_addk_i32" => format!(
            "{{ uint x = {d}, y = {k}; uint r = x + y; scc = ((~(x ^ y) & (x ^ r)) >> 31) != 0; {d} = r; }}"
        ),
        "s_mulk_i32" => format!("{d} = {d} * {k};"),
        "s_version" | "s_waitcnt_vscnt" | "s_waitcnt_vmcnt" | "s_waitcnt_expcnt"
        | "s_waitcnt_lgkmcnt" => {
            return Ok(None);
        }
        _ => {
            let (op, signed) = match name {
                "s_cmpk_eq_i32" => ("==", true),
                "s_cmpk_lg_i32" => ("!=", true),
                "s_cmpk_gt_i32" => (">", true),
                "s_cmpk_ge_i32" => (">=", true),
                "s_cmpk_lt_i32" => ("<", true),
                "s_cmpk_le_i32" => ("<=", true),
                "s_cmpk_eq_u32" => ("==", false),
                "s_cmpk_lg_u32" => ("!=", false),
                "s_cmpk_gt_u32" => (">", false),
                "s_cmpk_ge_u32" => (">=", false),
                "s_cmpk_lt_u32" => ("<", false),
                "s_cmpk_le_u32" => ("<=", false),
                _ => return Err(format!("untranslated instruction: {name}")),
            };
            if signed {
                format!("scc = asint({d}) {op} asint({k});")
            } else {
                format!("scc = {d} {op} {u};")
            }
        }
    };
    Ok(Some(line))
}
