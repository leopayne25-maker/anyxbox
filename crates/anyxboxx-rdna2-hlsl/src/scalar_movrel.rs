//! SOP1 scalar moves indexed by M0.
use crate::scalar::source;
use crate::scalar_compare::high;
use anyxboxx_rdna2_ref::scalar::fields;

pub fn supports(name: &str) -> bool {
    matches!(
        name,
        "s_movrels_b32" | "s_movrels_b64" | "s_movreld_b32" | "s_movreld_b64" | "s_movrelsd_2_b32"
    )
}

pub fn statement(name: &str, word: u32, literal: Option<u32>) -> Result<Option<String>, String> {
    let f = fields(word);
    let wide = name.ends_with("_b64");
    let direct = matches!(name, "s_movreld_b32" | "s_movreld_b64");
    let reads = if direct {
        if wide && f.ssrc0 == 255 {
            return Err(format!(
                "untranslated instruction: {name} with 64-bit literal"
            ));
        }
        let x = source(f.ssrc0, literal).map_err(|e| format!("{name}: {e}"))?;
        let y = if wide {
            high(f.ssrc0).map_err(|e| format!("{name}: {e}"))?
        } else {
            "0u".into()
        };
        format!("uint x = {x}, y = {y};")
    } else {
        if f.ssrc0 > 105 {
            return Err(format!(
                "untranslated instruction: {name}: indexed source must be an SGPR"
            ));
        }
        let offset = if name == "s_movrelsd_2_b32" {
            "(s[124] & 0x3FFu)"
        } else {
            "s[124]"
        };
        let last = if wide { 104 } else { 105 };
        let pair = if wide { " && (i & 1u) == 0u" } else { "" };
        let y = if wide { "ok ? s[i + 1u] : 0u" } else { "0u" };
        format!(
            "uint i = ({}u + {offset}); bool ok = i <= {last}u{pair}; uint x = ok ? s[i] : 0u, y = {y};",
            f.ssrc0
        )
    };
    if matches!(name, "s_movrels_b32" | "s_movrels_b64") {
        if f.sdst == 125 {
            return Ok(None);
        }
        let high = if wide {
            format!(" s[{}] = y;", f.sdst + 1)
        } else {
            String::new()
        };
        return Ok(Some(format!("{{ {reads} s[{}] = x;{high} }}", f.sdst)));
    }
    if f.sdst > 105 {
        return Err(format!(
            "untranslated instruction: {name}: indexed destination must be an SGPR"
        ));
    }
    let last = if wide { 104 } else { 105 };
    let pair = if wide { " && (j & 1u) == 0u" } else { "" };
    let high = if wide { " s[j + 1u] = y;" } else { "" };
    let offset = if name == "s_movrelsd_2_b32" {
        "((s[124] >> 16) & 0x3FFu)"
    } else {
        "s[124]"
    };
    Ok(Some(format!(
        "{{ {reads} uint j = ({}u + {offset}); if (j <= {last}u{pair}) {{ s[j] = x;{high} }} }}",
        f.sdst
    )))
}
