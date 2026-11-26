//! VOP1 moves and swaps indexed by M0, as emitted by the Python reference.

const ACTIVE: &str = "((s[126] >> lane) & 1u) != 0";

pub fn supports(name: &str) -> bool {
    matches!(
        name,
        "v_movrels_b32"
            | "v_movreld_b32"
            | "v_movrelsd_b32"
            | "v_movrelsd_2_b32"
            | "v_swap_b32"
            | "v_swaprel_b32"
    )
}

pub fn statement(name: &str, dest: u32, src: u32, operand: &str) -> Result<String, String> {
    let dest_offset = match name {
        "v_movreld_b32" | "v_movrelsd_b32" => Some("s[124]"),
        "v_movrelsd_2_b32" | "v_swaprel_b32" => Some("((s[124] >> 16) & 0x3FFu)"),
        _ => None,
    };
    let j = index(dest, dest_offset);
    if name == "v_movreld_b32" {
        return Ok(format!(
            "if ({ACTIVE}) {{ uint x = {operand}, j = {j}; if (j < 256u) v[j] = x; }}"
        ));
    }
    if src < 256 {
        return Err(format!(
            "untranslated instruction: {name}: indexed source must be a VGPR"
        ));
    }
    let src_offset = match name {
        "v_movrels_b32" | "v_movrelsd_b32" => Some("s[124]"),
        "v_movrelsd_2_b32" | "v_swaprel_b32" => Some("(s[124] & 0x3FFu)"),
        _ => None,
    };
    let i = index(src - 256, src_offset);
    let writes = if matches!(name, "v_swap_b32" | "v_swaprel_b32") {
        "if (i < 256u && j < 256u) { v[i] = v[j]; v[j] = x; }"
    } else {
        "if (j < 256u) v[j] = x;"
    };
    Ok(format!(
        "if ({ACTIVE}) {{ uint i = {i}, j = {j}; uint x = i < 256u ? v[i] : 0u; {writes} }}"
    ))
}

fn index(register: u32, offset: Option<&str>) -> String {
    offset.map_or_else(
        || format!("{register}u"),
        |offset| format!("({register}u + {offset})"),
    )
}
