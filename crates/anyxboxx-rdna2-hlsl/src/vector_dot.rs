//! Integer packed dot products (VOP3P).
use crate::scalar::source;

pub fn statement(name: &str, word: u64, literal: Option<u32>) -> Result<String, String> {
    let (width, signed) = match name {
        "v_dot2_i32_i16" => (16, true),
        "v_dot2_u32_u16" => (16, false),
        "v_dot4_i32_i8" => (8, true),
        "v_dot4_u32_u8" => (8, false),
        "v_dot8_i32_i4" => (4, true),
        "v_dot8_u32_u4" => (4, false),
        _ => return Err(format!("untranslated instruction: {name}")),
    };
    if (word >> 11) & 7 != 0
        || [59, 60, 14].iter().any(|shift| (word >> shift) & 1 == 0)
        || (word >> 61) & 7 != 0
        || (word >> 8) & 7 != 0
        || (word >> 15) & 1 != 0
    {
        return Err(format!(
            "untranslated instruction: {name} with VOP3P modifiers"
        ));
    }
    let read = |shift: u32| {
        let code = ((word >> shift) & 0x1ffu64) as u32;
        if code >= 256 {
            Ok(format!("v[{}]", code - 256))
        } else {
            source(code, literal).map_err(|e| format!("{name}: {e}"))
        }
    };
    let (a, b, c) = (read(32)?, read(41)?, read(50)?);
    let dest = word & 0xff;
    Ok(format!(
        "if (((s[126] >> lane) & 1u) != 0) v[{dest}] = DotPacked({a}, {b}, {c}, {width}u, {signed});"
    ))
}
