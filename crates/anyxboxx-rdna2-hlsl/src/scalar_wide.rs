//! SOP1/SOP2 64-bit scalar operations supported by the Python emitter.
use crate::scalar::source;
use crate::scalar_compare::high;
use anyxboxx_rdna2_ref::scalar::fields;

fn operation<'a>(name: &'a str, suffix: &str) -> Option<&'a str> {
    let op = name.strip_prefix("s_")?.strip_suffix(suffix)?;
    matches!(
        op,
        "and" | "or" | "xor" | "andn2" | "orn2" | "nand" | "nor" | "xnor" | "andn1" | "orn1"
    )
    .then_some(op)
}

pub fn supports(name: &str, family: &str) -> bool {
    match family {
        "SOP1" => {
            matches!(
                name,
                "s_mov_b64"
                    | "s_cmov_b64"
                    | "s_not_b64"
                    | "s_bcnt1_i32_b64"
                    | "s_bcnt0_i32_b64"
                    | "s_ff1_i32_b64"
                    | "s_ff0_i32_b64"
            ) || operation(name, "_saveexec_b64").is_some()
                || matches!(name, "s_andn1_wrexec_b64" | "s_andn2_wrexec_b64")
        }
        "SOP2" => {
            name == "s_cselect_b64"
                || operation(name, "_b64").is_some_and(|op| !matches!(op, "andn1" | "orn1"))
        }
        _ => false,
    }
}

fn pair(code: u32, name: &str) -> Result<(String, String), String> {
    if code == 255 {
        return Err(format!(
            "untranslated instruction: {name} with 64-bit literal"
        ));
    }
    let low = source(code, None).map_err(|e| format!("{name}: {e}"))?;
    let high = high(code).map_err(|e| format!("{name}: {e}"))?;
    Ok((low, high))
}

fn combine(op: &str, a: &str, b: &str) -> String {
    match op {
        "and" => format!("{a} & {b}"),
        "or" => format!("{a} | {b}"),
        "xor" => format!("{a} ^ {b}"),
        "andn2" => format!("{a} & ~{b}"),
        "orn2" => format!("{a} | ~{b}"),
        "nand" => format!("~({a} & {b})"),
        "nor" => format!("~({a} | {b})"),
        "xnor" => format!("~({a} ^ {b})"),
        "andn1" => format!("~{a} & {b}"),
        "orn1" => format!("~{a} | {b}"),
        _ => unreachable!(),
    }
}

pub fn statement(name: &str, family: &str, word: u32) -> Result<String, String> {
    let f = fields(word);
    let (a0, a1) = pair(f.ssrc0, name)?;
    let (d0, d1) = if f.sdst == 125 {
        ("s_null".into(), "s_null".into())
    } else {
        (format!("s[{}]", f.sdst), format!("s[{}]", f.sdst + 1))
    };
    if family == "SOP2" {
        let (b0, b1) = pair(f.ssrc1, name)?;
        if name == "s_cselect_b64" {
            return Ok(format!(
                "{{ uint r0 = scc ? {a0} : {b0}, r1 = scc ? {a1} : {b1}; {d0} = r0; {d1} = r1; }}"
            ));
        }
        let op = operation(name, "_b64").unwrap();
        let r0 = combine(op, &a0, &b0);
        let r1 = combine(op, &a1, &b1);
        return Ok(format!(
            "{{ uint r0 = {r0}, r1 = {r1}; scc = (r0 | r1) != 0u; {d0} = r0; {d1} = r1; }}"
        ));
    }
    if let Some(op) = operation(name, "_saveexec_b64").or_else(|| operation(name, "_wrexec_b64")) {
        let r0 = combine(op, &a0, "e0");
        let r1 = combine(op, &a1, "e1");
        let (w0, w1) = if name.contains("_wrexec_") {
            ("s[126]", "s[127]")
        } else {
            ("e0", "e1")
        };
        return Ok(format!(
            "{{ uint e0 = s[126], e1 = s[127]; uint r0 = {r0}, r1 = {r1}; s[126] = r0; s[127] = r1; scc = (r0 | r1) != 0u; {d0} = {w0}; {d1} = {w1}; }}"
        ));
    }
    let first = "(x0 != 0u ? firstbitlow(x0) : (x1 != 0u ? 32u + firstbitlow(x1) : 0xFFFFFFFFu))";
    Ok(match name {
        "s_mov_b64" => format!("{{ uint r0 = {a0}, r1 = {a1}; {d0} = r0; {d1} = r1; }}"),
        "s_cmov_b64" => format!("if (scc) {{ uint r0 = {a0}, r1 = {a1}; {d0} = r0; {d1} = r1; }}"),
        "s_not_b64" => format!(
            "{{ uint r0 = ~{a0}, r1 = ~{a1}; scc = (r0 | r1) != 0u; {d0} = r0; {d1} = r1; }}"
        ),
        "s_bcnt1_i32_b64" => {
            format!("{{ uint r = countbits({a0}) + countbits({a1}); scc = r != 0u; {d0} = r; }}")
        }
        "s_bcnt0_i32_b64" => format!(
            "{{ uint r = 64u - countbits({a0}) - countbits({a1}); scc = r != 0u; {d0} = r; }}"
        ),
        "s_ff1_i32_b64" => format!("{{ uint x0 = {a0}, x1 = {a1}; {d0} = {first}; }}"),
        "s_ff0_i32_b64" => format!("{{ uint x0 = ~{a0}, x1 = ~{a1}; {d0} = {first}; }}"),
        _ => return Err(format!("untranslated instruction: {name}")),
    })
}
