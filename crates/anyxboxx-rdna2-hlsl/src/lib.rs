//! RDNA 2 to HLSL translation. Unsupported instructions are rejected by name.

use anyxboxx_rdna2_ref::decode::{decode, mnemonic};

mod buffer_memory;
mod flat_atomic;
mod flat_memory;
mod lds;
mod lds_lanes;
mod scalar;
mod scalar_atomic;
mod scalar_binary;
mod scalar_compare;
mod scalar_immediate;
mod scalar_memory;
mod scalar_movrel;
mod scalar_wide;
mod vector_binary;
mod vector_compare;
mod vector_dot;
mod vector_movrel;
mod vector_ternary;
mod vector_unary;

const PRELUDE: &str = concat!(
    include_str!("prelude0.hlsl"),
    include_str!("prelude1.hlsl"),
    include_str!("prelude2.hlsl"),
    include_str!("prelude3.hlsl"),
);

/// Translate a straight-line wave32 program, rejecting unsupported instructions.
pub fn translate(data: &[u8]) -> Result<String, String> {
    if !data.len().is_multiple_of(4) {
        return Err("truncated instruction".into());
    }
    let instructions = decode(data);
    let mut lines = String::from(PRELUDE);
    for (index, instruction) in instructions.iter().enumerate() {
        let name = mnemonic(instruction.family, instruction.opcode)
            .map(str::to_owned)
            .unwrap_or_else(|| format!("{} opcode {}", instruction.family, instruction.opcode));
        match name.as_str() {
            "s_nop" => (),
            "s_endpgm" if index + 1 == instructions.len() => lines.push_str("  return;\n"),
            _ if instruction.family == "MUBUF" => {
                let offset = instruction.offset;
                if instruction.size < 2 || offset + instruction.size * 4 > data.len() {
                    return Err(format!("truncated instruction: {name}"));
                }
                let word = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
                for line in buffer_memory::statements(&name, word)? {
                    lines.push_str("  ");
                    lines.push_str(&line);
                    lines.push('\n');
                }
            }
            _ if matches!(instruction.family, "GLOBAL" | "FLAT") => {
                let offset = instruction.offset;
                if instruction.size < 2 || offset + instruction.size * 4 > data.len() {
                    return Err(format!("truncated instruction: {name}"));
                }
                let word = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
                for line in flat_memory::statements(&name, instruction.family, word)? {
                    lines.push_str("  ");
                    lines.push_str(&line);
                    lines.push('\n');
                }
            }
            _ if instruction.family == "DS" => {
                let offset = instruction.offset;
                if instruction.size < 2 || offset + instruction.size * 4 > data.len() {
                    return Err(format!("truncated instruction: {name}"));
                }
                let word = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
                for line in lds::statements(&name, word)? {
                    lines.push_str("  ");
                    lines.push_str(&line);
                    lines.push('\n');
                }
            }
            _ if instruction.family == "SMEM" => {
                let offset = instruction.offset;
                if instruction.size < 2 || offset + instruction.size * 4 > data.len() {
                    return Err(format!("truncated instruction: {name}"));
                }
                let word = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
                for line in scalar_memory::statements(&name, word)? {
                    lines.push_str("  ");
                    lines.push_str(&line);
                    lines.push('\n');
                }
            }
            _ if instruction.family == "VOP3P" => {
                let offset = instruction.offset;
                if instruction.size < 2 || offset + instruction.size * 4 > data.len() {
                    return Err(format!("truncated instruction: {name}"));
                }
                let word = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
                let literal = (instruction.size > 2)
                    .then(|| u32::from_le_bytes(data[offset + 8..offset + 12].try_into().unwrap()));
                lines.push_str("  ");
                lines.push_str(&vector_dot::statement(&name, word, literal)?);
                lines.push('\n');
            }
            _ if instruction.family == "VOP3" => {
                let offset = instruction.offset;
                if offset + instruction.size * 4 > data.len() || instruction.size < 2 {
                    return Err(format!("truncated instruction: {name}"));
                }
                let word = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
                let literal = (instruction.size > 2)
                    .then(|| u32::from_le_bytes(data[offset + 8..offset + 12].try_into().unwrap()));
                let line = vector_ternary::statement(&name, word, literal)?;
                lines.push_str("  ");
                lines.push_str(&line);
                lines.push('\n');
            }
            _ if instruction.family == "VOPC" => {
                let offset = instruction.offset;
                let word = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
                let literal = (instruction.size > 1)
                    .then(|| u32::from_le_bytes(data[offset + 4..offset + 8].try_into().unwrap()));
                let line = vector_compare::statement(&name, word, literal)?;
                lines.push_str("  ");
                lines.push_str(&line);
                lines.push('\n');
            }
            _ if instruction.family == "VOP2" => {
                let offset = instruction.offset;
                let word = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
                let literal = (instruction.size > 1)
                    .then(|| u32::from_le_bytes(data[offset + 4..offset + 8].try_into().unwrap()));
                let line = vector_binary::statement(&name, word, literal)?;
                lines.push_str("  ");
                lines.push_str(&line);
                lines.push('\n');
            }
            _ if instruction.family == "VOP1" => {
                let offset = instruction.offset;
                let word = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
                let literal = (instruction.size > 1)
                    .then(|| u32::from_le_bytes(data[offset + 4..offset + 8].try_into().unwrap()));
                if let Some(line) = vector_unary::statement(&name, word, literal)? {
                    lines.push_str("  ");
                    lines.push_str(&line);
                    lines.push('\n');
                }
            }
            _ if instruction.family == "SOPK" => {
                let offset = instruction.offset;
                let word = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
                if let Some(line) = scalar_immediate::statement(&name, word)? {
                    lines.push_str("  ");
                    lines.push_str(&line);
                    lines.push('\n');
                }
            }
            _ if instruction.family == "SOP1" && scalar_movrel::supports(&name) => {
                let offset = instruction.offset;
                let word = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
                let literal = (instruction.size > 1)
                    .then(|| u32::from_le_bytes(data[offset + 4..offset + 8].try_into().unwrap()));
                if let Some(line) = scalar_movrel::statement(&name, word, literal)? {
                    lines.push_str("  ");
                    lines.push_str(&line);
                    lines.push('\n');
                }
            }
            _ if matches!(instruction.family, "SOP1" | "SOP2" | "SOPC") => {
                let offset = instruction.offset;
                let word = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
                let literal = (instruction.size > 1)
                    .then(|| u32::from_le_bytes(data[offset + 4..offset + 8].try_into().unwrap()));
                lines.push_str("  ");
                let statement = if scalar_wide::supports(&name, instruction.family) {
                    scalar_wide::statement(&name, instruction.family, word)?
                } else if instruction.family == "SOP1" {
                    scalar::statement(&name, word, literal)?
                } else if instruction.family == "SOP2" {
                    scalar_binary::statement(&name, word, literal)?
                } else {
                    scalar_compare::statement(&name, word, literal)?
                };
                lines.push_str(&statement);
                lines.push('\n');
            }
            _ => return Err(format!("untranslated instruction: {name}")),
        }
    }
    lines.push_str("}\n");
    Ok(lines)
}

#[cfg(test)]
mod tests {
    use super::translate;

    #[test]
    fn rejects_untranslated_instruction_by_name() {
        let data = 0xbe80_0c80u32.to_le_bytes();
        assert!(translate(&data).unwrap_err().contains("s_brev_b64"));
    }

    #[test]
    fn rejects_truncated_input() {
        assert!(translate(&[0]).is_err());
    }
}
