//! The image as it lies in memory: headers and sections at their RVAs, base relocations, and the
//! protection of each page (`loader._map` and `_protection`).

use crate::headers::{Headers, Section};
use crate::{LoadError, PAGE, align};

/// `IMAGE_DIRECTORY_ENTRY_BASERELOC`.
pub const DIR_RELOC: usize = 5;

const SCN_EXEC: u32 = 0x2000_0000;
const SCN_READ: u32 = 0x4000_0000;
const SCN_WRITE: u32 = 0x8000_0000;

/// Page protection bits, the same values as Unicorn's `UC_PROT_*`.
pub const PROT_READ: u8 = 1;
pub const PROT_WRITE: u8 = 2;
pub const PROT_EXEC: u8 = 4;

const REL_ABSOLUTE: u16 = 0;
const REL_HIGHLOW: u16 = 3;
const REL_DIR64: u16 = 10;
const RELOC_BLOCK_HEADER: usize = 8;

/// Copies headers and sections at their RVAs into a zeroed image of `SizeOfImage` (page-aligned).
pub fn map(data: &[u8], headers: &Headers) -> Result<Vec<u8>, LoadError> {
    let mut image = vec![0u8; align(u64::from(headers.size_of_image)) as usize];
    let header_len = (headers.size_of_headers as usize)
        .min(data.len())
        .min(image.len());
    image[..header_len].copy_from_slice(&data[..header_len]);
    for section in &headers.sections {
        copy_section(data, section, &mut image)?;
    }
    Ok(image)
}

fn copy_section(data: &[u8], section: &Section, image: &mut [u8]) -> Result<(), LoadError> {
    if section.raw_size == 0 {
        return Ok(());
    }
    let start = section.raw_offset as usize;
    let source = start
        .checked_add(section.raw_size as usize)
        .and_then(|end| data.get(start..end))
        .ok_or_else(|| LoadError::Malformed("section raw data outside the file".into()))?;
    let target = section.virtual_address as usize;
    let room = image.len().checked_sub(target).filter(|&room| room > 0);
    let room = room.ok_or_else(|| LoadError::Malformed("section outside the image".into()))?;
    let length = source.len().min(room);
    image[target..target + length].copy_from_slice(&source[..length]);
    Ok(())
}

/// `(rva, type)` of every base relocation entry, from the relocation directory of `image`.
///
/// A block with `SizeOfBlock == 0` ends the table (pefile stops there too). Any other malformed
/// block (shorter than its header, or running past the directory or the image) is an error:
/// pefile only warned and dropped the rest of the table, which would leave pointers unrelocated.
pub fn relocations(image: &[u8], headers: &Headers) -> Result<Vec<(usize, u16)>, LoadError> {
    let (rva, size) = headers.directory(DIR_RELOC);
    let mut found = Vec::new();
    if size == 0 {
        return Ok(found);
    }
    let (mut at, end) = (rva as usize, rva as usize + size as usize);
    if end > image.len() {
        return Err(LoadError::Relocation("table outside the image".into()));
    }
    while at < end {
        if at + RELOC_BLOCK_HEADER > end {
            return Err(LoadError::Relocation(format!("truncated block at {at:#x}")));
        }
        let page = read_u32(image, at) as usize;
        let block = read_u32(image, at + 4) as usize;
        if block == 0 {
            break;
        }
        if block < RELOC_BLOCK_HEADER || at + block > end {
            return Err(LoadError::Relocation(format!("malformed block at {at:#x}")));
        }
        // A dangling odd byte at the end of a block is not an entry.
        for entry in (at + RELOC_BLOCK_HEADER..at + block - 1).step_by(2) {
            let value = u16::from_le_bytes([image[entry], image[entry + 1]]);
            found.push((page + usize::from(value & 0xFFF), value >> 12));
        }
        at += block;
    }
    Ok(found)
}

/// Adds `delta` to every relocated field; an unknown relocation type is an error, never skipped.
pub fn relocate(image: &mut [u8], entries: &[(usize, u16)], delta: u64) -> Result<(), LoadError> {
    for &(rva, kind) in entries {
        match kind {
            REL_ABSOLUTE => {}
            REL_DIR64 => {
                let field = field(image, rva, 8)?;
                let value = u64::from_le_bytes(field.try_into().expect("8-byte field"));
                field.copy_from_slice(&value.wrapping_add(delta).to_le_bytes());
            }
            REL_HIGHLOW => {
                let field = field(image, rva, 4)?;
                let value = u32::from_le_bytes(field.try_into().expect("4-byte field"));
                field.copy_from_slice(&value.wrapping_add(delta as u32).to_le_bytes());
            }
            other => return Err(LoadError::Relocation(format!("unsupported type {other}"))),
        }
    }
    Ok(())
}

fn field(image: &mut [u8], rva: usize, length: usize) -> Result<&mut [u8], LoadError> {
    image
        .get_mut(rva..rva + length)
        .ok_or_else(|| LoadError::Relocation(format!("field at {rva:#x} outside the image")))
}

/// Protection of every page of the image (`loader._map`): read-only by default, then each
/// section's own rights. A writable or executable section stays readable, as on Windows.
pub fn page_protections(headers: &Headers) -> Result<Vec<u8>, LoadError> {
    let size = align(u64::from(headers.size_of_image));
    let mut pages = vec![PROT_READ; (size / PAGE) as usize];
    for section in &headers.sections {
        let length = align(u64::from(section.virtual_size.max(section.raw_size)));
        let start = u64::from(section.virtual_address);
        if !start.is_multiple_of(PAGE) || start + length > size {
            return Err(LoadError::Section(start));
        }
        let first = (start / PAGE) as usize;
        let count = (length / PAGE) as usize;
        pages[first..first + count].fill(protection(section.characteristics));
    }
    Ok(pages)
}

fn protection(flags: u32) -> u8 {
    let readable = flags & (SCN_READ | SCN_WRITE | SCN_EXEC) != 0;
    (if readable { PROT_READ } else { 0 })
        | (if flags & SCN_WRITE != 0 {
            PROT_WRITE
        } else {
            0
        })
        | (if flags & SCN_EXEC != 0 { PROT_EXEC } else { 0 })
}

pub fn read_u32(image: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(image[at..at + 4].try_into().expect("4 bytes"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_writable_or_executable_section_stays_readable() {
        assert_eq!(protection(SCN_WRITE), PROT_READ | PROT_WRITE);
        assert_eq!(protection(SCN_EXEC), PROT_READ | PROT_EXEC);
        assert_eq!(protection(0), 0);
    }

    #[test]
    fn dir64_and_highlow_fields_move_by_the_delta() {
        let mut image = vec![0u8; 16];
        image[..8].copy_from_slice(&0x1_4000_1000u64.to_le_bytes());
        image[8..12].copy_from_slice(&0x1000u32.to_le_bytes());
        relocate(
            &mut image,
            &[(0, REL_DIR64), (8, REL_HIGHLOW), (4, REL_ABSOLUTE)],
            0x10_0000,
        )
        .unwrap();
        assert_eq!(&image[..8], &0x1_4010_1000u64.to_le_bytes());
        assert_eq!(&image[8..12], &0x10_1000u32.to_le_bytes());
    }

    #[test]
    fn unknown_relocation_types_are_refused() {
        let mut image = vec![0u8; 16];
        assert!(matches!(
            relocate(&mut image, &[(0, 4)], 1),
            Err(LoadError::Relocation(_))
        ));
        assert!(relocate(&mut image, &[(12, REL_DIR64)], 1).is_err());
    }
}
