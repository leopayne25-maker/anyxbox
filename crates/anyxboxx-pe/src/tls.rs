//! The TLS directory (`loader._tls_fields` and `_tls`): template, zero fill, index slot and
//! callbacks, moved to the chosen base.

use crate::headers::Headers;
use crate::{LoadError, Tls};

/// `IMAGE_DIRECTORY_ENTRY_TLS`.
pub const DIR_TLS: usize = 9;
pub const MAX_TLS_CALLBACKS: usize = 64;
pub const MAX_TLS_SIZE: u64 = 16 * 1024 * 1024;

// IMAGE_TLS_DIRECTORY64 fields.
const TLS_START: usize = 0;
const TLS_END: usize = 8;
const TLS_INDEX: usize = 16;
const TLS_CALLBACKS: usize = 24;
const TLS_ZERO_FILL: usize = 32;
const TLS_DIRECTORY_SIZE: usize = 40;

/// The directory's fields as written in the image, before relocation.
#[derive(Debug, Clone, Copy)]
pub struct Fields {
    start: u64,
    end: u64,
    zero_fill: u32,
    index: u64,
    callbacks: u64,
}

/// Reads the directory from the unrelocated image and shifts its addresses by `delta`.
pub fn fields(image: &[u8], headers: &Headers, delta: u64) -> Result<Option<Fields>, LoadError> {
    let (rva, _) = headers.directory(DIR_TLS);
    if rva == 0 {
        return Ok(None);
    }
    let at = rva as usize;
    let directory = image
        .get(at..at + TLS_DIRECTORY_SIZE)
        .ok_or_else(|| LoadError::Tls("TLS directory outside the image".into()))?;
    let u64_at = |offset: usize| {
        u64::from_le_bytes(directory[offset..offset + 8].try_into().expect("8 bytes"))
    };
    let shift = |value: u64| {
        if value == 0 {
            0
        } else {
            value.wrapping_add(delta)
        }
    };
    Ok(Some(Fields {
        start: shift(u64_at(TLS_START)),
        end: shift(u64_at(TLS_END)),
        zero_fill: u32::from_le_bytes(
            directory[TLS_ZERO_FILL..TLS_ZERO_FILL + 4]
                .try_into()
                .expect("4 bytes"),
        ),
        index: shift(u64_at(TLS_INDEX)),
        callbacks: shift(u64_at(TLS_CALLBACKS)),
    }))
}

/// Checks the shifted fields against the relocated `image` mapped at `base`, then reads the
/// callback array (zero-terminated, at most `MAX_TLS_CALLBACKS`).
pub fn resolve(fields: Fields, image: &[u8], base: u64) -> Result<Tls, LoadError> {
    let size = image.len() as u64;
    let Fields {
        start,
        end,
        zero_fill,
        index,
        callbacks,
    } = fields;
    if (start != 0 || end != 0)
        && !(inside(start, base, size, 0) && start <= end && inside(end, base, size, 0))
    {
        return Err(LoadError::Tls("TLS template outside the image".into()));
    }
    if end.wrapping_sub(start).saturating_add(u64::from(zero_fill)) > MAX_TLS_SIZE {
        return Err(LoadError::Tls("TLS block too large".into()));
    }
    if index != 0 && !inside(index, base, size, 4) {
        return Err(LoadError::Tls("TLS index outside the image".into()));
    }
    Ok(Tls {
        template_start: start,
        template_end: end,
        zero_fill,
        index_address: index,
        callbacks: callback_list(callbacks, image, base)?,
    })
}

fn callback_list(mut cursor: u64, image: &[u8], base: u64) -> Result<Vec<u64>, LoadError> {
    let mut found = Vec::new();
    while cursor != 0 && found.len() < MAX_TLS_CALLBACKS {
        let offset = cursor
            .checked_sub(base)
            .filter(|&offset| {
                offset
                    .checked_add(8)
                    .is_some_and(|end| end <= image.len() as u64)
            })
            .ok_or_else(|| LoadError::Tls("TLS callback array outside the image".into()))?
            as usize;
        let target = u64::from_le_bytes(image[offset..offset + 8].try_into().expect("8 bytes"));
        if target == 0 {
            break;
        }
        found.push(target);
        cursor = cursor
            .checked_add(8)
            .ok_or_else(|| LoadError::Tls("TLS callback array wraps around".into()))?;
    }
    Ok(found)
}

fn inside(address: u64, base: u64, size: u64, length: u64) -> bool {
    base <= address
        && address
            .checked_add(length)
            .is_some_and(|end| end <= base.saturating_add(size))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_callback_array_at_the_top_of_the_address_space_is_refused() {
        // Base near u64::MAX: the walk must stop with an error, not overflow.
        let base = u64::MAX - 15;
        let mut image = vec![0u8; 16];
        image[8..16].copy_from_slice(&1u64.to_le_bytes());
        assert!(matches!(
            callback_list(base + 8, &image, base),
            Err(LoadError::Tls(_))
        ));
    }
}
