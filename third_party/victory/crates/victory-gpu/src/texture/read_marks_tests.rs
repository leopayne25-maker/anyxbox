//! What a draw's read of a texture asks of the texture itself instead of
//! the shared tables (each behind a lock every recording thread took):
//! whether the host wrote it, the first level its staging buffer holds,
//! the placed texture a view names, and whether a target lies at its
//! memory. Each answer follows the table it stands for.

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;
use windows::core::Interface;

use crate::object::{S_OK, This, guid, slot};

const RGBA8_64: [u32; 14] = [3, 0, 0, 0, 64, 0, 64, 1 | 1 << 16, 28, 1, 0, 0, 0, 0];
const RENDER_TARGET: u32 = 0x1;

/// A 64x64 RGBA8 title texture at `address` with resource flags `flags`.
fn texture_at(address: u64, flags: u32) -> This {
    let mut words = RGBA8_64;
    words[12] = flags;
    let i = ID3D12Resource::IID;
    let iid = guid(i.data1, i.data2, i.data3, i.data4);
    let mut out = std::ptr::null();
    let made = crate::resource::create_at_address(
        std::ptr::null(),
        address,
        words.as_ptr().cast(),
        0,
        std::ptr::null(),
        &iid,
        &mut out,
    );
    assert_eq!(made, S_OK);
    out
}

/// Title memory of `bytes`, 64 KiB aligned; leaked, as title memory
/// outlives a test.
fn title_memory(bytes: usize) -> u64 {
    let layout = std::alloc::Layout::from_size_align(bytes, 1 << 16).unwrap();
    // SAFETY: a non-zero size.
    unsafe { std::alloc::alloc_zeroed(layout) as u64 }
}

fn release(t: This) {
    let _ = slot::<extern "system" fn(This) -> u32>(t, 0x10)(t);
}

/// The first level a texture's staging buffer holds, as a read finds it
/// without the staging table's lock, follows the table: made by the first
/// upload, gone when the buffer is let go.
#[test]
fn a_textures_staged_level_follows_its_staging_buffer() {
    let Some(host) = crate::host::host() else {
        return; // no host GPU here
    };
    let t = texture_at(title_memory(1 << 16), 0);
    assert_eq!(crate::resource::staged_first(t), u32::MAX, "nothing staged yet");
    let list = host.new_list().unwrap();
    let texture = crate::resource::host_resource(t).unwrap();
    super::upload(&list, t, texture);
    assert_eq!(crate::resource::staged_first(t), super::staged_from(t));
    assert_eq!(crate::resource::staged_first(t), 0, "the upload staged the whole texture");
    super::unstage(t);
    assert_eq!(crate::resource::staged_first(t), u32::MAX, "let go");
    assert_eq!(super::staged_from(t), u32::MAX);
    list.close();
    release(t);
}

/// A texture the host wrote is known to be so without the set's lock, and
/// is not uploaded from title memory.
#[test]
fn a_texture_the_host_wrote_is_marked_and_not_uploaded() {
    let Some(host) = crate::host::host() else {
        return; // no host GPU here
    };
    let t = texture_at(title_memory(1 << 16), 0);
    assert!(!crate::resource::host_written_mark(t));
    super::host_written(t);
    assert!(crate::resource::host_written_mark(t));
    assert!(super::host_wrote(t as usize));
    let list = host.new_list().unwrap();
    super::upload(&list, t, crate::resource::host_resource(t).unwrap());
    assert!(list.first_use(t as usize), "not uploaded");
    assert_eq!(crate::resource::staged_first(t), u32::MAX, "not staged");
    list.close();
    release(t);
}

/// A placed texture's kept "no target at its memory" answer does not hide
/// a render target made there later, nor one gone after it.
#[test]
fn a_placed_texture_finds_a_target_made_at_its_memory_later() {
    if crate::host::host().is_none() {
        return; // no host GPU here
    }
    let at = title_memory(1 << 16);
    let placed = crate::resource::texture_at(at, &RGBA8_64).unwrap();
    assert_eq!(crate::resource::alias_of(placed), None);
    assert_eq!(crate::resource::alias_of(placed), None, "the answer kept");
    let target = texture_at(at, RENDER_TARGET);
    assert_eq!(crate::resource::alias_of(placed), Some(target), "the target made since");
    release(target);
    assert_eq!(crate::resource::alias_of(placed), None, "the target gone");
    crate::resource::release_test_placed_texture(at, RGBA8_64);
}

/// What a thread found of a placed texture is the table's: the same
/// resource each time, and a new one once a test took the old one out.
#[test]
fn placed_textures_found_again_are_the_tables() {
    if crate::host::host().is_none() {
        return; // no host GPU here
    }
    let at = title_memory(1 << 16);
    let first = crate::resource::texture_at(at, &RGBA8_64).unwrap();
    assert_eq!(crate::resource::texture_at(at, &RGBA8_64), Some(first));
    let mut filler = RGBA8_64;
    filler[1] = 7;
    assert_eq!(crate::resource::texture_at(at, &filler), Some(first), "the filler is not part of the key");
    crate::resource::release_test_placed_texture(at, RGBA8_64);
    let again = crate::resource::texture_at(at, &RGBA8_64).unwrap();
    assert!(crate::resource::is_texture(again), "a live texture, not the one taken out");
    assert_eq!(crate::resource::texture_at(at, &filler), Some(again));
    crate::resource::release_test_placed_texture(at, RGBA8_64);
}
