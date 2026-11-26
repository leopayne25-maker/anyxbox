//! A draw's read of a texture through a view ([`readable`]): one lookup of
//! the host copy for the upload, the twin and the state, with the copy's
//! flags kept instead of asked of the driver.

use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC};

use super::*;
use crate::object::{S_OK, guid, slot};

const SHADER_RESOURCE: u32 = 0x40 | 0x80;
const COPY_SOURCE: u32 = 0x800;

/// A 64x64 RGBA8 title texture at `address` (one level, one layer).
fn texture_at(address: u64) -> This {
    let words: [u32; 14] = [3, 0, 0, 0, 64, 0, 64, 1 | 1 << 16, 28, 1, 0, 0, 0, 0];
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

/// A host texture's kept flags give what asking the driver for its
/// description gives: the legal states and whether title memory fills it,
/// for every kind of host texture.
#[test]
fn a_host_textures_kept_flags_give_what_its_description_does() {
    let Some(host) = crate::host::host() else {
        return; // no host GPU here
    };
    let kinds = [
        (D3D12_RESOURCE_FLAG_NONE, DXGI_FORMAT_R8G8B8A8_UNORM),
        (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, DXGI_FORMAT_R8G8B8A8_UNORM),
        (D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, DXGI_FORMAT_R8G8B8A8_UNORM),
        (D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, DXGI_FORMAT_D32_FLOAT),
        (
            D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL | D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE,
            DXGI_FORMAT_D32_FLOAT,
        ),
    ];
    // Every state bit a list asks for, and the common mixes.
    let states: Vec<u32> = (0..24).map(|b| 1u32 << b).chain([0, 0xc0, 0x10 | 0x40, 0x8 | 0xc0]).collect();
    for (flags, format) in kinds {
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 4,
            Height: 4,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Flags: flags,
            ..Default::default()
        };
        let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
        let mut texture: Option<ID3D12Resource> = None;
        // SAFETY: a valid description and output.
        unsafe {
            host.device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut texture)
                .unwrap();
        }
        let texture = texture.unwrap();
        let copy = crate::resource::HostTexture::new(texture.clone(), 0);
        assert_eq!(copy.flags, flags.0 as u32);
        assert_eq!(copy.title_written(), crate::texture::title_written(&texture), "flags {:#x}", flags.0);
        for &state in &states {
            assert_eq!(
                copy.legal(state),
                crate::host::legal_state(&texture, state),
                "flags {:#x}, state {state:#x}",
                flags.0
            );
        }
    }
}

/// A draw's read takes the texture's host copy once for the upload and the
/// state: the texture moves as a read by resource does, its upload is
/// recorded once in the list, and a texture the host wrote is not uploaded.
#[test]
fn a_texture_read_uploads_once_and_moves_as_a_read_by_resource() {
    let Some(host) = crate::host::host() else {
        return; // no host GPU here
    };
    let address = title_memory(2 << 16);
    let textures: Vec<This> = (0..2).map(|k| texture_at(address + (k << 16))).collect();
    crate::texture::host_written(textures[1]);
    let list = host.new_list().unwrap();
    let read = |k: usize| readable(&list, ResourceType::Srv, &View::Shader(textures[k] as usize, None));
    read(0);
    // The first use is the list's prelude's: no move recorded, the
    // texture is in the read state from there on.
    assert!(list.transition(textures[0], SHADER_RESOURCE).is_none(), "already a shader resource");
    assert_eq!(list.transition(textures[0], COPY_SOURCE).map(|(_, before)| before), Some(SHADER_RESOURCE));
    read(0);
    assert_eq!(
        list.transition(textures[0], COPY_SOURCE).map(|(_, before)| before),
        Some(SHADER_RESOURCE),
        "the read moved it back from the copy source state"
    );
    read(0);
    read(0);
    assert!(list.transition(textures[0], SHADER_RESOURCE).is_none(), "the last read left it a shader resource");
    assert!(!list.first_use(textures[0] as usize), "the title-filled texture's upload is recorded in the list");
    read(1);
    assert!(list.first_use(textures[1] as usize), "a texture the host wrote is not uploaded");
    assert!(list.transition(textures[1], SHADER_RESOURCE).is_none(), "the read still moved it");
    list.close();
    type Release = extern "system" fn(This) -> u32;
    for t in textures {
        let _ = slot::<Release>(t, 0x10)(t);
    }
}
