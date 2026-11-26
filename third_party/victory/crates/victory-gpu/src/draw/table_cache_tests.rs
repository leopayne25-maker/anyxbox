use super::*;
use crate::object::{Iid, S_OK, guid, slot};
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::core::s;

const HEAP_IID: Iid = guid(
    0x8efb471d,
    0x616c,
    0x4f49,
    [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
);
const RESOURCE_IID: Iid = guid(
    0x696442be,
    0xa72e,
    0x4059,
    [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
);

#[repr(C)]
struct ExtendedHeapDesc {
    kind: u32,
    count: u32,
    flags: u32,
    nodes: u32,
    address: u64,
    size: u64,
}

#[repr(C)]
struct HeapProps {
    kind: u32,
    cpu: u32,
    pool: u32,
    creation: u32,
    visible: u32,
}

#[repr(C)]
struct TitleResourceDesc {
    dimension: u32,
    alignment: u64,
    width: u64,
    height: u32,
    depth_or_array_size: u16,
    mip_levels: u16,
    format: u32,
    sample_count: u32,
    sample_quality: u32,
    layout: u32,
    flags: u32,
}

fn release_obj(obj: This) {
    slot::<extern "system" fn(This) -> u32>(obj, 0x10)(obj);
}

fn make_view_heap(count: u32) -> (This, u64) {
    let desc = ExtendedHeapDesc {
        kind: 0,
        count,
        flags: 1,
        nodes: 1,
        address: 0,
        size: 0,
    };
    let mut h = std::ptr::null();
    let hr = crate::descriptor::create_heap(
        std::ptr::null(),
        (&raw const desc).cast(),
        &HEAP_IID,
        &mut h,
    );
    assert_eq!(hr, S_OK);
    let start = crate::descriptor::heap(h).start;
    (h, start)
}

fn make_buffer(bytes: u64, flags: u32) -> (This, *mut u8) {
    let props = HeapProps {
        kind: 1,
        cpu: 0,
        pool: 0,
        creation: 1,
        visible: 1,
    };
    let desc = TitleResourceDesc {
        dimension: 1,
        alignment: 0,
        width: bytes,
        height: 1,
        depth_or_array_size: 1,
        mip_levels: 1,
        format: 0,
        sample_count: 1,
        sample_quality: 0,
        layout: 1,
        flags,
    };
    let mut r = std::ptr::null();
    let hr = crate::resource::create_committed(
        std::ptr::null(),
        (&raw const props).cast(),
        0,
        (&raw const desc).cast(),
        0,
        std::ptr::null(),
        &RESOURCE_IID,
        &mut r,
    );
    assert_eq!(hr, S_OK);
    let (ptr, len) = crate::resource::buffer_bytes(r).expect("buffer bytes");
    assert!(len >= bytes);
    (r, ptr)
}

fn make_rt_texture(width: u64, height: u32) -> This {
    let props = HeapProps {
        kind: 1,
        cpu: 0,
        pool: 0,
        creation: 1,
        visible: 1,
    };
    let desc = TitleResourceDesc {
        dimension: 3,
        alignment: 0,
        width,
        height,
        depth_or_array_size: 1,
        mip_levels: 1,
        format: 28, // R8G8B8A8_UNORM
        sample_count: 1,
        sample_quality: 0,
        layout: 0,
        flags: 1, // ALLOW_RENDER_TARGET
    };
    let mut r = std::ptr::null();
    let hr = crate::resource::create_committed(
        std::ptr::null(),
        (&raw const props).cast(),
        0,
        (&raw const desc).cast(),
        0x4, // RENDER_TARGET
        std::ptr::null(),
        &RESOURCE_IID,
        &mut r,
    );
    assert_eq!(hr, S_OK);
    r
}

fn tex2d_srv(resource: This, handle: u64) {
    // Format=R8G8B8A8_UNORM(28), ViewDimension=TEXTURE2D(4), Mapping=5768, pad,
    // MostDetailedMip=0, MipLevels=1, PlaneSlice=0, MinLOD=0.
    let desc = [28u32, 4, 5768, 0, 0, 1, 0, 0, 0, 0];
    crate::descriptor::create_srv(std::ptr::null(), resource, &desc, handle);
}

fn cbv(addr: u64, size: u32, handle: u64) {
    let desc = [addr as u32, (addr >> 32) as u32, size, 0];
    crate::descriptor::create_cbv(std::ptr::null(), &desc, handle);
}

fn used_entry(slot: u8, table_offset_dwords: u32, kind: ResourceType) -> UsedResource {
    UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride: 0,
        dwords: 8,
        user_data: 0,
        slot,
    }
}

fn compile_cs(device: &ID3D12Device, source: &str) -> (ID3D12RootSignature, ID3D12PipelineState) {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: source and entry strings outlive D3DCompile.
    unsafe {
        if let Err(e) = D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            s!("draw_test"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        ) {
            let msg = errors.map_or_else(String::new, |b| {
                String::from_utf8_lossy(std::slice::from_raw_parts(
                    b.GetBufferPointer().cast::<u8>(),
                    b.GetBufferSize(),
                ))
                .into_owned()
            });
            panic!("D3DCompile failed: {e}: {msg}");
        }
        let code = code.unwrap();
        let bytes =
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes).unwrap();
        let pso: ID3D12PipelineState = device
            .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            })
            .unwrap();
        (root, pso)
    }
}

fn readback_u32(device: &ID3D12Device) -> (ID3D12Resource, *mut u32) {
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_CUSTOM,
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
        MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: 256,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let mut buf: Option<ID3D12Resource> = None;
    let mut ptr = std::ptr::null_mut();
    // SAFETY: valid buffer creation and persistent map.
    unsafe {
        device
            .CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut buf,
            )
            .unwrap();
        let buf = buf.unwrap();
        buf.Map(0, None, Some(&mut ptr)).unwrap();
        ptr.cast::<u32>().write_volatile(0);
        (buf, ptr.cast())
    }
}

#[test]
fn ready_tables_refresh_a_second_root_slot_using_the_same_table() {
    let Some(host) = crate::host::host() else {
        return;
    };
    let _ = host.log_messages();
    let (heap_obj, heap_start) = make_view_heap(64);
    let (cb_res, cb_ptr) = make_buffer(256, 0);
    let tex_a = make_rt_texture(4, 4);
    let tex_b = make_rt_texture(4, 4);

    // Slot 0: CBV, Slot 1: tex_a SRV, Slot 2: tex_b SRV.
    cbv(cb_ptr as u64, 256, heap_start);
    tex2d_srv(tex_a, heap_start + 32);
    tex2d_srv(tex_b, heap_start + 64);

    let list = host.new_list().unwrap();
    let host_tex_a = crate::resource::host_resource(tex_a).unwrap();
    // Clear tex_a to bright green, leaving it tracked in RENDER_TARGET (0x4).
    let rtv = host.rtv(Some(host_tex_a), None);
    crate::list::ensure_state(&list, tex_a, 0x4);
    // SAFETY: open list, fresh RTV of live RT texture.
    unsafe {
        list.list
            .ClearRenderTargetView(rtv, &[0.0, 1.0, 0.0, 1.0], None);
    }

    let mut state = State::default();
    state.set_heaps(&[heap_obj]);
    state.set_table(false, 0, heap_start);

    // Both roots use one physical table and one plan. Slot 1 reads an
    // extra SRV, so slot 0's ready entry cannot satisfy it.
    let plan = TablePlan::new(
        &[
            used_entry(0, 0, ResourceType::Cbv),
            used_entry(0, 16, ResourceType::Srv),
            used_entry(1, 0, ResourceType::Cbv),
            used_entry(1, 8, ResourceType::Srv),
            used_entry(1, 16, ResourceType::Srv),
        ],
        &[],
    );
    let _ = state.tables(&list, &plan, false);
    let _ = state.tables(&list, &plan, false);
    state.set_table(false, 1, heap_start);
    let made = state.tables(&list, &plan, false);
    let first = made[1].1 / 32;

    // Sample tex_a from dynamic slot (first + 1) on the GPU and write RGBA8 to readback.
    const SAMPLE_CS: &str = r#"
#define RS "DescriptorTable(SRV(t0)), UAV(u0)"
Texture2D<float4> src : register(t0);
RWByteAddressBuffer dst : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    float4 c = src.Load(int3(0, 0, 0));
    uint r = (uint)(c.r * 255.0 + 0.5);
    uint g = (uint)(c.g * 255.0 + 0.5);
    uint b = (uint)(c.b * 255.0 + 0.5);
    uint a = (uint)(c.a * 255.0 + 0.5);
    dst.Store(0, r | (g << 8) | (b << 16) | (a << 24));
}
"#;
    let (root, pso) = compile_cs(&host.device, SAMPLE_CS);
    let (out_buf, out_ptr) = readback_u32(&host.device);
    let host_heap = crate::descriptor::heap(heap_obj).host().unwrap();
    // SAFETY: open list, live heap/PSO/buffers.
    unsafe {
        list.list
            .SetDescriptorHeaps(&[Some(host_heap.heap.clone())]);
        list.list.SetComputeRootSignature(&root);
        list.list.SetPipelineState(&pso);
        list.list
            .SetComputeRootDescriptorTable(0, host_heap.gpu_handle(first + 1));
        list.list
            .SetComputeRootUnorderedAccessView(1, out_buf.GetGPUVirtualAddress());
        list.list.Dispatch(1, 1, 1);
    }
    // Verify tex_a was transitioned out of RENDER_TARGET (0x4) to SHADER_RESOURCE (0xc0).
    let after_state = list.transition(tex_a, SHADER_RESOURCE);
    assert!(
        after_state.is_none(),
        "tex_a should already be in SHADER_RESOURCE after root slot 1 bound it, but was left in {:#x}",
        after_state.unwrap().1
    );

    list.close();
    host.execute(&[&list]);
    if let Some(err) = host.log_messages() {
        panic!("D3D12 debug error: {err}");
    }
    // SAFETY: GPU finished executing `list`.
    let pixel = unsafe { out_ptr.read_volatile() };
    assert_eq!(
        pixel, 0xff00_ff00,
        "expected cleared green texel (0,255,0,255)"
    );

    drop(state);
    drop(list);
    release_obj(tex_a);
    release_obj(tex_b);
    release_obj(cb_res);
    release_obj(heap_obj);
}
