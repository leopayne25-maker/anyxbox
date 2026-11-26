//! The list's argument buffer stays readable as arguments from one
//! execution to the next clamp, and is common again at the close.

use super::*;
use crate::host::barrier_log::{issued, replay};
use windows::core::Interface;

const COMMON: u32 = 0;

/// Each group adds one to the word of its execution.
const COUNTING: &str = r#"
#define RS "RootConstants(num32BitConstants=1, b0), UAV(u0)"
cbuffer Execution : register(b0) { uint execution; };
RWByteAddressBuffer groups : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    uint before;
    groups.InterlockedAdd(execution * 4, 1, before);
}
"#;

fn dispatch_signature(device: &ID3D12Device) -> ID3D12CommandSignature {
    let argument = D3D12_INDIRECT_ARGUMENT_DESC { Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH, ..Default::default() };
    let desc = D3D12_COMMAND_SIGNATURE_DESC { ByteStride: 12, NumArgumentDescs: 1, pArgumentDescs: &argument, NodeMask: 0 };
    let mut signature: Option<ID3D12CommandSignature> = None;
    // SAFETY: one argument, no root arguments.
    unsafe { device.CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut signature) }.unwrap();
    signature.unwrap()
}

/// Indirect dispatches in one list, each with its own records, through the
/// list's one argument buffer (left readable as arguments from one
/// execution to the clamp of the next, and replaced by a bigger one while
/// readable): every execution runs the groups its own records ask for, and
/// every barrier starts from the buffer's current state, all common at the
/// close. The list runs twice, so the second run starts from what the
/// first left.
#[test]
fn each_execution_reads_its_own_clamped_records() {
    let Some(host) = host() else {
        return; // no host GPU here
    };
    let counting = make_from(&host.device, COUNTING, s!("count")).unwrap();
    let signature = dispatch_signature(&host.device);
    // 6,000 records (past the first argument buffer's 64 KiB), of which
    // only the first has groups; then the bigger buffer again.
    let big = 6_000;
    let executions: [(Vec<[u32; 3]>, u32); 4] = [
        (vec![[3, 2, 1]], 6),
        (vec![[4, 1, 1]], 4),
        (std::iter::once([5, 1, 1]).chain(std::iter::repeat_n([0, 0, 0], big - 1)).collect(), 5),
        (vec![[7, 1, 1]], 7),
    ];
    let words: usize = executions.iter().map(|(r, _)| r.len() * 3).sum();
    let (title, data) = host.shared_buffer(words as u64 * 4).unwrap();
    let (counts, counted) = host.shared_buffer(256).unwrap();
    let list = host.new_list().unwrap();
    let mut states: std::collections::HashMap<usize, u32> = std::collections::HashMap::new();
    for run in 0..2 {
        let _ = issued();
        let mut outs: Vec<ID3D12Resource> = Vec::new();
        // SAFETY: our mapped buffers, written before the list runs and read
        // after it has; an open list, our pipelines and live buffers.
        unsafe {
            counted.cast::<[u32; 64]>().write_volatile([0; 64]);
            let mut at = 0;
            for (i, (records, _)) in executions.iter().enumerate() {
                for (k, r) in records.iter().enumerate() {
                    data.cast::<[u32; 3]>().add(at / 3 + k).write_volatile(*r);
                }
                let out = bound(&list, (&title, at as u64 * 4), None, (2, 12, records.len() as u32)).unwrap();
                at += records.len() * 3;
                let l = &list.list;
                l.SetComputeRootSignature(&counting.root);
                l.SetPipelineState(&counting.pso);
                l.SetComputeRoot32BitConstants(0, 1, [i as u32].as_ptr().cast(), 0);
                l.SetComputeRootUnorderedAccessView(1, counts.GetGPUVirtualAddress());
                l.ExecuteIndirect(&signature, records.len() as u32, &out, 0, None, 0);
                if outs.last().is_none_or(|o| o.as_raw() != out.as_raw()) {
                    outs.push(out);
                }
            }
            list.close();
            for o in &outs {
                states.entry(o.as_raw() as usize).or_insert(COMMON);
            }
            let recorded = issued();
            host.execute(&[&list]);
            replay(&recorded, &mut states).unwrap_or_else(|e| panic!("run {run}: {e}"));
            assert!(states.values().all(|&s| s == COMMON), "run {run}: all common again");
            if run == 0 {
                assert_eq!(outs.len(), 2, "the argument buffer was replaced by a bigger one");
            }
            let ran = counted.cast::<[u32; 64]>().read_volatile();
            let expected: Vec<u32> = executions.iter().map(|(_, groups)| *groups).collect();
            assert_eq!(ran[..executions.len()], expected[..], "run {run}");
        }
        list.reset();
    }
}

/// The argument buffer's state as the list reports it to the clamp: common
/// at first, after a reset and for a bigger buffer; readable as arguments
/// after a clamp was recorded into it.
#[test]
fn the_argument_buffer_is_common_until_a_clamp_and_after_the_close() {
    let Some(host) = host() else {
        return; // no host GPU here
    };
    let (title, _) = host.shared_buffer(1 << 16).unwrap();
    let list = host.new_list().unwrap();
    let state = |bytes: u64| list.indirect_scratch(bytes).unwrap();
    let (first, before) = state(12);
    assert_eq!(before, D3D12_RESOURCE_STATE_COMMON);
    assert_eq!(state(12).1, D3D12_RESOURCE_STATE_COMMON, "no clamp yet");
    bound(&list, (&title, 0), None, (2, 12, 1)).unwrap();
    let (same, after) = state(12);
    assert_eq!(same.as_raw(), first.as_raw());
    assert_eq!(after, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
    let (bigger, from) = state(1 << 20);
    assert_ne!(bigger.as_raw(), first.as_raw());
    assert_eq!(from, D3D12_RESOURCE_STATE_COMMON, "a new buffer");
    bound(&list, (&title, 0), None, (2, 12, 1)).unwrap();
    list.close();
    host.execute(&[&list]);
    list.reset();
    assert_eq!(state(12).1, D3D12_RESOURCE_STATE_COMMON, "after the close and reset");
}
