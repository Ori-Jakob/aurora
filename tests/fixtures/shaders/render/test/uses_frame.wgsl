//#import aurora/frame

@group(1) @binding(0) var<storage, read_write> out_index: array<u32>;

@compute @workgroup_size(1)
fn main() {
  out_index[0] = aurora_frame.frame_index;
}
