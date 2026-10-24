//#import aurora/stream
@compute @workgroup_size(aurora_wg_x, aurora_wg_y, aurora_wg_z)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (!stream_active(id)) {
    return;
  }
  let i = stream_index(id);
  stream_store(i, stream_load(0u, i));
}
