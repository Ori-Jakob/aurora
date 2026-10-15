//#import test/pattern_common

override scale: u32 = 1u;

@group(0) @binding(0) var<storage, read_write> out_values: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= arrayLength(&out_values)) {
    return;
  }
  //#if FILL_SQUARES
  out_values[i] = pattern_square(i) * scale + PATTERN_BIAS;
  //#else
  out_values[i] = i * scale + PATTERN_BIAS;
  //#endif
}
