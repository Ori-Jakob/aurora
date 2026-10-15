//#import test/pattern_common
//#if AURORA_F16
enable f16;
//#endif

@group(0) @binding(0) var<storage, read_write> out_value: u32;

@compute @workgroup_size(1)
fn main() {
  //#if AURORA_F16
  let half_bias = f16(PATTERN_BIAS);
  out_value = u32(half_bias);
  //#else
  out_value = PATTERN_BIAS;
  //#endif
}
