@group(1) @binding(0) var hzb_source: texture_2d<f32>;
@group(1) @binding(1) var hzb_target: texture_storage_2d<r32float, write>;

// Texel u covers source [2u, 2u+1]; the last texel also takes the odd remainder
fn hzb_footprint(u: u32, out_size: u32, in_size: u32) -> vec2u {
  let last = select(2u * u + 1u, in_size - 1u, u + 1u == out_size);
  return vec2u(2u * u, min(last, in_size - 1u));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let out_size = textureDimensions(hzb_target);
  if (id.x >= out_size.x || id.y >= out_size.y) {
    return;
  }
  let in_size = textureDimensions(hzb_source, 0);
  let fx = hzb_footprint(id.x, out_size.x, in_size.x);
  let fy = hzb_footprint(id.y, out_size.y, in_size.y);
  // Reversed Z: 0 is the far plane, so an unreducible footprint occludes nothing
  var farthest = 0.0;
  if (fx.x <= fx.y && fy.x <= fy.y && fx.y - fx.x < 3u && fy.y - fy.x < 3u) {
    farthest = textureLoad(hzb_source, vec2u(fx.x, fy.x), 0).r;
    for (var y = fy.x; y <= fy.y; y++) {
      for (var x = fx.x; x <= fx.y; x++) {
        farthest = min(farthest, textureLoad(hzb_source, vec2u(x, y), 0).r);
      }
    }
  }
  textureStore(hzb_target, id.xy, vec4f(farthest, 0.0, 0.0, 0.0));
}
