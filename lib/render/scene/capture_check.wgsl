@group(0) @binding(0) var scene_color: texture_2d<f32>;
@group(0) @binding(1) var replay_color: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> counts: array<atomic<u32>, 2>;

@compute @workgroup_size(8, 8)
fn compare(@builtin(global_invocation_id) id: vec3u) {
  let dims = textureDimensions(scene_color);
  if (any(id.xy >= dims) || any(id.xy >= textureDimensions(replay_color))) {
    return;
  }
  if (any(textureLoad(scene_color, id.xy, 0) != textureLoad(replay_color, id.xy, 0))) {
    atomicAdd(&counts[0], 1u);
    atomicMax(&counts[1], id.y * dims.x + id.x + 1u);
  }
}
