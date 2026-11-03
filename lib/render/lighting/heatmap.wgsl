//#import aurora/lighting

@group(0) @binding(1) var<uniform> heatmap_params: LightClusterParams;
@group(1) @binding(0) var heatmap_depth: texture_2d<f32>;
@group(1) @binding(1) var<storage, read> light_list: LightList;
@group(1) @binding(2) var<storage, read> light_grid: array<u32>;
@group(1) @binding(3) var heatmap_out: texture_storage_2d<rgba8unorm, write>;

fn heatmap_ramp(count: u32) -> vec3f {
  if (count == 0u) {
    return vec3f(40.0, 40.0, 40.0) / 255.0;
  }
  if (count <= 2u) {
    return vec3f(30.0, 90.0, 255.0) / 255.0;
  }
  if (count <= 4u) {
    return vec3f(40.0, 200.0, 60.0) / 255.0;
  }
  if (count <= 8u) {
    return vec3f(255.0, 230.0, 40.0) / 255.0;
  }
  if (count <= 16u) {
    return vec3f(255.0, 140.0, 20.0) / 255.0;
  }
  if (count < 32u) {
    return vec3f(255.0, 80.0, 20.0) / 255.0;
  }
  return vec3f(230.0, 20.0, 20.0) / 255.0;
}

fn heatmap_border(p: u32, size: u32, tiles: u32) -> bool {
  return p > 0u && light_tile(p, size, tiles) != light_tile(p - 1u, size, tiles);
}

@compute @workgroup_size(8, 8)
fn heatmap(@builtin(global_invocation_id) id: vec3u) {
  let dims = textureDimensions(heatmap_out);
  if (id.x >= dims.x || id.y >= dims.y) {
    return;
  }
  let d = textureLoad(heatmap_depth, min(id.xy, textureDimensions(heatmap_depth) - 1u), 0).r;
  let h = light_list.header;
  let counts = light_counts(h);
  let base = light_cluster(id.xy, dims, camera_view_depth(heatmap_params.camera, d), h) * LIGHT_MASK_WORDS;
  var count = 0u;
  for (var w = 0u; w < counts.words; w++) {
    let valid = select(0xffffffffu, (1u << (counts.locals % 32u)) - 1u, w == counts.locals / 32u);
    count += countOneBits(light_grid[base + w] & valid);
  }
  var rgb = heatmap_ramp(count);
  if (heatmap_border(id.x, dims.x, LIGHT_TILES_X) || heatmap_border(id.y, dims.y, LIGHT_TILES_Y)) {
    rgb *= 0.5;
  }
  if (d <= 0.0) {
    rgb *= 0.35;
  }
  textureStore(heatmap_out, id.xy, vec4f(rgb, f32(min(count, 255u)) / 255.0));
}
