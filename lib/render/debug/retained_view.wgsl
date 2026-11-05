//#import aurora/retained

const VIEW_HEADER_WORDS: u32 = 64u;
const VIEW_INSTANCE_WORDS: u32 = 4u;
const VIEW_PROJ: u32 = 0u;
const VIEW_ROTATION: u32 = 16u;
const VIEW_MODE: u32 = 28u;
const VIEW_FLAGS: u32 = 29u;
const VIEW_MARKER: u32 = 32u;
const VIEW_TARGET: u32 = 44u;
const VIEW_ORTHO: u32 = 1u;
const VIEW_MODE_WIREFRAME: u32 = 0u;
const VIEW_MODE_CLUSTERS: u32 = 1u;

@group(0) @binding(0) var<storage, read> vertices: array<u32>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read> instances: array<u32>;
@group(0) @binding(3) var<storage, read> frame_words: array<u32>;
@group(0) @binding(4) var<storage, read> clusters: array<u32>;
@group(0) @binding(5) var<storage, read> view_data: array<u32>;
@group(0) @binding(6) var<storage, read> view_pairs: array<u32>;
@group(0) @binding(7) var<storage, read> materials: array<u32>;
@group(0) @binding(8) var atlas_page: texture_2d<f32>;
@group(0) @binding(9) var atlas_sampler: sampler;

struct ViewVarying {
  @builtin(position) position: vec4f,
  @location(0) rel: vec3f,
  @location(1) bary: vec3f,
  @location(2) @interpolate(flat) color: vec3f,
  @location(3) @interpolate(flat) style: vec3f,
}

fn header_vec4(word: u32) -> vec4f {
  return vec4f(bitcast<f32>(view_data[word]), bitcast<f32>(view_data[word + 1u]), bitcast<f32>(view_data[word + 2u]),
               bitcast<f32>(view_data[word + 3u]));
}

fn project(rel: vec3f) -> vec4f {
  let rotation = mat3x4f(header_vec4(VIEW_ROTATION), header_vec4(VIEW_ROTATION + 4u), header_vec4(VIEW_ROTATION + 8u));
  let proj = mat4x4f(header_vec4(VIEW_PROJ), header_vec4(VIEW_PROJ + 4u), header_vec4(VIEW_PROJ + 8u),
                     header_vec4(VIEW_PROJ + 12u));
  return vec4f(vec4f(rel, 1.0) * rotation, 1.0) * proj;
}

fn hidden() -> ViewVarying {
  var out: ViewVarying;
  out.position = vec4f(2.0, 2.0, 2.0, 1.0);
  out.rel = vec3f(0.0);
  out.bary = vec3f(0.0);
  out.color = vec3f(0.0);
  out.style = vec3f(0.0);
  return out;
}

fn ortho_flag() -> f32 {
  return select(0.0, 1.0, (view_data[VIEW_FLAGS] & VIEW_ORTHO) != 0u);
}

fn emit(slot: u32, vertex: u32, corner: u32, color: vec3f, fill: f32, edge: f32) -> ViewVarying {
  let base = vertex * RETAINED_VERTEX_WORDS;
  let pos = vec3f(bitcast<f32>(vertices[base]), bitcast<f32>(vertices[base + 1u]), bitcast<f32>(vertices[base + 2u]));
  let hi = retained_vec4(&frame_words, RETAINED_FRAME_CAM_HI).xyz;
  let lo = retained_vec4(&frame_words, RETAINED_FRAME_CAM_LO).xyz;
  let rel = vec4f(pos, 1.0) * retained_world_rel(retained_world(&instances, slot), hi, lo);
  var out: ViewVarying;
  out.position = project(rel);
  out.rel = rel;
  out.bary = vec3f(select(0.0, 1.0, corner == 0u), select(0.0, 1.0, corner == 1u), select(0.0, 1.0, corner == 2u));
  out.color = color;
  out.style = vec3f(fill, edge, ortho_flag());
  return out;
}

fn instance_valid(slot: u32) -> bool {
  return (retained_record_word(&instances, slot, RETAINED_REC_FLAGS) & RETAINED_VALID) != 0u;
}

@vertex
fn vs_instance(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> ViewVarying {
  let base = VIEW_HEADER_WORDS + ii * VIEW_INSTANCE_WORDS;
  let slot = view_data[base];
  if (!instance_valid(slot) || vi >= retained_record_word(&instances, slot, RETAINED_REC_INDEX_COUNT)) {
    return hidden();
  }
  let local = indices[retained_record_word(&instances, slot, RETAINED_REC_INDEX_FIRST) + vi];
  if (local >= retained_record_word(&instances, slot, RETAINED_REC_VERTEX_COUNT)) {
    return hidden();
  }
  let mode = view_data[VIEW_MODE];
  var fill = 1.0;
  var edge = 0.0;
  if (mode == VIEW_MODE_WIREFRAME) {
    fill = 0.15;
    edge = 1.0;
  } else if (mode == VIEW_MODE_CLUSTERS) {
    fill = 0.4;
  }
  let color = unpack4x8unorm(view_data[base + 1u]).rgb;
  return emit(slot, retained_record_word(&instances, slot, RETAINED_REC_VERTEX_FIRST) + local, vi % 3u, color, fill,
              edge);
}

fn cluster_color(index: u32) -> vec3f {
  var x = index * 0x9E3779B9u;
  x = (x ^ (x >> 16u)) * 0x85EBCA6Bu;
  x = (x ^ (x >> 13u)) * 0xC2B2AE35u;
  x = x ^ (x >> 16u);
  let c = vec3f(f32(x & 0xFFu), f32((x >> 8u) & 0xFFu), f32((x >> 16u) & 0xFFu)) / 255.0;
  return 0.25 + 0.75 * c;
}

@vertex
fn vs_cluster(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> ViewVarying {
  let index = view_pairs[ii * 2u + 1u];
  let slot = view_data[VIEW_HEADER_WORDS + view_pairs[ii * 2u] * VIEW_INSTANCE_WORDS];
  let range = retained_cluster_range(&instances, slot);
  if (!instance_valid(slot) || index < range.x || index - range.x >= range.y) {
    return hidden();
  }
  let c = retained_cluster(&clusters, index);
  let t = vi / 3u;
  if (t >= c.triangle_count) {
    return hidden();
  }
  let tri = retained_cluster_triangle(&clusters, c, t);
  let local = retained_cluster_vertex(&clusters, c, tri[vi % 3u]);
  if (local >= retained_record_word(&instances, slot, RETAINED_REC_VERTEX_COUNT)) {
    return hidden();
  }
  return emit(slot, retained_record_word(&instances, slot, RETAINED_REC_VERTEX_FIRST) + local, vi % 3u,
              cluster_color(index), 1.0, 0.3);
}

@fragment
fn fs_view(in: ViewVarying) -> @location(0) vec4f {
  let n = normalize(cross(dpdx(in.rel), dpdy(in.rel)));
  let to_eye = select(-normalize(in.rel), vec3f(0.0, 1.0, 0.0), in.style.z > 0.5);
  let shade = 0.35 + 0.65 * abs(dot(n, to_eye));
  let b = smoothstep(vec3f(0.0), fwidth(in.bary) * 1.5, in.bary);
  let line = 1.0 - min(b.x, min(b.y, b.z));
  return vec4f(mix(in.color * in.style.x * shade, in.color, line * in.style.y), 1.0);
}

struct MarkerVarying {
  @builtin(position) position: vec4f,
}

@vertex
fn vs_marker(@builtin(vertex_index) vi: u32) -> MarkerVarying {
  var out: MarkerVarying;
  out.position = project(header_vec4(VIEW_MARKER + min(vi, 2u) * 4u).xyz);
  return out;
}

@fragment
fn fs_marker() -> @location(0) vec4f {
  return vec4f(1.0);
}

struct PageVarying {
  @builtin(position) position: vec4f,
  @location(0) @interpolate(flat) target_size: vec2f,
}

@vertex
fn vs_page(@builtin(vertex_index) vi: u32) -> PageVarying {
  var out: PageVarying;
  let uv = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  out.position = vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
  out.target_size = vec2f(f32(view_data[VIEW_TARGET]), f32(view_data[VIEW_TARGET + 1u]));
  return out;
}

@fragment
fn fs_page(in: PageVarying) -> @location(0) vec4f {
  let side = min(in.target_size.x, in.target_size.y);
  let uv = (in.position.xy - (in.target_size - side) * 0.5) / side;
  let t = uv * f32(RETAINED_ATLAS_TILES_PER_ROW);
  let border = any(fract(t) < fwidth(t));
  let page = textureSampleLevel(atlas_page, atlas_sampler, saturate(uv), 0.0).rgb;
  let inside = all(uv >= vec2f(0.0)) && all(uv <= vec2f(1.0));
  return vec4f(select(vec3f(0.0), select(page, vec3f(0.25), border), inside), 1.0);
}

struct AlbedoVarying {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
  @location(1) alpha_uv: vec2f,
  @location(2) @interpolate(flat) material: u32,
  @location(3) @interpolate(flat) const_color: vec4f,
  @location(4) @interpolate(flat) flags: u32,
}

fn albedo_hidden() -> AlbedoVarying {
  var out: AlbedoVarying;
  out.position = vec4f(2.0, 2.0, 2.0, 1.0);
  out.uv = vec2f(0.0);
  out.alpha_uv = vec2f(0.0);
  out.material = RETAINED_NO_MATERIAL;
  out.const_color = vec4f(0.0);
  out.flags = 0u;
  return out;
}

@vertex
fn vs_albedo(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> AlbedoVarying {
  let slot = view_data[VIEW_HEADER_WORDS + ii * VIEW_INSTANCE_WORDS];
  if (!instance_valid(slot) || vi >= retained_record_word(&instances, slot, RETAINED_REC_INDEX_COUNT)) {
    return albedo_hidden();
  }
  let local = indices[retained_record_word(&instances, slot, RETAINED_REC_INDEX_FIRST) + vi];
  if (local >= retained_record_word(&instances, slot, RETAINED_REC_VERTEX_COUNT)) {
    return albedo_hidden();
  }
  let base = (retained_record_word(&instances, slot, RETAINED_REC_VERTEX_FIRST) + local) * RETAINED_VERTEX_WORDS;
  var raw: array<u32, 8>;
  for (var k = 0u; k < 8u; k++) {
    raw[k] = vertices[base + k];
  }
  let v = retained_unpack_vertex(raw);
  let flags = retained_record_word(&instances, slot, RETAINED_REC_FLAGS);
  let hi = retained_vec4(&frame_words, RETAINED_FRAME_CAM_HI).xyz;
  let lo = retained_vec4(&frame_words, RETAINED_FRAME_CAM_LO).xyz;
  let rel = vec4f(v.pos, 1.0) * retained_world_rel(retained_world(&instances, slot), hi, lo);
  var out: AlbedoVarying;
  out.position = project(rel);
  out.uv = retained_uv(retained_uv_xform(&instances, slot, false), v.uv0);
  let alpha_uv = retained_uv(retained_uv_xform(&instances, slot, true), retained_alpha_uv(v, flags));
  out.alpha_uv = select(alpha_uv, out.uv, (flags & RETAINED_ALPHA_UV_IS_UV0) != 0u);
  out.material = retained_record_word(&instances, slot, RETAINED_REC_MATERIAL) & 0xFFFFu;
  out.const_color = retained_draw_attributes(&instances, slot).const_color;
  out.flags = flags;
  return out;
}

@fragment
fn fs_albedo(in: AlbedoVarying) -> @location(0) vec4f {
  if (in.material == RETAINED_NO_MATERIAL) {
    return vec4f(0.5, 0.5, 0.5, 1.0);
  }
  let m = retained_material(&materials, in.material);
  let albedo_blocked = RETAINED_MAT_ALBEDO_NO_UV | RETAINED_MAT_ALBEDO_PENDING | RETAINED_MAT_ALBEDO_MISSING;
  var color = in.const_color.rgb;
  if (m.albedo_tile != RETAINED_NO_TILE && (m.flags & albedo_blocked) == 0u && (in.flags & RETAINED_UV_NON_TEX) == 0u) {
    let uv = atlas_uv(m.albedo_tile, m.albedo_wrap.x, m.albedo_wrap.y, in.uv);
    color = textureSampleLevel(atlas_page, atlas_sampler, uv, 0.0).rgb;
  }
  let alpha_blocked = RETAINED_MAT_ALPHA_NO_UV | RETAINED_MAT_ALPHA_PENDING | RETAINED_MAT_ALPHA_MISSING;
  if (m.alpha_tile != RETAINED_NO_TILE && (m.flags & alpha_blocked) == 0u) {
    let uv = atlas_uv(m.alpha_tile, m.alpha_wrap.x, m.alpha_wrap.y, in.alpha_uv);
    let texel = textureSampleLevel(atlas_page, atlas_sampler, uv, 0.0);
    if (!retained_alpha_pass(m, texel[m.alpha_channel])) {
      discard;
    }
  }
  return vec4f(color, 1.0);
}
