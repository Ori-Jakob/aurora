//#import aurora/retained

const VIEW_HEADER_WORDS: u32 = 64u;
const VIEW_INSTANCE_WORDS: u32 = 4u;
const VIEW_PROJ: u32 = 0u;
const VIEW_ROTATION: u32 = 16u;
const VIEW_MODE: u32 = 28u;
const VIEW_FLAGS: u32 = 29u;
const VIEW_MARKER: u32 = 32u;
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
