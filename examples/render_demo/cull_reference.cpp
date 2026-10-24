#include "cull_reference.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <string_view>

namespace render_demo {
namespace {
constexpr u32 PadWords = 64;
constexpr u32 MaskWordsPerGroup = CullWorkgroupSize / 32;

constexpr u32 round_up(u32 value, u32 multiple) noexcept { return (value + multiple - 1) / multiple * multiple; }

constexpr u32 cull_hash(u32 x) noexcept {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

// The same expression and order as the kernel, so exact inputs give the kernel's flags
bool inside(const Plane& plane, const CullInstance& s) noexcept {
  const f32 distance = plane[0] * s.x + plane[1] * s.y + plane[2] * s.z;
  return distance + plane[3] >= -s.radius;
}

bool kept(const CullInstance& s, std::span<const Plane, 6> planes) noexcept {
  bool keep = true;
  for (const Plane& plane : planes) {
    keep = keep && inside(plane, s);
  }
  return keep;
}

constexpr std::string_view CullComputeBody = R"(
struct CullParams {
  planes: array<vec4f, 6>,
  frame_tag: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
};
@group(0) @binding(1) var<uniform> params: CullParams;
@group(1) @binding(0) var<storage, read_write> outw: array<u32>;
@group(2) @binding(0) var<storage, read> instances: array<vec4f>;

var<workgroup> flags: array<u32, 128>;
var<workgroup> packed: array<u32, 4>;
var<workgroup> sums: array<u32, 128>;

fn inside(p: vec4f, s: vec4f) -> bool {
  return dot(p.xyz, s.xyz) + p.w >= -s.w;
}

@compute @workgroup_size(128)
fn cs_count(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) lane: u32) {
  let id = wg.x * 128u + lane;
  var flag = 0u;
  if (id < N) {
    let s = instances[id];
    flag = select(0u, 1u, inside(params.planes[0], s) && inside(params.planes[1], s) &&
                          inside(params.planes[2], s) && inside(params.planes[3], s) &&
                          inside(params.planes[4], s) && inside(params.planes[5], s));
  }
  flags[lane] = flag;
  workgroupBarrier();
  if (lane < 4u) {
    var word = 0u;
    for (var b = 0u; b < 32u; b++) {
      word |= flags[lane * 32u + b] << b;
    }
    packed[lane] = word;
    outw[MASK + wg.x * 4u + lane] = word;
  }
  workgroupBarrier();
  if (lane == 0u) {
    outw[COUNTS + wg.x] =
        countOneBits(packed[0]) + countOneBits(packed[1]) + countOneBits(packed[2]) + countOneBits(packed[3]);
  }
}

@compute @workgroup_size(128)
fn cs_scan(@builtin(local_invocation_index) lane: u32) {
  let chunk = (GROUPS + 127u) / 128u;
  let first = min(lane * chunk, GROUPS);
  let last = min(first + chunk, GROUPS);
  var sum = 0u;
  for (var i = first; i < last; i++) {
    sum += outw[COUNTS + i];
  }
  sums[lane] = sum;
  workgroupBarrier();
  if (lane == 0u) {
    var running = 0u;
    for (var i = 0u; i < 128u; i++) {
      let value = sums[i];
      sums[i] = running;
      running += value;
    }
    outw[0] = 24u;
    outw[1] = min(running, N);
    outw[2] = 0u;
    outw[3] = 0u;
    outw[4] = 0u;
    outw[5] = params.frame_tag;
    outw[6] = running;
    outw[7] = GROUPS;
  }
  workgroupBarrier();
  var offset = sums[lane];
  for (var i = first; i < last; i++) {
    let count = outw[COUNTS + i];
    outw[COUNTS + i] = offset;
    offset += count;
  }
}

@compute @workgroup_size(128)
fn cs_scatter(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) lane: u32) {
  let id = wg.x * 128u + lane;
  if (id >= N) {
    return;
  }
  let w = lane / 32u;
  let bit = 1u << (lane % 32u);
  let word = outw[MASK + wg.x * 4u + w];
  if ((word & bit) == 0u) {
    return;
  }
  var dst = outw[COUNTS + wg.x];
  for (var i = 0u; i < w; i++) {
    dst += countOneBits(outw[MASK + wg.x * 4u + i]);
  }
  dst += countOneBits(word & (bit - 1u));
  if (dst < N) {
    outw[IDS + dst] = id;
  }
}
)";

constexpr std::string_view CullDrawBody = R"(
struct CullDrawParams {
  view_proj: mat4x4f,
};
@group(0) @binding(1) var<uniform> params: CullDrawParams;
@group(2) @binding(0) var<storage, read> instances: array<vec4f>;

struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) color: vec4f,
};

@vertex fn vs_main(@location(0) offset: vec3f, @location(1) id: u32) -> VertexOut {
  let s = instances[min(id, N - 1u)];
  var h = id * 0x9e3779b9u;
  h ^= h >> 15u;
  h *= 0x2c1b3c6du;
  h ^= h >> 12u;
  var out: VertexOut;
  out.position = params.view_proj * vec4f(s.xyz + s.w * offset, 1.0);
  out.color = vec4f(vec3f(f32(h & 255u), f32((h >> 8u) & 255u), f32((h >> 16u) & 255u)) / 255.0, 1.0);
  return out;
}

@fragment fn fs_main(input: VertexOut) -> @location(0) vec4f {
  return input.color;
}
)";
} // namespace

u64 CullLayout::bytes() const noexcept { return static_cast<u64>(totalWords) * 4; }

u64 CullLayout::ids_offset() const noexcept { return static_cast<u64>(idsWord) * 4; }

CullLayout cull_layout(u32 instances) noexcept {
  CullLayout layout{.instances = instances, .groups = (instances + CullWorkgroupSize - 1) / CullWorkgroupSize};
  layout.countsWord = HeaderWords;
  layout.maskWord = layout.countsWord + round_up(layout.groups, PadWords);
  layout.idsWord = layout.maskWord + round_up(layout.groups * MaskWordsPerGroup, PadWords);
  layout.totalWords = layout.idsWord + instances;
  return layout;
}

CullInstance cull_instance(u32 id) noexcept {
  const u32 hash = cull_hash(id);
  return {
      .x = static_cast<f32>(static_cast<s32>(id % 1024) - 512),
      .y = static_cast<f32>(hash & 7) * 0.25f,
      .z = static_cast<f32>(static_cast<s32>(id / 1024) - 512),
      .radius = 0.125f + static_cast<f32>((hash >> 3) & 3) * 0.0625f,
  };
}

void fill_cull_instances(std::span<CullInstance> out) noexcept {
  for (size_t i = 0; i < out.size(); ++i) {
    out[i] = cull_instance(static_cast<u32>(i));
  }
}

std::array<f32, 16> view_projection(const f32 (&proj)[4][4], const f32 (&view)[3][4], bool reversedZ) noexcept {
  f32 p[4][4];
  for (u32 r = 0; r < 4; ++r) {
    for (u32 c = 0; c < 4; ++c) {
      p[r][c] = proj[r][c];
    }
  }
  for (u32 c = 0; c < 4; ++c) {
    p[2][c] = reversedZ ? -proj[2][c] : proj[2][c] + proj[3][c];
  }
  std::array<f32, 16> out{};
  for (u32 r = 0; r < 4; ++r) {
    for (u32 c = 0; c < 4; ++c) {
      f32 sum = c == 3 ? p[r][3] : 0.f;
      for (u32 k = 0; k < 3; ++k) {
        sum += p[r][k] * view[k][c];
      }
      out[c * 4 + r] = sum;
    }
  }
  return out;
}

std::array<Plane, 6> frustum_planes(const std::array<f32, 16>& viewProj) noexcept {
  const auto row = [&](u32 r) {
    return std::array<f64, 4>{viewProj[r], viewProj[4 + r], viewProj[8 + r], viewProj[12 + r]};
  };
  const auto combine = [](const std::array<f64, 4>& a, const std::array<f64, 4>& b, f64 sign) {
    std::array<f64, 4> plane{};
    for (u32 i = 0; i < 4; ++i) {
      plane[i] = a[i] + sign * b[i];
    }
    const f64 length = std::sqrt(plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2]);
    Plane out{};
    for (u32 i = 0; i < 4; ++i) {
      out[i] = static_cast<f32>(length > 0.0 ? plane[i] / length : plane[i]);
    }
    return out;
  };
  const auto r0 = row(0);
  const auto r1 = row(1);
  const auto r2 = row(2);
  const auto r3 = row(3);
  constexpr std::array<f64, 4> Zero{};
  return {combine(r3, r0, 1.0),  combine(r3, r0, -1.0),  combine(r3, r1, 1.0),
          combine(r3, r1, -1.0), combine(r2, Zero, 0.0), combine(r3, r2, -1.0)};
}

CullClass classify(const CullInstance& instance, std::span<const Plane, 6> planes) noexcept {
  bool ambiguous = false;
  for (const Plane& plane : planes) {
    const f64 terms[3] = {static_cast<f64>(plane[0]) * instance.x, static_cast<f64>(plane[1]) * instance.y,
                          static_cast<f64>(plane[2]) * instance.z};
    const f64 value = terms[0] + terms[1] + terms[2] + plane[3] + instance.radius;
    const f64 scale = std::abs(terms[0]) + std::abs(terms[1]) + std::abs(terms[2]) + std::abs(plane[3]) +
                      static_cast<f64>(instance.radius);
    if (std::abs(value) <= std::ldexp(scale, -16)) {
      ambiguous = true;
    } else if (value < 0.0) {
      return CullClass::Culled;
    }
  }
  return ambiguous ? CullClass::Ambiguous : CullClass::Visible;
}

std::vector<u32> simulate_cull(std::span<const CullInstance> instances, std::span<const Plane, 6> planes,
                               u32 frameTag) {
  const auto n = static_cast<u32>(instances.size());
  const CullLayout layout = cull_layout(n);
  std::vector<u32> words(layout.totalWords, 0);
  for (u32 group = 0; group < layout.groups; ++group) {
    u32 count = 0;
    for (u32 w = 0; w < MaskWordsPerGroup; ++w) {
      u32 word = 0;
      for (u32 b = 0; b < 32; ++b) {
        const u32 id = group * CullWorkgroupSize + w * 32 + b;
        if (id < n && kept(instances[id], planes)) {
          word |= 1u << b;
        }
      }
      words[layout.maskWord + group * MaskWordsPerGroup + w] = word;
      count += static_cast<u32>(std::popcount(word));
    }
    words[layout.countsWord + group] = count;
  }
  const u32 chunk = (layout.groups + CullWorkgroupSize - 1) / CullWorkgroupSize;
  u32 running = 0;
  for (u32 lane = 0; lane < CullWorkgroupSize; ++lane) {
    const u32 first = std::min(lane * chunk, layout.groups);
    const u32 last = std::min(first + chunk, layout.groups);
    for (u32 i = first; i < last; ++i) {
      const u32 count = words[layout.countsWord + i];
      words[layout.countsWord + i] = running;
      running += count;
    }
  }
  words[0] = CullIndexCount;
  words[1] = std::min(running, n);
  words[HeaderFrameTag] = frameTag;
  words[HeaderTotal] = running;
  words[HeaderGroups] = layout.groups;
  for (u32 id = 0; id < n; ++id) {
    const u32 group = id / CullWorkgroupSize;
    const u32 lane = id % CullWorkgroupSize;
    const u32 w = lane / 32;
    const u32 bit = 1u << (lane % 32);
    const u32 word = words[layout.maskWord + group * MaskWordsPerGroup + w];
    if ((word & bit) == 0) {
      continue;
    }
    u32 dst = words[layout.countsWord + group];
    for (u32 i = 0; i < w; ++i) {
      dst += static_cast<u32>(std::popcount(words[layout.maskWord + group * MaskWordsPerGroup + i]));
    }
    dst += static_cast<u32>(std::popcount(word & (bit - 1)));
    if (dst < n) {
      words[layout.idsWord + dst] = id;
    }
  }
  return words;
}

std::string cull_compute_wgsl(const CullLayout& layout) {
  char constants[256];
  std::snprintf(constants, sizeof(constants),
                "const N: u32 = %uu;\nconst GROUPS: u32 = %uu;\nconst COUNTS: u32 = %uu;\nconst MASK: u32 = %uu;\n"
                "const IDS: u32 = %uu;\n",
                layout.instances, layout.groups, layout.countsWord, layout.maskWord, layout.idsWord);
  return std::string{constants} + std::string{CullComputeBody};
}

std::string cull_draw_wgsl(u32 instances) {
  char constants[64];
  std::snprintf(constants, sizeof(constants), "const N: u32 = %uu;\n", instances);
  return std::string{constants} + std::string{CullDrawBody};
}

CullCheckResult check_cull_readback(std::span<const u32> words, const CullLayout& layout,
                                    std::span<const CullInstance> instances, std::span<const Plane, 6> planes,
                                    u32 expectedFrameTag) {
  CullCheckResult result;
  char text[160];
  const auto fail = [&]() {
    result.failure = std::string{text};
    return result;
  };
  const u32 n = layout.instances;
  if (words.size() < layout.totalWords || instances.size() != n) {
    std::snprintf(text, sizeof(text), "readback holds %zu words for %zu instances, expected %u and %u", words.size(),
                  instances.size(), layout.totalWords, n);
    return fail();
  }
  if (words[HeaderFrameTag] != expectedFrameTag) {
    std::snprintf(text, sizeof(text), "frame tag %u, expected %u", words[HeaderFrameTag], expectedFrameTag);
    return fail();
  }
  if (words[0] != CullIndexCount) {
    std::snprintf(text, sizeof(text), "index count %u, expected %u", words[0], CullIndexCount);
    return fail();
  }
  if (words[2] != 0 || words[3] != 0) {
    std::snprintf(text, sizeof(text), "first index %u and base vertex %u, expected 0", words[2], words[3]);
    return fail();
  }
  if (words[4] != 0) {
    std::snprintf(text, sizeof(text), "first instance %u, expected 0", words[4]);
    return fail();
  }
  const u32 total = words[HeaderTotal];
  if (total > n) {
    std::snprintf(text, sizeof(text), "total %u exceeds %u instances", total, n);
    return fail();
  }
  if (words[1] != total) {
    std::snprintf(text, sizeof(text), "instance count %u does not match total %u", words[1], total);
    return fail();
  }
  if (words[HeaderGroups] != layout.groups) {
    std::snprintf(text, sizeof(text), "groups %u, expected %u", words[HeaderGroups], layout.groups);
    return fail();
  }
  std::vector<CullClass> classes(n);
  for (u32 id = 0; id < n; ++id) {
    classes[id] = classify(instances[id], planes);
    result.visible += classes[id] == CullClass::Visible ? 1 : 0;
    result.ambiguous += classes[id] == CullClass::Ambiguous ? 1 : 0;
  }
  if (result.ambiguous > n / 100) {
    std::snprintf(text, sizeof(text), "ill-conditioned reference: %u of %u instances are ambiguous", result.ambiguous,
                  n);
    return fail();
  }
  const std::span<const u32> ids = words.subspan(layout.idsWord, total);
  std::vector<bool> present(n, false);
  for (u32 slot = 0; slot < total; ++slot) {
    const u32 id = ids[slot];
    if (id >= n) {
      std::snprintf(text, sizeof(text), "id %u at slot %u is out of range", id, slot);
      return fail();
    }
    if (slot != 0 && id <= ids[slot - 1]) {
      std::snprintf(text, sizeof(text), "ids are not strictly increasing at slot %u (%u after %u)", slot, id,
                    ids[slot - 1]);
      return fail();
    }
    if (classes[id] == CullClass::Culled) {
      std::snprintf(text, sizeof(text), "culled instance %u was kept", id);
      return fail();
    }
    present[id] = true;
  }
  for (u32 id = 0; id < n; ++id) {
    if (classes[id] == CullClass::Visible && !present[id]) {
      std::snprintf(text, sizeof(text), "visible instance %u is missing", id);
      return fail();
    }
  }
  return result;
}
} // namespace render_demo
