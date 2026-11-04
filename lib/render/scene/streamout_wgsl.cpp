#include "streamout.hpp"

#include "../../gx/vertex_fetch.hpp"

#include <fmt/format.h>

#include <string>

namespace aurora::render::scene::streamout {
namespace {
constexpr std::string_view Prefix = "scene/streamout_variant_";

std::string tex_name(uint8_t tex) { return fmt::format("in_tex{}_uv", tex); }
} // namespace

std::string_view variant_shader_prefix() noexcept { return Prefix; }

std::string variant_source(std::span<const gx::AttrConfig, gx::MaxVtxAttr> attrs, uint8_t stride, uint8_t uvTex,
                           uint8_t alphaTex) {
  gx::ShaderConfig config{};
  config.vtxStride = stride;
  config.attrs[GX_VA_POS] = attrs[GX_VA_POS];
  config.attrs[GX_VA_NRM] = attrs[GX_VA_NRM];
  config.attrs[GX_VA_CLR0] = attrs[GX_VA_CLR0];
  const bool hasNormal = attrs[GX_VA_NRM].attrType != GX_NONE;
  const bool hasUv = uvTex < 8 && attrs[GX_VA_TEX0 + uvTex].attrType != GX_NONE;
  const bool hasAlpha = alphaTex < 8 && attrs[GX_VA_TEX0 + alphaTex].attrType != GX_NONE;
  if (hasUv) {
    config.attrs[GX_VA_TEX0 + uvTex] = attrs[GX_VA_TEX0 + uvTex];
  }
  if (hasAlpha) {
    config.attrs[GX_VA_TEX0 + alphaTex] = attrs[GX_VA_TEX0 + alphaTex];
  }

  std::string loads = fmt::format("\n    let in_pos = {};", gx::vertex_fetch::attr_load(config, GX_VA_POS, "vidx"));
  std::string body;
  if (hasNormal) {
    loads += fmt::format("\n    let in_nrm = {};", gx::vertex_fetch::attr_load(config, GX_VA_NRM, "vidx"));
    body += fmt::format(
        "\n    if ((pflags & {}u) != 0u) {{"
        "\n      let nlen = length(in_nrm);"
        "\n      nrm_word = oct_encode(select(vec3f(0.0, 0.0, 1.0), in_nrm / nlen, nlen > 0.0));"
        "\n      aux = aux | {}u;"
        "\n    }}",
        block::PieceHasNormal, vertex::HasNormal);
  }
  if (hasUv) {
    loads += fmt::format("\n    let {} = {};", tex_name(uvTex),
                         gx::vertex_fetch::attr_load(config, GXAttr(GX_VA_TEX0 + uvTex), "vidx"));
    body += fmt::format("\n    uv0 = {};", tex_name(uvTex));
  }
  if (hasAlpha) {
    if (alphaTex != uvTex) {
      loads += fmt::format("\n    let {} = {};", tex_name(alphaTex),
                           gx::vertex_fetch::attr_load(config, GXAttr(GX_VA_TEX0 + alphaTex), "vidx"));
    }
    body += fmt::format(
        "\n    if ((pflags & {0}u) != 0u && (pflags & {1}u) == 0u) {{"
        "\n      let src = {2};"
        "\n      let fixed = clamp(round(src * {3:.1f}), vec2f(-32767.0), vec2f(32767.0));"
        "\n      alpha_word = (u32(i32(fixed.x)) & 0xFFFFu) | (u32(i32(fixed.y)) << 16u);"
        "\n      if (any(abs(src) >= vec2f({4:.1f}))) {{"
        "\n        atomicOr(&instances[inst * {5}u + {6}u], {7}u);"
        "\n      }}"
        "\n    }}",
        block::PieceAlphaUv, block::PieceAlphaUvIsUv0, tex_name(alphaTex), vertex::AlphaUvScale, vertex::AlphaUvLimit,
        record::Words, record::GpuFlags, gpu_flag::AlphaUvClamped);
  }

  return fmt::format(R"""(const STREAMOUT_REVISION: u32 = {0}u;

{1}

{2}

@group(0) @binding({3}) var<storage, read> vbuf: array<u32>;
@group(0) @binding({4}) var<storage, read> abuf: array<u32>;
@group(0) @binding({5}) var<storage, read_write> vertices: array<u32>;
@group(0) @binding({6}) var<storage, read_write> instances: array<atomic<u32>>;

struct StreamPush {{
  block_word: u32,
  first: u32,
  count: u32,
  value: u32,
}}
var<immediate> push: StreamPush;

var<workgroup> wg_min: array<atomic<u32>, 3>;
var<workgroup> wg_max: array<atomic<u32>, 3>;

fn ordered_from_f32(v: f32) -> u32 {{
  let b = bitcast<u32>(v);
  return select(b | 0x80000000u, ~b, (b & 0x80000000u) != 0u);
}}

fn oct_encode(n: vec3f) -> u32 {{
  let p = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));
  let s = select(vec2f(-1.0), vec2f(1.0), p >= vec2f(0.0));
  return pack2x16snorm(select(p, (vec2f(1.0) - abs(p.yx)) * s, n.z < 0.0));
}}

@compute @workgroup_size({7})
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {{
  if (li < 3u) {{
    atomicStore(&wg_min[li], {8}u);
    atomicStore(&wg_max[li], {9}u);
  }}
  workgroupBarrier();
  let hw = push.block_word;
  let chunk = hw + abuf[hw + {10}u] + (push.first + wid.x) * {11}u;
  let pw = hw + abuf[hw + {12}u] + abuf[chunk] * {13}u;
  let imm = Immediate(abuf[pw], abuf[pw + 1u], abuf[pw + 2u], abuf[pw + 3u],
                      vec4u(abuf[pw + 4u], abuf[pw + 5u], abuf[pw + 6u], abuf[pw + 7u]),
                      vec4u(abuf[pw + 8u], abuf[pw + 9u], abuf[pw + 10u], abuf[pw + 11u]),
                      vec4u(abuf[pw + 12u], abuf[pw + 13u], abuf[pw + 14u], abuf[pw + 15u]));
  let inst = abuf[pw + {14}u];
  let pflags = abuf[pw + {15}u];
  let vidx = abuf[chunk + 1u] + li;
  // Every thread reaches the barrier, so out-of-range lanes add an empty box instead of returning
  if (vidx < abuf[pw + {16}u]) {{{17}
    var aux = 0u;
    var nrm_word = 0u;
    var uv0 = vec2f(0.0);
    var alpha_word = 0u;{18}
    let o = (abuf[pw + {19}u] + vidx) * {20}u;
    vertices[o] = bitcast<u32>(in_pos.x);
    vertices[o + 1u] = bitcast<u32>(in_pos.y);
    vertices[o + 2u] = bitcast<u32>(in_pos.z);
    vertices[o + 3u] = nrm_word;
    vertices[o + 4u] = bitcast<u32>(uv0.x);
    vertices[o + 5u] = bitcast<u32>(uv0.y);
    vertices[o + 6u] = alpha_word;
    vertices[o + 7u] = aux;
    atomicMin(&wg_min[0], ordered_from_f32(in_pos.x));
    atomicMin(&wg_min[1], ordered_from_f32(in_pos.y));
    atomicMin(&wg_min[2], ordered_from_f32(in_pos.z));
    atomicMax(&wg_max[0], ordered_from_f32(in_pos.x));
    atomicMax(&wg_max[1], ordered_from_f32(in_pos.y));
    atomicMax(&wg_max[2], ordered_from_f32(in_pos.z));
  }}
  workgroupBarrier();
  if (li == 0u) {{
    let lo = vec3u(atomicLoad(&wg_min[0]), atomicLoad(&wg_min[1]), atomicLoad(&wg_min[2]));
    let hi = vec3u(atomicLoad(&wg_max[0]), atomicLoad(&wg_max[1]), atomicLoad(&wg_max[2]));
    if (lo.x != {8}u) {{
      let base = inst * {21}u + {22}u;
      atomicMin(&instances[base], lo.x);
      atomicMin(&instances[base + 1u], lo.y);
      atomicMin(&instances[base + 2u], lo.z);
      atomicMax(&instances[base + 3u], hi.x);
      atomicMax(&instances[base + 4u], hi.y);
      atomicMax(&instances[base + 5u], hi.z);
    }}
  }}
}}
)""",
                     GeneratorRevision, gx::vertex_fetch::wgsl_helpers(), gx::vertex_fetch::wgsl_immediate_struct(),
                     binding::GxVertex, binding::GxStorage, binding::Vertices, binding::Instances, WorkgroupSize,
                     EmptyMin, EmptyMax, block::VtxChunksWord, block::ChunkWords, block::PiecesWord, block::PieceWords,
                     block::PieceInstance, block::PieceFlags, block::PieceVtxCount, loads, body,
                     block::PieceOutVertexFirst, vertex::Words, record::Words, record::CurAabb);
}
} // namespace aurora::render::scene::streamout
