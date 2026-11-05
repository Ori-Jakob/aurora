#include "vertex_fetch.hpp"

#include "../internal.hpp"
#include "gx_fmt.hpp"

#include <algorithm>
#include <bit>
#include <string>
#include <string_view>

namespace aurora::gx::vertex_fetch {
using namespace std::string_literals;
using namespace std::string_view_literals;

namespace {
constexpr Module Log{"aurora::gfx::gx"};

constexpr std::string_view HelpersWgsl = R"""(fn bswap32(v: u32, le: bool) -> u32 {
  if (le) {
    return v;
  }
  return ((v & 0x000000FFu) << 24u) |
         ((v & 0x0000FF00u) << 8u) |
         ((v & 0x00FF0000u) >> 8u) |
         ((v & 0xFF000000u) >> 24u);
}

fn bswap16(v: u32, le: bool) -> u32 {
  return select(((v & 0xFFu) << 8u) | (v >> 8u), v, le);
}

fn load_word(p: ptr<storage, array<u32>>, word_idx: u32) -> u32 {
  // This guard is not expected to handle routine out-of-bounds accesses.
  // It appears to discourage some Adreno drivers/optimizers from storage buffer
  // optimizations that can cause visual artifacts, including vertex explosions
  // in Dusklight.
  if (word_idx < arrayLength(p)) {
    return p[word_idx];
  }
  return 0u;
}

fn load_u8(p: ptr<storage, array<u32>>, byte_off: u32) -> u32 {
  let word = load_word(p, byte_off / 4u);
  let shift = (byte_off & 3u) * 8u;
  return (word >> shift) & 0xFFu;
}

fn load_u32_raw(p: ptr<storage, array<u32>>, byte_off: u32) -> u32 {
  let word_idx = byte_off >> 2u;
  let sub = byte_off & 3u;
  let lo = load_word(p, word_idx);
  if (sub == 0u) {
    return lo;
  }
  let hi = load_word(p, word_idx + 1u);
  let shift = sub * 8u;
  return (lo >> shift) | (hi << (32u - shift));
}

fn load_u16(p: ptr<storage, array<u32>>, byte_off: u32, le: bool) -> u32 {
  let word_idx = byte_off >> 2u;
  let sub = byte_off & 3u;
  let word = load_word(p, word_idx);
  if (sub <= 2u) {
    return bswap16(extractBits(word, sub * 8u, 16u), le);
  }
  let next = load_word(p, word_idx + 1u);
  let raw = extractBits(word, 24u, 8u) | (extractBits(next, 0u, 8u) << 8u);
  return bswap16(raw, le);
}

fn load_u24(p: ptr<storage, array<u32>>, byte_off: u32, le: bool) -> u32 {
  let raw = load_u32_raw(p, byte_off) & 0x00FFFFFFu;
  if (le) {
    return raw;
  }
  return ((raw & 0x0000FFu) << 16u) |
         (raw & 0x00FF00u) |
         ((raw & 0xFF0000u) >> 16u);
}

fn load_u32(p: ptr<storage, array<u32>>, byte_off: u32, le: bool) -> u32 {
  return bswap32(load_u32_raw(p, byte_off), le);
}

fn load_f32(p: ptr<storage, array<u32>>, byte_off: u32, le: bool) -> f32 {
  return bitcast<f32>(load_u32(p, byte_off, le));
}

fn raw_fetch_u8_1(p: ptr<storage, array<u32>>, byte_off: u32) -> u32 {
  return load_u8(p, byte_off);
}

fn raw_fetch_u8_2(p: ptr<storage, array<u32>>, byte_off: u32) -> vec2u {
  let word_idx = byte_off >> 2u;
  let sub = byte_off & 3u;
  let word = load_word(p, word_idx);
  if (sub <= 2u) {
    let shift = sub * 8u;
    return vec2u(
      extractBits(word, shift + 0u, 8u),
      extractBits(word, shift + 8u, 8u),
    );
  }
  let next = load_word(p, word_idx + 1u);
  return vec2u(
    extractBits(word, 24u, 8u),
    extractBits(next, 0u, 8u),
  );
}

fn raw_fetch_u8_3(p: ptr<storage, array<u32>>, byte_off: u32) -> vec3u {
  let raw = load_u32_raw(p, byte_off);
  return vec3u(
    extractBits(raw, 0u, 8u),
    extractBits(raw, 8u, 8u),
    extractBits(raw, 16u, 8u),
  );
}

fn raw_fetch_u8_4(p: ptr<storage, array<u32>>, byte_off: u32) -> vec4u {
  let raw = load_u32_raw(p, byte_off);
  return vec4u(
    extractBits(raw, 0u, 8u),
    extractBits(raw, 8u, 8u),
    extractBits(raw, 16u, 8u),
    extractBits(raw, 24u, 8u),
  );
}

fn raw_fetch_u16_1(p: ptr<storage, array<u32>>, byte_off: u32, le: bool) -> u32 {
  return load_u16(p, byte_off, le);
}

fn raw_fetch_u16_2(p: ptr<storage, array<u32>>, byte_off: u32, le: bool) -> vec2u {
  return vec2u(
    load_u16(p, byte_off + 0u, le),
    load_u16(p, byte_off + 2u, le),
  );
}

fn raw_fetch_u16_3(p: ptr<storage, array<u32>>, byte_off: u32, le: bool) -> vec3u {
  return vec3u(
    load_u16(p, byte_off + 0u, le),
    load_u16(p, byte_off + 2u, le),
    load_u16(p, byte_off + 4u, le),
  );
}

fn raw_fetch_u16_4(p: ptr<storage, array<u32>>, byte_off: u32, le: bool) -> vec4u {
  return vec4u(
    load_u16(p, byte_off + 0u, le),
    load_u16(p, byte_off + 2u, le),
    load_u16(p, byte_off + 4u, le),
    load_u16(p, byte_off + 6u, le),
  );
}

fn raw_fetch_f32_1(p: ptr<storage, array<u32>>, byte_off: u32, le: bool) -> f32 {
  return load_f32(p, byte_off, le);
}

fn raw_fetch_f32_2(p: ptr<storage, array<u32>>, byte_off: u32, le: bool) -> vec2f {
  return vec2f(
    load_f32(p, byte_off + 0u, le),
    load_f32(p, byte_off + 4u, le),
  );
}

fn raw_fetch_f32_3(p: ptr<storage, array<u32>>, byte_off: u32, le: bool) -> vec3f {
  return vec3f(
    load_f32(p, byte_off + 0u, le),
    load_f32(p, byte_off + 4u, le),
    load_f32(p, byte_off + 8u, le),
  );
}

fn raw_fetch_f32_4(p: ptr<storage, array<u32>>, byte_off: u32, le: bool) -> vec4f {
  return vec4f(
    load_f32(p, byte_off + 0u, le),
    load_f32(p, byte_off + 4u, le),
    load_f32(p, byte_off + 8u, le),
    load_f32(p, byte_off + 12u, le),
  );
}

fn fetch_u8_1(p: ptr<storage, array<u32>>, byte_off: u32, frac: u32, le: bool) -> f32 {
  let v = raw_fetch_u8_1(p, byte_off);
  return f32(v) / f32(1u << frac);
}

fn fetch_s8_1(p: ptr<storage, array<u32>>, byte_off: u32, frac: u32, le: bool) -> f32 {
  let v = (bitcast<i32>(raw_fetch_u8_1(p, byte_off)) << 24) >> 24;
  return f32(v) / f32(1u << frac);
}

fn fetch_u8_2(p: ptr<storage, array<u32>>, byte_off: u32, frac: u32, le: bool) -> vec2f {
  let v = raw_fetch_u8_2(p, byte_off);
  return vec2f(v) / f32(1u << frac);
}

fn fetch_s8_2(p: ptr<storage, array<u32>>, byte_off: u32, frac: u32, le: bool) -> vec2f {
  let v = (bitcast<vec2i>(raw_fetch_u8_2(p, byte_off)) << vec2u(24u)) >> vec2u(24u);
  return vec2f(v) / f32(1u << frac);
}

fn fetch_u8_3(p: ptr<storage, array<u32>>, byte_off: u32, frac: u32, le: bool) -> vec3f {
  let v = raw_fetch_u8_3(p, byte_off);
  return vec3f(v) / f32(1u << frac);
}

fn fetch_s8_3(p: ptr<storage, array<u32>>, byte_off: u32, frac: u32, le: bool) -> vec3f {
  let v = (bitcast<vec3i>(raw_fetch_u8_3(p, byte_off)) << vec3u(24u)) >> vec3u(24u);
  return vec3f(v) / f32(1u << frac);
}

fn fetch_u8_4(p: ptr<storage, array<u32>>, byte_off: u32, frac: u32, le: bool) -> vec4f {
  let v = raw_fetch_u8_4(p, byte_off);
  return vec4f(v) / f32(1u << frac);
}

fn fetch_s8_4(p: ptr<storage, array<u32>>, byte_off: u32, frac: u32, le: bool) -> vec4f {
  let v = (bitcast<vec4i>(raw_fetch_u8_4(p, byte_off)) << vec4u(24u)) >> vec4u(24u);
  return vec4f(v) / f32(1u << frac);
}

fn fetch_u16_1(p: ptr<storage, array<u32>>, byte_off: u32, frac: u32, le: bool) -> f32 {
  let v = raw_fetch_u16_1(p, byte_off, le);
  return f32(v) / f32(1u << frac);
}

fn fetch_s16_1(p: ptr<storage, array<u32>>, byte_off: u32, frac: u32, le: bool) -> f32 {
  let v = bitcast<i32>(raw_fetch_u16_1(p, byte_off, le) << 16u) >> 16;
  return f32(v) / f32(1u << frac);
}

fn fetch_u16_2(p: ptr<storage, array<u32>>, byte_off: u32, frac: u32, le: bool) -> vec2f {
  let v = raw_fetch_u16_2(p, byte_off, le);
  return vec2f(v) / f32(1u << frac);
}

fn fetch_s16_2(p: ptr<storage, array<u32>>, byte_off: u32, frac: u32, le: bool) -> vec2f {
  let v = (bitcast<vec2i>(raw_fetch_u16_2(p, byte_off, le)) << vec2u(16u)) >> vec2u(16u);
  return vec2f(v) / f32(1u << frac);
}

fn fetch_u16_3(p: ptr<storage, array<u32>>, byte_off: u32, frac: u32, le: bool) -> vec3f {
  let v = raw_fetch_u16_3(p, byte_off, le);
  return vec3f(v) / f32(1u << frac);
}

fn fetch_s16_3(p: ptr<storage, array<u32>>, byte_off: u32, frac: u32, le: bool) -> vec3f {
  let v = (bitcast<vec3i>(raw_fetch_u16_3(p, byte_off, le)) << vec3u(16u)) >> vec3u(16u);
  return vec3f(v) / f32(1u << frac);
}

fn fetch_u16_4(p: ptr<storage, array<u32>>, byte_off: u32, frac: u32, le: bool) -> vec4f {
  let v = raw_fetch_u16_4(p, byte_off, le);
  return vec4f(v) / f32(1u << frac);
}

fn fetch_s16_4(p: ptr<storage, array<u32>>, byte_off: u32, frac: u32, le: bool) -> vec4f {
  let v = (bitcast<vec4i>(raw_fetch_u16_4(p, byte_off, le)) << vec4u(16u)) >> vec4u(16u);
  return vec4f(v) / f32(1u << frac);
}

fn fetch_f32_1(p: ptr<storage, array<u32>>, byte_off: u32, le: bool) -> f32 {
  return raw_fetch_f32_1(p, byte_off, le);
}

fn fetch_f32_2(p: ptr<storage, array<u32>>, byte_off: u32, le: bool) -> vec2f {
  return raw_fetch_f32_2(p, byte_off, le);
}

fn fetch_f32_3(p: ptr<storage, array<u32>>, byte_off: u32, le: bool) -> vec3f {
  return raw_fetch_f32_3(p, byte_off, le);
}

fn fetch_f32_4(p: ptr<storage, array<u32>>, byte_off: u32, le: bool) -> vec4f {
  return raw_fetch_f32_4(p, byte_off, le);
}

fn fetch_rgb565(p: ptr<storage, array<u32>>, byte_off: u32, le: bool) -> vec4f {
  let v = load_u16(p, byte_off, le);
  return vec4f(
    f32((v >> 11u) & 0x1Fu) / f32(0x1Fu),
    f32((v >>  5u) & 0x3Fu) / f32(0x3Fu),
    f32((v >>  0u) & 0x1Fu) / f32(0x1Fu),
    1.0,
  );
}

fn fetch_rgb8(p: ptr<storage, array<u32>>, byte_off: u32, le: bool) -> vec4f {
  let v = raw_fetch_u8_3(p, byte_off);
  return vec4f(f32(v.x), f32(v.y), f32(v.z), 255.0) / 255.0;
}

fn fetch_rgbx8(p: ptr<storage, array<u32>>, byte_off: u32, le: bool) -> vec4f {
  let v = raw_fetch_u8_4(p, byte_off);
  return vec4f(f32(v.x), f32(v.y), f32(v.z), 255.0) / 255.0;
}

fn fetch_rgba4(p: ptr<storage, array<u32>>, byte_off: u32, le: bool) -> vec4f {
  let v = load_u16(p, byte_off, le);
  return vec4f(
    f32((v >> 12u) & 0x0Fu) / f32(0x0Fu),
    f32((v >>  8u) & 0x0Fu) / f32(0x0Fu),
    f32((v >>  4u) & 0x0Fu) / f32(0x0Fu),
    f32((v >>  0u) & 0x0Fu) / f32(0x0Fu),
  );
}

fn fetch_rgba6(p: ptr<storage, array<u32>>, byte_off: u32, le: bool) -> vec4f {
  let v = load_u24(p, byte_off, le);
  return vec4f(
    f32((v >> 18u) & 0x3Fu) / f32(0x3Fu),
    f32((v >> 12u) & 0x3Fu) / f32(0x3Fu),
    f32((v >>  6u) & 0x3Fu) / f32(0x3Fu),
    f32((v >>  0u) & 0x3Fu) / f32(0x3Fu),
  );
}

fn fetch_rgba8(p: ptr<storage, array<u32>>, byte_off: u32, le: bool) -> vec4f {
  let v = raw_fetch_u8_4(p, byte_off);
  return vec4f(v) / 255.0;
})""";

constexpr std::string_view ImmediateStructWgsl = R"""(struct Immediate {
    vtx_start: u32,
    current_pnmtx: u32,
    fog_range_base: u32,
    _pad: u32,
    array_start0: vec4u,
    array_start1: vec4u,
    array_start2: vec4u,
};)""";

auto fetch_fixed16_attr(std::string_view fetchFn, const AttrConfig& mapping, std::string_view buf,
                        std::string_view offs, bool le) -> std::string {
  // Some Adreno drivers appear sensitive to generated shaders that route 2- and
  // 3-component fixed-16 vertex attributes through reusable vector fetch helpers.
  // Emitting scalar component fetches at the call site avoids observed artifacts.
  if (mapping.cnt == 2) {
    const auto comp0 = fmt::format("{}_1(&{}, {} + 0u, {}u, {})", fetchFn, buf, offs, mapping.frac, le);
    const auto comp1 = fmt::format("{}_1(&{}, {} + 2u, {}u, {})", fetchFn, buf, offs, mapping.frac, le);
    return fmt::format("vec2f({}, {})", comp0, comp1);
  }
  if (mapping.cnt == 3) {
    const auto comp0 = fmt::format("{}_1(&{}, {} + 0u, {}u, {})", fetchFn, buf, offs, mapping.frac, le);
    const auto comp1 = fmt::format("{}_1(&{}, {} + 2u, {}u, {})", fetchFn, buf, offs, mapping.frac, le);
    const auto comp2 = fmt::format("{}_1(&{}, {} + 4u, {}u, {})", fetchFn, buf, offs, mapping.frac, le);
    return fmt::format("vec3f({}, {}, {})", comp0, comp1, comp2);
  }
  return fmt::format("{}_{}(&{}, {}, {}u, {})", fetchFn, mapping.cnt, buf, offs, mapping.frac, le);
}

auto fetch_attr(const AttrConfig& mapping, std::string_view buf, std::string_view offs, bool le) -> std::string {
  switch (mapping.compType) {
  case GX_U8:
    return fmt::format("fetch_u8_{}(&{}, {}, {}, {})", mapping.cnt, buf, offs, mapping.frac, le);
  case GX_S8:
    return fmt::format("fetch_s8_{}(&{}, {}, {}, {})", mapping.cnt, buf, offs, mapping.frac, le);
  case GX_U16:
    return fetch_fixed16_attr("fetch_u16"sv, mapping, buf, offs, le);
  case GX_S16:
    return fetch_fixed16_attr("fetch_s16"sv, mapping, buf, offs, le);
  case GX_F32:
    return fmt::format("fetch_f32_{}(&{}, {}, {})", mapping.cnt, buf, offs, le);
  case GX_RGBA8:
    return fmt::format("unpack4x8unorm(load_u32_raw(&{}, {}))", buf, offs);
  default:
    Log.fatal("fetch_attr: Unimplemented {}", static_cast<GXCompType>(mapping.compType));
  }
}

auto fetch_color_attr(const AttrConfig& mapping, std::string_view buf, std::string_view offs, bool le) -> std::string {
  switch (mapping.compType) {
  case GX_RGB565:
    return fmt::format("fetch_rgb565(&{}, {}, {})", buf, offs, le);
  case GX_RGB8:
    return fmt::format("fetch_rgb8(&{}, {}, {})", buf, offs, le);
  case GX_RGBX8:
    return fmt::format("fetch_rgbx8(&{}, {}, {})", buf, offs, le);
  case GX_RGBA4:
    return fmt::format("fetch_rgba4(&{}, {}, {})", buf, offs, le);
  case GX_RGBA6:
    return fmt::format("fetch_rgba6(&{}, {}, {})", buf, offs, le);
  case GX_RGBA8:
    return fmt::format("fetch_rgba8(&{}, {}, {})", buf, offs, le);
  default:
    Log.fatal("fetch_color_attr: Unimplemented {}", static_cast<GXCompType>(mapping.compType));
  }
}

struct AttrAddress {
  std::string offs;
  std::string_view buf;
  bool le;
};

// Immediates cannot contain arrays, so array_start is packed as three vec4u.
std::string imm_array_start(GXAttr attr) noexcept {
  const u32 idx = attr - GX_VA_POS;
  return fmt::format("imm.array_start{}.{}", idx / 4, "xyzw"[idx % 4]);
}

auto attr_address(const AttrConfig& mapping, GXAttr attr, std::string_view vidx, u32 vtxStride, u32 dlExtra, u32 within)
    -> AttrAddress {
  const u32 dlOffset = mapping.offset + dlExtra;
  if (mapping.attrType == GX_INDEX8) {
    return {fmt::format("{} + raw_fetch_u8_1(&vbuf, imm.vtx_start + {} * {}u + {}u) * {}u + {}u", imm_array_start(attr),
                        vidx, vtxStride, dlOffset, mapping.stride, within),
            "abuf"sv, mapping.le};
  }
  if (mapping.attrType == GX_INDEX16) {
    return {fmt::format("{} + raw_fetch_u16_1(&vbuf, imm.vtx_start + {} * {}u + {}u, false) * {}u + {}u",
                        imm_array_start(attr), vidx, vtxStride, dlOffset, mapping.stride, within),
            "abuf"sv, mapping.le};
  }
  return {fmt::format("imm.vtx_start + {} * {}u + {}u", vidx, vtxStride, dlOffset + within), "vbuf"sv, false};
}

struct CpuSource {
  std::span<const uint8_t> bytes;
  uint32_t first = 0;
  bool le = false;
};

const uint8_t* cpu_bytes(const CpuSource& src, uint32_t offset, uint32_t size) noexcept {
  if (offset < src.first) {
    return nullptr;
  }
  const size_t at = offset - src.first;
  if (at > src.bytes.size() || size > src.bytes.size() - at) {
    return nullptr;
  }
  return src.bytes.data() + at;
}

uint32_t cpu_load(const uint8_t* p, uint32_t size, bool le) noexcept {
  uint32_t value = 0;
  for (uint32_t i = 0; i < size; ++i) {
    value |= uint32_t{p[le ? i : size - 1 - i]} << (8 * i);
  }
  return value;
}

bool cpu_address(const AttrConfig& mapping, GXAttr attr, const CpuVertexSource& source, uint32_t vidx, uint32_t stride,
                 CpuSource& out, uint32_t& offset) noexcept {
  const uint32_t at = vidx * stride + mapping.offset;
  if (mapping.attrType == GX_DIRECT) {
    out = {.bytes = source.records};
    offset = at;
    return true;
  }
  if (attr < GX_VA_POS || attr > GX_VA_TEX7) {
    return false;
  }
  const uint32_t idxSize = mapping.attrType == GX_INDEX8 ? 1 : 2;
  const uint8_t* p = cpu_bytes({.bytes = source.records}, at, idxSize);
  if (p == nullptr) {
    return false;
  }
  const uint32_t idx = idxSize == 1 ? p[0] : cpu_load(p, 2, false);
  const CpuArray& array = source.arrays[attr - GX_VA_POS];
  out = {.bytes = array.bytes, .first = array.first, .le = mapping.le};
  offset = idx * mapping.stride;
  return true;
}

bool cpu_fetch(const AttrConfig& mapping, const CpuSource& src, uint32_t offset, uint32_t cnt, float* out) noexcept {
  const float scale = static_cast<float>(1u << mapping.frac);
  for (uint32_t i = 0; i < cnt; ++i) {
    const uint8_t* p = nullptr;
    switch (mapping.compType) {
    case GX_U8:
    case GX_S8:
      p = cpu_bytes(src, offset + i, 1);
      if (p == nullptr) {
        return false;
      }
      out[i] = (mapping.compType == GX_U8 ? static_cast<float>(p[0]) : static_cast<float>(static_cast<int8_t>(p[0]))) /
               scale;
      break;
    case GX_U16:
    case GX_S16: {
      p = cpu_bytes(src, offset + i * 2, 2);
      if (p == nullptr) {
        return false;
      }
      const uint32_t raw = cpu_load(p, 2, src.le);
      out[i] = (mapping.compType == GX_U16 ? static_cast<float>(raw)
                                           : static_cast<float>(static_cast<int16_t>(static_cast<uint16_t>(raw)))) /
               scale;
      break;
    }
    case GX_F32:
      p = cpu_bytes(src, offset + i * 4, 4);
      if (p == nullptr) {
        return false;
      }
      out[i] = std::bit_cast<float>(cpu_load(p, 4, src.le));
      break;
    case GX_RGBA8:
      p = cpu_bytes(src, offset + i, 1);
      if (p == nullptr) {
        return false;
      }
      out[i] = static_cast<float>(p[0]) / 255.f;
      break;
    default:
      return false;
    }
  }
  return true;
}

bool cpu_color(const AttrConfig& mapping, const CpuSource& src, uint32_t offset, std::array<float, 4>& out) noexcept {
  const auto unorm = [](uint32_t value, uint32_t shift, uint32_t mask) {
    return static_cast<float>((value >> shift) & mask) / static_cast<float>(mask);
  };
  const uint8_t* p = nullptr;
  switch (mapping.compType) {
  case GX_RGB565: {
    if ((p = cpu_bytes(src, offset, 2)) == nullptr) {
      return false;
    }
    const uint32_t v = cpu_load(p, 2, src.le);
    out = {unorm(v, 11, 0x1F), unorm(v, 5, 0x3F), unorm(v, 0, 0x1F), 1.f};
    return true;
  }
  case GX_RGB8:
  case GX_RGBX8:
    if ((p = cpu_bytes(src, offset, mapping.compType == GX_RGB8 ? 3 : 4)) == nullptr) {
      return false;
    }
    out = {static_cast<float>(p[0]) / 255.f, static_cast<float>(p[1]) / 255.f, static_cast<float>(p[2]) / 255.f, 1.f};
    return true;
  case GX_RGBA4: {
    if ((p = cpu_bytes(src, offset, 2)) == nullptr) {
      return false;
    }
    const uint32_t v = cpu_load(p, 2, src.le);
    out = {unorm(v, 12, 0xF), unorm(v, 8, 0xF), unorm(v, 4, 0xF), unorm(v, 0, 0xF)};
    return true;
  }
  case GX_RGBA6: {
    if ((p = cpu_bytes(src, offset, 3)) == nullptr) {
      return false;
    }
    const uint32_t v = cpu_load(p, 3, src.le);
    out = {unorm(v, 18, 0x3F), unorm(v, 12, 0x3F), unorm(v, 6, 0x3F), unorm(v, 0, 0x3F)};
    return true;
  }
  case GX_RGBA8:
    if ((p = cpu_bytes(src, offset, 4)) == nullptr) {
      return false;
    }
    for (int i = 0; i < 4; ++i) {
      out[i] = static_cast<float>(p[i]) / 255.f;
    }
    return true;
  default:
    return false;
  }
}
} // namespace

std::string_view wgsl_helpers() noexcept { return HelpersWgsl; }

std::string_view wgsl_immediate_struct() noexcept { return ImmediateStructWgsl; }

std::string vtx_attr(const ShaderConfig& config, GXAttr attr) {
  const auto type = config.attrs[attr].attrType;
  if (type == GX_NONE) {
    if (attr == GX_VA_PNMTXIDX) {
      return "imm.current_pnmtx";
    }
    if (attr == GX_VA_NRM) {
      // Default normal
      return "vec3f(1.0, 0.0, 0.0)"s;
    }
    if (attr == GX_VA_CLR0 || attr == GX_VA_CLR1) {
      return "vec4f(0.0, 0.0, 0.0, 0.0)"s;
    }
    UNLIKELY FATAL("unmapped vtx attr {}", underlying(attr));
  }
  if (attr == GX_VA_POS) {
    return "in_pos"s;
  }
  if (attr == GX_VA_NRM) {
    return "in_nrm"s;
  }
  if (attr == GX_VA_CLR0 || attr == GX_VA_CLR1) {
    const auto idx = attr - GX_VA_CLR0;
    return fmt::format("in_clr{}", idx);
  }
  if (attr >= GX_VA_TEX0 && attr <= GX_VA_TEX7) {
    const auto idx = attr - GX_VA_TEX0;
    return fmt::format("in_tex{}_uv", idx);
  }
  if (attr == GX_VA_PNMTXIDX) {
    return "in_pnmtxidx"s;
  }
  if (attr >= GX_VA_TEX0MTXIDX && attr <= GX_VA_TEX7MTXIDX) {
    const auto idx = attr - GX_VA_TEX0MTXIDX;
    return fmt::format("in_texmtxidx{}", idx);
  }
  UNLIKELY FATAL("unhandled vtx attr {}", underlying(attr));
}

auto attr_load(const ShaderConfig& config, GXAttr attr, std::string_view vidx) -> std::string {
  const auto& mapping = config.attrs[attr];
  if (mapping.attrType == GX_NONE) {
    return vtx_attr(config, attr);
  }
  const auto [offs, buf, le] = attr_address(mapping, attr, vidx, config.vtxStride, 0u, 0u);
  switch (attr) {
  case GX_VA_PNMTXIDX:
    return fmt::format("(raw_fetch_u8_1(&{}, {}) / 3u)", buf, offs);
  case GX_VA_TEX0MTXIDX:
  case GX_VA_TEX1MTXIDX:
  case GX_VA_TEX2MTXIDX:
  case GX_VA_TEX3MTXIDX:
  case GX_VA_TEX4MTXIDX:
  case GX_VA_TEX5MTXIDX:
  case GX_VA_TEX6MTXIDX:
  case GX_VA_TEX7MTXIDX:
    return fmt::format("raw_fetch_u8_1(&{}, {})", buf, offs);
  case GX_VA_POS: {
    const auto posLoad = fetch_attr(mapping, buf, offs, le);
    if (mapping.cnt == 2) {
      return fmt::format("vec3f({}, 0.0)", posLoad);
    }
    return posLoad;
  }
  case GX_VA_NRM:
    // NBT: normal only here; binormal/tangent loaded via attr_load_nbt_slice
    if (mapping.cnt > 3) {
      auto nrmMapping = mapping;
      nrmMapping.cnt = 3;
      return fetch_attr(nrmMapping, buf, offs, le);
    }
    return fetch_attr(mapping, buf, offs, le);
  case GX_VA_CLR0:
  case GX_VA_CLR1:
    return fetch_color_attr(mapping, buf, offs, le);
  case GX_VA_TEX0:
  case GX_VA_TEX1:
  case GX_VA_TEX2:
  case GX_VA_TEX3:
  case GX_VA_TEX4:
  case GX_VA_TEX5:
  case GX_VA_TEX6:
  case GX_VA_TEX7: {
    const auto texLoad = fetch_attr(mapping, buf, offs, le);
    if (mapping.cnt == 1) {
      return fmt::format("vec2f({}, 0.0)", texLoad);
    }
    return texLoad;
  }
  default:
    Log.fatal("attr_load: Unimplemented {}", attr);
  }
}

auto attr_load_nbt_slice(const ShaderConfig& config, NbtSlice slice, std::string_view vidx) -> std::string {
  const auto& mapping = config.attrs[GX_VA_NRM];
  if (mapping.attrType == GX_NONE || mapping.cnt != 9) {
    Log.fatal("attr_load_nbt_slice: GX_TG_BINRM/TANGENT requires GX_NRM_NBT or GX_NRM_NBT3");
  }
  const auto sliceIdx = static_cast<u32>(slice);
  const auto compsize = comp_type_size(GX_VA_NRM, static_cast<GXCompType>(mapping.compType));
  u32 dlExtra = 0;
  if (mapping.nbt3) {
    if (mapping.attrType == GX_INDEX8) {
      dlExtra = sliceIdx;
    } else if (mapping.attrType == GX_INDEX16) {
      dlExtra = sliceIdx * 2u;
    }
  }
  const u32 within = sliceIdx * 3u * compsize;
  const auto [offs, buf, le] = attr_address(mapping, GX_VA_NRM, vidx, config.vtxStride, dlExtra, within);
  auto sliceMapping = mapping;
  sliceMapping.cnt = 3;
  return fetch_attr(sliceMapping, buf, offs, le);
}

std::string_view nbt_slice_local(NbtSlice slice) noexcept { return slice == NbtSlice::B ? "in_binrm" : "in_tangent"; }

bool decode_vertex(const ShaderConfig& config, const CpuVertexSource& source, uint32_t vidx, CpuVertex& out) noexcept {
  out = {};
  out.pnMtx = source.currentPnMtx;
  for (uint32_t a = GX_VA_PNMTXIDX; a <= GX_VA_TEX7; ++a) {
    const auto attr = static_cast<GXAttr>(a);
    const AttrConfig& mapping = config.attrs[attr];
    if (mapping.attrType == GX_NONE || (attr >= GX_VA_TEX0MTXIDX && attr <= GX_VA_TEX7MTXIDX)) {
      continue;
    }
    CpuSource src;
    uint32_t offset = 0;
    if (!cpu_address(mapping, attr, source, vidx, config.vtxStride, src, offset)) {
      return false;
    }
    bool ok = true;
    if (attr == GX_VA_PNMTXIDX) {
      const uint8_t* p = cpu_bytes(src, offset, 1);
      ok = p != nullptr;
      out.pnMtx = ok ? p[0] / 3u : 0u;
    } else if (attr == GX_VA_POS) {
      ok = (mapping.cnt == 2 || mapping.cnt == 3) && cpu_fetch(mapping, src, offset, mapping.cnt, out.pos.data());
      if (mapping.cnt == 2) {
        out.pos[2] = 0.f;
      }
    } else if (attr == GX_VA_NRM) {
      ok = cpu_fetch(mapping, src, offset, std::min<uint32_t>(mapping.cnt, 3), out.nrm.data());
    } else if (attr == GX_VA_CLR0 || attr == GX_VA_CLR1) {
      ok = cpu_color(mapping, src, offset, out.clr[attr - GX_VA_CLR0]);
    } else {
      auto& tex = out.tex[attr - GX_VA_TEX0];
      ok = (mapping.cnt == 1 || mapping.cnt == 2) && cpu_fetch(mapping, src, offset, mapping.cnt, tex.data());
      if (mapping.cnt == 1) {
        tex[1] = 0.f;
      }
    }
    if (!ok) {
      return false;
    }
  }
  return true;
}
} // namespace aurora::gx::vertex_fetch
