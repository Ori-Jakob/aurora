#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>

// Surface v2; the WGSL in aurora/surface and the generator's copy must match this bit for bit
namespace aurora::gx::surface {
inline constexpr uint32_t LayoutVersion = 2;

inline constexpr uint8_t ClassNone = 0;
inline constexpr uint8_t ClassDefault = 1;
inline constexpr uint8_t ClassReserved = 255;

inline constexpr uint8_t CloudReceiver = 1u << 0;
inline constexpr uint8_t NoShadowReceive = 1u << 1;
inline constexpr uint8_t ThinTranslucent = 1u << 2;
inline constexpr uint8_t TwoSided = 1u << 3;
inline constexpr uint8_t NoSsr = 1u << 4;
inline constexpr uint8_t Unlit = 1u << 5;
inline constexpr uint8_t BakedLight = 1u << 6;

inline constexpr uint32_t EntryExtAmbient = 1u << 24;
inline constexpr uint32_t EntryExtEmissive = 1u << 25;
inline constexpr uint32_t SlotShift = 16;
inline constexpr uint32_t SlotMax = 0xFFFF;

inline constexpr uint32_t RgMax = 0x7BF;
inline constexpr uint32_t BMax = 0x3DF;
inline constexpr float InvUnorm8 = std::bit_cast<float>(0x3B808081u);

struct Fields {
  std::array<float, 3> albedo{};
  float ao = 1.f;
  float roughness = 0.f;
  float metalness = 0.f;
  uint8_t classId = 0;
  uint8_t flags = 0;
  std::array<float, 3> ambient{};
  uint8_t category = 0;
  std::array<float, 3> emissive{};

  bool operator==(const Fields&) const = default;
};

[[nodiscard]] constexpr bool is_nan_bits(float x) noexcept {
  return (std::bit_cast<uint32_t>(x) & 0x7FFFFFFFu) > 0x7F800000u;
}

[[nodiscard]] constexpr uint8_t unorm8(float x) noexcept {
  if (is_nan_bits(x)) {
    return 0;
  }
  const float c = x < 0.f ? 0.f : x > 1.f ? 1.f : x;
  return static_cast<uint8_t>(static_cast<uint32_t>(0.5f + 255.f * c));
}

[[nodiscard]] constexpr float unorm8_decode(uint32_t b) noexcept { return static_cast<float>(b & 0xFFu) * InvUnorm8; }

namespace detail {
[[nodiscard]] constexpr uint32_t pack_ufloat(float x, uint32_t mantissaBits, uint32_t maxCode) noexcept {
  const uint32_t bits = std::bit_cast<uint32_t>(x);
  if (bits > 0x7F800000u || bits < 0x38800000u) {
    return 0;
  }
  const uint32_t shift = 23 - mantissaBits;
  const uint32_t code = (((bits >> 23) - 112) << mantissaBits) | ((bits >> shift) & ((1u << mantissaBits) - 1));
  return std::min(code + ((bits >> (shift - 1)) & 1), maxCode);
}

[[nodiscard]] constexpr float unpack_ufloat(uint32_t code, uint32_t mantissaBits) noexcept {
  const uint32_t e = code >> mantissaBits;
  if (e == 0) {
    return 0.f;
  }
  const uint32_t m = code & ((1u << mantissaBits) - 1);
  return std::bit_cast<float>(((e + 112) << 23) | (m << (23 - mantissaBits)));
}
} // namespace detail

[[nodiscard]] constexpr uint32_t pack_r11g11b10(const std::array<float, 3>& v) noexcept {
  return detail::pack_ufloat(v[0], 6, RgMax) | (detail::pack_ufloat(v[1], 6, RgMax) << 11) |
         (detail::pack_ufloat(v[2], 5, BMax) << 22);
}

[[nodiscard]] constexpr std::array<float, 3> unpack_r11g11b10(uint32_t w) noexcept {
  return {detail::unpack_ufloat(w & 0x7FFu, 6), detail::unpack_ufloat((w >> 11) & 0x7FFu, 6),
          detail::unpack_ufloat(w >> 22, 5)};
}

[[nodiscard]] constexpr uint32_t pack_unorm4(float a, float b, float c, float d) noexcept {
  return uint32_t{unorm8(a)} | (uint32_t{unorm8(b)} << 8) | (uint32_t{unorm8(c)} << 16) | (uint32_t{unorm8(d)} << 24);
}

// v1 words, which the debug expand pass rebuilds for tools
[[nodiscard]] constexpr std::array<uint32_t, 4> pack(const Fields& f) noexcept {
  return {
      pack_unorm4(f.albedo[0], f.albedo[1], f.albedo[2], f.ao),
      pack_unorm4(f.roughness, f.metalness, 0.f, 0.f) | (uint32_t{f.classId} << 16) | (uint32_t{f.flags} << 24),
      (pack_unorm4(f.ambient[0], f.ambient[1], f.ambient[2], 0.f) & 0xFFFFFFu) | (uint32_t{f.category} << 24),
      pack_r11g11b10(f.emissive),
  };
}

[[nodiscard]] constexpr Fields unpack(const std::array<uint32_t, 4>& words) noexcept {
  const auto byte = [](uint32_t word, uint32_t index) { return (word >> (8 * index)) & 0xFFu; };
  return {
      .albedo = {unorm8_decode(byte(words[0], 0)), unorm8_decode(byte(words[0], 1)), unorm8_decode(byte(words[0], 2))},
      .ao = unorm8_decode(byte(words[0], 3)),
      .roughness = unorm8_decode(byte(words[1], 0)),
      .metalness = unorm8_decode(byte(words[1], 1)),
      .classId = static_cast<uint8_t>(byte(words[1], 2)),
      .flags = static_cast<uint8_t>(byte(words[1], 3)),
      .ambient = {unorm8_decode(byte(words[2], 0)), unorm8_decode(byte(words[2], 1)), unorm8_decode(byte(words[2], 2))},
      .category = static_cast<uint8_t>(byte(words[2], 3)),
      .emissive = unpack_r11g11b10(words[3]),
  };
}

using Words = std::array<uint32_t, 2>;
using Channels = std::array<uint16_t, 4>;
using Entry = std::array<uint32_t, 4>;

// RGBA16Uint writes cheaper than RG32Uint, so each 32-bit word spans two channels, low half first
[[nodiscard]] constexpr Channels split(const Words& w) noexcept {
  return {static_cast<uint16_t>(w[0]), static_cast<uint16_t>(w[0] >> 16), static_cast<uint16_t>(w[1]),
          static_cast<uint16_t>(w[1] >> 16)};
}

[[nodiscard]] constexpr Words join(const Channels& c) noexcept {
  return {uint32_t{c[0]} | (uint32_t{c[1]} << 16), uint32_t{c[2]} | (uint32_t{c[3]} << 16)};
}

[[nodiscard]] constexpr Words pack_record(const Fields& f, uint32_t slot) noexcept {
  return {pack_unorm4(f.albedo[0], f.albedo[1], f.albedo[2], f.ao),
          (pack_unorm4(f.roughness, f.metalness, 0.f, 0.f) & 0xFFFFu) | ((slot & SlotMax) << SlotShift)};
}

[[nodiscard]] constexpr Words pack_ext(const Fields& f) noexcept {
  return {pack_unorm4(f.ambient[0], f.ambient[1], f.ambient[2], 0.f) & 0xFFFFFFu, pack_r11g11b10(f.emissive)};
}

[[nodiscard]] constexpr Entry pack_entry(const Fields& f, uint32_t extBits) noexcept {
  return {uint32_t{f.classId} | (uint32_t{f.flags} << 8) | (uint32_t{f.category} << 16) |
              (extBits & (EntryExtAmbient | EntryExtEmissive)),
          pack_unorm4(f.ambient[0], f.ambient[1], f.ambient[2], 0.f) & 0xFFFFFFu, pack_r11g11b10(f.emissive), 0};
}

[[nodiscard]] constexpr uint32_t record_slot(const Words& record) noexcept { return record[1] >> SlotShift; }

[[nodiscard]] constexpr Fields decode(const Words& record, const Entry& entry, const Words& ext) noexcept {
  const auto byte = [](uint32_t word, uint32_t index) { return (word >> (8 * index)) & 0xFFu; };
  const uint32_t ambient = (entry[0] & EntryExtAmbient) != 0 ? ext[0] : entry[1];
  return {
      .albedo = {unorm8_decode(byte(record[0], 0)), unorm8_decode(byte(record[0], 1)),
                 unorm8_decode(byte(record[0], 2))},
      .ao = unorm8_decode(byte(record[0], 3)),
      .roughness = unorm8_decode(byte(record[1], 0)),
      .metalness = unorm8_decode(byte(record[1], 1)),
      .classId = static_cast<uint8_t>(byte(entry[0], 0)),
      .flags = static_cast<uint8_t>(byte(entry[0], 1)),
      .ambient = {unorm8_decode(byte(ambient, 0)), unorm8_decode(byte(ambient, 1)), unorm8_decode(byte(ambient, 2))},
      .category = static_cast<uint8_t>(byte(entry[0], 2)),
      .emissive = unpack_r11g11b10((entry[0] & EntryExtEmissive) != 0 ? ext[1] : entry[2]),
  };
}

// Mirrors gbuf_surface_v1_expand, which keeps tools and views on v1 words
[[nodiscard]] constexpr std::array<uint32_t, 4> expand_v1(const Words& record, const Entry& entry,
                                                          const Words& ext) noexcept {
  const uint32_t ambient = (entry[0] & EntryExtAmbient) != 0 ? ext[0] : entry[1];
  return {record[0], (record[1] & 0xFFFFu) | ((entry[0] & 0xFFFFu) << 16),
          (ambient & 0xFFFFFFu) | (((entry[0] >> 16) & 0xFFu) << 24),
          (entry[0] & EntryExtEmissive) != 0 ? ext[1] : entry[2]};
}
} // namespace aurora::gx::surface
