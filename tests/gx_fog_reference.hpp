#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>

namespace aurora::gx::fog_reference {
struct Registers {
  float a = 0.f;
  float b = 0.f;
  float c = 0.f;
};

inline float truncate_fog_float(float value) noexcept {
  return std::bit_cast<float>(std::bit_cast<uint32_t>(value) & 0xFFFFF000u);
}

inline Registers fog_params(uint32_t type, float startZ, float endZ, float nearZ, float farZ) noexcept {
  if ((type & 0x08) != 0) {
    float a = 0.f;
    float c = 0.f;
    if (farZ != nearZ && endZ != startZ) {
      const float scale = 1.f / (endZ - startZ);
      a = scale * (farZ - nearZ);
      c = scale * (startZ - nearZ);
    }
    return {.a = truncate_fog_float(a), .b = 0.f, .c = truncate_fog_float(c)};
  }
  float aCoeff = 0.f;
  float bCoeff = 0.5f;
  float cCoeff = 0.f;
  if (farZ != nearZ && endZ != startZ) {
    aCoeff = (farZ * nearZ) / ((farZ - nearZ) * (endZ - startZ));
    bCoeff = farZ / (farZ - nearZ);
    cCoeff = startZ / (endZ - startZ);
  }
  float bMantissa = bCoeff;
  int bExponent = 0;
  while (bMantissa > 1.f) {
    bMantissa *= 0.5f;
    ++bExponent;
  }
  while (bMantissa > 0.f && bMantissa < 0.5f) {
    bMantissa *= 2.f;
    --bExponent;
  }
  const float a = aCoeff / std::ldexp(1.f, bExponent + 1);
  const uint32_t bMantissaRaw = static_cast<uint32_t>(8.388638e6f * bMantissa) & 0xFFFFFFu;
  const int bScale = (bExponent + 1) & 0x1F;
  return {
      .a = std::ldexp(truncate_fog_float(a), bScale),
      .b = std::ldexp(static_cast<float>(bMantissaRaw) / 8388638.f, bScale - 1),
      .c = truncate_fog_float(cCoeff),
  };
}

inline double range_factor(const std::array<uint16_t, 10>& rangeK, double rangeCenter, double renderWidth,
                           uint32_t x) noexcept {
  std::array<double, 10> k{};
  for (uint32_t i = 0; i < k.size(); ++i) {
    const uint32_t source = (i & ~1u) | (1u - (i & 1u));
    k[i] = static_cast<double>(rangeK[source]) / 64.0;
  }
  const double screenX = ((static_cast<double>(x) + 0.5) / renderWidth) * 2.0 - 1.0;
  const double offset = screenX - rangeCenter;
  const double rangeIndex = std::clamp(9.0 - std::abs(offset) * 9.0, 0.0, 9.0);
  const uint32_t lower = static_cast<uint32_t>(rangeIndex);
  const uint32_t upper = std::min(lower + 1, 9u);
  const double fraction = rangeIndex - static_cast<double>(lower);
  const double kk = std::max(k[lower] * (1.0 - fraction) + k[upper] * fraction, 0.000001);
  return std::sqrt(offset * offset + kk * kk) / kk;
}

inline double weight(uint32_t type, double a, double b, double c, double zw, double rangeFactor) noexcept {
  if ((type & 7) == 0) {
    return 0.0;
  }
  double fogBase = (type & 8) != 0 ? a * zw : a / (b - zw);
  fogBase *= rangeFactor;
  double fogF = std::clamp(fogBase - c, 0.0, 1.0);
  double fogZ = 0.0;
  switch (type & 7) {
  case 2:
    fogZ = fogF;
    break;
  case 4:
    fogZ = 1.0 - std::exp2(-8.0 * fogF);
    break;
  case 5:
    fogZ = 1.0 - std::exp2(-8.0 * fogF * fogF);
    break;
  case 6:
    fogZ = std::exp2(-8.0 * (1.0 - fogF));
    break;
  case 7:
    fogF = 1.0 - fogF;
    fogZ = std::exp2(-8.0 * fogF * fogF);
    break;
  default:
    return 0.0;
  }
  return std::clamp(fogZ, 0.0, 1.0);
}

inline uint32_t quantize(double fogZ) noexcept {
  return std::min(static_cast<uint32_t>(std::nearbyint(std::clamp(fogZ, 0.0, 1.0) * 127.0)), 127u);
}

inline std::array<uint32_t, 4> record(const std::array<uint32_t, 3>& rgb, double fogZ, uint32_t type) noexcept {
  if ((type & 7) == 0) {
    return {};
  }
  return {rgb[0], rgb[1], rgb[2], (quantize(fogZ) << 1) | (type != 2 ? 1u : 0u)};
}

inline double zw_ref(double ndc, double znear, double zfar) noexcept { return zfar - ndc * (zfar - znear); }
} // namespace aurora::gx::fog_reference
