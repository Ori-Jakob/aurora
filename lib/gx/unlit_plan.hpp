#pragma once

#include "gx.hpp"

#include <cstdint>
#include <string_view>
#include <type_traits>

namespace aurora::gx::unlit {
// SP1 rule table v1 plus the ras-swap and indirect-texgen extensions; a verdict change bumps RenderShaderRev
inline constexpr uint32_t RuleTableVersion = 1;

enum class Verdict : uint8_t { Passthrough, Linear, Scale, FailSpecOnly, FailOther, FailNonlinear, FailLightTexgen };

inline constexpr uint8_t RewriteNone = 0;
inline constexpr uint8_t RewriteMaterial = 1;
inline constexpr uint8_t RewriteVertex = 2;
inline constexpr uint8_t RewriteZero = 3;
inline constexpr uint8_t NoStage = 0xFF;

struct Plan {
  Verdict verdict = Verdict::Passthrough;
  uint8_t failRule = 0;
  uint8_t failStage = NoStage;
  uint8_t rawVtx = 0;
  uint16_t scaleDivide = 0;
  uint16_t layerStages = 0;
  uint8_t neutralOne = 0;
  uint8_t neutralZero = 0;
  uint8_t rewrite[2]{};

  bool pass() const noexcept {
    return verdict == Verdict::Passthrough || verdict == Verdict::Linear || verdict == Verdict::Scale;
  }
  bool operator==(const Plan&) const = default;
};
static_assert(std::is_trivially_copyable_v<Plan> && sizeof(Plan) <= 16);

enum class AmbientSource : uint8_t { None, Register, Vertex, Unknown };

struct Ambient {
  AmbientSource source = AmbientSource::None;
  uint8_t channel = 0;
  bool rawVertex = false;

  bool operator==(const Ambient&) const = default;
};

Plan analyse(const ShaderConfig& config) noexcept;
Plan plan(const ShaderConfig& config) noexcept;
bool admitted(const Plan& p, const ShaderConfig& config, uint32_t baseVaryings) noexcept;
uint32_t raw_vertex_varyings(const Plan& p) noexcept;
Ambient ambient_channel(const ShaderConfig& config, const Plan& p) noexcept;
Ambient ambient_of(const ShaderConfig& config, const ShaderInfo& info) noexcept;
std::string_view verdict_name(Verdict verdict) noexcept;
} // namespace aurora::gx::unlit
