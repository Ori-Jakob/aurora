#include "vsm_filter.hpp"

#include "../../logging.hpp"

#include <SDL3/SDL_stdinc.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <string>

namespace aurora::render::shadows::vsm {
namespace {
constexpr Module Log{"aurora::render::shadows"};

constexpr std::array<std::array<float, 2>, 1> OneTap{{{0.f, 0.f}}};
constexpr std::array<std::array<float, 2>, 4> FourTaps{{{-0.5f, -0.5f}, {0.5f, -0.5f}, {-0.5f, 0.5f}, {0.5f, 0.5f}}};
constexpr std::array<std::array<float, 2>, 9> NineTaps{{
    {-1.f, -1.f},
    {0.f, -1.f},
    {1.f, -1.f},
    {-1.f, 0.f},
    {0.f, 0.f},
    {1.f, 0.f},
    {-1.f, 1.f},
    {0.f, 1.f},
    {1.f, 1.f},
}};

std::optional<float> parse_float(std::string_view text) noexcept {
  float value = 0.f;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size() || !std::isfinite(value) || value < 0.f) {
    return std::nullopt;
  }
  return value;
}

std::array<int32_t, 2> page_of(const ViewRecord& view, const std::array<float, 2>& c,
                               std::array<float, 2>& local) noexcept {
  std::array<int32_t, 2> page{};
  for (size_t i = 0; i < 2; ++i) {
    const float lp =
        std::clamp(std::floor(c[i] / static_cast<float>(PageTexels)), 0.f, static_cast<float>(WindowPages - 1));
    page[i] = view.origin[i] + static_cast<int32_t>(lp);
    local[i] = c[i] - lp * static_cast<float>(PageTexels);
  }
  return page;
}
} // namespace

std::optional<Bias> parse_bias(std::string_view text) noexcept {
  const size_t comma = text.find(',');
  if (comma == std::string_view::npos) {
    return std::nullopt;
  }
  const std::optional<float> normal = parse_float(text.substr(0, comma));
  const std::optional<float> depth = parse_float(text.substr(comma + 1));
  if (!normal || !depth) {
    return std::nullopt;
  }
  return Bias{.normal = *normal, .depth = *depth};
}

const std::optional<Bias>& env_bias() noexcept {
  static const std::optional<Bias> bias = []() -> std::optional<Bias> {
    const char* value = SDL_getenv(std::string{BiasEnv}.c_str());
    if (value == nullptr) {
      return std::nullopt;
    }
    const std::optional<Bias> parsed = parse_bias(value);
    if (!parsed) {
      Log.warn("{}={} is not <normal scale>,<depth texels>; ignored", BiasEnv, value);
      return std::nullopt;
    }
    Log.warn("{}={}: the sun shadow filter uses a test bias, for verification only", BiasEnv, value);
    return parsed;
  }();
  return bias;
}

FilterParams filter_params(Quality quality, const std::optional<Bias>& bias) noexcept {
  FilterParams out;
  switch (quality) {
  case Quality::Low:
    out.taps = 1;
    break;
  case Quality::Medium:
    out.taps = 4;
    break;
  default:
    out.taps = MaxFilterTaps;
    break;
  }
  if (bias) {
    out.normalScale = bias->normal;
    out.depthBias = bias->depth;
  }
  return out;
}

uint32_t filter_taps(uint32_t taps) noexcept { return taps <= 1 ? 1u : taps <= 4 ? 4u : MaxFilterTaps; }

std::span<const std::array<float, 2>> tap_offsets(uint32_t taps) noexcept {
  switch (filter_taps(taps)) {
  case 1:
    return OneTap;
  case 4:
    return FourTaps;
  default:
    return NineTaps;
  }
}

float normal_offset_texels(const FilterParams& params, float nDotL) noexcept {
  return (NormalOffsetBase + NormalOffsetSlope * (1.f - std::clamp(nDotL, 0.f, 1.f))) * params.normalScale;
}

float tap_radius(uint32_t taps) noexcept {
  float radius = 0.f;
  for (const std::array<float, 2>& o : tap_offsets(taps)) {
    radius = std::max({radius, std::abs(o[0]), std::abs(o[1])});
  }
  return radius + 1.f;
}

float receiver_ref(float depth, bool& outOfRange) noexcept {
  outOfRange = !(depth >= 0.f && depth <= 1.f);
  return std::isfinite(depth) ? std::clamp(depth, 0.f, 1.f) : 0.f;
}

ReceiverPage receiver_page(const ViewRecord& view, const std::array<float, 3>& pRel, float biasTexels) noexcept {
  const std::array<float, 3> c = view_coords(view, pRel);
  ReceiverPage out;
  out.texel = {c[0], c[1]};
  out.page = page_of(view, out.texel, out.local);
  out.ref = receiver_ref(c[2] + biasTexels * view.texel * view.depthScale, out.outOfRange);
  return out;
}

TapPage tap_page(const ViewRecord& view, const std::array<float, 2>& texel,
                 const std::array<float, 2>& offset) noexcept {
  TapPage out;
  out.page = page_of(view, {texel[0] + offset[0], texel[1] + offset[1]}, out.local);
  for (float& l : out.local) {
    l = std::clamp(l, TapInset, static_cast<float>(PageTexels) - TapInset);
  }
  return out;
}
} // namespace aurora::render::shadows::vsm
