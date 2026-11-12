#include "decomp_view.hpp"

#include "gbuf_view.hpp"

#include "../fakeshadow/lsm.hpp"
#include "../post/composite.hpp"
#include "../../webgpu/gpu.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace aurora::render::debug::decomp_view {
namespace {
constexpr std::array<std::string_view, 4> ModeDefines{"DECOMP_IDENTITY", "DECOMP_COVERAGE", "DECOMP_MULTIPLY",
                                                      "DECOMP_LSM"};

struct Decoded {
  std::array<float, 3> s{};
  bool bad = false;
};

std::span<const std::string_view> mode_defines(DecompMode mode) noexcept {
  if (mode == DecompMode::Defog) {
    return {};
  }
  return std::span{ModeDefines}.subspan(static_cast<size_t>(mode) - 1, 1);
}

std::string_view mode_input(DecompMode mode) noexcept {
  switch (mode) {
  case DecompMode::Identity:
    return detail::composite::LegacyTex;
  case DecompMode::Coverage:
    return resource::Reactive;
  case DecompMode::Multiply:
    return resource::AoFinal;
  case DecompMode::Lsm:
    return fakeshadow::lsm::Resource;
  case DecompMode::Defog:
    break;
  }
  return {};
}

float unorm(uint8_t value) noexcept { return static_cast<float>(value) / 255.f; }

uint8_t to_unorm(float value) noexcept {
  return static_cast<uint8_t>(std::lround(std::clamp(value, 0.f, 1.f) * 255.f));
}

Decoded decode(Texel legacy, Texel rec) noexcept {
  Decoded out;
  const float q = static_cast<float>(rec[3] >> 1) / 127.f;
  for (size_t c = 0; c < 3; ++c) {
    out.s[c] = unorm(legacy[c]) - q * unorm(rec[c]);
    out.bad = out.bad || out.s[c] < -2.f / 255.f;
  }
  return out;
}

bool live_creator(const detail::CompiledGraph& graph, std::string_view name) noexcept {
  const detail::ResourceId id = graph.find_resource(name);
  if (id == detail::InvalidResource) {
    return false;
  }
  const detail::NodeId creator = graph.resources[id].creator;
  return creator != detail::InvalidNode && graph.nodes[creator].status == detail::CullReason::None;
}
} // namespace

bool declarable(const GbufView& row, const detail::PassSetup& setup) noexcept {
  if (!setup.gx_available(detail::GxResource::Fog) || !setup.gx_available(detail::GxResource::Normal)) {
    return false;
  }
  return decomp_mode(row) != DecompMode::Coverage || setup.gx_available(detail::GxResource::Reactive);
}

void declare(detail::InternalPassBuilder& builder, const detail::PassSetup& setup, const GbufView& row) {
  if (!declarable(row, setup)) {
    builder.skip();
    return;
  }
  const detail::UseOptions sampled{.missing = Missing::DisablePass,
                                   .sampleType = wgpu::TextureSampleType::UnfilterableFloat};
  // Each mode declares only what it decodes, so the view never keeps a culled producer alive
  builder.use(resource::GxColor, Access::Sampled, sampled);
  builder.use(resource::Fog, Access::Sampled,
              detail::UseOptions{.missing = Missing::DisablePass, .sampleType = wgpu::TextureSampleType::Uint});
  builder.use(resource::GxNormal, Access::Sampled, sampled);
  if (decomp_mode(row) == DecompMode::Lsm) {
    // A frame without kept fakes has no mask, which reads as white like the composite's dummy
    builder.use(fakeshadow::lsm::Resource, Access::Sampled, Missing::BindDummyWhite);
    builder.use(resource::SceneHdr, Access::Sampled, Missing::BindDummyBlack);
  } else if (const std::string_view input = mode_input(decomp_mode(row)); !input.empty()) {
    builder.use(input, Access::Sampled, sampled);
  }
  builder.create(gbuf_view::SnapshotResource, {.format = wgpu::TextureFormat::RGBA8Unorm, .sizeClass = SizeClass::Efb},
                 Access::ColorWrite);
}

shaders::RenderPipelineDesc pipeline_desc(DecompMode mode) {
  shaders::RenderPipelineDesc desc{
      .label = gbuf_view::SnapshotPass,
      .shader = Shader,
      .vertexEntry = "vs_main",
      .fragmentEntry = "fs_main",
      .defines = mode_defines(mode),
      .colorCount = 1,
  };
  desc.colors[0] = {.format = wgpu::TextureFormat::RGBA8Unorm};
  return desc;
}

void pipelines(const detail::PipelineSetup& setup, const GbufView& row,
               std::span<uint32_t, detail::MaxPassPipelines> ids) {
  // One pass per selected row, and a pass holds at most four pipelines
  shaders::RenderPipelineDesc desc = pipeline_desc(decomp_mode(row));
  desc.layout = setup.layout;
  desc.layoutKey = setup.layoutHash;
  ids[0] = shaders::request_render(desc).value;
}

bool available(const GbufView& row, const Settings& settings, const detail::CompiledGraph* graph) {
  if (row.kind != GbufViewKind::Decomposition || !webgpu::g_graphicsConfig.gbuffer || !webgpu::normal_present()) {
    return false;
  }
  switch (decomp_mode(row)) {
  case DecompMode::Multiply:
    return feature_enabled(settings, Feature::Ao);
  case DecompMode::Identity:
    return graph != nullptr && live_creator(*graph, detail::composite::LegacyTex);
  case DecompMode::Lsm:
    return feature_enabled(settings, Feature::Pbr);
  case DecompMode::Defog:
  case DecompMode::Coverage:
    break;
  }
  return true;
}

Rgb8 identity_color(Texel legacy, Texel out, Texel rec, uint32_t code) noexcept {
  if (out[3] != legacy[3]) {
    return identity_class_color(IdentityClass::AlphaChanged);
  }
  if (out[0] == legacy[0] && out[1] == legacy[1] && out[2] == legacy[2]) {
    return identity_class_color(IdentityClass::Unchanged);
  }
  if (code == 0) {
    return identity_class_color(IdentityClass::CodeZero);
  }
  const Decoded decoded = decode(legacy, rec);
  bool up = false;
  bool explained = true;
  uint32_t d = 0;
  for (size_t c = 0; c < 3; ++c) {
    const bool lifted = out[c] > legacy[c];
    up = up || lifted;
    explained = explained && (!lifted || decoded.s[c] < 0.f);
    d = std::max(d, static_cast<uint32_t>(std::abs(int{out[c]} - int{legacy[c]})));
  }
  if (up) {
    return identity_class_color(explained ? IdentityClass::Lifted : IdentityClass::Brightened);
  }
  const uint8_t grey = identity_grey(d);
  return {grey, grey, grey};
}

Rgb8 coverage_color(Texel rec, uint32_t code, Texel legacy, uint8_t reactive) noexcept {
  if (decode(legacy, rec).bad) {
    return coverage_class_color(CoverageClass::Bad);
  }
  const uint32_t q7 = rec[3] >> 1;
  if (code == 0) {
    return coverage_class_color(q7 > 0 ? CoverageClass::FoggedNonReceiver : CoverageClass::Untouched);
  }
  const bool lock = (rec[3] & 1u) != 0;
  const CoverageClass base =
      q7 == 0 ? CoverageClass::Receiver : (lock ? CoverageClass::LockedReceiver : CoverageClass::FoggedReceiver);
  return reactive > 0 ? coverage_touched(coverage_class_color(base)) : coverage_class_color(base);
}

Rgb8 defog_color(Texel legacy, Texel rec) noexcept {
  const Decoded decoded = decode(legacy, rec);
  if (decoded.bad) {
    return DefogBadColor;
  }
  return {to_unorm(decoded.s[0]), to_unorm(decoded.s[1]), to_unorm(decoded.s[2])};
}

Rgb8 multiply_color(Texel legacy, uint32_t code, uint8_t ao) noexcept {
  if (code == 0) {
    return {legacy[0], legacy[1], legacy[2]};
  }
  const float a = unorm(ao);
  return {to_unorm(unorm(legacy[0]) * a), to_unorm(unorm(legacy[1]) * a), to_unorm(unorm(legacy[2]) * a)};
}
} // namespace aurora::render::debug::decomp_view
