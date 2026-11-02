#pragma once

#include "../graph.hpp"
#include "../shaders.hpp"

#include <aurora/render.hpp>
#include <webgpu/webgpu_cpp.h>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace aurora::render::detail {
struct FrameState;
struct SceneHints;
} // namespace aurora::render::detail

namespace aurora::render::detail::composite {
inline constexpr std::string_view CopyPass = "legacy.copy";
inline constexpr std::string_view Pass = "composite.main";
inline constexpr std::string_view LegacyTex = "legacyTex";
inline constexpr std::string_view Shader = "post/composite";
inline constexpr std::string_view NoFogEnv = "AURORA_RENDER_COMPOSITE_NOFOG";
inline constexpr wgpu::TextureFormat AoFormat = wgpu::TextureFormat::RGBA8Unorm;

// Bindings follow declaration order, so later terms append and never renumber
enum Binding : uint32_t { BindLegacy = 0, BindFog = 1, BindNormal = 2, BindAo = 3 };

enum ParamFlags : uint32_t {
  FlagIdentity = 1u << 0,
  FlagNoFog = 1u << 1,
};

struct Params {
  uint32_t flags = 0;
  uint32_t pad[3]{};
};
static_assert(sizeof(Params) == 16);

[[nodiscard]] bool wanted(const Settings& settings) noexcept;
[[nodiscard]] bool declare_wanted(const PassSetup& setup) noexcept;
[[nodiscard]] uint32_t params_flags(bool identity, const char* noFog) noexcept;
[[nodiscard]] std::optional<std::string> check_contract(const CompiledGraph& graph);
[[nodiscard]] shaders::RenderPipelineDesc pipeline_desc(wgpu::TextureFormat efb, const wgpu::PipelineLayout& layout,
                                                        uint64_t layoutKey);
PassDesc copy_pass_desc();
PassDesc pass_desc();
void initialize();
void shutdown() noexcept;
void on_latch(const PassSetup& latched, const CompiledGraph* graph);
[[nodiscard]] bool ready(const Settings& settings) noexcept;
[[nodiscard]] bool live() noexcept;
[[nodiscard]] bool enabled_this_frame(uint64_t frameId) noexcept;
[[nodiscard]] bool identity_forced() noexcept;
[[nodiscard]] bool ao_gate(const FrameState& state) noexcept;

namespace testing {
struct Counters {
  uint32_t copies = 0;
  uint32_t draws = 0;
  uint32_t skippedMismatch = 0;
  uint32_t skippedNoScene = 0;
  uint32_t skippedView = 0;
  uint32_t contractFailures = 0;
};
Counters counters() noexcept;
void reset() noexcept;
void set_flags(std::optional<uint32_t> flags) noexcept;
void set_camera_override(std::shared_ptr<const SceneHints> hints) noexcept;
[[nodiscard]] std::array<float, 3> legacy_ao(std::array<float, 3> legacy, std::array<float, 3> fogColor, uint32_t q7,
                                             float ao, float normalAlpha) noexcept;
} // namespace testing
} // namespace aurora::render::detail::composite
