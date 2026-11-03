#pragma once

#include "../graph.hpp"
#include "../stage_camera.hpp"
#include "clusters.hpp"

#include <aurora/render.hpp>
#include <webgpu/webgpu_cpp.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>

namespace aurora::render::material::classes {
struct GpuRow;
} // namespace aurora::render::material::classes

namespace aurora::render::detail {
struct SceneHints;
struct StageData;
} // namespace aurora::render::detail

namespace aurora::render::detail::lighting {
[[nodiscard]] bool ready(const Settings& settings) noexcept;

namespace testing {
void set_ready_override(std::optional<bool> ready) noexcept;
} // namespace testing
} // namespace aurora::render::detail::lighting

namespace aurora::render::detail::lighting::deferred {
inline constexpr std::string_view Pass = "light.deferred";
inline constexpr std::string_view Shader = "lighting/deferred";
inline constexpr std::string_view ModeEnv = "AURORA_RENDER_LIGHTING";
inline constexpr wgpu::TextureFormat HdrFormat = wgpu::TextureFormat::RGBA16Float;
inline constexpr uint32_t WorkgroupSize = 8;
inline constexpr float HdrMax = 65504.f;

// Later milestones append bindings 14 and up and never renumber these
enum Binding : uint32_t {
  BindSurface = 0,
  BindNormal = 1,
  BindDepth = 2,
  BindAo = 3,
  BindList = 4,
  BindGrid = 5,
  BindClasses = 6,
  BindLut3d = 7,
  BindLut2d = 8,
  BindHdr = 9,
  BindSurfaceExt = 10,
  BindSurfaceTable = 11,
  BindSkyProbe = 12,
  BindSkySh9 = 13,
};

enum class Mode : uint32_t { Full = 0, Grey = 1, Direct = 2, Indirect = 3, Furnace = 4, FurnaceMetal = 5 };

enum ParamFlags : uint32_t {
  FlagSurfaceAmbient = 1u << 0,
};
inline constexpr uint32_t ModeShift = 1;
inline constexpr uint32_t ModeMask = 7u << ModeShift;

struct Params {
  StageCamera camera;
  float ambient[3]{};
  uint32_t flags = 0;
  // The sky probe is stored over world directions, and shading runs in the view-forward frame
  ViewRotation viewToWorld;
  uint32_t reserved[4]{};
};
static_assert(sizeof(Params) == 128 && sizeof(Params) <= MaxPassParamsSize);
static_assert(offsetof(Params, ambient) == 48 && offsetof(Params, viewToWorld) == 64);

[[nodiscard]] std::optional<Mode> parse_mode(std::string_view text) noexcept;
[[nodiscard]] std::array<float, 3> srgb_to_linear(const float (&stored)[3]) noexcept;
void fill_params(const SceneHints* hints, bool surfaceAmbient, Mode mode, Params& out) noexcept;

// The pass's enabled hook and the composite's per-frame term, so H exists exactly when the composite expects it
[[nodiscard]] bool frame_enabled(const StageData& stage) noexcept;
[[nodiscard]] bool ready(const Settings& settings) noexcept;
[[nodiscard]] bool ran(const StageData& opaqueDone) noexcept;
PassDesc pass_desc();
void initialize();
void shutdown() noexcept;
void on_latch(const PassSetup& latched, const CompiledGraph* graph);

namespace testing {
struct Counters {
  uint32_t ran = 0;
  uint32_t armed = 0;
  uint32_t identity = 0;
  uint32_t noLights = 0;
  uint32_t notResident = 0;
  uint32_t pending = 0;
};
[[nodiscard]] Counters counters() noexcept;
void reset() noexcept;
void set_hints_override(std::shared_ptr<const SceneHints> hints) noexcept;
void set_mode(std::optional<Mode> mode) noexcept;
void force_enabled(std::optional<bool> enabled) noexcept;

using Vec3 = std::array<double, 3>;

// One pixel's decoded inputs, in the view-forward frame light.list uses
struct Pixel {
  Vec3 albedo{};
  double ao = 1.0;
  double roughness = 0.85;
  double metalness = 0.0;
  uint32_t classId = 1;
  uint32_t flags = 0;
  uint32_t category = 0;
  Vec3 ambient{};
  Vec3 emissive{};
  Vec3 normal{0.0, 0.0, -1.0};
  Vec3 position{0.0, 0.0, 100.0};
  double depth = 0.5;
  double aoFinal = 1.0;
  uint32_t receiver = 1;
  bool cameraValid = true;
};

struct Result {
  Vec3 rgb{};
  double alpha = 0.0;
};

[[nodiscard]] Result shade(const Pixel& pixel, const clusters::LightListHeader& header,
                           std::span<const clusters::GpuLight> lights, const Params& params,
                           std::span<const material::classes::GpuRow, 256> rows);
[[nodiscard]] double attenuation(const clusters::GpuLight& light, double distance) noexcept;
[[nodiscard]] bool category_match(const clusters::GpuLight& light, uint32_t category) noexcept;
[[nodiscard]] Vec3 bend_normal(const Vec3& normal, const Vec3& view) noexcept;
} // namespace testing
} // namespace aurora::render::detail::lighting::deferred
