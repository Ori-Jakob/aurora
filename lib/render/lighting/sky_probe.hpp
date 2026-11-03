#pragma once

#include "../graph.hpp"
#include "../stage_camera.hpp"

#include <aurora/render.hpp>
#include <webgpu/webgpu_cpp.h>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

namespace aurora::render::detail {
struct SceneHints;
struct StageData;
} // namespace aurora::render::detail

namespace aurora::render::detail::lighting::sky_probe {
inline constexpr std::string_view ProjectPass = "sky.probe_project";
inline constexpr std::string_view FilterPass = "sky.probe_filter";
inline constexpr std::string_view ViewPass = "sky.probe_view";
inline constexpr std::string_view Radiance = "sky.probe.radiance";
inline constexpr std::string_view Probe = "sky.probe";
inline constexpr std::string_view Sh9 = "sky.sh9";
inline constexpr std::string_view ViewResource = "sky.probe.view";
inline constexpr std::string_view Shader = "lighting/sky_probe";
inline constexpr std::string_view ViewShader = "lighting/sky_probe_view";
inline constexpr std::string_view HookEnv = "AURORA_RENDER_SKY_PROBE";
inline constexpr std::string_view MaskRow = "sky.probe.mask";
inline constexpr wgpu::TextureFormat Format = wgpu::TextureFormat::RGBA16Float;

inline constexpr uint32_t ProbeSize = 128;
inline constexpr uint32_t ProbeMips = 6;
inline constexpr uint32_t ShMip = 2;
inline constexpr uint32_t ShSize = ProbeSize >> ShMip;
inline constexpr uint32_t TapsPerAxis = 4;
inline constexpr uint32_t Sh9Bytes = 224;
inline constexpr float BlendTau = 0.1f;
inline constexpr float MaxDeltaMs = 250.f;
inline constexpr uint32_t ProjectOrder = 3;
inline constexpr uint32_t FilterOrder = 5;
inline constexpr float GainMin = 0x1p-8f;
inline constexpr float GainMax = 0x1p4f;
inline constexpr float AmbientFloor = 0x1p-10f;
inline constexpr float ScaleMax = 16.f;
inline constexpr uint32_t ProjectGroup = 16;
inline constexpr uint32_t FilterGroup = 8;
inline constexpr uint32_t ViewGroup = 8;
static_assert(ProbeSize >> (ProbeMips - 1) == 4 && ShSize == 32);

enum class Mode : uint32_t { Raw = 0, Pixel = 1, PixelFlat = 2 };
enum class Policy : uint32_t { MatchAmbient, Physical };
enum class HookMode : uint8_t { Default, Off, Constant, Physical, Global };

struct Hook {
  HookMode mode = HookMode::Default;
  float constant = 0.f;
  bool operator==(const Hook&) const = default;
};

enum ProjectFlags : uint32_t {
  ProjectObserve = 1u << 0,
  ProjectReset = 1u << 1,
};

enum FilterFlags : uint32_t {
  FilterConstant = 1u << 0,
  FilterHasEnv = 1u << 1,
  FilterPhysical = 1u << 2,
  FilterGlobal = 1u << 3,
  FilterObserved = 1u << 4,
};

struct ProjectParams {
  StageCamera camera;
  ViewRotation rotation;
  uint32_t flags = 0;
  float alpha = 1.f;
  uint32_t pad[2]{};
};
static_assert(sizeof(ProjectParams) == 112 && sizeof(ProjectParams) <= MaxPassParamsSize);

struct FilterParams {
  ViewRotation rotation;
  float fog[4]{};
  float ambient[4]{};
  uint32_t flags = 0;
  float constant = 0.f;
  uint32_t pad[2]{};
};
static_assert(sizeof(FilterParams) == 96);

struct ViewParams {
  uint32_t mode = 0;
  uint32_t pad[3]{};
};
static_assert(sizeof(ViewParams) == 16);

// Mirrors aurora/sky's SkySh9
struct Sh9Block {
  float c[9][4]{};
  float info[4]{};
  float amb[4]{};
  float rot[3][4]{};
};
static_assert(sizeof(Sh9Block) == Sh9Bytes);

constexpr TexDesc radiance_desc() noexcept {
  // Sampled through manual groups, so the usage cannot depend on a consumer being active
  return {
      .format = Format,
      .sizeClass = SizeClass::Absolute,
      .width = ProbeSize,
      .height = ProbeSize,
      .extraUsage = wgpu::TextureUsage::TextureBinding,
  };
}

constexpr TexDesc probe_desc() noexcept {
  TexDesc desc = radiance_desc();
  desc.mips = ProbeMips;
  return desc;
}

[[nodiscard]] std::optional<Hook> parse_hook(std::string_view text) noexcept;
[[nodiscard]] Hook hook() noexcept;
[[nodiscard]] Policy policy_of(const Hook& hook) noexcept;
[[nodiscard]] float blend_alpha(float deltaMs) noexcept;
void fill_project_params(const SceneHints* hints, uint32_t skyDraws, bool constant, float alpha,
                         ProjectParams& out) noexcept;
bool fill_filter_params(const SceneHints* hints, const Hook& hook, bool observed, FilterParams& out) noexcept;

[[nodiscard]] bool wanted(const PassSetup& setup) noexcept;
[[nodiscard]] bool armed() noexcept;
[[nodiscard]] bool ready(const Settings& settings) noexcept;
[[nodiscard]] bool producing() noexcept;
PassDesc project_pass_desc();
PassDesc filter_pass_desc();
PassDesc view_pass_desc();
void initialize();
void shutdown() noexcept;
void on_latch(const PassSetup& latched, const CompiledGraph* graph);
void worker_tick(uint64_t frameId) noexcept;

namespace testing {
struct Counters {
  uint32_t observed = 0;
  uint32_t held = 0;
  uint32_t resets = 0;
  uint32_t filtered = 0;
  uint32_t constant = 0;
  uint32_t viewDraws = 0;
};
[[nodiscard]] Counters counters() noexcept;
void reset() noexcept;
void set_hints_override(std::shared_ptr<const SceneHints> hints) noexcept;
void set_hook(std::optional<Hook> hook) noexcept;
void set_sky_draws(std::optional<uint32_t> draws) noexcept;
void set_alpha(std::optional<float> alpha) noexcept;

using Vec3 = std::array<double, 3>;
using Sh = std::array<Vec3, 9>;

// CPU mirrors of aurora/sky and lighting/sky_probe, in double
[[nodiscard]] Vec3 oct_decode(double u, double v) noexcept;
[[nodiscard]] std::array<double, 2> oct_encode(const Vec3& dir) noexcept;
[[nodiscard]] double texel_weight(double u, double v, double size) noexcept;
[[nodiscard]] std::array<double, 9> sh_basis(const Vec3& d) noexcept;
[[nodiscard]] Sh sh_project(const std::vector<Vec3>& texels, uint32_t size, const Vec3& base = {});
[[nodiscard]] Vec3 sh_irradiance(const Sh& sh, const Vec3& n) noexcept;
[[nodiscard]] std::array<double, 3> project_fwd(const StageCamera& camera, const Vec3& fwd, double width,
                                                double height) noexcept;
[[nodiscard]] Vec3 view_dir_fwd(const ViewRotation& rotation, const Vec3& world) noexcept;
[[nodiscard]] Vec3 world_dir(const ViewRotation& rotation, const Vec3& fwd) noexcept;
[[nodiscard]] double gain(Policy policy, bool hasEnv, const Vec3& linearAmbient, double upperMean) noexcept;
[[nodiscard]] Mode mode_for(const Hook& hook, bool hasEnv, const Vec3& linearAmbient) noexcept;
[[nodiscard]] Vec3 pixel_scale(Mode mode, const Vec3& probeAmbient, const Vec3& a) noexcept;
[[nodiscard]] Vec3 env_irradiance(Mode mode, const Sh& sh, const Vec3& probeAmbient, const Vec3& n, const Vec3& a,
                                  bool baked) noexcept;
[[nodiscard]] double luminance(const Vec3& rgb) noexcept;
[[nodiscard]] double srgb_to_linear(double stored) noexcept;
} // namespace testing
} // namespace aurora::render::detail::lighting::sky_probe
