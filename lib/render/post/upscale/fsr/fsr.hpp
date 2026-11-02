#pragma once

#include "../../../graph.hpp"
#include "../../../shaders.hpp"

#include <aurora/math.hpp>
#include <aurora/render.hpp>
#include <webgpu/webgpu_cpp.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

namespace aurora::render::detail::upscale::fsr {
inline constexpr uint32_t MinRenderSize = 32;
inline constexpr uint32_t BasePhases = 8;
inline constexpr uint32_t TileSize = 64;
inline constexpr uint32_t GroupSize = 8;
inline constexpr uint32_t MaxSpdMips = 12;
inline constexpr uint32_t LumaTailMip = 6;
inline constexpr uint32_t MaxStorageTextures = 4;
inline constexpr float NominalFrameMs = 1000.f / 60.f;
inline constexpr float MotionScale[2] = {-0.5f, 0.5f};

enum class Stage : uint8_t {
  PrepareInputs,
  LumaPyramidA,
  LumaPyramidB,
  ShadingChangePyramid,
  ShadingChange,
  PrepareReactivity,
  LumaInstability,
  Accumulate,
};
inline constexpr size_t StageCount = static_cast<size_t>(Stage::Accumulate) + 1;

struct Params {
  float jitter[2] = {0.f, 0.f};
  float prevJitter[2] = {0.f, 0.f};
  float motionScale[2] = {MotionScale[0], MotionScale[1]};
  float deviceToView[2] = {0.f, 1.f};
  float jitterSequenceLength = static_cast<float>(BasePhases);
  float deltaTime = NominalFrameMs / 1000.f;
  float viewToMeters = 1.f;
  float frameIndex = 0.f;
  float velocityFactor = 1.f;
  float reactivenessScale = 1.f;
  float shadingChangeScale = 1.f;
  float accumulationPerFrame = 1.f / 3.f;
  float minDisocclusionAccumulation = -1.f / 3.f;
  float reactiveMax = 1.f;
  uint32_t spdMips = 0;
  uint32_t pad = 0;
};
static_assert(sizeof(Params) == 80);

struct Camera {
  float nearZ = 0.f;
  float farZ = 0.f;
  float fovY = 0.f;
  float viewToMeters = 1.f;
};

struct FrameInputs {
  float jitter[2] = {0.f, 0.f};
  float prevJitter[2] = {0.f, 0.f};
  float motionScale[2] = {MotionScale[0], MotionScale[1]};
  Camera camera;
  float frameTimeMs = NominalFrameMs;
  uint32_t framesSinceReset = 0;
  uint32_t jitterPhases = BasePhases;
  GraphExtent render;
};

struct Selection {
  UpscalerQuality quality = UpscalerQuality::Quality;
  GraphExtent render;
  GraphExtent output;
  bool locksR8 = false;
};

struct Refs {
  TexRef dilatedDepth;
  TexRef dilatedMotion;
  TexRef farthestDepth;
  TexRef farthestDepthMip1;
  TexRef lumaMip5;
  TexRef spd;
  TexRef shadingChange;
  TexRef dilatedReactive;
  TexRef lumaInstability;
  TexRef luma;
  TexRef accumulation;
  TexRef frameInfo;
  TexRef lumaHistory;
  TexRef upscaled;
  TexRef locks;
  BufRef recon;
};

struct Pipelines {
  std::array<shaders::PipelineId, StageCount> ids{};
  bool locksR8 = false;
  bool linear = false;
};

struct FrameGroup {
  wgpu::BindGroup group;
  std::array<uint32_t, 2> offsets{};
  uint32_t offsetCount = 0;
};

struct Bindings {
  wgpu::TextureView color;
  wgpu::TextureView depth;
  wgpu::TextureView motion;
  wgpu::TextureView reactive;
  wgpu::TextureView dilatedDepth;
  wgpu::TextureView dilatedMotion;
  wgpu::TextureView farthestDepth;
  wgpu::TextureView farthestDepthMip1;
  wgpu::TextureView lumaMip5;
  wgpu::TextureView spd;
  std::array<wgpu::TextureView, 3> spdMips;
  wgpu::TextureView shadingChange;
  wgpu::TextureView dilatedReactive;
  wgpu::TextureView lumaInstability;
  wgpu::TextureView luma;
  wgpu::TextureView prevLuma;
  wgpu::TextureView accumulation;
  wgpu::TextureView prevAccumulation;
  wgpu::TextureView frameInfo;
  wgpu::TextureView prevFrameInfo;
  wgpu::TextureView lumaHistory;
  wgpu::TextureView prevLumaHistory;
  wgpu::TextureView upscaled;
  wgpu::TextureView prevUpscaled;
  wgpu::TextureView locks;
  BufferView recon;
};

[[nodiscard]] float mode_ratio(UpscalerQuality quality) noexcept;
[[nodiscard]] GraphExtent render_size(GraphExtent output, UpscalerQuality quality) noexcept;
[[nodiscard]] uint32_t jitter_phases(float ratio) noexcept;
[[nodiscard]] uint32_t spd_mips(GraphExtent render) noexcept;
[[nodiscard]] GraphExtent spd_tiles(GraphExtent render) noexcept;
[[nodiscard]] std::optional<Camera> camera_from_projection(const Mat4x4<float>& gxProj, float unitsPerMeter) noexcept;
[[nodiscard]] float frame_delta_ms(std::optional<std::chrono::steady_clock::time_point>& last,
                                   std::chrono::steady_clock::time_point now, bool blocking) noexcept;
[[nodiscard]] Params make_params(const FrameInputs& inputs) noexcept;
[[nodiscard]] std::string_view shader_id(Stage stage) noexcept;

Refs declare(PassBuilder& builder, const Selection& selection);
bool request_pipelines(const wgpu::BindGroupLayout& frameLayout, uint64_t layoutKey, bool locksR8, bool linear,
                       Pipelines& out);
[[nodiscard]] bool pipelines_ready(const Pipelines& pipelines) noexcept;
bool encode(const wgpu::CommandEncoder& cmd, const Pipelines& pipelines, const FrameGroup& frame,
            const Bindings& bindings, GraphExtent render, GraphExtent output, Stage last = Stage::Accumulate);
void shutdown() noexcept;

namespace testing {
enum class Kind : uint8_t { Texture, UnfilterableTexture, StorageWrite, StorageReadWrite, Buffer, ReadOnlyBuffer };

struct Binding {
  Kind kind = Kind::Texture;
  wgpu::TextureFormat format = wgpu::TextureFormat::Undefined;
};

struct Inputs {
  wgpu::Texture color;
  wgpu::Texture depth;
  wgpu::Texture motion;
  wgpu::Texture reactive;
  FrameInputs frame;
};

struct State {
  GraphExtent render;
  GraphExtent output;
  bool locksR8 = false;
  bool linear = false;
  uint32_t frames = 0;
  wgpu::Texture dilatedDepth;
  wgpu::Texture dilatedMotion;
  wgpu::Texture farthestDepth;
  wgpu::Texture farthestDepthMip1;
  wgpu::Texture lumaMip5;
  wgpu::Texture spd;
  wgpu::Texture shadingChange;
  wgpu::Texture dilatedReactive;
  wgpu::Texture lumaInstability;
  std::array<wgpu::Texture, 2> luma;
  std::array<wgpu::Texture, 2> accumulation;
  std::array<wgpu::Texture, 2> frameInfo;
  std::array<wgpu::Texture, 2> lumaHistory;
  std::array<wgpu::Texture, 2> upscaled;
  wgpu::Texture locks;
  wgpu::Buffer recon;

  [[nodiscard]] size_t current() const noexcept { return (frames + 1) % 2; }
};

[[nodiscard]] std::span<const Binding> layout(Stage stage, bool locksR8) noexcept;
bool run(State& state, const Inputs& inputs, Stage last = Stage::Accumulate);
} // namespace testing
} // namespace aurora::render::detail::upscale::fsr
