#pragma once

#include <aurora/render.hpp>
#include <webgpu/webgpu_cpp.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#define AURORA_RENDER_GRAPH_VERSION 1

namespace aurora::render {
namespace detail {
struct PassBuilderState;
struct PassContextState;
} // namespace detail

enum class Access : uint8_t {
  Sampled,
  Uniform,
  StorageRead,
  StorageWrite,
  StorageRW,
  ColorWrite,
  ColorRMW,
  DepthRead,
  DepthWrite,
  CopySrc,
  CopyDst,
  Vertex,
  Index,
  Indirect,
};

enum class Missing : uint8_t {
  DisablePass,
  BindDummyBlack,
  BindDummyWhite,
};

enum class PassKind : uint8_t {
  Compute,
  Render,
  Copy,
  Encoder,
  Native,
};

enum class SizeClass : uint8_t {
  Absolute,
  Efb,
  Output,
};

struct TexDesc {
  wgpu::TextureFormat format = wgpu::TextureFormat::Undefined;
  wgpu::TextureDimension dimension = wgpu::TextureDimension::e2D;
  SizeClass sizeClass = SizeClass::Efb;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t depthOrLayers = 1;
  uint16_t divisor = 1;
  uint32_t mips = 1;
  wgpu::TextureUsage extraUsage = wgpu::TextureUsage::None;

  bool operator==(const TexDesc&) const = default;
};

struct BufDesc {
  uint64_t size = 0;
  SizeClass sizeClass = SizeClass::Absolute;
  uint32_t bytesPerTexel = 0;
  uint16_t divisor = 1;
  wgpu::BufferUsage extraUsage = wgpu::BufferUsage::None;
};

struct TexRef {
  uint16_t index = UINT16_MAX;

  constexpr bool valid() const noexcept { return index != UINT16_MAX; }
};

struct BufRef {
  uint16_t index = UINT16_MAX;

  constexpr bool valid() const noexcept { return index != UINT16_MAX; }
};

struct BufferView {
  wgpu::Buffer buffer;
  uint64_t offset = 0;
  uint64_t size = 0;
};

inline constexpr uint32_t AllMips = UINT32_MAX;

class PassBuilder {
public:
  PassBuilder(const PassBuilder&) = delete;
  PassBuilder& operator=(const PassBuilder&) = delete;

  TexRef create(std::string_view name, const TexDesc& desc, Access access);
  BufRef create_buffer(std::string_view name, const BufDesc& desc, Access access);
  TexRef history(std::string_view name, const TexDesc& desc, Access access);
  TexRef use(std::string_view name, Access access, Missing missing = Missing::DisablePass);
  BufRef use_buffer(std::string_view name, Access access, Missing missing = Missing::DisablePass);
  void after(std::string_view passOrPhase);
  void before(std::string_view passOrPhase);
  void side_effect();
  void manual_bindings();
  Stage stage() const noexcept;

protected:
  explicit PassBuilder(detail::PassBuilderState& state) noexcept : m_state(&state) {}
  ~PassBuilder() = default;

  detail::PassBuilderState* m_state;
};

class PassContext {
public:
  PassContext(const PassContext&) = delete;
  PassContext& operator=(const PassContext&) = delete;

  Stage stage() const noexcept;
  std::string_view name() const noexcept;
  uint32_t frame() const noexcept;
  std::span<const std::byte> params() const noexcept;
  const wgpu::Device& device() const noexcept;
  const wgpu::CommandEncoder& encoder() const noexcept;
  wgpu::ComputePassEncoder begin_compute();
  wgpu::RenderPassEncoder begin_render();
  wgpu::Texture texture(TexRef ref) const;
  wgpu::TextureView view(TexRef ref, uint32_t mip = AllMips) const;
  wgpu::TextureView prev(TexRef ref) const;
  bool prev_valid(TexRef ref) const noexcept;
  BufferView buffer(BufRef ref) const;
  wgpu::Extent3D extent(TexRef ref) const;
  wgpu::TextureFormat format(TexRef ref) const;
  uint32_t sample_count(TexRef ref) const;
  wgpu::ColorTargetState color_target(TexRef ref) const;
  const wgpu::BindGroupLayout& frame_layout() const noexcept;
  const wgpu::BindGroupLayout& pass_layout() const noexcept;
  const wgpu::PipelineLayout& pipeline_layout() const noexcept;

protected:
  explicit PassContext(detail::PassContextState& state) noexcept : m_state(&state) {}
  ~PassContext() = default;

  detail::PassContextState* m_state;
};

using PassId = uint64_t;
inline constexpr PassId InvalidPass = 0;
inline constexpr size_t MaxPassParamsSize = 128;

using PassDeclareCallback = void (*)(PassBuilder& builder, void* userdata);
using PassEnabledCallback = bool (*)(std::span<const std::byte> params, void* userdata);
using PassEncodeCallback = void (*)(PassContext& ctx, void* userdata);
using PassSubmittedCallback = void (*)(uint32_t frame, void* userdata);

struct ExternalPassDesc {
  const char* name = nullptr;
  Stage stage = Stage::BeforeHud;
  PassKind kind = PassKind::Render;
  uint16_t order = 0;
  uint32_t paramsSize = 0;
  PassDeclareCallback declare = nullptr;
  PassEnabledCallback enabled = nullptr;
  PassEncodeCallback encode = nullptr;
  PassSubmittedCallback afterSubmit = nullptr;
  void* userdata = nullptr;
};

PassId register_pass(const ExternalPassDesc& desc);
void unregister_pass(PassId id) noexcept;
bool set_pass_params(PassId id, const void* data, size_t size) noexcept;

namespace phase {
inline constexpr std::string_view ScenePrep = "phase.scene_prep";
inline constexpr std::string_view Capture = "phase.capture";
inline constexpr std::string_view Geometry = "phase.geometry";
inline constexpr std::string_view Prepare = "phase.prepare";
inline constexpr std::string_view Ao = "phase.ao";
inline constexpr std::string_view CloudShadows = "phase.cloud_shadows";
inline constexpr std::string_view Shadows = "phase.shadows";
inline constexpr std::string_view Gi = "phase.gi";
inline constexpr std::string_view Lighting = "phase.lighting";
inline constexpr std::string_view Reflections = "phase.reflections";
inline constexpr std::string_view Exposure = "phase.exposure";
inline constexpr std::string_view Sky = "phase.sky";
inline constexpr std::string_view Volumetrics = "phase.volumetrics";
inline constexpr std::string_view Composite = "phase.composite";
inline constexpr std::string_view Overlay = "gx.overlay";
inline constexpr std::string_view PreTaa = "phase.pre_taa";
inline constexpr std::string_view Taa = "phase.taa";
inline constexpr std::string_view PostTaa = "phase.post_taa";
inline constexpr std::string_view SceneEnd = "phase.scene_end";
inline constexpr std::string_view BeforeHud = "phase.before_hud";
inline constexpr std::string_view FrameEnd = "phase.frame_end";
} // namespace phase

namespace resource {
inline constexpr std::string_view GxColor = "gx.color";
inline constexpr std::string_view GxDepth = "gx.depth";
inline constexpr std::string_view GxNormal = "gx.normal";
inline constexpr std::string_view Motion = "gbuf.motion";
inline constexpr std::string_view Reactive = "gbuf.reactive";
inline constexpr std::string_view Fog = "gbuf.fog";
inline constexpr std::string_view Surface = "gbuf.surface";
inline constexpr std::string_view AoFinal = "ao.final";
inline constexpr std::string_view SceneHdr = "scene.hdr";
inline constexpr std::string_view SunVisibility = "vis.sun";
inline constexpr std::string_view RtShadow = "rt.shadow";
inline constexpr std::string_view GiIrradiance = "gi.irr";
inline constexpr std::string_view SsilIrradiance = "ssil.irr";
inline constexpr std::string_view ReflectionSpecular = "refl.spec";
inline constexpr std::string_view FogIntegrated = "fog.integrated";
inline constexpr std::string_view Clouds = "clouds.full";
inline constexpr std::string_view RadianceDefogged = "radiance.defogged";
} // namespace resource

enum class FrameBinding : uint32_t {
  Constants = 0,
  Params = 1,
  LinearClamp = 2,
  NearestClamp = 3,
  LinearRepeat = 4,
  DepthCompare = 5,
};

struct FrameConstants {
  uint32_t efbSize[2];
  uint32_t outputSize[2];
  float invEfbSize[2];
  float invOutputSize[2];
  uint32_t frameIndex;
  uint32_t flags;
  uint32_t msaaSamples;
  uint32_t pad0;
};
static_assert(sizeof(FrameConstants) == 48);
inline constexpr uint32_t FrameFlagReversedZ = 1u << 0;
inline constexpr uint32_t FrameFlagMultisampled = 1u << 1;

inline constexpr std::string_view FrameWgsl = R"(struct AuroraFrame {
  efb_size: vec2u,
  output_size: vec2u,
  inv_efb_size: vec2f,
  inv_output_size: vec2f,
  frame_index: u32,
  flags: u32,
  msaa_samples: u32,
  pad0: u32,
};
@group(0) @binding(0) var<uniform> aurora_frame: AuroraFrame;
)";
} // namespace aurora::render
