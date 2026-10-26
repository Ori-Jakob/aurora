#pragma once

#include "../../graph.hpp"
#include "../../shaders.hpp"
#include "../../stage.hpp"

#include <aurora/render.hpp>
#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace aurora::render::detail::temporal_resolve {
struct Resolver;
} // namespace aurora::render::detail::temporal_resolve

namespace aurora::render::detail::upscale {
struct Signs;
} // namespace aurora::render::detail::upscale

namespace aurora::render::detail::upscale::native_upscale {
inline constexpr std::string_view Pass = "aurora.native_upscale";
inline constexpr std::string_view InputShader = "post/upscale/native_inputs";

enum class Backend : uint8_t { D3D12 };
enum class Warmup : uint8_t { Ready, Pending, Failed };

struct Adapter {
  Backend backend = Backend::D3D12;
  void* device = nullptr;
  void* queue = nullptr;
  void* physical = nullptr;
  uint64_t luid = 0;
  uint32_t vendorId = 0;
  uint32_t deviceId = 0;
  uint32_t subSysId = 0;
  uint32_t revision = 0;
  uint64_t driverVersion = 0;
};

struct ProbeResult {
  FeatureSupport support{false, Reason::UnsupportedGpu};
  std::string path;
  bool autoPick = true;
};

struct Image {
  void* resource = nullptr;
  wgpu::TextureFormat format = wgpu::TextureFormat::Undefined;
  GraphExtent size;
};

struct Camera {
  float nearZ = 0.f;
  float farZ = 0.f;
  float fovY = 0.f;
  float viewToMeters = 0.f;
};

struct Inputs {
  void* commandList = nullptr;
  Image color;
  Image depth;
  Image motion;
  Image reactive;
  Image output;
  float jitter[2]{};
  float motionScale[2]{};
  Camera camera;
  float frameTimeMs = 0.f;
  float sharpness = 0.f;
  bool reset = true;
};

class NativeUpscaler {
public:
  virtual ~NativeUpscaler() = default;

  [[nodiscard]] virtual Upscaler option() const noexcept = 0;
  [[nodiscard]] virtual std::string_view name() const noexcept = 0;
  [[nodiscard]] virtual std::string_view runtime_file() const noexcept { return {}; }
  [[nodiscard]] virtual bool configured() const noexcept { return true; }
  virtual ProbeResult probe(const Adapter& adapter) noexcept = 0;
  [[nodiscard]] virtual std::string_view display_name(UpscalerQuality quality) const noexcept = 0;
  [[nodiscard]] virtual ColorEncoding color_encoding() const noexcept = 0;
  [[nodiscard]] virtual bool wants_reactive() const noexcept = 0;
  // A pass-through evaluator copies In_color to Out, so Out keeps the render size and In_color's format
  [[nodiscard]] virtual bool passthrough() const noexcept { return false; }
  virtual std::optional<GraphExtent> prepare(const Adapter& adapter, GraphExtent output,
                                             UpscalerQuality quality) noexcept = 0;
  [[nodiscard]] virtual Reason prepare_failure() const noexcept { return Reason::NativeUnavailable; }
  virtual Warmup warmup() noexcept { return Warmup::Ready; }
  virtual bool evaluate(const Inputs& inputs) noexcept = 0;
  virtual void destroy() noexcept = 0;
  [[nodiscard]] virtual uint64_t vram_bytes() const noexcept = 0;
  [[nodiscard]] virtual uint32_t jitter_phases(float ratio) const noexcept = 0;
  [[nodiscard]] virtual float mip_bias(float ratio) const noexcept = 0;
};

struct Plan {
  NativeUpscaler* upscaler = nullptr;
  UpscalerQuality quality = UpscalerQuality::Quality;
  GraphExtent output;

  bool operator==(const Plan&) const = default;
};

struct Frame {
  GraphExtent render;
  GraphExtent output;
  wgpu::TextureFormat colorFormat = wgpu::TextureFormat::Undefined;
  float jitter[2]{};
  float motionScale[2]{};
  Camera camera;
  float frameTimeMs = 0.f;
  float sharpness = 0.f;
  bool reset = true;
};

struct Targets {
  wgpu::Texture color;
  wgpu::TextureView colorView;
  wgpu::TextureView depthView;
  wgpu::Texture motion;
  wgpu::Texture reactive;
  ColorEncoding encoding = ColorEncoding::Stored;
};

struct Selection {
  const temporal_resolve::Resolver* resolver = nullptr;
  GraphExtent scene;
  ColorEncoding encoding = ColorEncoding::Stored;
};

struct Params {
  float frameTimeMs = 1000.f / 60.f;
  float sharpness = 0.f;
  float jitterSign[2] = {1.f, 1.f};
  float motionSign[2] = {1.f, 1.f};
  uint32_t pad[2]{};
};
static_assert(sizeof(Params) == 32);

PassDesc pass_desc();
void initialize();
void shutdown() noexcept;
[[nodiscard]] NativeUpscaler* stand_in(float ratio) noexcept;
[[nodiscard]] std::optional<Selection> select(NativeUpscaler& upscaler, UpscalerQuality quality, GraphExtent output);
bool prepare_pipelines(ColorEncoding encoding);
void installed(const temporal_resolve::Resolver* resolver) noexcept;
void release() noexcept;
void on_latch(const Settings& settings, const CompiledGraph* graph, const Signs& signs);
[[nodiscard]] bool is_native(const temporal_resolve::Resolver& resolver) noexcept;
[[nodiscard]] bool jitter_ready() noexcept;
[[nodiscard]] bool ready() noexcept;
[[nodiscard]] Upscaler active() noexcept;
[[nodiscard]] std::string_view display_name(UpscalerQuality quality) noexcept;
[[nodiscard]] std::string_view path() noexcept;

namespace testing {
struct Counters {
  uint32_t encodes = 0;
  uint32_t resets = 0;
  uint32_t skipped = 0;
  uint32_t skippedNoScene = 0;
};

Counters counters() noexcept;
void set_stand_in(NativeUpscaler* upscaler) noexcept;
void reset() noexcept;
shaders::RenderPipelineDesc input_desc(bool depth);
} // namespace testing
} // namespace aurora::render::detail::upscale::native_upscale
