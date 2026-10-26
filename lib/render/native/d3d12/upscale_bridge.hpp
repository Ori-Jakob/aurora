#pragma once

#include "shared_texture.hpp"

#include "../../post/upscale/native_upscale.hpp"

#include <cstdint>
#include <optional>
#include <string_view>

namespace aurora::render::detail::native::d3d12::upscale_bridge {
using upscale::native_upscale::NativeUpscaler;

class PassthroughUpscaler final : public NativeUpscaler {
public:
  explicit PassthroughUpscaler(float ratio = 1.f, ColorEncoding encoding = ColorEncoding::Stored,
                               bool reactive = false) noexcept
  : m_ratio(ratio), m_encoding(encoding), m_reactive(reactive) {}

  void set_ratio(float ratio) noexcept { m_ratio = ratio; }
  void set_failing(bool failing) noexcept { m_failing = failing; }
  [[nodiscard]] uint32_t prepares() const noexcept { return m_prepares; }
  [[nodiscard]] uint32_t evaluations() const noexcept { return m_evaluations; }
  [[nodiscard]] uint32_t destroys() const noexcept { return m_destroys; }

  [[nodiscard]] Upscaler option() const noexcept override { return Upscaler::Off; }
  [[nodiscard]] std::string_view name() const noexcept override { return "copy"; }
  upscale::native_upscale::ProbeResult probe(const upscale::native_upscale::Adapter& adapter) noexcept override;
  [[nodiscard]] std::string_view display_name(UpscalerQuality quality) const noexcept override;
  [[nodiscard]] ColorEncoding color_encoding() const noexcept override { return m_encoding; }
  [[nodiscard]] bool wants_reactive() const noexcept override { return m_reactive; }
  [[nodiscard]] bool passthrough() const noexcept override { return true; }
  std::optional<GraphExtent> prepare(const upscale::native_upscale::Adapter& adapter, GraphExtent output,
                                     UpscalerQuality quality) noexcept override;
  bool evaluate(const upscale::native_upscale::Inputs& inputs) noexcept override;
  void destroy() noexcept override { ++m_destroys; }
  [[nodiscard]] uint64_t vram_bytes() const noexcept override { return 0; }
  [[nodiscard]] uint32_t jitter_phases(float ratio) const noexcept override;
  [[nodiscard]] float mip_bias(float ratio) const noexcept override;

private:
  float m_ratio = 1.f;
  ColorEncoding m_encoding = ColorEncoding::Stored;
  bool m_reactive = false;
  bool m_failing = false;
  uint32_t m_prepares = 0;
  uint32_t m_evaluations = 0;
  uint32_t m_destroys = 0;
};

void register_upscaler(NativeUpscaler& upscaler) noexcept;
[[nodiscard]] NativeUpscaler* upscaler(Upscaler option) noexcept;
[[nodiscard]] upscale::native_upscale::Adapter adapter() noexcept;
bool prepare(const upscale::native_upscale::Plan& plan, GraphExtent& render) noexcept;
void release() noexcept;
void worker_frame_begin() noexcept;
void wait_idle() noexcept;
bool begin(const upscale::native_upscale::Frame& frame, upscale::native_upscale::Targets& targets) noexcept;

namespace testing {
struct Shared {
  SharedTexture* color = nullptr;
  SharedTexture* depth = nullptr;
  SharedTexture* motion = nullptr;
  SharedTexture* reactive = nullptr;
  SharedTexture* out = nullptr;
};

Shared shared() noexcept;
void set_upscaler(Upscaler option, NativeUpscaler* upscaler) noexcept;
void reset() noexcept;
} // namespace testing
} // namespace aurora::render::detail::native::d3d12::upscale_bridge
