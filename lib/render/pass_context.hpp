#pragma once

#include "graph.hpp"

#include <webgpu/webgpu_cpp.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

namespace aurora::render::shaders {
struct PipelineId;
} // namespace aurora::render::shaders

namespace aurora::render::detail {
struct StageData;
struct BoundPass;

struct ColorTarget {
  TexRef ref;
  uint32_t mip = 0;
  uint32_t layer = 0;
  std::optional<wgpu::Color> clear;
};

struct DepthTarget {
  TexRef ref;
  uint32_t mip = 0;
  uint32_t layer = 0;
  std::optional<float> clear;
};

class PassContext {
public:
  PassContext(const wgpu::CommandEncoder& encoder, const StageData& stage, const CompiledNode& pass,
              const BoundPass& bound) noexcept
  : m_encoder(encoder), m_stage(stage), m_pass(pass), m_bound(bound) {}

  [[nodiscard]] Stage stage() const noexcept;
  [[nodiscard]] const StageData& stage_data() const noexcept { return m_stage; }
  [[nodiscard]] const CompiledNode& pass() const noexcept { return m_pass; }
  [[nodiscard]] std::string_view name() const noexcept { return m_pass.name; }
  [[nodiscard]] uint64_t frame_id() const noexcept;
  [[nodiscard]] uint32_t frame_index() const noexcept;
  [[nodiscard]] const wgpu::Device& device() const noexcept;
  [[nodiscard]] const wgpu::CommandEncoder& encoder() const noexcept { return m_encoder; }
  [[nodiscard]] std::span<const std::byte, PassLocalSize> local() const noexcept { return m_pass.local; }
  [[nodiscard]] std::span<const std::byte> params() const noexcept;

  wgpu::ComputePassEncoder begin_compute(const char* label = nullptr) const;
  wgpu::RenderPassEncoder begin_render(const char* label = nullptr) const;
  wgpu::RenderPassEncoder begin_render(std::span<const ColorTarget> colors, const DepthTarget* depth,
                                       const char* label = nullptr) const;
  void bind_groups(const wgpu::ComputePassEncoder& pass) const;
  void bind_groups(const wgpu::RenderPassEncoder& pass) const;

  [[nodiscard]] wgpu::TextureView view(TexRef ref, uint32_t mip = AllMips) const;
  [[nodiscard]] wgpu::Texture texture(TexRef ref) const;
  [[nodiscard]] wgpu::TextureView prev(TexRef ref) const;
  [[nodiscard]] bool prev_valid(TexRef ref) const noexcept;
  [[nodiscard]] BufferView buffer(BufRef ref) const;
  [[nodiscard]] wgpu::Extent3D extent(TexRef ref) const;
  [[nodiscard]] wgpu::TextureFormat format(TexRef ref) const;
  [[nodiscard]] uint32_t sample_count(TexRef ref) const;

  [[nodiscard]] const wgpu::BindGroup& frame_group() const noexcept;
  [[nodiscard]] uint32_t frame_offset() const noexcept;
  [[nodiscard]] const wgpu::BindGroup& pass_group() const noexcept;
  [[nodiscard]] uint32_t params_offset() const noexcept;
  [[nodiscard]] const wgpu::BindGroupLayout& frame_layout() const noexcept;
  [[nodiscard]] const wgpu::BindGroupLayout& pass_layout() const noexcept;
  [[nodiscard]] const wgpu::PipelineLayout& pipeline_layout() const noexcept;
  [[nodiscard]] uint64_t layout_hash() const noexcept;
  [[nodiscard]] shaders::PipelineId pipeline(uint32_t index = 0) const noexcept;

private:
  template <typename Encoder>
  void bind_groups_to(const Encoder& pass) const;

  const wgpu::CommandEncoder& m_encoder;
  const StageData& m_stage;
  const CompiledNode& m_pass;
  const BoundPass& m_bound;
};
} // namespace aurora::render::detail
