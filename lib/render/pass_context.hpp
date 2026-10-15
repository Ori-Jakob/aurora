#pragma once

#include <webgpu/webgpu_cpp.h>

namespace aurora::render::detail {
struct CompiledPass;
struct StageData;

class PassContext {
public:
  PassContext(const wgpu::CommandEncoder& encoder, const StageData& stage) noexcept
  : m_encoder(encoder), m_stage(stage) {}

  [[nodiscard]] const wgpu::CommandEncoder& encoder() const noexcept { return m_encoder; }
  [[nodiscard]] const StageData& stage() const noexcept { return m_stage; }
  [[nodiscard]] const CompiledPass& pass() const noexcept { return *m_pass; }
  void set_pass(const CompiledPass& pass) noexcept { m_pass = &pass; }

private:
  const wgpu::CommandEncoder& m_encoder;
  const StageData& m_stage;
  const CompiledPass* m_pass = nullptr;
};
} // namespace aurora::render::detail
