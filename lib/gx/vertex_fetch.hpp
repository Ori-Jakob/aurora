#pragma once

#include "gx.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace aurora::gx::vertex_fetch {
enum class NbtSlice : u8 {
  N,
  B,
  T,
};

struct CpuArray {
  std::span<const uint8_t> bytes;
  uint32_t first = 0;
};

struct CpuVertexSource {
  std::span<const uint8_t> records;
  std::array<CpuArray, MaxIndexAttr> arrays{};
  uint32_t currentPnMtx = 0;
};

struct CpuVertex {
  uint32_t pnMtx = 0;
  std::array<float, 3> pos{};
  std::array<float, 3> nrm{1.f, 0.f, 0.f};
  std::array<std::array<float, 4>, 2> clr{};
  std::array<std::array<float, 2>, 8> tex{};
};

// GX shaders and stream-out kernels both emit this text, so fetch results match bit for bit
std::string_view wgsl_helpers() noexcept;
std::string_view wgsl_immediate_struct() noexcept;
std::string vtx_attr(const ShaderConfig& config, GXAttr attr);
std::string attr_load(const ShaderConfig& config, GXAttr attr, std::string_view vidx);
std::string attr_load_nbt_slice(const ShaderConfig& config, NbtSlice slice, std::string_view vidx);
std::string_view nbt_slice_local(NbtSlice slice) noexcept;

[[nodiscard]] bool decode_vertex(const ShaderConfig& config, const CpuVertexSource& source, uint32_t vidx,
                                 CpuVertex& out) noexcept;
} // namespace aurora::gx::vertex_fetch
