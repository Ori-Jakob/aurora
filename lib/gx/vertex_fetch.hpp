#pragma once

#include "gx.hpp"

#include <string>
#include <string_view>

namespace aurora::gx::vertex_fetch {
enum class NbtSlice : u8 {
  N,
  B,
  T,
};

// GX shaders and stream-out kernels both emit this text, so fetch results match bit for bit
std::string_view wgsl_helpers() noexcept;
std::string_view wgsl_immediate_struct() noexcept;
std::string vtx_attr(const ShaderConfig& config, GXAttr attr);
std::string attr_load(const ShaderConfig& config, GXAttr attr, std::string_view vidx);
std::string attr_load_nbt_slice(const ShaderConfig& config, NbtSlice slice, std::string_view vidx);
std::string_view nbt_slice_local(NbtSlice slice) noexcept;
} // namespace aurora::gx::vertex_fetch
