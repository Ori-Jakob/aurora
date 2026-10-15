#pragma once

#include <cstddef>
#include <span>

namespace aurora::render::shaders {
struct EmbeddedShader {
  const char* id;
  const unsigned char* data;
  std::size_t size;
};

std::span<const EmbeddedShader> embedded_library() noexcept;
} // namespace aurora::render::shaders
