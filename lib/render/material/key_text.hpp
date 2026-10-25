#pragma once

#include <dolphin/gx/GXStruct.h>
#include <aurora/texture.hpp>

#include <fmt/format.h>

#include <iterator>
#include <optional>
#include <string>
#include <string_view>

namespace aurora::render::material {
inline std::string format_key_text(const texture::TextureSourceKey& key) {
  std::string out = fmt::format("tex1_{}x{}_", key.width, key.height);
  if (key.textureHash == texture::kWildcardTextureHash) {
    out += '$';
  } else {
    fmt::format_to(std::back_inserter(out), "{:016x}", key.textureHash);
  }
  if (key.hasTlut) {
    if (key.tlutHash == texture::kWildcardTlutHash) {
      out += "_$";
    } else {
      fmt::format_to(std::back_inserter(out), "_{:016x}", key.tlutHash);
    }
  }
  fmt::format_to(std::back_inserter(out), "_{}", key.format);
  return out;
}

inline std::optional<texture::TextureSourceKey> parse_key_text(std::string_view text) {
  std::string name{text};
  name += ".dds";
  return texture::parse_replacement_filename(name);
}
} // namespace aurora::render::material
