#pragma once

#include "../../gfx/texture.hpp"

#include <aurora/texture.hpp>

namespace aurora::render::material {
void install_gx_lookup() noexcept;
void remove_gx_lookup() noexcept;

namespace testing {
gfx::TextureMaterialRef gx_find(const texture::TextureSourceKey& key) noexcept;
void reset_map_warning() noexcept;
} // namespace testing
} // namespace aurora::render::material
