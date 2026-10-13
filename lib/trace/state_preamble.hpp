#pragma once

#include "../gx/gx.hpp"

#include <cstdint>
#include <vector>

namespace aurora::trace {
std::vector<uint8_t> build_state_preamble(const gx::GXState& state);
} // namespace aurora::trace
