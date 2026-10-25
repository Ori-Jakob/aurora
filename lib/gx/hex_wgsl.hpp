#pragma once

#include <cstdint>
#include <string>

namespace aurora::gx {
void append_hex_common(std::string& out);
void append_hex_module(std::string& out, uint32_t maps);
std::string hex_site(uint32_t map, uint32_t coord);

namespace testing {
// Tests only: makes every hex_tex fn reference an undefined name so ext creation fails
void set_hex_corrupt(bool corrupt) noexcept;
} // namespace testing
} // namespace aurora::gx
