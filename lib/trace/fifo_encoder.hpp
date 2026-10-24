#pragma once

#include <dolphin/gx/GXCommandList.h>

#include <bit>
#include <cstdint>
#include <span>
#include <vector>

namespace aurora::trace {
class FifoEncoder {
public:
  void u8(uint8_t v) { m_bytes.push_back(v); }
  void u16(uint16_t v) {
    u8(static_cast<uint8_t>(v >> 8));
    u8(static_cast<uint8_t>(v));
  }
  void u32(uint32_t v) {
    u16(static_cast<uint16_t>(v >> 16));
    u16(static_cast<uint16_t>(v));
  }
  void u64(uint64_t v) {
    u32(static_cast<uint32_t>(v >> 32));
    u32(static_cast<uint32_t>(v));
  }
  void f32(float v) { u32(std::bit_cast<uint32_t>(v)); }

  void bp(uint32_t value) {
    u8(GX_LOAD_BP_REG);
    u32(value);
  }
  void cp(uint8_t addr, uint32_t value) {
    u8(GX_LOAD_CP_REG);
    u8(addr);
    u32(value);
  }
  void xf(uint16_t addr, std::span<const uint32_t> words) {
    if (words.empty()) {
      return;
    }
    u8(GX_LOAD_XF_REG);
    u32(static_cast<uint32_t>(words.size() - 1) << 16 | addr);
    for (const uint32_t word : words) {
      u32(word);
    }
  }
  void aurora(uint16_t subCmd) {
    u8(GX_AURORA);
    u16(subCmd);
  }
  void raw(std::span<const uint8_t> bytes) { m_bytes.insert(m_bytes.end(), bytes.begin(), bytes.end()); }

  [[nodiscard]] std::vector<uint8_t>& bytes() noexcept { return m_bytes; }

private:
  std::vector<uint8_t> m_bytes;
};
} // namespace aurora::trace
