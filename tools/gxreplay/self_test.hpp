#pragma once

#include "trace/trace_format.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace aurora::gxreplay {
inline constexpr std::array<uint16_t, 5> SelfTestHintIds{0x0050, 0x0051, 0x00FF, 0x0100, 0x01FF};
inline constexpr std::array<uint32_t, 4> SelfTestPayloadSizes{0, 1, 7, 4096};
inline constexpr uint32_t SelfTestFrameCount = 3;
inline constexpr uint64_t SelfTestTextureAddress = 0x80100000;
inline constexpr uint64_t SelfTestPositionAddress = 0x80200000;
inline constexpr uint64_t SelfTestColorAddress = 0x80200100;
inline constexpr uint64_t SelfTestCopyAddress = 0x80300000;
inline constexpr uint32_t SelfTestWidth = 640;
inline constexpr uint32_t SelfTestHeight = 480;

trace::TraceInfo self_test_info();
uint8_t self_test_hint_byte(size_t index) noexcept;
bool write_self_test_trace(const std::filesystem::path& path, bool withHints, std::string& error);
bool check_self_test_frames(std::span<const std::vector<uint8_t>> withHints,
                            std::span<const std::vector<uint8_t>> plain, bool pixelProbes, std::string& error);
} // namespace aurora::gxreplay
