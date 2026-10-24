#pragma once

#include "bench_report.hpp"

#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <string>

namespace render_demo {
struct BenchDevice {
  wgpu::Instance instance;
  wgpu::Device device;
  wgpu::Queue queue;
  wgpu::TextureFormat colorFormat = wgpu::TextureFormat::Undefined;
  uint32_t sampleCount = 1;
  uint32_t minUniformBufferOffsetAlignment = 256;
  bool multiDrawIndirect = false;
  bool timestampQuery = false;
  bool nullBackend = false;
};

struct BenchOptions {
  uint32_t blocks = 150;
  uint32_t width = 0, height = 0;
  uint32_t uniformPermille = 1000, texturePermille = 500;
};

using BenchPump = bool (*)(void* userdata);

bool run_indirect_bench(const BenchDevice& device, const BenchOptions& options, BenchPump pump, void* userdata,
                        BenchRun& run, std::string& error);
} // namespace render_demo
