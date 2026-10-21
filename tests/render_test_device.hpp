#pragma once

#include <aurora/aurora.h>
#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace aurora::test {
struct TestDeviceOptions {
  wgpu::BackendType backend = wgpu::BackendType::Null;
  bool core = true;
  bool allowCpu = true;
  bool optionalFeatures = true;
  wgpu::Extent3D efbSize{1280, 720, 1};
  uint32_t msaaSamples = 1;
  bool createEfb = false;
  uint64_t uniformBufferSize = 64 * 1024;
  bool productionToggles = false;
  std::optional<AuroraFeatureProfile> profile;
};

struct TestDevice {
  wgpu::Instance instance;
  wgpu::Adapter adapter;
  wgpu::AdapterInfo adapterInfo;
  wgpu::Device device;
  wgpu::Queue queue;
  bool core = true;
  AuroraFeatureProfile profile = FEATURE_PROFILE_FULL;
  std::mutex errorMutex;
  std::vector<std::string> uncapturedErrors;
};

std::unique_ptr<TestDevice> create_test_device(const TestDeviceOptions& options, std::string& error);
std::optional<std::string> pop_validation_error(TestDevice& device);
std::vector<std::string> take_uncaptured_errors(TestDevice& device);
std::optional<wgpu::BackendType> parse_test_backend(const char* value) noexcept;
std::optional<std::vector<uint32_t>> read_buffer(TestDevice& device, const wgpu::Buffer& buffer, uint64_t offset,
                                                 uint64_t size);
bool submit_and_wait(TestDevice& device, const wgpu::CommandBuffer& commands);

class ScopedGlobalDevice {
public:
  ScopedGlobalDevice(TestDevice& device, const TestDeviceOptions& options);
  ~ScopedGlobalDevice();
  ScopedGlobalDevice(const ScopedGlobalDevice&) = delete;
  ScopedGlobalDevice& operator=(const ScopedGlobalDevice&) = delete;

private:
  struct Saved;
  std::unique_ptr<Saved> m_saved;
};
} // namespace aurora::test
