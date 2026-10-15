#include "render_test_device.hpp"

#include "gfx/resources.hpp"
#include "webgpu/caps.hpp"
#include "webgpu/gpu.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <span>
#include <string_view>
#include <utility>

namespace aurora::test {
namespace {
constexpr uint64_t WaitTimeoutNs = 5'000'000'000;

constexpr std::array OptionalFeatures{
    wgpu::FeatureName::ShaderF16,
    wgpu::FeatureName::Subgroups,
    wgpu::FeatureName::TimestampQuery,
    wgpu::FeatureName::Float32Filterable,
    wgpu::FeatureName::TextureFormatsTier1,
    wgpu::FeatureName::TextureFormatsTier2,
    wgpu::FeatureName::RG11B10UfloatRenderable,
    wgpu::FeatureName::TextureCompressionBC,
};

std::string to_string(wgpu::StringView text) { return std::string{std::string_view{text}}; }
} // namespace

std::unique_ptr<TestDevice> create_test_device(const TestDeviceOptions& options, std::string& error) {
  auto result = std::make_unique<TestDevice>();
  constexpr std::array instanceFeatures{wgpu::InstanceFeatureName::TimedWaitAny};
  constexpr std::array instanceToggles{"allow_unsafe_apis"};
  const wgpu::DawnTogglesDescriptor instanceTogglesDescriptor{wgpu::DawnTogglesDescriptor::Init{
      .enabledToggleCount = instanceToggles.size(),
      .enabledToggles = instanceToggles.data(),
  }};
  const wgpu::InstanceDescriptor instanceDescriptor{
      .nextInChain = &instanceTogglesDescriptor,
      .requiredFeatureCount = instanceFeatures.size(),
      .requiredFeatures = instanceFeatures.data(),
  };
  result->instance = wgpu::CreateInstance(&instanceDescriptor);
  if (!result->instance) {
    error = "failed to create a WebGPU instance";
    return nullptr;
  }

  const wgpu::RequestAdapterOptions adapterOptions{
      .featureLevel = options.core ? wgpu::FeatureLevel::Core : wgpu::FeatureLevel::Compatibility,
      .backendType = options.backend,
  };
  std::string message;
  const wgpu::Future adapterFuture = result->instance.RequestAdapter(
      &adapterOptions, wgpu::CallbackMode::WaitAnyOnly,
      [&](wgpu::RequestAdapterStatus status, wgpu::Adapter adapter, wgpu::StringView text) {
        if (status == wgpu::RequestAdapterStatus::Success) {
          result->adapter = std::move(adapter);
        } else {
          message = to_string(text);
        }
      });
  if (result->instance.WaitAny(adapterFuture, WaitTimeoutNs) != wgpu::WaitStatus::Success || !result->adapter) {
    error = "no adapter: " + message;
    return nullptr;
  }
  result->adapter.GetInfo(&result->adapterInfo);
  if (!options.allowCpu && result->adapterInfo.adapterType == wgpu::AdapterType::CPU) {
    error = "only a CPU adapter is available";
    return nullptr;
  }

  std::vector<wgpu::FeatureName> features;
  if (options.core && result->adapter.HasFeature(wgpu::FeatureName::CoreFeaturesAndLimits)) {
    features.push_back(wgpu::FeatureName::CoreFeaturesAndLimits);
  }
  if (options.optionalFeatures) {
    for (const auto feature : OptionalFeatures) {
      if (result->adapter.HasFeature(feature)) {
        features.push_back(feature);
      }
    }
  }
  // Errors reach the callback during the failing call, so a test sees them before its next check
  constexpr std::array deviceToggles{"enable_immediate_error_handling"};
  wgpu::DawnTogglesDescriptor deviceTogglesDescriptor{wgpu::DawnTogglesDescriptor::Init{
      .enabledToggleCount = deviceToggles.size(),
      .enabledToggles = deviceToggles.data(),
  }};
  wgpu::DeviceDescriptor deviceDescriptor({
      .nextInChain = &deviceTogglesDescriptor,
      .requiredFeatureCount = features.size(),
      .requiredFeatures = features.data(),
  });
  deviceDescriptor.SetUncapturedErrorCallback(
      [](const wgpu::Device&, wgpu::ErrorType type, wgpu::StringView text, TestDevice* device) {
        std::lock_guard lock{device->errorMutex};
        device->uncapturedErrors.push_back("error " + std::to_string(static_cast<uint32_t>(type)) + ": " +
                                           to_string(text));
      },
      result.get());
  deviceDescriptor.SetDeviceLostCallback(wgpu::CallbackMode::AllowSpontaneous,
                                         [](const wgpu::Device&, wgpu::DeviceLostReason, wgpu::StringView) {});
  const wgpu::Future deviceFuture =
      result->adapter.RequestDevice(&deviceDescriptor, wgpu::CallbackMode::WaitAnyOnly,
                                    [&](wgpu::RequestDeviceStatus status, wgpu::Device device, wgpu::StringView text) {
                                      if (status == wgpu::RequestDeviceStatus::Success) {
                                        result->device = std::move(device);
                                      } else {
                                        message = to_string(text);
                                      }
                                    });
  if (result->instance.WaitAny(deviceFuture, WaitTimeoutNs) != wgpu::WaitStatus::Success || !result->device) {
    error = "no device: " + message;
    return nullptr;
  }
  result->queue = result->device.GetQueue();
  result->core = result->device.HasFeature(wgpu::FeatureName::CoreFeaturesAndLimits);
  return result;
}

std::optional<std::string> pop_validation_error(TestDevice& device) {
  std::optional<std::string> result;
  const wgpu::Future future =
      device.device.PopErrorScope(wgpu::CallbackMode::WaitAnyOnly, [&](wgpu::PopErrorScopeStatus status,
                                                                       wgpu::ErrorType type, wgpu::StringView message) {
        if (status != wgpu::PopErrorScopeStatus::Success) {
          result = "no error scope to pop";
        } else if (type != wgpu::ErrorType::NoError) {
          result = to_string(message);
        }
      });
  if (device.instance.WaitAny(future, WaitTimeoutNs) != wgpu::WaitStatus::Success) {
    return "error scope pop timed out";
  }
  return result;
}

std::vector<std::string> take_uncaptured_errors(TestDevice& device) {
  std::lock_guard lock{device.errorMutex};
  return std::exchange(device.uncapturedErrors, {});
}

std::optional<wgpu::BackendType> parse_test_backend(const char* value) noexcept {
  if (value == nullptr) {
    return std::nullopt;
  }
  constexpr std::array<std::pair<std::string_view, wgpu::BackendType>, 8> Backends{{
      {"null", wgpu::BackendType::Null},
      {"d3d12", wgpu::BackendType::D3D12},
      {"d3d11", wgpu::BackendType::D3D11},
      {"vulkan", wgpu::BackendType::Vulkan},
      {"metal", wgpu::BackendType::Metal},
      {"opengl", wgpu::BackendType::OpenGL},
      {"opengles", wgpu::BackendType::OpenGLES},
      {"default", wgpu::BackendType::Undefined},
  }};
  const std::string_view text{value};
  for (const auto& [name, backend] : Backends) {
    const bool same = std::ranges::equal(
        text, name, [](char lhs, char rhs) { return std::tolower(static_cast<unsigned char>(lhs)) == rhs; });
    if (same) {
      return backend;
    }
  }
  return std::nullopt;
}

struct ScopedGlobalDevice::Saved {
  wgpu::Instance instance;
  wgpu::AdapterInfo adapterInfo;
  wgpu::Device device;
  wgpu::Queue queue;
  wgpu::BackendType backendType = wgpu::BackendType::Undefined;
  bool hasCoreFeatures = false;
  wgpu::Limits limits;
  wgpu::Buffer uniformBuffer;
  webgpu::GraphicsConfig graphicsConfig{};
  webgpu::TextureWithSampler frameBuffer;
  webgpu::TextureWithSampler frameBufferResolved;
  webgpu::TextureWithSampler depthBuffer;
  webgpu::TextureWithSampler normalBuffer;
};

ScopedGlobalDevice::ScopedGlobalDevice(TestDevice& device, const TestDeviceOptions& options)
: m_saved(std::make_unique<Saved>()) {
  Saved& saved = *m_saved;
  saved.instance = std::exchange(webgpu::g_instance, device.instance);
  saved.adapterInfo = std::move(webgpu::g_adapterInfo);
  device.adapter.GetInfo(&webgpu::g_adapterInfo);
  saved.device = std::exchange(webgpu::g_device, device.device);
  saved.queue = std::exchange(webgpu::g_queue, device.queue);
  saved.backendType = std::exchange(webgpu::g_backendType, webgpu::g_adapterInfo.backendType);
  saved.hasCoreFeatures =
      std::exchange(webgpu::g_hasCoreFeatures, device.device.HasFeature(wgpu::FeatureName::CoreFeaturesAndLimits));

  auto& resources = gfx::detail::resources();
  saved.limits = std::exchange(resources.limits, {});
  device.device.GetLimits(&resources.limits);
  const wgpu::BufferDescriptor uniformDescriptor{
      .label = "Test uniform buffer",
      .usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
      .size = options.uniformBufferSize,
  };
  saved.uniformBuffer = std::exchange(resources.uniformBuffer, device.device.CreateBuffer(&uniformDescriptor));

  const webgpu::GraphicsConfig graphicsConfig{
      .surfaceConfiguration{.format = wgpu::TextureFormat::BGRA8Unorm},
      .depthFormat = wgpu::TextureFormat::Depth32Float,
      .msaaSamples = options.msaaSamples,
      .textureAnisotropy = 1,
      .normalBuffer = false,
  };
  saved.graphicsConfig = std::exchange(webgpu::g_graphicsConfig, graphicsConfig);
  saved.frameBuffer = std::exchange(webgpu::g_frameBuffer, {});
  saved.frameBufferResolved = std::exchange(webgpu::g_frameBufferResolved, {});
  saved.depthBuffer = std::exchange(webgpu::g_depthBuffer, {});
  saved.normalBuffer = std::exchange(webgpu::g_normalBuffer, {});
  if (options.createEfb) {
    const auto& size = options.efbSize;
    webgpu::g_frameBuffer = webgpu::create_render_texture(size.width, size.height, true);
    if (options.msaaSamples > 1) {
      webgpu::g_frameBufferResolved = webgpu::create_render_texture(size.width, size.height, false);
    }
  } else {
    webgpu::g_frameBuffer.size = options.efbSize;
    webgpu::g_frameBuffer.format = wgpu::TextureFormat::BGRA8Unorm;
  }

  wgpu::SupportedFeatures features;
  device.device.GetFeatures(&features);
  wgpu::CompatibilityModeLimits compatLimits{};
  wgpu::Limits limits{.nextInChain = &compatLimits};
  device.device.GetLimits(&limits);
  limits.nextInChain = nullptr;
  const auto& info = webgpu::g_adapterInfo;
  webgpu::set_caps(webgpu::build_caps({
      .backend = info.backendType,
      .adapterType = info.adapterType,
      .profile = device.core ? FEATURE_PROFILE_FULL : FEATURE_PROFILE_COMPAT,
      .vendorId = info.vendorID,
      .deviceId = info.deviceID,
      .adapterName = info.device.IsUndefined() ? std::string_view{"Unknown"} : std::string_view{info.device},
      .features = std::span<const wgpu::FeatureName>{features.features, features.featureCount},
      .limits = limits,
      .compatLimits = compatLimits,
      .subgroupMinSize = info.subgroupMinSize,
      .subgroupMaxSize = info.subgroupMaxSize,
  }));
}

ScopedGlobalDevice::~ScopedGlobalDevice() {
  Saved& saved = *m_saved;
  webgpu::reset_caps();
  webgpu::g_normalBuffer = std::move(saved.normalBuffer);
  webgpu::g_depthBuffer = std::move(saved.depthBuffer);
  webgpu::g_frameBufferResolved = std::move(saved.frameBufferResolved);
  webgpu::g_frameBuffer = std::move(saved.frameBuffer);
  webgpu::g_graphicsConfig = std::move(saved.graphicsConfig);
  auto& resources = gfx::detail::resources();
  resources.uniformBuffer = std::move(saved.uniformBuffer);
  resources.limits = saved.limits;
  webgpu::g_hasCoreFeatures = saved.hasCoreFeatures;
  webgpu::g_backendType = saved.backendType;
  webgpu::g_queue = std::move(saved.queue);
  webgpu::g_device = std::move(saved.device);
  webgpu::g_adapterInfo = std::move(saved.adapterInfo);
  webgpu::g_instance = std::move(saved.instance);
}
} // namespace aurora::test
