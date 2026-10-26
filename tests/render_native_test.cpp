#include <gtest/gtest.h>

#include "render_test_caps.hpp"

#include "gfx/frame.hpp"
#include "gfx/render_worker.hpp"
#include "render/native/native.hpp"
#include "render/settings.hpp"

#if AURORA_ENABLE_NATIVE_D3D12
#include "io.hpp"
#include "render/frame_state.hpp"
#include "render/native/d3d12/dlss.hpp"
#include "render/native/d3d12/probe.hpp"
#include "render/native/d3d12/sentinel.hpp"
#include "render/native/d3d12/teardown.hpp"
#include "render/native/d3d12/upscale_bridge.hpp"
#include "render/post/upscale/fsr/fsr.hpp"

#include <array>
#include <cmath>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#endif

#if AURORA_ENABLE_NATIVE_D3D12 && AURORA_ENABLE_XESS
#include "render/native/d3d12/xess.hpp"
#include "render/post/upscale/upscale.hpp"
#endif

#include <atomic>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace aurora::render::detail {
namespace {
TEST(NativeD3D12, NativeTraceStillNotImplemented) {
  Caps caps = test::full_caps();
  caps.nativeD3D12 = true;
  Settings request;
  request.rt = RtMode::Shadows;
  request.rtBackend = RtBackend::Dxr;
  const SupportEnv env{caps, 1, default_built_options(), true};
  EXPECT_EQ(evaluate_support(Feature::NativeTrace, 0, request, env), (FeatureSupport{false, Reason::NotImplemented}));
}

#if AURORA_ENABLE_NATIVE_D3D12
namespace sentinel = native::d3d12::sentinel;

class NativeSentinel : public ::testing::Test {
protected:
  void SetUp() override {
    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    m_dir = std::filesystem::temp_directory_path() / ("aurora_native_sentinel_" + std::string{info->name()});
    std::error_code ec;
    std::filesystem::remove_all(m_dir, ec);
    ASSERT_TRUE(std::filesystem::create_directories(m_dir, ec)) << ec.message();
    sentinel::open(m_dir);
  }

  void TearDown() override {
    sentinel::open({});
    std::error_code ec;
    std::filesystem::remove_all(m_dir, ec);
  }

  [[nodiscard]] bool file_exists() const { return std::filesystem::exists(m_dir / "native_upscaler.sentinel"); }

  std::filesystem::path m_dir;
};

TEST_F(NativeSentinel, ArmThenDisarmLeavesNoFile) {
  EXPECT_EQ(sentinel::stale_at_startup(), 0u);
  sentinel::arm(sentinel::Probe);
  EXPECT_TRUE(file_exists());
  sentinel::disarm(sentinel::Probe);
  EXPECT_FALSE(file_exists());
  sentinel::arm(sentinel::Session);
  sentinel::arm(sentinel::Probe);
  sentinel::disarm(sentinel::Probe);
  EXPECT_TRUE(file_exists());
  sentinel::disarm(sentinel::Session);
  EXPECT_FALSE(file_exists());
  sentinel::open(m_dir);
  EXPECT_EQ(sentinel::stale_at_startup(), 0u);
}

TEST_F(NativeSentinel, StaleProbeReportedAtStartup) {
  sentinel::arm(sentinel::Probe);
  sentinel::open(m_dir);
  EXPECT_EQ(sentinel::stale_at_startup(), uint8_t{sentinel::Probe});
}

TEST_F(NativeSentinel, StaleSessionReportedAtStartup) {
  sentinel::arm(sentinel::Session);
  sentinel::open(m_dir);
  EXPECT_EQ(sentinel::stale_at_startup(), uint8_t{sentinel::Session});
  sentinel::arm(sentinel::Probe);
  sentinel::disarm(sentinel::Probe);
  EXPECT_TRUE(file_exists());
  sentinel::open(m_dir);
  EXPECT_EQ(sentinel::stale_at_startup(), uint8_t{sentinel::Session});
}

TEST_F(NativeSentinel, ClearRemovesFile) {
  sentinel::arm(sentinel::Probe);
  sentinel::arm(sentinel::Session);
  sentinel::open(m_dir);
  EXPECT_EQ(sentinel::stale_at_startup(), uint8_t{sentinel::Probe | sentinel::Session});
  sentinel::clear();
  EXPECT_FALSE(file_exists());
  EXPECT_EQ(sentinel::stale_at_startup(), 0u);
  sentinel::open(m_dir);
  EXPECT_EQ(sentinel::stale_at_startup(), 0u);
}

std::atomic<bool> g_firstOnWorker{false};
std::vector<int> g_teardownOrder;

TEST(NativeTeardown, RunsOnWorkerBeforeWorkerStops) {
  g_firstOnWorker = false;
  g_teardownOrder.clear();
  gfx::render_worker::initialize();
  native::d3d12::register_teardown([]() noexcept {
    g_firstOnWorker = gfx::render_worker::is_worker_thread();
    g_teardownOrder.push_back(1);
  });
  native::d3d12::register_teardown([]() noexcept { g_teardownOrder.push_back(2); });
  gfx::detail::stop_render_worker();
  EXPECT_TRUE(g_firstOnWorker);
  EXPECT_EQ(g_teardownOrder, (std::vector<int>{2, 1}));

  g_teardownOrder.clear();
  gfx::render_worker::initialize();
  gfx::detail::stop_render_worker();
  EXPECT_TRUE(g_teardownOrder.empty());
}
#endif

#if AURORA_ENABLE_NATIVE_D3D12 && AURORA_ENABLE_XESS
namespace xess = native::d3d12::xess;
namespace probe = native::d3d12::probe;
namespace bridge = native::d3d12::upscale_bridge;

class Xess : public ::testing::Test {
protected:
  void SetUp() override {
    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    m_dir = std::filesystem::temp_directory_path() / ("aurora_xess_" + std::string{info->name()});
    std::error_code ec;
    std::filesystem::remove_all(m_dir, ec);
    ASSERT_TRUE(std::filesystem::create_directories(m_dir, ec)) << ec.message();
    testing::reset();
    probe::testing::reset();
    sentinel::open(m_dir);
    Caps deviceCaps = test::full_caps();
    deviceCaps.d3d12SharedTexture = true;
    deviceCaps.d3d12SharedFence = true;
    testing::set_caps(deviceCaps);
    probe::testing::set_identity(
        probe::Identity{.vendorId = 0x10DE, .deviceId = 0x2860, .subSysId = 0x3CBC17AA, .driverVersion = 1});
    bridge::testing::set_upscaler(Upscaler::Xess, &xess::upscaler());
  }

  void TearDown() override {
    bridge::testing::reset();
    probe::testing::reset();
    xess::testing::set_library(std::nullopt);
    native::d3d12::run_teardowns();
    sentinel::open({});
    testing::reset();
    std::error_code ec;
    std::filesystem::remove_all(m_dir, ec);
  }

  static FeatureSupport support(Upscaler option) {
    return current_support(Feature::Upscaler, static_cast<uint8_t>(option), nullptr);
  }

  static upscale::native_upscale::ProbeResult probe_on_worker() {
    upscale::native_upscale::ProbeResult result;
    gfx::render_worker::initialize();
    gfx::render_worker::enqueue_work([&] { result = xess::upscaler().probe(bridge::adapter()); });
    gfx::render_worker::synchronize();
    gfx::render_worker::shutdown();
    return result;
  }

  std::filesystem::path m_dir;
};

TEST_F(Xess, ResultMapping) {
  for (const xess_result_t result :
       {XESS_RESULT_SUCCESS, XESS_RESULT_WARNING_OLD_DRIVER, XESS_RESULT_WARNING_NONEXISTING_FOLDER}) {
    EXPECT_EQ(xess::support_from_result(result), (FeatureSupport{true, Reason::None})) << static_cast<int>(result);
  }
  EXPECT_EQ(xess::support_from_result(XESS_RESULT_ERROR_UNSUPPORTED_DEVICE),
            (FeatureSupport{false, Reason::UnsupportedGpu}));
  EXPECT_EQ(xess::support_from_result(XESS_RESULT_ERROR_UNSUPPORTED_DRIVER),
            (FeatureSupport{false, Reason::DriverTooOld}));
  EXPECT_EQ(xess::support_from_result(XESS_RESULT_ERROR_CANT_LOAD_LIBRARY),
            (FeatureSupport{false, Reason::RuntimeMissing}));
  for (const xess_result_t result : {XESS_RESULT_ERROR_DEVICE, XESS_RESULT_ERROR_DEVICE_OUT_OF_MEMORY,
                                     XESS_RESULT_ERROR_INVALID_ARGUMENT, XESS_RESULT_ERROR_UNKNOWN}) {
    EXPECT_EQ(xess::support_from_result(result), (FeatureSupport{false, Reason::NativeUnavailable}))
        << static_cast<int>(result);
  }
}

TEST_F(Xess, PathNameXeFxVsCrossVendor) {
  const xess_version_t none{};
  const xess_version_t xefx{.major = 1, .minor = 3, .patch = 2};
  EXPECT_EQ(xess::path_for(xess::IntelVendorId, xefx).name, "XeFX");
  EXPECT_TRUE(xess::path_for(xess::IntelVendorId, xefx).xefx);
  for (const uint32_t vendor : {xess::IntelVendorId, 0x10DEu, 0x1002u}) {
    EXPECT_EQ(xess::path_for(vendor, none).name, "cross-vendor") << vendor;
    EXPECT_FALSE(xess::path_for(vendor, none).xefx) << vendor;
  }
  EXPECT_EQ(xess::path_for(0x10DE, xefx).name, "cross-vendor");
  EXPECT_FALSE(xess::path_for(0x10DE, xefx).xefx);
  const xess::XessUpscaler& upscaler = xess::upscaler();
  EXPECT_EQ(upscaler.option(), Upscaler::Xess);
  EXPECT_EQ(upscaler.name(), "xess");
  EXPECT_EQ(upscaler.runtime_file(), "libxess.dll");
  EXPECT_EQ(upscaler.display_name(UpscalerQuality::Quality), "XeSS-SR");
}

TEST_F(Xess, MissingDllReportsRuntimeMissing) {
  const uint32_t loads = xess::testing::loads();
  probe::testing::set_runtime_directory(m_dir);
  probe::initialize(m_dir);
  EXPECT_EQ(support(Upscaler::Xess), (FeatureSupport{false, Reason::RuntimeMissing}));
  native_worker_frame_begin();
  EXPECT_EQ(native::testing::sdk_calls(), 0u);

  xess::testing::set_library(m_dir / "libxess.dll");
  const upscale::native_upscale::ProbeResult result = probe_on_worker();
  EXPECT_EQ(result.support, (FeatureSupport{false, Reason::RuntimeMissing}));
  EXPECT_FALSE(result.autoPick);
  EXPECT_TRUE(result.path.empty());
  EXPECT_FALSE(xess::loaded());
  EXPECT_EQ(xess::testing::loads(), loads);
}

TEST_F(Xess, NotLoadedWithUpscalerOff) {
  const uint32_t loads = xess::testing::loads();
  probe::initialize(m_dir);
  submit_request(Settings{});
  for (int i = 0; i < 100; ++i) {
    static_cast<void>(latch_frame());
    native_worker_frame_begin();
    static_cast<void>(support(Upscaler::Off));
    static_cast<void>(support(Upscaler::Fsr));
  }
  EXPECT_FALSE(xess::loaded());
  EXPECT_EQ(xess::testing::loads(), loads);
  EXPECT_EQ(native::testing::sdk_calls(), 0u);
  EXPECT_FALSE(std::filesystem::exists(m_dir / probe::CacheFile));
  EXPECT_EQ(support(Upscaler::Xess).reason, Reason::Probing);
  EXPECT_FALSE(xess::loaded());
}

TEST_F(Xess, QualityMapsToXessEnum) {
  EXPECT_EQ(xess::quality_setting(UpscalerQuality::NativeAa), XESS_QUALITY_SETTING_AA);
  EXPECT_EQ(xess::quality_setting(UpscalerQuality::Quality), XESS_QUALITY_SETTING_QUALITY);
  EXPECT_EQ(xess::quality_setting(UpscalerQuality::Balanced), XESS_QUALITY_SETTING_BALANCED);
  EXPECT_EQ(xess::quality_setting(UpscalerQuality::Performance), XESS_QUALITY_SETTING_PERFORMANCE);
  EXPECT_EQ(xess::quality_setting(UpscalerQuality::UltraPerformance), XESS_QUALITY_SETTING_ULTRA_PERFORMANCE);
  const std::vector<int> spec{106, 103, 102, 101, 100};
  for (uint8_t i = 0; i < spec.size(); ++i) {
    EXPECT_EQ(static_cast<int>(xess::quality_setting(static_cast<UpscalerQuality>(i))), spec[i]) << int{i};
  }
}

TEST_F(Xess, JitterPhasesAtLeastEightRatioSquared) {
  const xess::XessUpscaler& upscaler = xess::upscaler();
  for (const auto& [ratio, phases] : std::vector<std::pair<float, uint32_t>>{
           {1.f, 8}, {1.3f, 14}, {1.5f, 18}, {1.7f, 24}, {2.f, 32}, {2.3f, 43}, {3.f, 72}}) {
    EXPECT_EQ(upscaler.jitter_phases(ratio), phases) << ratio;
    EXPECT_GE(upscaler.jitter_phases(ratio), 8u) << ratio;
    EXPECT_GE(static_cast<float>(upscaler.jitter_phases(ratio)), 8.f * ratio * ratio) << ratio;
    EXPECT_FLOAT_EQ(upscaler.mip_bias(ratio), std::log2(1.f / ratio)) << ratio;
  }
}

TEST_F(Xess, VelocityScaleFollowsRenderSize) {
  EXPECT_EQ(xess::velocity_scale({1706, 960}), (std::array{-853.f, 480.f}));
  EXPECT_EQ(xess::velocity_scale({2811, 1581}), (std::array{-1405.5f, 790.5f}));
  EXPECT_EQ(xess::velocity_scale({64, 32}), (std::array{-32.f, 16.f}));
  EXPECT_EQ(xess::velocity_scale({}), (std::array{-0.f, 0.f}));
}

TEST_F(Xess, InitFlagsNoResponsiveByDefault) {
  const uint32_t flags = xess::init_flags(false);
  EXPECT_EQ(flags, static_cast<uint32_t>(XESS_INIT_FLAG_INVERTED_DEPTH | XESS_INIT_FLAG_LDR_INPUT_COLOR));
  EXPECT_EQ(flags & XESS_INIT_FLAG_RESPONSIVE_PIXEL_MASK, 0u);
  EXPECT_EQ(flags & (XESS_INIT_FLAG_HIGH_RES_MV | XESS_INIT_FLAG_USE_NDC_VELOCITY | XESS_INIT_FLAG_JITTERED_MV), 0u);
  EXPECT_EQ(xess::init_flags(true), flags | XESS_INIT_FLAG_RESPONSIVE_PIXEL_MASK);
  EXPECT_FALSE(xess::upscaler().wants_reactive());
}

TEST_F(Xess, ColorFollowsEnvOverride) {
  EXPECT_EQ(xess::upscaler().color_encoding(), ColorEncoding::Linear);
  upscale::testing::set_color_env(ColorEncoding::Stored);
  EXPECT_EQ(xess::upscaler().color_encoding(), ColorEncoding::Stored);
  upscale::testing::set_color_env(ColorEncoding::Linear);
  EXPECT_EQ(xess::upscaler().color_encoding(), ColorEncoding::Linear);
  upscale::testing::reset();
  EXPECT_EQ(xess::upscaler().color_encoding(), ColorEncoding::Linear);
}
#endif

#if AURORA_ENABLE_NATIVE_D3D12
namespace dlss = native::d3d12::dlss;
namespace ngx_probe = native::d3d12::probe;
namespace ngx_bridge = native::d3d12::upscale_bridge;

constexpr std::string_view TestProjectId = "3f6c2a91-58d4-4e0b-9a7e-2c41d8b5f063";
constexpr GraphExtent TestOutput{2560, 1440};
constexpr std::array<float, dlss::ModeCount> FakeRatios{1.f, 1.5f, 1.72f, 2.f, 3.f};

struct FakeNgx {
  uint32_t requirements = 0;
  uint32_t inits = 0;
  uint32_t capabilities = 0;
  uint32_t optimals = 0;
  uint32_t shutdowns = 0;
  uint32_t waits = 0;
  uint32_t creates = 0;
  uint32_t evaluates = 0;
  uint32_t releases = 0;
  std::optional<dlss::Requirements> requirement = dlss::Requirements{};
  bool initOk = true;
  bool createOk = true;
  std::optional<dlss::Capability> capability = dlss::Capability{.available = true};
  uint64_t vram = uint64_t{150} << 20;
  native::Identity identity;
  std::filesystem::path data;
  std::optional<dlss::FeatureDesc> created;
  void* createList = nullptr;
  void* released = nullptr;
  dlss::Evaluation evaluation;
  std::string events;
  int feature = 0;

  [[nodiscard]] uint32_t calls() const noexcept {
    return requirements + inits + capabilities + optimals + shutdowns + creates + evaluates + releases;
  }
};

FakeNgx g_ngx;

dlss::Api fake_ngx_api() {
  dlss::Api api;
  api.requirements = [](IDXGIAdapter*, const native::Identity&,
                        const std::filesystem::path&) noexcept -> std::optional<dlss::Requirements> {
    ++g_ngx.requirements;
    return g_ngx.requirement;
  };
  api.init = [](ID3D12Device*, const native::Identity& identity, const std::filesystem::path& data) noexcept {
    ++g_ngx.inits;
    g_ngx.identity = identity;
    g_ngx.data = data;
    return g_ngx.initOk;
  };
  api.capability = []() noexcept {
    ++g_ngx.capabilities;
    return g_ngx.capability;
  };
  api.optimal = [](GraphExtent output, UpscalerQuality quality) noexcept -> std::optional<dlss::ModeSize> {
    ++g_ngx.optimals;
    const float ratio = FakeRatios[static_cast<size_t>(quality)];
    const GraphExtent optimal{static_cast<uint32_t>(std::lround(static_cast<float>(output.width) / ratio)),
                              static_cast<uint32_t>(std::lround(static_cast<float>(output.height) / ratio))};
    return dlss::ModeSize{.optimal = optimal, .minimum = optimal, .maximum = output};
  };
  api.shutdown = [](ID3D12Device*) noexcept {
    ++g_ngx.shutdowns;
    g_ngx.events += 's';
  };
  api.create = [](ID3D12GraphicsCommandList* list, const dlss::FeatureDesc& desc) noexcept -> void* {
    ++g_ngx.creates;
    g_ngx.events += 'c';
    g_ngx.created = desc;
    g_ngx.createList = list;
    return g_ngx.createOk ? &g_ngx.feature : nullptr;
  };
  api.evaluate = [](ID3D12GraphicsCommandList*, void* feature, const dlss::Evaluation& evaluation) noexcept {
    ++g_ngx.evaluates;
    g_ngx.evaluation = evaluation;
    return feature == &g_ngx.feature;
  };
  api.release = [](void* feature) noexcept {
    ++g_ngx.releases;
    g_ngx.events += 'r';
    g_ngx.released = feature;
  };
  api.vram = []() noexcept -> std::optional<uint64_t> { return g_ngx.vram; };
  return api;
}

void count_wait() noexcept {
  ++g_ngx.waits;
  g_ngx.events += 'w';
}

std::array<int, 5> g_fakeResources{};

void* fake_resource(size_t index) noexcept { return &g_fakeResources[index]; }

upscale::native_upscale::Inputs fake_inputs(GraphExtent render, GraphExtent output = TestOutput) {
  return {
      .commandList = fake_resource(0),
      .color = {.resource = fake_resource(1), .format = wgpu::TextureFormat::BGRA8Unorm, .size = render},
      .depth = {.resource = fake_resource(2), .format = wgpu::TextureFormat::R32Float, .size = render},
      .motion = {.resource = fake_resource(3), .format = wgpu::TextureFormat::RG16Float, .size = render},
      .output = {.resource = fake_resource(4), .format = wgpu::TextureFormat::RGBA8Unorm, .size = output},
      .motionScale = {upscale::fsr::MotionScale[0] * static_cast<float>(render.width),
                      upscale::fsr::MotionScale[1] * static_cast<float>(render.height)},
  };
}

class Dlss : public ::testing::Test {
protected:
  void SetUp() override {
    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    m_dir = std::filesystem::temp_directory_path() / ("aurora_dlss_" + std::string{info->name()});
    std::error_code ec;
    std::filesystem::remove_all(m_dir, ec);
    ASSERT_TRUE(std::filesystem::create_directories(m_dir, ec)) << ec.message();
    constexpr std::string_view Runtime = "stand-in runtime";
    ASSERT_TRUE(io::write_file(m_dir / std::string{dlss::RuntimeFile},
                               std::span{reinterpret_cast<const uint8_t*>(Runtime.data()), Runtime.size()}));
    testing::reset();
    ngx_probe::testing::reset();
    sentinel::open(m_dir);
    Caps deviceCaps = test::full_caps();
    deviceCaps.d3d12SharedTexture = true;
    deviceCaps.d3d12SharedFence = true;
    testing::set_caps(deviceCaps);
    ngx_probe::testing::set_identity(
        ngx_probe::Identity{.vendorId = 0x10DE, .deviceId = 0x2860, .subSysId = 0x3CBC17AA, .driverVersion = 1});
    ngx_probe::testing::set_runtime_directory(m_dir);
    g_ngx = {};
    m_api = fake_ngx_api();
    dlss::set_api(&m_api);
    dlss::testing::set_project_env(std::nullopt);
    dlss::testing::set_data_directory(m_dir / std::string{dlss::DataFolder});
    dlss::testing::set_fence_wait(&count_wait);
    native::testing::clear_app_identity();
    ngx_bridge::testing::set_upscaler(Upscaler::Dlss, &dlss::upscaler());
    ngx_bridge::testing::set_adapter(adapter());
    gfx::render_worker::initialize();
    m_worker = true;
  }

  void TearDown() override {
    if (m_worker) {
      on_worker([] {
        ngx_bridge::testing::reset();
        native::d3d12::run_teardowns();
        dlss::upscaler().destroy();
      });
      gfx::render_worker::shutdown();
    }
    EXPECT_FALSE(dlss::upscaler().session());
    ngx_probe::testing::reset();
    dlss::set_api(nullptr);
    dlss::testing::reset();
    native::testing::clear_app_identity();
    sentinel::open({});
    testing::reset();
    std::error_code ec;
    std::filesystem::remove_all(m_dir, ec);
  }

  template <typename Work>
  static void on_worker(Work&& work) {
    gfx::render_worker::enqueue_work([&] { work(); });
    gfx::render_worker::synchronize();
  }

  static FeatureSupport support(Upscaler option) {
    return current_support(Feature::Upscaler, static_cast<uint8_t>(option), nullptr);
  }

  static upscale::native_upscale::Adapter adapter(uint32_t vendorId = dlss::NvidiaVendorId) {
    return {.device = &g_ngx, .vendorId = vendorId};
  }

  static upscale::native_upscale::ProbeResult probe(uint32_t vendorId = dlss::NvidiaVendorId) {
    upscale::native_upscale::ProbeResult result;
    on_worker([&] { result = dlss::upscaler().probe(adapter(vendorId)); });
    return result;
  }

  static std::optional<GraphExtent> prepare(GraphExtent output, UpscalerQuality quality = UpscalerQuality::Quality) {
    std::optional<GraphExtent> render;
    on_worker([&] { render = dlss::upscaler().prepare(adapter(), output, quality); });
    return render;
  }

  static bool evaluate(const upscale::native_upscale::Inputs& inputs) {
    bool ok = false;
    on_worker([&] { ok = dlss::upscaler().evaluate(inputs); });
    return ok;
  }

  static void request(Upscaler upscaler) {
    Settings settings;
    settings.upscaler = upscaler;
    static_cast<void>(submit_request(settings));
  }

  static void use_test_identity() { set_app_identity({.projectId = TestProjectId, .engineVersion = "1.2.3"}); }

  [[nodiscard]] bool cache_has_dlss() const {
    const auto bytes = io::read_file(m_dir / ngx_probe::CacheFile);
    return bytes && std::string_view{reinterpret_cast<const char*>(bytes->data()), bytes->size()}.find("\"dlss\"") !=
                        std::string_view::npos;
  }

  std::filesystem::path m_dir;
  dlss::Api m_api;
  bool m_worker = false;
};

TEST_F(Dlss, CapabilityMappingTable) {
  using dlss::Capability;
  using dlss::InitResult;
  const std::vector<std::pair<Capability, FeatureSupport>> capabilities{
      {{.available = true}, {true, Reason::None}},
      {{.available = false}, {false, Reason::UnsupportedGpu}},
      {{.available = true, .needsUpdatedDriver = true}, {false, Reason::DriverTooOld}},
      {{.available = false, .needsUpdatedDriver = true}, {false, Reason::DriverTooOld}},
      {{.available = false, .initResult = InitResult::Denied}, {false, Reason::NativeUnavailable}},
      {{.needsUpdatedDriver = true, .initResult = InitResult::Denied}, {false, Reason::NativeUnavailable}},
      {{.available = false, .initResult = InitResult::OutOfDate}, {false, Reason::DriverTooOld}},
      {{.available = false, .initResult = InitResult::Failed}, {false, Reason::UnsupportedGpu}},
  };
  for (size_t i = 0; i < capabilities.size(); ++i) {
    EXPECT_EQ(dlss::support_from_capability(capabilities[i].first), capabilities[i].second) << i;
  }
  EXPECT_EQ(dlss::refusal_from_requirements({}), std::nullopt);
  EXPECT_EQ(dlss::refusal_from_requirements({.driverTooOld = true}), (FeatureSupport{false, Reason::DriverTooOld}));
  EXPECT_EQ(dlss::refusal_from_requirements({.adapterUnsupported = true}),
            (FeatureSupport{false, Reason::UnsupportedGpu}));
  EXPECT_EQ(dlss::refusal_from_requirements({.osTooOld = true}), (FeatureSupport{false, Reason::NativeUnavailable}));
  EXPECT_EQ(dlss::refusal_from_requirements({.driverTooOld = true, .adapterUnsupported = true}),
            (FeatureSupport{false, Reason::DriverTooOld}));

  use_test_identity();
  for (const uint32_t vendor : {0x8086u, 0x1002u, 0u}) {
    EXPECT_EQ(probe(vendor).support, (FeatureSupport{false, Reason::UnsupportedGpu})) << vendor;
  }
  EXPECT_EQ(g_ngx.calls(), 0u);

  g_ngx.requirement = dlss::Requirements{.adapterUnsupported = true};
  EXPECT_EQ(probe().support, (FeatureSupport{false, Reason::UnsupportedGpu}));
  EXPECT_EQ(g_ngx.inits, 0u);

  g_ngx.requirement = std::nullopt;
  g_ngx.capability = Capability{.needsUpdatedDriver = true, .minDriverMajor = 580, .minDriverMinor = 88};
  request(Upscaler::Dlss);
  EXPECT_EQ(probe().support, (FeatureSupport{false, Reason::DriverTooOld}));
  EXPECT_EQ(g_ngx.inits, 1u);
  EXPECT_EQ(g_ngx.shutdowns, 1u);
  EXPECT_FALSE(dlss::upscaler().session());

  g_ngx.capability = std::nullopt;
  EXPECT_EQ(probe().support, (FeatureSupport{false, Reason::NativeUnavailable}));
  g_ngx.capability = Capability{.available = true};
  g_ngx.initOk = false;
  EXPECT_EQ(probe().support, (FeatureSupport{false, Reason::NativeUnavailable}));
  EXPECT_FALSE(dlss::upscaler().session());
}

TEST_F(Dlss, OutputLimits) {
  using dlss::OutputFit;
  const std::vector<std::pair<GraphExtent, OutputFit>> fits{
      {{64, 32}, OutputFit::Fits},          {{63, 32}, OutputFit::TooSmall},     {{64, 31}, OutputFit::TooSmall},
      {{0, 0}, OutputFit::TooSmall},        {{2560, 1440}, OutputFit::Fits},     {{4779, 2688}, OutputFit::Fits},
      {{7680, 4320}, OutputFit::Fits},      {{7681, 4320}, OutputFit::TooLarge}, {{7680, 4321}, OutputFit::TooLarge},
      {{10240, 1440}, OutputFit::TooLarge},
  };
  for (const auto& [output, fit] : fits) {
    EXPECT_EQ(dlss::output_fit(output), fit) << output.width << "x" << output.height;
  }

  use_test_identity();
  EXPECT_EQ(prepare({7681, 4320}), std::nullopt);
  EXPECT_EQ(dlss::upscaler().prepare_failure(), Reason::OutputTooLarge);
  EXPECT_EQ(prepare({32, 16}), (GraphExtent{32, 16}));
  on_worker([] { EXPECT_EQ(dlss::upscaler().warmup(), upscale::native_upscale::Warmup::Pending); });
  EXPECT_EQ(g_ngx.calls(), 0u);
  EXPECT_EQ(prepare(TestOutput), (GraphExtent{1707, 960}));
  EXPECT_EQ(dlss::upscaler().prepare_failure(), Reason::NativeUnavailable);
  on_worker([] { EXPECT_EQ(dlss::upscaler().warmup(), upscale::native_upscale::Warmup::Ready); });
  EXPECT_EQ(prepare({7680, 4320}, UpscalerQuality::Performance), (GraphExtent{3840, 2160}));

  ngx_probe::initialize(m_dir);
  ngx_probe::request();
  on_worker([] { native_worker_frame_begin(); });
  ASSERT_TRUE(support(Upscaler::Dlss).supported);
  ASSERT_TRUE(cache_has_dlss());
  ngx_probe::fail(Upscaler::Dlss, Reason::OutputTooLarge);
  EXPECT_EQ(support(Upscaler::Dlss), (FeatureSupport{false, Reason::OutputTooLarge}));
  EXPECT_TRUE(cache_has_dlss());
  ngx_probe::fail(Upscaler::Dlss);
  EXPECT_EQ(support(Upscaler::Dlss), (FeatureSupport{false, Reason::NativeUnavailable}));
  EXPECT_FALSE(cache_has_dlss());
}

TEST_F(Dlss, MissingIdentityRefuses) {
  const std::string engine = std::to_string(ApiVersion);
  const native::Identity configured{.projectId = std::string{TestProjectId}, .engineVersion = "2.0"};
  const std::optional<std::string> env = std::string{"0d6f1c5e-3b7a-4f29-8e61-9a2b4c7d1e08"};
  EXPECT_EQ(dlss::resolve_identity(std::nullopt, std::nullopt), std::nullopt);
  EXPECT_EQ(dlss::resolve_identity(std::nullopt, std::string{}), std::nullopt);
  EXPECT_EQ(dlss::resolve_identity(std::nullopt, env), (native::Identity{.projectId = *env, .engineVersion = engine}));
  EXPECT_EQ(dlss::resolve_identity(configured, env), configured);
  EXPECT_EQ(dlss::resolve_identity(native::Identity{}, env), std::nullopt);
  EXPECT_EQ(dlss::resolve_identity(native::Identity{.projectId = *env}, std::nullopt),
            (native::Identity{.projectId = *env, .engineVersion = engine}));
  EXPECT_TRUE(dlss::valid_project_id(TestProjectId));
  EXPECT_TRUE(dlss::valid_project_id("A0F57B54-1DAF-4934-90AE-C4035C19DF04"));
  for (const std::string_view id :
       {"", "dusklight", "{3f6c2a91-58d4-4e0b-9a7e-2c41d8b5f063}", "3f6c2a9158d44e0b9a7e2c41d8b5f063",
        "3f6c2a91-58d4-4e0b-9a7e-2c41d8b5f06g", "3f6c2a91-58d4-4e0b-9a7e_2c41d8b5f063"}) {
    EXPECT_FALSE(dlss::valid_project_id(id)) << id;
  }

  ngx_probe::initialize(m_dir);
  EXPECT_EQ(support(Upscaler::Dlss), (FeatureSupport{false, Reason::NativeUnavailable}));
  request(Upscaler::Dlss);
  for (int i = 0; i < 10; ++i) {
    static_cast<void>(latch_frame());
    on_worker([] { native_worker_frame_begin(); });
  }
  EXPECT_EQ(probe().support, (FeatureSupport{false, Reason::NativeUnavailable}));
  EXPECT_EQ(g_ngx.calls(), 0u);
  EXPECT_FALSE(std::filesystem::exists(m_dir / std::string{dlss::DataFolder}));

  dlss::testing::set_project_env(*env);
  ngx_probe::initialize(m_dir);
  EXPECT_EQ(support(Upscaler::Dlss).reason, Reason::Probing);
  set_app_identity({.projectId = {}, .engineVersion = "2.0"});
  ngx_probe::initialize(m_dir);
  EXPECT_EQ(support(Upscaler::Dlss), (FeatureSupport{false, Reason::NativeUnavailable}));
  set_app_identity({.projectId = "dusklight"});
  ngx_probe::initialize(m_dir);
  EXPECT_EQ(support(Upscaler::Dlss), (FeatureSupport{false, Reason::NativeUnavailable}));
  EXPECT_EQ(g_ngx.calls(), 0u);

  use_test_identity();
  ngx_probe::initialize(m_dir);
  EXPECT_EQ(support(Upscaler::Dlss).reason, Reason::Probing);
  ngx_probe::request();
  on_worker([] { native_worker_frame_begin(); });
  EXPECT_TRUE(support(Upscaler::Dlss).supported);
  EXPECT_EQ(g_ngx.inits, 1u);
  EXPECT_EQ(g_ngx.identity, (native::Identity{.projectId = std::string{TestProjectId}, .engineVersion = "1.2.3"}));
  EXPECT_EQ(g_ngx.data, m_dir / std::string{dlss::DataFolder});
}

TEST_F(Dlss, ModesFromOptimalSettings) {
  use_test_identity();
  EXPECT_EQ(prepare(TestOutput), (GraphExtent{1707, 960}));
  EXPECT_EQ(g_ngx.optimals, dlss::ModeCount);
  const dlss::Modes& modes = dlss::upscaler().modes();
  const std::array<GraphExtent, dlss::ModeCount> expected{GraphExtent{2560, 1440}, GraphExtent{1707, 960},
                                                          GraphExtent{1488, 837}, GraphExtent{1280, 720},
                                                          GraphExtent{853, 480}};
  for (size_t i = 0; i < dlss::ModeCount; ++i) {
    const auto quality = static_cast<UpscalerQuality>(i);
    ASSERT_TRUE(modes[i].has_value()) << i;
    EXPECT_EQ(modes[i]->optimal, expected[i]) << i;
    EXPECT_EQ(dlss::render_size(modes, quality, TestOutput), expected[i]) << i;
    EXPECT_EQ(prepare(TestOutput, quality), expected[i]) << i;
  }
  EXPECT_EQ(g_ngx.optimals, dlss::ModeCount);
  EXPECT_EQ(g_ngx.inits, 1u);
  EXPECT_EQ(prepare({1920, 1080}, UpscalerQuality::Performance), (GraphExtent{960, 540}));
  EXPECT_EQ(g_ngx.optimals, 2 * dlss::ModeCount);

  dlss::Modes partial{};
  partial[1] = dlss::ModeSize{.optimal = {3000, 2000}};
  partial[3] = dlss::ModeSize{.optimal = {0, 0}};
  EXPECT_EQ(dlss::render_size(partial, UpscalerQuality::NativeAa, TestOutput), TestOutput);
  EXPECT_EQ(dlss::render_size(partial, UpscalerQuality::Quality, TestOutput), TestOutput);
  EXPECT_EQ(dlss::render_size(partial, UpscalerQuality::Balanced, TestOutput), std::nullopt);
  EXPECT_EQ(dlss::render_size(partial, UpscalerQuality::Performance, TestOutput), std::nullopt);
  EXPECT_EQ(dlss::upscaler().display_name(UpscalerQuality::NativeAa), "DLAA");
  EXPECT_EQ(dlss::upscaler().display_name(UpscalerQuality::Quality), "DLSS");
  EXPECT_EQ(dlss::upscaler().color_encoding(), ColorEncoding::Stored);
  EXPECT_FALSE(dlss::upscaler().wants_reactive());
}

TEST_F(Dlss, ProbeOnlySessionShutsDown) {
  use_test_identity();
  for (const Upscaler requested : {Upscaler::Off, Upscaler::Fsr, Upscaler::Xess}) {
    request(requested);
    EXPECT_TRUE(probe().support.supported) << int{static_cast<uint8_t>(requested)};
    EXPECT_FALSE(dlss::upscaler().session());
  }
  EXPECT_EQ(g_ngx.requirements, 3u);
  EXPECT_EQ(g_ngx.inits, 0u);
  EXPECT_EQ(g_ngx.shutdowns, 0u);

  g_ngx.requirement = std::nullopt;
  request(Upscaler::Off);
  EXPECT_TRUE(probe().support.supported);
  EXPECT_FALSE(dlss::upscaler().session());
  EXPECT_EQ(g_ngx.inits, 1u);
  EXPECT_EQ(g_ngx.shutdowns, 1u);
  EXPECT_EQ(g_ngx.waits, 1u);

  for (const Upscaler requested : {Upscaler::Dlss, Upscaler::Auto}) {
    request(requested);
    const uint32_t requirements = g_ngx.requirements;
    const uint32_t inits = g_ngx.inits;
    const uint32_t shutdowns = g_ngx.shutdowns;
    EXPECT_TRUE(probe().support.supported);
    EXPECT_TRUE(dlss::upscaler().session());
    EXPECT_EQ(prepare(TestOutput), (GraphExtent{1707, 960}));
    EXPECT_EQ(g_ngx.requirements, requirements);
    EXPECT_EQ(g_ngx.inits, inits + 1);
    EXPECT_EQ(g_ngx.shutdowns, shutdowns);
    on_worker([] { dlss::upscaler().destroy(); });
    EXPECT_FALSE(dlss::upscaler().session());
    EXPECT_EQ(g_ngx.shutdowns, shutdowns + 1);
  }
  EXPECT_EQ(g_ngx.waits, g_ngx.shutdowns);

  g_ngx.capability = dlss::Capability{.available = false};
  request(Upscaler::Dlss);
  EXPECT_EQ(probe().support, (FeatureSupport{false, Reason::UnsupportedGpu}));
  EXPECT_FALSE(dlss::upscaler().session());

  g_ngx.capability = dlss::Capability{.available = true};
  g_ngx.requirement = dlss::Requirements{};
  const uint32_t inits = g_ngx.inits;
  ngx_probe::initialize(m_dir);
  request(Upscaler::Off);
  ngx_probe::request();
  on_worker([] { native_worker_frame_begin(); });
  EXPECT_TRUE(support(Upscaler::Dlss).supported);
  EXPECT_FALSE(dlss::upscaler().session());
  EXPECT_EQ(g_ngx.inits, inits);

  request(Upscaler::Dlss);
  static_cast<void>(probe());
  ASSERT_TRUE(dlss::upscaler().session());
  const uint32_t shutdowns = g_ngx.shutdowns;
  on_worker([] { native::d3d12::run_teardowns(); });
  EXPECT_FALSE(dlss::upscaler().session());
  EXPECT_EQ(g_ngx.shutdowns, shutdowns + 1);
}

TEST_F(Dlss, ModeRatiosAndPhases) {
  const dlss::DlssUpscaler& upscaler = dlss::upscaler();
  const std::vector<std::pair<float, uint32_t>> phases{{1.f, 16},  {1.3f, 16}, {1.5f, 18},
                                                       {1.7f, 23}, {2.f, 32},  {3.f, 72}};
  for (const auto& [ratio, expected] : phases) {
    EXPECT_EQ(upscaler.jitter_phases(ratio), expected) << ratio;
    EXPECT_FLOAT_EQ(upscaler.mip_bias(ratio), std::log2(1.f / ratio)) << ratio;
  }
  EXPECT_FLOAT_EQ(upscaler.mip_bias(1.f), 0.f);

  use_test_identity();
  for (size_t i = 0; i < dlss::ModeCount; ++i) {
    const auto quality = static_cast<UpscalerQuality>(i);
    const std::optional<GraphExtent> render = prepare(TestOutput, quality);
    ASSERT_TRUE(render.has_value()) << i;
    EXPECT_NEAR(static_cast<float>(TestOutput.width) / static_cast<float>(render->width), FakeRatios[i], 0.01f) << i;
    EXPECT_NEAR(static_cast<float>(TestOutput.height) / static_cast<float>(render->height), FakeRatios[i], 0.01f) << i;
    ASSERT_TRUE(evaluate(fake_inputs(*render)));
    ASSERT_TRUE(g_ngx.created.has_value());
    EXPECT_EQ(g_ngx.created->quality, quality);
    EXPECT_EQ(g_ngx.created->render, *render);
    EXPECT_EQ(g_ngx.created->output, TestOutput);
  }
  EXPECT_EQ(g_ngx.creates, dlss::ModeCount);
  EXPECT_EQ(g_ngx.releases, dlss::ModeCount - 1);
}

TEST_F(Dlss, MvScaleSigns) {
  // aurora's motion is NDC(cur) - NDC(prev) with y up; NGX wants previous minus current in render pixels, y down
  EXPECT_EQ(upscale::fsr::MotionScale[0], -0.5f);
  EXPECT_EQ(upscale::fsr::MotionScale[1], 0.5f);
  use_test_identity();
  const std::optional<GraphExtent> render = prepare(TestOutput);
  ASSERT_EQ(render, (GraphExtent{1707, 960}));
  upscale::native_upscale::Inputs inputs = fake_inputs(*render);
  inputs.jitter[0] = 0.25f;
  inputs.jitter[1] = -0.375f;
  inputs.frameTimeMs = 1000.f / 60.f;
  ASSERT_TRUE(evaluate(inputs));
  EXPECT_EQ(g_ngx.createList, fake_resource(0));
  EXPECT_EQ(g_ngx.evaluation.motionScale, (std::array{-853.5f, 480.f}));
  EXPECT_EQ(g_ngx.evaluation.jitter, (std::array{0.25f, -0.375f}));
  EXPECT_EQ(g_ngx.evaluation.render, *render);
  EXPECT_FLOAT_EQ(g_ngx.evaluation.frameTimeMs, 1000.f / 60.f);
  EXPECT_TRUE(g_ngx.evaluation.reset);
  EXPECT_EQ(static_cast<void*>(g_ngx.evaluation.color), fake_resource(1));
  EXPECT_EQ(static_cast<void*>(g_ngx.evaluation.depth), fake_resource(2));
  EXPECT_EQ(static_cast<void*>(g_ngx.evaluation.motion), fake_resource(3));
  EXPECT_EQ(static_cast<void*>(g_ngx.evaluation.output), fake_resource(4));

  inputs.reset = false;
  inputs.motionScale[1] = -inputs.motionScale[1];
  ASSERT_TRUE(evaluate(inputs));
  EXPECT_FALSE(g_ngx.evaluation.reset);
  EXPECT_EQ(g_ngx.evaluation.motionScale, (std::array{-853.5f, -480.f}));
  EXPECT_EQ(g_ngx.creates, 1u);
  EXPECT_EQ(g_ngx.evaluates, 2u);

  EXPECT_FALSE(evaluate(fake_inputs({1280, 720})));
  EXPECT_FALSE(evaluate(fake_inputs(*render, {1920, 1080})));
  upscale::native_upscale::Inputs noList = fake_inputs(*render);
  noList.commandList = nullptr;
  EXPECT_FALSE(evaluate(noList));
  EXPECT_EQ(g_ngx.evaluates, 2u);
}

TEST_F(Dlss, FlagsLdrInverted) {
  const dlss::FeatureFlags flags = dlss::feature_flags();
  EXPECT_TRUE(flags.mvLowRes);
  EXPECT_TRUE(flags.depthInverted);
  EXPECT_FALSE(flags.hdr);
  EXPECT_FALSE(flags.mvJittered);
  EXPECT_FALSE(flags.autoExposure);
  EXPECT_EQ(dlss::upscaler().color_encoding(), ColorEncoding::Stored);

  use_test_identity();
  const std::optional<GraphExtent> render = prepare(TestOutput, UpscalerQuality::Performance);
  ASSERT_EQ(render, (GraphExtent{1280, 720}));
  EXPECT_EQ(g_ngx.creates, 0u);
  EXPECT_EQ(dlss::upscaler().vram_bytes(), 0u);
  ASSERT_TRUE(evaluate(fake_inputs(*render)));
  EXPECT_EQ(g_ngx.created,
            (dlss::FeatureDesc{
                .render = {1280, 720}, .output = TestOutput, .quality = UpscalerQuality::Performance, .flags = flags}));
  EXPECT_EQ(dlss::upscaler().vram_bytes(), g_ngx.vram);

  on_worker([] { dlss::upscaler().destroy(); });
  g_ngx.createOk = false;
  ASSERT_EQ(prepare(TestOutput), (GraphExtent{1707, 960}));
  EXPECT_FALSE(evaluate(fake_inputs({1707, 960})));
  EXPECT_FALSE(dlss::upscaler().has_feature());
  EXPECT_EQ(g_ngx.evaluates, 1u);
  EXPECT_EQ(dlss::upscaler().vram_bytes(), 0u);
}

TEST_F(Dlss, DisplayNameDlaaAtNative) {
  const dlss::DlssUpscaler& upscaler = dlss::upscaler();
  EXPECT_EQ(upscaler.display_name(UpscalerQuality::NativeAa), "DLAA");
  for (const UpscalerQuality quality : {UpscalerQuality::Quality, UpscalerQuality::Balanced,
                                        UpscalerQuality::Performance, UpscalerQuality::UltraPerformance}) {
    EXPECT_EQ(upscaler.display_name(quality), "DLSS") << int{static_cast<uint8_t>(quality)};
  }
  EXPECT_EQ(dlss::mode_name(UpscalerQuality::NativeAa), "DLAA");
  use_test_identity();
  ASSERT_EQ(prepare(TestOutput, UpscalerQuality::NativeAa), TestOutput);
  ASSERT_TRUE(evaluate(fake_inputs(TestOutput)));
  ASSERT_TRUE(g_ngx.created.has_value());
  EXPECT_EQ(g_ngx.created->quality, UpscalerQuality::NativeAa);
  EXPECT_EQ(g_ngx.created->render, TestOutput);
}

TEST_F(Dlss, RecreateWaitsForFence) {
  use_test_identity();
  const std::optional<GraphExtent> quality = prepare(TestOutput);
  ASSERT_TRUE(quality.has_value());
  ASSERT_TRUE(evaluate(fake_inputs(*quality)));
  ASSERT_TRUE(evaluate(fake_inputs(*quality)));
  EXPECT_EQ(g_ngx.events, "c");

  g_ngx.events.clear();
  EXPECT_EQ(prepare(TestOutput), quality);
  EXPECT_EQ(g_ngx.events, "");
  EXPECT_TRUE(dlss::upscaler().has_feature());

  const std::optional<GraphExtent> performance = prepare(TestOutput, UpscalerQuality::Performance);
  ASSERT_TRUE(performance.has_value());
  EXPECT_EQ(g_ngx.events, "wr");
  EXPECT_EQ(g_ngx.released, &g_ngx.feature);
  EXPECT_FALSE(dlss::upscaler().has_feature());
  EXPECT_EQ(dlss::upscaler().vram_bytes(), 0u);
  EXPECT_FALSE(evaluate(fake_inputs(*quality)));
  ASSERT_TRUE(evaluate(fake_inputs(*performance)));
  EXPECT_EQ(g_ngx.events, "wrc");

  const GraphExtent resized{1920, 1080};
  const std::optional<GraphExtent> smaller = prepare(resized, UpscalerQuality::Performance);
  ASSERT_EQ(smaller, (GraphExtent{960, 540}));
  ASSERT_TRUE(evaluate(fake_inputs(*smaller, resized)));
  EXPECT_EQ(g_ngx.events, "wrcwrc");
  EXPECT_EQ(g_ngx.created->output, resized);

  on_worker([] { dlss::upscaler().destroy(); });
  EXPECT_EQ(g_ngx.events, "wrcwrcwrs");
  EXPECT_EQ(g_ngx.releases, g_ngx.creates);
  EXPECT_FALSE(dlss::upscaler().session());

  ASSERT_EQ(prepare(TestOutput), quality);
  ASSERT_TRUE(evaluate(fake_inputs(*quality)));
  g_ngx.events.clear();
  on_worker([] { native::d3d12::run_teardowns(); });
  EXPECT_EQ(g_ngx.events, "wrs");
  EXPECT_EQ(g_ngx.releases, g_ngx.creates);
}
#endif
} // namespace
} // namespace aurora::render::detail
