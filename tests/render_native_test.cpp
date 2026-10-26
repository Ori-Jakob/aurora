#include <gtest/gtest.h>

#include "render_test_caps.hpp"

#include "gfx/frame.hpp"
#include "gfx/render_worker.hpp"
#include "render/native/native.hpp"
#include "render/settings.hpp"

#if AURORA_ENABLE_NATIVE_D3D12
#include "render/native/d3d12/sentinel.hpp"
#include "render/native/d3d12/teardown.hpp"
#endif

#if AURORA_ENABLE_NATIVE_D3D12 && AURORA_ENABLE_XESS
#include "render/frame_state.hpp"
#include "render/native/d3d12/probe.hpp"
#include "render/native/d3d12/upscale_bridge.hpp"
#include "render/native/d3d12/xess.hpp"

#include <optional>
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
#endif
} // namespace
} // namespace aurora::render::detail
