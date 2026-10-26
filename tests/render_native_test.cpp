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
} // namespace
} // namespace aurora::render::detail
