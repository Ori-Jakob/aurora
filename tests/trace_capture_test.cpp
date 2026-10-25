#include <aurora/gfx.hpp>
#include <aurora/gpu_systems.hpp>
#include <aurora/trace.hpp>
#include <gtest/gtest.h>

#include "gx/command_processor.hpp"
#include "gx/fifo.hpp"
#include "gx/gpu_channel.hpp"
#include "gx/guest_memory.hpp"
#include "gx/hints.hpp"
#include "internal.hpp"
#if AURORA_ENABLE_RENDER
#include "render/graph_api.hpp"

#include <aurora/render_graph.hpp>
#endif
#include "trace/capture.hpp"
#include "trace/trace_reader.hpp"
#include "webgpu/gpu.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace aurora::trace {
namespace {
std::filesystem::path temp_trace(const std::string& name) {
  const auto dir = std::filesystem::temp_directory_path() / "aurora_trace_tests";
  std::filesystem::create_directories(dir);
  const auto path = dir / (name + ".gxtrace");
  std::filesystem::remove(path);
  return path;
}

void run_frame() {
  detail::on_frame_begin();
  detail::on_frame_end();
}

class TraceCapture : public ::testing::Test {
protected:
  void SetUp() override { webgpu::g_frameBuffer.size = {640, 480, 1}; }

  void TearDown() override {
    stop_capture();
    detail::shutdown();
  }
};

TEST_F(TraceCapture, StartRejectsInvalidOptions) {
  const auto path = temp_trace("capture_invalid");
  EXPECT_FALSE(start_capture({.path = path, .frameCount = 0}));
  EXPECT_EQ(capture_status().state, CaptureState::Failed);
  EXPECT_FALSE(capture_status().error.empty());
  EXPECT_FALSE(start_capture({.frameCount = 1}));
  EXPECT_FALSE(start_capture({.path = path, .compressionLevel = 20}));
  EXPECT_FALSE(start_capture({.path = path, .compressionLevel = -1}));
  EXPECT_FALSE(is_capturing());

  ASSERT_TRUE(start_capture({.path = path}));
  EXPECT_EQ(capture_status().state, CaptureState::Pending);
  EXPECT_TRUE(is_capturing());
  EXPECT_FALSE(start_capture({.path = temp_trace("capture_second")}));
  EXPECT_EQ(capture_status().state, CaptureState::Pending);
  EXPECT_EQ(capture_status().path, path);
}

TEST_F(TraceCapture, CapturesRequestedFramesAndFinalizes) {
  const auto path = temp_trace("capture_frames");
  ASSERT_TRUE(start_capture({.path = path, .frameCount = 2, .label = "unit"}));
  run_frame();
  EXPECT_EQ(capture_status().state, CaptureState::Capturing);
  EXPECT_NE(gx::guest_memory::g_hooks.load(), nullptr);
  EXPECT_NE(gx::fifo::process_observer(), nullptr);
  run_frame();
  run_frame();

  const auto status = capture_status();
  EXPECT_EQ(status.state, CaptureState::Finished) << status.error;
  EXPECT_EQ(status.framesRequested, 2u);
  EXPECT_EQ(status.framesCaptured, 2u);
  ASSERT_TRUE(std::filesystem::exists(path));
  EXPECT_EQ(status.bytesWritten, std::filesystem::file_size(path));
  EXPECT_FALSE(is_capturing());
  EXPECT_EQ(gx::guest_memory::g_hooks.load(), nullptr);
  EXPECT_EQ(gx::fifo::process_observer(), nullptr);

  TraceReader reader;
  std::string error;
  ASSERT_TRUE(reader.open(path, error)) << error;
  EXPECT_TRUE(reader.complete());
  EXPECT_EQ(reader.info().captureFlags & CaptureReplacementsBypassed, CaptureReplacementsBypassed);
  EXPECT_EQ(reader.info().captureFlags & CaptureSynchronousPublish, CaptureSynchronousPublish);
  EXPECT_EQ(reader.info().requestedFrames, 2u);
  EXPECT_EQ(reader.info().label, "unit");
  EXPECT_FALSE(reader.state_preamble().empty());
  uint32_t frames = 0;
  FrameView frame;
  while (reader.next(frame, {}, error) == TraceReader::Status::Frame) {
    EXPECT_EQ(frame.config.efbWidth, 640u);
    EXPECT_EQ(frame.config.efbHeight, 480u);
    ++frames;
  }
  EXPECT_TRUE(error.empty()) << error;
  EXPECT_EQ(frames, 2u);
  ASSERT_TRUE(reader.trailer());
  EXPECT_EQ(reader.trailer()->frameCount, 2u);
}

TEST_F(TraceCapture, UntraceableCallDiscardsTrace) {
  const auto path = temp_trace("capture_untraceable");
  ASSERT_TRUE(start_capture({.path = path, .frameCount = 2}));
  detail::on_frame_begin();
  EXPECT_FALSE(gfx::create_pass(320, 180));
  EXPECT_NE(capture_status().error.find("create_pass"), std::string::npos);
  detail::on_frame_end();

  const auto status = capture_status();
  EXPECT_EQ(status.state, CaptureState::Failed);
  EXPECT_NE(status.error.find("create_pass"), std::string::npos) << status.error;
  EXPECT_FALSE(std::filesystem::exists(path));
  EXPECT_EQ(gx::guest_memory::g_hooks.load(), nullptr);
  EXPECT_FALSE(is_capturing());
}

#if AURORA_ENABLE_RENDER
class TraceCaptureWithExternal : public TraceCapture {
protected:
  void TearDown() override {
    TraceCapture::TearDown();
    render::detail::testing::reset_external_passes();
  }

  static void declare_sink(render::PassBuilder& builder, void*) { builder.side_effect(); }

  static void encode_nothing(render::PassContext&, void*) {}
};

TEST_F(TraceCaptureWithExternal, ExternalPassDiscardsCapture) {
  const auto path = temp_trace("capture_external");
  ASSERT_TRUE(start_capture({.path = path, .frameCount = 2}));
  ASSERT_NE(render::register_pass({
                .name = "demo.test",
                .kind = render::PassKind::Compute,
                .declare = declare_sink,
                .encode = encode_nothing,
            }),
            render::InvalidPass);
  render::detail::latch_external_passes();
  run_frame();

  const auto status = capture_status();
  EXPECT_NE(status.state, CaptureState::Finished);
  EXPECT_NE(status.error.find("render graph pass demo.test"), std::string::npos) << status.error;
  EXPECT_FALSE(std::filesystem::exists(path));
  EXPECT_EQ(gx::guest_memory::g_hooks.load(), nullptr);
  EXPECT_EQ(gx::fifo::process_observer(), nullptr);
  EXPECT_FALSE(is_capturing());
}
#endif

TEST_F(TraceCapture, StopWhilePendingCancels) {
  const auto path = temp_trace("capture_cancel");
  ASSERT_TRUE(start_capture({.path = path}));
  stop_capture();
  EXPECT_EQ(capture_status().state, CaptureState::Idle);
  EXPECT_FALSE(is_capturing());
  run_frame();
  EXPECT_FALSE(std::filesystem::exists(path));
}

TEST_F(TraceCapture, StopWhileCapturingFinalizesAfterFrame) {
  const auto path = temp_trace("capture_stop");
  ASSERT_TRUE(start_capture({.path = path, .frameCount = 5}));
  detail::on_frame_begin();
  stop_capture();
  EXPECT_EQ(capture_status().state, CaptureState::Capturing);
  detail::on_frame_end();

  const auto status = capture_status();
  EXPECT_EQ(status.state, CaptureState::Finished);
  EXPECT_EQ(status.framesCaptured, 1u);
  TraceReader reader;
  std::string error;
  ASSERT_TRUE(reader.open(path, error)) << error;
  EXPECT_EQ(reader.header().frameCount, 1u);
}

TEST_F(TraceCapture, ShutdownKeepsAbortedPartialTrace) {
  const auto path = temp_trace("capture_shutdown");
  ASSERT_TRUE(start_capture({.path = path, .frameCount = 3}));
  run_frame();
  detail::shutdown();
  EXPECT_FALSE(is_capturing());

  TraceReader reader;
  std::string error;
  ASSERT_TRUE(reader.open(path, error)) << error;
  EXPECT_EQ(reader.header().flags & (FileFinalized | FileAborted), FileFinalized | FileAborted);
  EXPECT_EQ(reader.header().frameCount, 1u);
  EXPECT_FALSE(reader.complete());
  FrameView frame;
  EXPECT_EQ(reader.next(frame, {}, error), TraceReader::Status::Frame) << error;
  EXPECT_EQ(reader.next(frame, {}, error), TraceReader::Status::End) << error;
  ASSERT_TRUE(reader.trailer());
  EXPECT_EQ(reader.trailer()->endFlags & EndAborted, EndAborted);
}

std::vector<std::string> g_warnings;

void capture_warning(AuroraLogLevel level, const char*, const char* message, unsigned int len) {
  if (level == LOG_WARNING) {
    g_warnings.emplace_back(message, len);
  }
}

size_t warnings_with(std::string_view text) {
  return static_cast<size_t>(std::ranges::count_if(
      g_warnings, [text](const std::string& line) { return line.find(text) != std::string::npos; }));
}

render::gpu::SystemStatus ready_status(void*, uint64_t) noexcept { return render::gpu::SystemStatus::Ready; }

bool accept_dispatch(void*, const gx::gpu_channel::DispatchDecode&) noexcept { return true; }

constexpr gx::gpu_channel::SystemBackend RecordingBackend{.status = ready_status, .dispatch = accept_dispatch};

void decode_pending() {
  if (const uint32_t size = gx::fifo::get_buffer_size(); size != 0) {
    gx::fifo::process(gx::fifo::get_buffer_data(), size);
  }
  gx::fifo::clear_buffer();
}

std::vector<uint8_t> upload_hint(uint64_t handle) {
  constexpr uint32_t PayloadSize = gx::hints::wire::GpuUploadHeader + 4;
  std::vector<uint8_t> out(GX_AURORA_HINT_HEADER_SIZE + PayloadSize);
  gx::hints::wire::Writer writer{out};
  writer.header(GX_AURORA_GPU_UPLOAD, PayloadSize);
  writer.u16(AURORA_HINT_GPU_UPLOAD_VERSION);
  writer.u64(handle);
  writer.u32(0);
  return out;
}

std::vector<uint8_t> update_hint(uint64_t system) {
  constexpr std::array<uint8_t, 16> State{1, 2, 3, 4};
  constexpr auto PayloadSize = static_cast<uint32_t>(gx::hints::wire::GpuUpdateHeader + State.size());
  std::vector<uint8_t> out(GX_AURORA_HINT_HEADER_SIZE + PayloadSize);
  gx::hints::wire::Writer writer{out};
  writer.header(GX_AURORA_GPU_UPDATE_PERSISTENT, PayloadSize);
  writer.u16(AURORA_HINT_GPU_UPDATE_PERSISTENT_VERSION);
  writer.u64(system);
  writer.u32(0);
  writer.bytes(State);
  return out;
}

class TraceCaptureGpu : public TraceCapture {
protected:
  void SetUp() override {
    TraceCapture::SetUp();
    gx::gpu_channel::reset();
    gx::gpu_channel::testing::reset_stats();
    gx::gpu_channel::set_system_backend(&RecordingBackend);
    g_warnings.clear();
    g_config.logCallback = capture_warning;
  }

  void TearDown() override {
    g_config.logCallback = nullptr;
    TraceCapture::TearDown();
    gx::gpu_channel::set_system_backend(nullptr);
    gx::gpu_channel::reset();
    gx::fifo::clear_buffer();
  }
};

TEST_F(TraceCaptureGpu, UncapturedStateAndStorageMissesFlagTheTrailer) {
  namespace gpu = render::gpu;
  const std::array<gpu::KernelDesc, 1> kernels{{{.entryPoint = "main"}}};
  const gpu::SystemId system =
      gpu::create_system({.name = "kept", .wgsl = "fn main() {}", .kernels = kernels, .persistentBytes = 16});
  ASSERT_NE(system, gpu::InvalidSystem);
  decode_pending();
  gx::gpu_channel::end_frame();
  ASSERT_TRUE(gpu::dispatch(system, 0, {}, gpu::InvalidHandle, {1, 1, 1}));
  decode_pending();
  ASSERT_EQ(gpu::stats().dispatches, 1u);
  ASSERT_TRUE(gx::gpu_channel::uncaptured_gpu_state());
  const auto update = update_hint(system);
  gx::fifo::process(update.data(), static_cast<uint32_t>(update.size()));
  ASSERT_EQ(gpu::stats().updates, 1u);

  const auto path = temp_trace("capture_gpu_flags");
  ASSERT_TRUE(start_capture({.path = path, .frameCount = 2}));
  for (uint32_t index = 1; index <= 2; ++index) {
    detail::on_frame_begin();
    const auto upload = upload_hint(uint64_t{gx::gpu_channel::current_serial()} << 32 | index);
    gx::fifo::process(upload.data(), static_cast<uint32_t>(upload.size()));
    detail::on_frame_end();
  }
  EXPECT_EQ(gpu::stats().storageMisses, 2u);
  EXPECT_EQ(capture_status().state, CaptureState::Finished) << capture_status().error;
  EXPECT_EQ(warnings_with("keeps state on the GPU only"), 1u);
  EXPECT_EQ(warnings_with("missed streaming storage during captured frame"), 1u);

  TraceReader reader;
  std::string error;
  ASSERT_TRUE(reader.open(path, error)) << error;
  FrameView frame;
  std::vector<uint32_t> storageMiB;
  while (reader.next(frame, {}, error) == TraceReader::Status::Frame) {
    storageMiB.push_back(frame.config.streamingStorageMiB);
  }
  EXPECT_TRUE(error.empty()) << error;
  ASSERT_EQ(storageMiB.size(), 2u);
  EXPECT_EQ(storageMiB[0], storageMiB[1] + 1);
  ASSERT_TRUE(reader.trailer());
  EXPECT_EQ(reader.trailer()->endFlags & (EndGpuStateUncaptured | EndGpuStorageMissed),
            EndGpuStateUncaptured | EndGpuStorageMissed);
}
} // namespace
} // namespace aurora::trace
