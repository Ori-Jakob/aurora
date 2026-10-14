#include <aurora/gfx.hpp>
#include <aurora/trace.hpp>
#include <gtest/gtest.h>

#include "gx/fifo.hpp"
#include "gx/guest_memory.hpp"
#include "trace/capture.hpp"
#include "trace/trace_reader.hpp"
#include "webgpu/gpu.hpp"

#include <filesystem>
#include <string>

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
} // namespace
} // namespace aurora::trace
