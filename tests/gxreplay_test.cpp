#include <gtest/gtest.h>

#include "options.hpp"
#include "report.hpp"

#include <optional>
#include <string>
#include <vector>

using namespace aurora::gxreplay;

namespace {
std::optional<Options> parse(std::vector<const char*> args, const char* envBackend = nullptr) {
  args.insert(args.begin(), "aurora_gxreplay");
  std::string error;
  return parse_options(static_cast<int>(args.size()), args.data(), envBackend, error);
}
} // namespace

TEST(GxReplayOptions, Defaults) {
  const Options defaults;
  EXPECT_TRUE(defaults.trace.empty());
  EXPECT_TRUE(defaults.outDir.empty());
  EXPECT_TRUE(defaults.dataDir.empty());
  EXPECT_TRUE(defaults.replacementsDir.empty());
  EXPECT_TRUE(defaults.writeSelfTest.empty());
  EXPECT_EQ(defaults.backend, BACKEND_AUTO);
  EXPECT_EQ(defaults.profile, FEATURE_PROFILE_AUTO);
  EXPECT_EQ(defaults.frames.first, 0u);
  EXPECT_EQ(defaults.frames.last, UINT32_MAX);
  EXPECT_EQ(defaults.repeat, 1u);
  EXPECT_TRUE(defaults.writePng);
  EXPECT_FALSE(defaults.selfTest);
  EXPECT_FALSE(defaults.info);
  EXPECT_FALSE(defaults.allowIncomplete);
  EXPECT_FALSE(defaults.allowDeviceMismatch);
  EXPECT_FALSE(defaults.allowMisses);
  EXPECT_FALSE(defaults.allowCpu);
  EXPECT_FALSE(defaults.showWindow);
  EXPECT_FALSE(defaults.verbose);
  EXPECT_FALSE(defaults.help);

  const auto parsed = parse({"demo.gxtrace", "--out", "out"});
  ASSERT_TRUE(parsed);
  EXPECT_EQ(parsed->trace, "demo.gxtrace");
  EXPECT_EQ(parsed->outDir, "out");
  EXPECT_EQ(parsed->repeat, 1u);
  EXPECT_EQ(parsed->frames, FrameRange{});
}

TEST(GxReplayOptions, ParsesEveryFlag) {
  const auto options = parse({"demo.gxtrace",
                              "--out",
                              "r1",
                              "--backend",
                              "vulkan",
                              "--profile",
                              "compat",
                              "--frames",
                              "3:9",
                              "--repeat",
                              "2",
                              "--no-png",
                              "--replacements",
                              "textures",
                              "--data-dir",
                              "data",
                              "--allow-incomplete",
                              "--allow-device-mismatch",
                              "--allow-misses",
                              "--allow-cpu",
                              "--show-window",
                              "-v"});
  ASSERT_TRUE(options);
  EXPECT_EQ(options->trace, "demo.gxtrace");
  EXPECT_EQ(options->outDir, "r1");
  EXPECT_EQ(options->backend, BACKEND_VULKAN);
  EXPECT_EQ(options->profile, FEATURE_PROFILE_COMPAT);
  EXPECT_EQ(options->frames, (FrameRange{3, 9}));
  EXPECT_EQ(options->repeat, 2u);
  EXPECT_FALSE(options->writePng);
  EXPECT_EQ(options->replacementsDir, "textures");
  EXPECT_EQ(options->dataDir, "data");
  EXPECT_TRUE(options->allowIncomplete);
  EXPECT_TRUE(options->allowDeviceMismatch);
  EXPECT_TRUE(options->allowMisses);
  EXPECT_TRUE(options->allowCpu);
  EXPECT_TRUE(options->showWindow);
  EXPECT_TRUE(options->verbose);

  const auto selfTest = parse({"--self-test", "--out", "st", "--verbose"});
  ASSERT_TRUE(selfTest);
  EXPECT_TRUE(selfTest->selfTest);
  EXPECT_TRUE(selfTest->verbose);

  const auto writeSelfTest = parse({"--write-self-test", "traces/selftest.gxtrace"});
  ASSERT_TRUE(writeSelfTest);
  EXPECT_EQ(writeSelfTest->writeSelfTest, "traces/selftest.gxtrace");

  const auto info = parse({"demo.gxtrace", "--info"});
  ASSERT_TRUE(info);
  EXPECT_TRUE(info->info);

  for (const char* help : {"-h", "--help"}) {
    const auto parsed = parse({help});
    ASSERT_TRUE(parsed);
    EXPECT_TRUE(parsed->help);
  }
}

TEST(GxReplayOptions, EnvironmentBackendIsFallback) {
  const auto fromEnv = parse({"demo.gxtrace", "--out", "o"}, "null");
  ASSERT_TRUE(fromEnv);
  EXPECT_EQ(fromEnv->backend, BACKEND_NULL);

  const auto flagWins = parse({"demo.gxtrace", "--out", "o", "--backend", "d3d12"}, "null");
  ASSERT_TRUE(flagWins);
  EXPECT_EQ(flagWins->backend, BACKEND_D3D12);

  EXPECT_FALSE(parse({"demo.gxtrace", "--out", "o"}, "dx9"));
}

TEST(GxReplayOptions, RejectsInvalidInput) {
  EXPECT_FALSE(parse({"demo.gxtrace", "--out", "o", "--repeat", "0"}));
  EXPECT_FALSE(parse({"demo.gxtrace", "--out", "o", "--frames", "5:2"}));
  EXPECT_FALSE(parse({"demo.gxtrace", "--out", "o", "--frames", "4:4"}));
  EXPECT_FALSE(parse({"demo.gxtrace", "--out", "o", "--frames", "x"}));
  EXPECT_FALSE(parse({"demo.gxtrace", "--out", "o", "--backend", "dx9"}));
  EXPECT_FALSE(parse({"demo.gxtrace", "--out", "o", "--bogus"}));
  EXPECT_FALSE(parse({"demo.gxtrace", "--out"}));
  EXPECT_FALSE(parse({"demo.gxtrace", "--self-test", "--out", "o"}));
  EXPECT_FALSE(parse({"--self-test", "--write-self-test", "t.gxtrace", "--out", "o"}));
  EXPECT_FALSE(parse({"a.gxtrace", "b.gxtrace", "--out", "o"}));
  EXPECT_FALSE(parse({"demo.gxtrace"}));
  EXPECT_FALSE(parse({"--self-test"}));
  EXPECT_FALSE(parse({"--info", "--self-test"}));
  EXPECT_FALSE(parse({"--out", "o"}));
}

TEST(GxReplayOptions, FrameRanges) {
  EXPECT_EQ(parse_frame_range("7"), (FrameRange{7, 8}));
  EXPECT_EQ(parse_frame_range("3:9"), (FrameRange{3, 9}));
  EXPECT_EQ(parse_frame_range("4:"), (FrameRange{4, UINT32_MAX}));
  EXPECT_EQ(parse_frame_range("0:1"), (FrameRange{0, 1}));
  EXPECT_FALSE(parse_frame_range(""));
  EXPECT_FALSE(parse_frame_range(":5"));
  EXPECT_FALSE(parse_frame_range("9:3"));
  EXPECT_FALSE(parse_frame_range("-1"));
  EXPECT_TRUE(FrameRange({3, 9}).contains(3));
  EXPECT_FALSE(FrameRange({3, 9}).contains(9));
}

TEST(GxReplayReport, FramesJsonIsStable) {
  const ReportContext context{
      .traceFile = "C:\\traces\\demo \"one\".gxtrace",
      .traceVersion = "1.0",
      .backend = "d3d12",
      .adapter = "Test GPU",
      .synthetic = false,
      .runs = 2,
      .deterministic = true,
  };
  const FrameResult frames[]{
      {.frame = 0,
       .sourceFrame = 10,
       .width = 1280,
       .height = 960,
       .rgbaHash = 0x0123456789ABCDEFull,
       .fifoBytes = 4096,
       .mapRecords = 4,
       .misses = 0,
       .draws = 19},
      {.frame = 1,
       .sourceFrame = 11,
       .width = 1280,
       .height = 960,
       .rgbaHash = 0xFEull,
       .fifoBytes = 3000,
       .mapRecords = 0,
       .misses = 0,
       .draws = 18},
  };
  const std::string expected = R"({
  "tool": "aurora_gxreplay",
  "traceFile": "C:\\traces\\demo \"one\".gxtrace",
  "traceVersion": "1.0",
  "synthetic": false,
  "backend": "d3d12",
  "adapter": "Test GPU",
  "runs": 2,
  "deterministic": true,
  "frames": [
    {
      "frame": 0,
      "sourceFrame": 10,
      "width": 1280,
      "height": 960,
      "rgbaXxh3": "0123456789abcdef",
      "fifoBytes": 4096,
      "mapRecords": 4,
      "misses": 0,
      "draws": 19
    },
    {
      "frame": 1,
      "sourceFrame": 11,
      "width": 1280,
      "height": 960,
      "rgbaXxh3": "00000000000000fe",
      "fifoBytes": 3000,
      "mapRecords": 0,
      "misses": 0,
      "draws": 18
    }
  ]
}
)";
  EXPECT_EQ(format_frames_json(context, frames), expected);
  EXPECT_EQ(json_escape("a\tb\n\x01"), "a\\tb\\n\\u0001");
  EXPECT_EQ(hex64(0xAB), "00000000000000ab");
}

TEST(GxReplayReport, TimingJsonFormatsThreeDecimals) {
  const FrameTiming timings[]{
      {.run = 0, .frame = 0, .processMs = 1.23456, .finishMs = 0.5, .gpuWaitMs = 2.0, .totalMs = 3.7346},
      {.run = 0, .frame = 1, .processMs = 1.0, .finishMs = 0.25, .gpuWaitMs = 1.0, .totalMs = 2.25},
      {.run = 1, .frame = 0, .processMs = 0.001, .finishMs = 0.0, .gpuWaitMs = 0.0, .totalMs = 10.0},
  };
  const std::string json = format_timing_json(timings);
  EXPECT_NE(json.find("\"processMs\": 1.235"), std::string::npos) << json;
  EXPECT_NE(json.find("\"finishMs\": 0.500"), std::string::npos) << json;
  EXPECT_NE(json.find("\"gpuWaitMs\": 2.000"), std::string::npos) << json;
  EXPECT_NE(json.find("\"totalMs\": 3.735"), std::string::npos) << json;
  EXPECT_NE(json.find("\"meanMs\": 2.992"), std::string::npos) << json;
  EXPECT_NE(json.find("\"p95Ms\": 3.735"), std::string::npos) << json;
  EXPECT_NE(json.find("\"totalMs\": 5.985"), std::string::npos) << json;
  EXPECT_NE(json.find("\"run\": 1"), std::string::npos) << json;
  EXPECT_NE(json.find("\"processMs\": 0.001"), std::string::npos) << json;
  EXPECT_NE(json.find("\"meanMs\": 10.000"), std::string::npos) << json;
}
