#include <gtest/gtest.h>

#include "options.hpp"
#include "report.hpp"

#include <optional>
#include <string>
#include <string_view>
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
  EXPECT_FALSE(defaults.gpuTimings);
  EXPECT_FALSE(defaults.cpuZones);
  EXPECT_FALSE(defaults.cpuDrawZones);
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
  const auto badEnvWithFlag = parse({"demo.gxtrace", "--out", "o", "--backend", "null"}, "dx9");
  ASSERT_TRUE(badEnvWithFlag);
  EXPECT_EQ(badEnvWithFlag->backend, BACKEND_NULL);
  const auto badEnvWithHelp = parse({"--help"}, "dx9");
  ASSERT_TRUE(badEnvWithHelp);
  EXPECT_TRUE(badEnvWithHelp->help);
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
       .hintBytes = 41,
       .mapRecords = 4,
       .misses = 0,
       .draws = 19},
      {.frame = 1,
       .sourceFrame = 11,
       .width = 1280,
       .height = 960,
       .rgbaHash = 0xFEull,
       .fifoBytes = 3000,
       .hintBytes = 0,
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
      "hintBytes": 41,
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
      "hintBytes": 0,
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

TEST(GxReplayOptions, GpuTimingsFlag) {
  const auto plain = parse({"demo.gxtrace", "--out", "o"});
  ASSERT_TRUE(plain);
  EXPECT_FALSE(plain->gpuTimings);
  const auto timed = parse({"demo.gxtrace", "--out", "o", "--gpu-timings", "--no-png"});
  ASSERT_TRUE(timed);
  EXPECT_TRUE(timed->gpuTimings);
  EXPECT_FALSE(timed->writePng);
  EXPECT_NE(usage().find("--gpu-timings"), std::string_view::npos);
}

TEST(GxReplayReport, GpuTimingJsonFormat) {
  const GpuFrameTiming timings[]{
      {.run = 0, .frame = 0, .frameMs = 3.2154, .dropped = 0, .passes = {{"A", 1.1116}, {"B", 0.5}}},
      {.run = 0, .frame = 1, .frameMs = 4.0, .dropped = 2, .passes = {{"A", 2.0}}},
      {.run = 1, .frame = 0, .frameMs = 1.0, .dropped = 0, .passes = {}},
  };
  const std::string json = format_gpu_timing_json(true, timings);
  EXPECT_EQ(json.find("{\n  \"supported\": true,\n  \"runs\": [\n"), 0u) << json;
  EXPECT_NE(json.find("{\"frame\": 0, \"gpuFrameMs\": 3.215, \"dropped\": 0, \"passes\": "
                      "[{\"name\": \"A\", \"ms\": 1.112}, {\"name\": \"B\", \"ms\": 0.500}]},"),
            std::string::npos)
      << json;
  EXPECT_NE(json.find("{\"frame\": 1, \"gpuFrameMs\": 4.000, \"dropped\": 2, \"passes\": "
                      "[{\"name\": \"A\", \"ms\": 2.000}]}\n"),
            std::string::npos)
      << json;
  EXPECT_NE(json.find("\"meanGpuFrameMs\": 3.608"), std::string::npos) << json;
  EXPECT_NE(json.find("\"p95GpuFrameMs\": 4.000"), std::string::npos) << json;
  EXPECT_NE(json.find("{\"name\": \"A\", \"meanMs\": 1.556, \"p95Ms\": 2.000, \"frames\": 2},"), std::string::npos)
      << json;
  EXPECT_NE(json.find("{\"name\": \"B\", \"meanMs\": 0.500, \"p95Ms\": 0.500, \"frames\": 1}\n"), std::string::npos)
      << json;
  EXPECT_NE(json.find("\"run\": 1,"), std::string::npos) << json;
  EXPECT_NE(json.find("{\"frame\": 0, \"gpuFrameMs\": 1.000, \"dropped\": 0, \"passes\": []}\n"), std::string::npos)
      << json;
  EXPECT_NE(json.find("\"meanGpuFrameMs\": 1.000"), std::string::npos) << json;
  EXPECT_NE(json.find("      \"passes\": []\n"), std::string::npos) << json;
  EXPECT_TRUE(json.ends_with("  ]\n}\n")) << json;
}

TEST(GxReplayReport, GpuTimingUnsupported) {
  EXPECT_EQ(format_gpu_timing_json(false, {}), "{\"supported\": false, \"runs\": []}\n");
  const GpuFrameTiming timings[]{{.run = 0, .frame = 0, .frameMs = 1.0}};
  EXPECT_EQ(format_gpu_timing_json(false, timings), "{\"supported\": false, \"runs\": []}\n");
  EXPECT_EQ(format_gpu_timing_json(true, {}), "{\"supported\": true, \"runs\": []}\n");
}

TEST(GxReplayOptions, CpuZoneFlags) {
  const auto zones = parse({"demo.gxtrace", "--out", "o", "--cpu-zones"});
  ASSERT_TRUE(zones);
  EXPECT_TRUE(zones->cpuZones);
  EXPECT_FALSE(zones->cpuDrawZones);
  const auto drawZones = parse({"demo.gxtrace", "--out", "o", "--cpu-draw-zones", "--no-png"});
  ASSERT_TRUE(drawZones);
  EXPECT_TRUE(drawZones->cpuZones);
  EXPECT_TRUE(drawZones->cpuDrawZones);
  EXPECT_FALSE(drawZones->writePng);
  EXPECT_NE(usage().find("--cpu-zones "), std::string_view::npos);
  EXPECT_NE(usage().find("--cpu-draw-zones "), std::string_view::npos);
}

namespace {
CpuFrameZones zone_row(uint32_t run, uint32_t frame, double fifoMs) {
  return {.run = run, .frame = frame, .fifoMs = fifoMs, .fifoCalls = 1};
}

std::vector<CpuFrameZones> zone_rows() {
  CpuFrameZones first{
      .run = 0,
      .frame = 0,
      .fifoMs = 10.0,
      .pushDrawMs = 4.0,
      .arraysMs = 0.5,
      .pipelineBuildMs = 0.25,
      .bindGroupBuildMs = 0.125,
      .uniformBuildMs = 1.0,
      .fogLutMs = 0.0,
      .fifoCalls = 3,
      .pushDraws = 365,
      .arrayMisses = 115,
      .arrayBytes = 880000,
      .pipelineBuilds = 12,
      .bindGroupBuilds = 40,
      .uniformBuilds = 200,
      .fogLutPushes = 1,
      .fogLutBytes = 17204,
      .resolves = 130,
      .uniqueArrayBytes = 455404,
      .vertBytes = 1000,
      .indexBytes = 2000,
      .uniformBytes = 3000,
      .storageBytes = 900000,
      .residentHits = 7,
      .residentStages = 3,
      .encodedDraws = 365,
      .texturedDraws = 300,
      .uniformBinds = 330,
      .textureBinds = 150,
      .indexBinds = 6,
      .encodeMs = 0.75,
  };
  CpuFrameZones warmA = zone_row(1, 1, 1.0);
  warmA.arraysMs = 0.5;
  warmA.pushDraws = 10;
  warmA.storageBytes = 1000;
  warmA.fogLutBytes = 100;
  warmA.uniqueArrayBytes = 450;
  warmA.residentHits = 3;
  warmA.indexBinds = 4;
  warmA.encodeMs = 0.5;
  CpuFrameZones warmB = zone_row(1, 2, 3.0);
  warmB.arraysMs = 1.5;
  warmB.pushDraws = 21;
  warmB.storageBytes = 600;
  warmB.fogLutBytes = 100;
  warmB.uniqueArrayBytes = 400;
  warmB.residentHits = 5;
  warmB.indexBinds = 6;
  warmB.encodeMs = 1.0;
  return {first, zone_row(0, 1, 9.0), zone_row(1, 0, 50.0), warmA, warmB};
}
} // namespace

TEST(GxReplayReport, CpuZonesJsonFormat) {
  const std::vector<CpuFrameZones> rows = zone_rows();
  const std::string json = format_cpu_zones_json(false, rows);
  EXPECT_EQ(json.find("{\n  \"drawZones\": false,\n  \"runs\": [\n"), 0u) << json;
  EXPECT_NE(
      json.find("        {\"frame\": 0, \"fifoMs\": 10.000, \"pushDrawMs\": 4.000, \"arraysMs\": 0.500, "
                "\"pipelineBuildMs\": 0.250, \"bindGroupBuildMs\": 0.125, \"uniformBuildMs\": 1.000, "
                "\"fogLutMs\": 0.000, \"fifoCalls\": 3, \"pushDraws\": 365, \"arrayMisses\": 115, "
                "\"arrayBytes\": 880000, \"pipelineBuilds\": 12, \"bindGroupBuilds\": 40, \"uniformBuilds\": 200, "
                "\"fogLutPushes\": 1, \"fogLutBytes\": 17204, \"resolves\": 130, \"uniqueArrayBytes\": 455404, "
                "\"vertBytes\": 1000, \"indexBytes\": 2000, \"uniformBytes\": 3000, \"storageBytes\": 900000, "
                "\"residentHits\": 7, \"residentStages\": 3, \"encodedDraws\": 365, \"texturedDraws\": 300, "
                "\"uniformBinds\": 330, \"textureBinds\": 150, \"indexBinds\": 6, \"encodeMs\": 0.750},\n"),
      std::string::npos)
      << json;
  EXPECT_NE(json.find("      \"meanFifoMs\": 18.000,\n      \"p95FifoMs\": 50.000,\n"), std::string::npos) << json;
  EXPECT_NE(json.find("  \"warm\": {\n    \"firstRun\": 1,\n    \"runs\": 1,\n    \"frames\": 2,\n"
                      "    \"meanFifoMs\": 2.000,\n    \"p95FifoMs\": 3.000,\n"),
            std::string::npos)
      << json;
  EXPECT_NE(json.find("    \"p95PushDrawMs\": 0.000,\n    \"meanEncodeMs\": 0.750,\n    \"sectionMeanMs\": "),
            std::string::npos)
      << json;
  EXPECT_NE(json.find("\"pushDraws\": 15.5"), std::string::npos) << json;
  EXPECT_NE(json.find("\"sectionMeanMs\": {\"arrays\": 1.000, "), std::string::npos) << json;
  EXPECT_NE(json.find("\"storageRatio\": null,"), std::string::npos) << json;
  EXPECT_TRUE(
      json.ends_with("\"storageBytes\": 800.0, \"residentHits\": 4.0, \"residentStages\": 0.0, "
                     "\"encodedDraws\": 0.0, \"texturedDraws\": 0.0, \"uniformBinds\": 0.0, "
                     "\"textureBinds\": 0.0, \"indexBinds\": 5.0}\n  }\n}\n"))
      << json;

  const std::string drawZones = format_cpu_zones_json(true, rows);
  EXPECT_EQ(drawZones.find("{\n  \"drawZones\": true,\n"), 0u) << drawZones;
  EXPECT_NE(drawZones.find("\"storageRatio\": {\"mean\": 1.625, \"max\": 2.000},"), std::string::npos) << drawZones;
}

TEST(GxReplayReport, CpuZonesWarmNeedsTwoRuns) {
  const CpuFrameZones oneRun[]{zone_row(0, 0, 1.0), zone_row(0, 1, 1.0), zone_row(0, 2, 1.0)};
  EXPECT_TRUE(format_cpu_zones_json(false, oneRun).ends_with("  ],\n  \"warm\": null\n}\n"));
  const CpuFrameZones coldOnly[]{zone_row(0, 0, 1.0), zone_row(0, 1, 1.0), zone_row(1, 0, 1.0)};
  EXPECT_TRUE(format_cpu_zones_json(true, coldOnly).ends_with("  ],\n  \"warm\": null\n}\n"));
}

TEST(GxReplayReport, CpuZonesJsonEmpty) {
  EXPECT_EQ(format_cpu_zones_json(false, {}), "{\"drawZones\": false, \"runs\": [], \"warm\": null}\n");
  EXPECT_EQ(format_cpu_zones_json(true, {}), "{\"drawZones\": true, \"runs\": [], \"warm\": null}\n");
}

TEST(GxReplayReport, CpuZonesCheck) {
  std::vector<CpuFrameZones> rows{zone_row(0, 0, 1.0), zone_row(0, 1, 0.0), zone_row(1, 0, 1.0), zone_row(1, 1, 1.0)};
  std::string error = "untouched";
  EXPECT_TRUE(check_cpu_zones(rows, 2, false, error));
  EXPECT_EQ(error, "untouched");

  EXPECT_FALSE(check_cpu_zones({}, 2, false, error));
  EXPECT_NE(error, "untouched");
  error.clear();
  EXPECT_FALSE(check_cpu_zones(rows, 3, false, error));
  EXPECT_FALSE(error.empty());

  std::vector<CpuFrameZones> uneven = rows;
  uneven.pop_back();
  error.clear();
  EXPECT_FALSE(check_cpu_zones(uneven, 2, false, error));
  EXPECT_FALSE(error.empty());

  std::vector<CpuFrameZones> noCalls = rows;
  noCalls[2].fifoCalls = 0;
  error.clear();
  EXPECT_FALSE(check_cpu_zones(noCalls, 2, false, error));
  EXPECT_FALSE(error.empty());

  std::vector<CpuFrameZones> untimed = rows;
  for (auto& row : untimed) {
    row.fifoMs = 0.0;
  }
  error.clear();
  EXPECT_FALSE(check_cpu_zones(untimed, 2, false, error));
  EXPECT_FALSE(error.empty());

  error.clear();
  EXPECT_FALSE(check_cpu_zones(rows, 2, true, error));
  EXPECT_FALSE(error.empty());

  std::vector<CpuFrameZones> drawTimed = rows;
  drawTimed[1].pushDrawMs = 0.5;
  error = "untouched";
  EXPECT_TRUE(check_cpu_zones(drawTimed, 2, true, error));
  EXPECT_EQ(error, "untouched");
  error.clear();
  EXPECT_FALSE(check_cpu_zones(drawTimed, 2, false, error));
  EXPECT_FALSE(error.empty());
}

TEST(GxReplayReport, CpuZonesCheckNeedsTextureBinds) {
  std::vector<CpuFrameZones> rows{zone_row(0, 0, 1.0), zone_row(0, 1, 1.0), zone_row(1, 0, 1.0), zone_row(1, 1, 1.0)};
  for (auto& row : rows) {
    row.encodedDraws = 4;
    row.texturedDraws = 3;
    row.textureBinds = 1;
  }
  std::string error = "untouched";
  EXPECT_TRUE(check_cpu_zones(rows, 2, false, error));
  EXPECT_EQ(error, "untouched");

  rows[3].textureBinds = 0;
  EXPECT_FALSE(check_cpu_zones(rows, 2, false, error));
  EXPECT_EQ(error, "cpu zones: run 1 frame 1 has 3 textured draws but no texture binds");

  // The bind counts come with plain --cpu-zones, so draw zones must not gate the check
  for (auto& row : rows) {
    row.pushDrawMs = 0.5;
  }
  error.clear();
  EXPECT_FALSE(check_cpu_zones(rows, 2, true, error));
  EXPECT_EQ(error, "cpu zones: run 1 frame 1 has 3 textured draws but no texture binds");

  rows[3].texturedDraws = 0;
  error = "untouched";
  EXPECT_TRUE(check_cpu_zones(rows, 2, true, error));
  EXPECT_EQ(error, "untouched");
}

TEST(GxReplayOptions, ParsesDebugView) {
  const auto role = parse({"demo.gxtrace", "--out", "o", "--debug-view", "role"});
  ASSERT_TRUE(role);
  EXPECT_EQ(role->debugView, 1u);
  EXPECT_EQ(role->debugViewBit, 0u);
  EXPECT_EQ(role->debugViewName, "class.role");
  EXPECT_TRUE(role->writes_classify_json());
  EXPECT_TRUE(role->classifies());
  EXPECT_FALSE(role->classifyJson);

  const auto byId = parse({"demo.gxtrace", "--out", "o", "--debug-view", "6"});
  ASSERT_TRUE(byId);
  EXPECT_EQ(byId->debugView, 6u);
  EXPECT_EQ(byId->debugViewName, "class.object");

  const auto bit = parse({"demo.gxtrace", "--out", "o", "--debug-view", "bit"});
  ASSERT_TRUE(bit);
  EXPECT_EQ(bit->debugView, 8u);
  EXPECT_EQ(bit->debugViewBit, AURORA_CAT_STATIC);
  EXPECT_EQ(bit->debugViewName, "class.bit");

  const auto decal = parse({"demo.gxtrace", "--out", "o", "--debug-view", "bit:decal"});
  ASSERT_TRUE(decal);
  EXPECT_EQ(decal->debugView, 8u);
  EXPECT_EQ(decal->debugViewBit, AURORA_CAT_DECAL);

  const auto reserved = parse({"demo.gxtrace", "--out", "o", "--debug-view", "100"});
  ASSERT_TRUE(reserved);
  EXPECT_EQ(reserved->debugView, 100u);
  EXPECT_TRUE(reserved->debugViewName.empty());

  std::vector<const char*> args{"aurora_gxreplay", "demo.gxtrace", "--out", "o", "--debug-view", "nope"};
  std::string error;
  EXPECT_FALSE(parse_options(static_cast<int>(args.size()), args.data(), nullptr, error));
  EXPECT_EQ(error, "unknown debug view: nope");
  args.back() = "bit:nope";
  EXPECT_FALSE(parse_options(static_cast<int>(args.size()), args.data(), nullptr, error));
  EXPECT_EQ(error, "unknown debug view: bit:nope");
  EXPECT_NE(usage().find("--debug-view VIEW"), std::string_view::npos);
}

TEST(GxReplayOptions, ParsesClassifyFlags) {
  const Options defaults;
  EXPECT_FALSE(defaults.classifies());
  EXPECT_FALSE(defaults.writes_classify_json());

  const auto json = parse({"demo.gxtrace", "--out", "o", "--classify-json"});
  ASSERT_TRUE(json);
  EXPECT_TRUE(json->classifyJson);
  EXPECT_FALSE(json->classifyObjects);
  EXPECT_FALSE(json->classifyDraws);

  const auto objects = parse({"demo.gxtrace", "--out", "o", "--classify-objects"});
  ASSERT_TRUE(objects);
  EXPECT_TRUE(objects->classifyObjects);
  EXPECT_TRUE(objects->classifyJson);
  EXPECT_TRUE(objects->writes_classify_json());

  const auto draws = parse({"demo.gxtrace", "--out", "o", "--classify-draws"});
  ASSERT_TRUE(draws);
  EXPECT_TRUE(draws->classifyDraws);
  EXPECT_FALSE(draws->writes_classify_json());
  EXPECT_TRUE(draws->classifies());
}

namespace {
namespace scene = aurora::render::scene;

ClassifyFrame crafted_frame() {
  ClassifyFrame frame{.frame = 0, .uniformBytes = 4096};
  scene::FrameReport& report = frame.report;
  scene::FrameStats& stats = report.stats;
  stats.draws = 3;
  stats.suppressed = 1;
  stats.mergeBreaks = 2;
  stats.unclassified = 1;
  stats.tevPolicyDraws = 1;
  stats.kinds[static_cast<size_t>(scene::Kind::Foreign)] = 2;
  stats.roles[static_cast<size_t>(scene::Role::Main)] = 2;
  stats.checks[static_cast<size_t>(scene::MainCheck::Ortho)] = 2;
  stats.windows[static_cast<size_t>(scene::Window::Opaque)] = 2;
  stats.categories[3] = 1;
  stats.categories[24] = 1;
  stats.underScope[static_cast<size_t>(scene::Role::Main)] = {.draws = 1, .asRole = 1};
  stats.markerDraw.fill(-1);
  stats.markerDraw[0] = 0;
  stats.markerDraw[1] = 2;
  stats.hints = {.cameras = 1, .mainCameras = 1, .objectBegins = 1, .objectEnds = 1, .markers = 2, .nestedObjects = 1};
  stats.view = {.substituted = 2, .untouched = 1, .uniformBytes = 512};
  report.mainCamera = true;
  report.cutEpoch = 7;
  report.sceneEpochKnown = true;
  report.sceneEpoch = 3;
  report.objects.push_back({
      .ref = {.id = 0x100000001ull, .subId = 0, .generation = 1, .valid = true},
      .categories = 9,
      .flags = 64,
      .roles = 0b11,
      .draws = 1,
      .mainDraws = 1,
  });
  return frame;
}
} // namespace

TEST(GxReplayReport, FormatsClassifyJson) {
  const ReportContext context{
      .traceFile = "C:\\traces\\t1.gxtrace",
      .backend = "d3d12",
      .adapter = "Test GPU",
  };
  const ClassifyFrame frames[]{crafted_frame()};
  const std::string zeros = R"({"draws": 0, "asRole": 0, "main": 0, "roleOnly": 0, "captures": 0, "skyKind": 0})";
  const std::string expected =
      "{\n"
      "  \"schema\": \"aurora.classify\",\n"
      "  \"version\": 1,\n"
      "  \"trace\": \"C:\\\\traces\\\\t1.gxtrace\",\n"
      "  \"backend\": \"d3d12\",\n"
      "  \"adapter\": \"Test GPU\",\n"
      "  \"view\": \"class.role\",\n"
      "  \"viewBit\": null,\n"
      "  \"frames\": [\n"
      "    {\n"
      "      \"frame\": 0,\n"
      "      \"draws\": 3,\n"
      "      \"suppressed\": 1,\n"
      "      \"mergeBreaks\": 2,\n"
      "      \"captureCandidates\": 0,\n"
      "      \"fakeShadow\": 0,\n"
      "      \"fakeShadowOverlay\": 0,\n"
      "      \"unclassified\": 1,\n"
      "      \"tevPolicyDraws\": 1,\n"
      "      \"skyMisses\": 0,\n"
      "      \"droppedRecords\": 0,\n"
      "      \"droppedObjects\": 0,\n"
      "      \"kinds\": {\"mainOpaque\": 0, \"mainBlended\": 0, \"overlay\": 0, \"sky\": 0, \"foreign\": 2, "
      "\"mainDepthOnly\": 0},\n"
      "      \"roles\": {\"main\": 2, \"sky\": 0, \"reflection\": 0, \"shadow\": 0, \"offscreen\": 0, \"ui\": 0, "
      "\"none\": 0},\n"
      "      \"checks\": {\"passed\": 0, \"role\": 0, \"target\": 0, \"window\": 0, \"ortho\": 2, \"noCamera\": 0, "
      "\"projection\": 0},\n"
      "      \"windows\": {\"preScene\": 0, \"opaque\": 2, \"translucent\": 0, \"post\": 0, \"sceneEnd\": 0, "
      "\"hud\": 0, \"final\": 0},\n"
      "      \"categories\": {\"static\": 0, \"dynamic\": 0, \"skinned\": 0, \"terrain\": 1, \"foliage\": 0, "
      "\"water\": 0, \"sky\": 0, \"particle\": 0, \"decal\": 0, \"ui\": 0, \"player\": 0, \"emissive\": 0, "
      "\"overlay\": 0, \"noShadowCast\": 0, \"noShadowReceive\": 0, \"noPbr\": 0, \"noCapture\": 0, \"noRt\": 0, "
      "\"screenRead\": 0, \"fakeShadow\": 0, \"cloudReceiver\": 0, \"noAo\": 1},\n"
      "      \"underScope\": {\"main\": {\"draws\": 1, \"asRole\": 1, \"main\": 0, \"roleOnly\": 0, \"captures\": 0, "
      "\"skyKind\": 0}, \"sky\": " +
      zeros + ", \"reflection\": " + zeros + ", \"shadow\": " + zeros + ", \"offscreen\": " + zeros +
      ", \"ui\": " + zeros + ", \"none\": " + zeros +
      "},\n"
      "      \"markers\": {\"sceneBegin\": 0, \"opaqueDone\": 2, \"prePost\": -1, \"sceneEnd\": -1, \"beforeHud\": -1, "
      "\"frameEnd\": -1},\n"
      "      \"hints\": {\"cameras\": 1, \"mainCameras\": 1, \"scopePushes\": 0, \"scopePops\": 0, \"objectBegins\": "
      "1, "
      "\"objectEnds\": 1, \"markers\": 2, \"staleMarkers\": 0, \"scopeOverflow\": 0, \"scopeUnderflow\": 0, "
      "\"nestedObjects\": 1, \"objectUnderflow\": 0, \"openScopes\": 0, \"openObjects\": 0, \"errors\": 1},\n"
      "      \"camera\": {\"main\": true, \"cutEpoch\": 7},\n"
      "      \"sceneEpoch\": 3,\n"
      "      \"viewCounts\": {\"substituted\": 2, \"untouched\": 1, \"pending\": 0, \"overBudget\": 0, "
      "\"uniformBytes\": 512},\n"
      "      \"uniformBytes\": 4096,\n"
      "      \"objects\": [\n"
      "        {\"id\": \"0x0000000100000001\", \"subId\": 0, \"generation\": 1, \"categories\": 9, \"flags\": 64, "
      "\"roles\": [\"main\", \"sky\"], \"draws\": 1, \"mainDraws\": 1}\n"
      "      ]\n"
      "    }\n"
      "  ]\n"
      "}\n";
  EXPECT_EQ(format_classify_json(context, "class.role", "", frames, true), expected);

  ClassifyFrame second = crafted_frame();
  second.frame = 1;
  second.report.sceneEpochKnown = false;
  const ClassifyFrame two[]{crafted_frame(), second};
  const std::string plain = format_classify_json(context, "", "decal", two, false);
  EXPECT_NE(plain.find("  \"view\": null,\n  \"viewBit\": \"decal\",\n"), std::string::npos) << plain;
  EXPECT_EQ(plain.find("\"objects\""), std::string::npos) << plain;
  EXPECT_NE(plain.find("      \"sceneEpoch\": 3,\n"), std::string::npos) << plain;
  EXPECT_NE(plain.find("      \"sceneEpoch\": null,\n"), std::string::npos) << plain;
  EXPECT_NE(plain.find("      \"uniformBytes\": 4096\n    },\n    {\n      \"frame\": 1,\n"), std::string::npos)
      << plain;
  EXPECT_TRUE(plain.ends_with("      \"uniformBytes\": 4096\n    }\n  ]\n}\n")) << plain;

  ClassifyFrame empty = crafted_frame();
  empty.report.objects.clear();
  const ClassifyFrame noObjects[]{empty};
  EXPECT_NE(format_classify_json(context, "", "", noObjects, true).find("      \"objects\": []\n    }\n"),
            std::string::npos);
  EXPECT_TRUE(
      format_classify_json(context, "", "", {}, false).ends_with("  \"viewBit\": null,\n  \"frames\": []\n}\n"));
}

TEST(GxReplayReport, FormatsClassifyDrawLine) {
  scene::ClassifiedDraw draw;
  draw.cls.index = 4;
  draw.cls.kind = scene::Kind::MainOpaque;
  draw.cls.check = scene::MainCheck::Passed;
  draw.cls.key.role = scene::Role::Main;
  draw.cls.key.window = scene::Window::Translucent;
  draw.cls.key.object = {.id = 0x12, .subId = 3, .generation = 2, .valid = true};
  draw.cls.key.categories = 9;
  draw.cls.key.flags = 2;
  draw.cls.key.materialClass = 1;
  draw.cls.key.tevPolicy = 16;
  draw.cls.key.scope = 0xDEADBEEF;
  draw.vertices = 36;
  draw.indexed = true;
  EXPECT_EQ(format_classify_draw_line(5, draw),
            "{\"frame\": 5, \"index\": 4, \"kind\": \"mainOpaque\", \"role\": \"main\", \"check\": \"passed\", "
            "\"window\": \"translucent\", \"object\": \"0x0000000000000012\", \"subId\": 3, \"generation\": 2, "
            "\"categories\": 9, \"flags\": 2, \"material\": 1, \"tevPolicy\": 16, \"scope\": 3735928559, "
            "\"vertices\": 36, \"indexed\": true, \"suppressed\": false}\n");

  scene::ClassifiedDraw loose;
  loose.cls.suppressed = true;
  EXPECT_EQ(format_classify_draw_line(0, loose),
            "{\"frame\": 0, \"index\": 0, \"kind\": \"foreign\", \"role\": \"none\", \"check\": \"role\", "
            "\"window\": \"preScene\", \"object\": null, \"subId\": 0, \"generation\": 0, \"categories\": 0, "
            "\"flags\": 0, \"material\": 0, \"tevPolicy\": 0, \"scope\": 0, \"vertices\": 0, \"indexed\": false, "
            "\"suppressed\": true}\n");
}

TEST(GxReplayReport, FormatsClassifySummary) {
  ClassifyFrame first = crafted_frame();
  first.report.stats.kinds[static_cast<size_t>(scene::Kind::MainOpaque)] = 4;
  first.report.stats.kinds[static_cast<size_t>(scene::Kind::MainDepthOnly)] = 1;
  first.report.stats.kinds[static_cast<size_t>(scene::Kind::Sky)] = 2;
  first.report.stats.skyMisses = 1;
  const ClassifyFrame frames[]{first, crafted_frame()};
  EXPECT_EQ(format_classify_summary(frames, 3),
            "aurora_gxreplay: classification: frames 2, draws 6, main 5, sky 2, sky misses 1, suppressed 2, "
            "hint errors 2, malformed hints 3, view substituted 4\n");
  EXPECT_EQ(format_classify_summary({}, 0),
            "aurora_gxreplay: classification: frames 0, draws 0, main 0, sky 0, sky misses 0, suppressed 0, "
            "hint errors 0, malformed hints 0, view substituted 0\n");
}
