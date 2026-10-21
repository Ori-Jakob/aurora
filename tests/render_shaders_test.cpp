#include <gtest/gtest.h>

#include "render_test_device.hpp"

#include "gfx/pipeline_cache.hpp"
#include "logging.hpp"
#include "render/geometry/indirect.hpp"
#include "render/graph_layout.hpp"
#include "render/jobs.hpp"
#include "render/shader_resolve.hpp"
#include "render/shaders.hpp"
#include "webgpu/gpu.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace aurora::render::shaders {
std::span<const EmbeddedShader> embedded_test_shaders() noexcept;

namespace {
constexpr uint64_t WaitNs = 5'000'000'000;

using Sources = std::map<std::string, std::string, std::less<>>;

std::string_view text_of(const EmbeddedShader& shader) {
  return {reinterpret_cast<const char*>(shader.data), shader.size};
}

SourceLookup lookup_in(const Sources& sources) {
  return [&sources](std::string_view id) -> std::optional<std::string_view> {
    const auto it = sources.find(id);
    return it != sources.end() ? std::optional{std::string_view{it->second}} : std::nullopt;
  };
}

SourceLookup test_table_lookup() {
  return [](std::string_view id) -> std::optional<std::string_view> {
    for (const EmbeddedShader& shader : embedded_test_shaders()) {
      if (shader.id == id) {
        return text_of(shader);
      }
    }
    return std::nullopt;
  };
}

struct Resolved {
  bool ok = false;
  ResolvedSource source;
  std::string error;
};

Resolved resolve(const Sources& sources, std::string_view root, std::span<const std::string_view> defines = {}) {
  Resolved out;
  out.ok = resolve_source(root, defines, lookup_in(sources), out.source, out.error);
  return out;
}

std::string resolve_error(const Sources& sources, std::string_view root = "t") {
  const Resolved result = resolve(sources, root);
  EXPECT_FALSE(result.ok);
  return result.error;
}

TEST(ShaderResolve, ImportsOnceInDepthFirstOrder) {
  const Sources sources{
      {"a", "//#import b\n//#import c\na-body\n"},
      {"b", "//#import c\nb-body\n"},
      {"c", "c-body\n"},
  };
  const Resolved result = resolve(sources, "a");
  ASSERT_TRUE(result.ok) << result.error;
  EXPECT_EQ(result.source.text, "c-body\nb-body\na-body\n");
  EXPECT_EQ(result.source.files, (std::vector<std::string>{"a", "b", "c"}));
}

TEST(ShaderResolve, ImportCycleNamesChain) {
  const Sources sources{
      {"a", "//#import b\n"},
      {"b", "//#import a\n"},
  };
  EXPECT_EQ(resolve_error(sources, "a"), "a.wgsl:1: import cycle: a -> b -> a");
}

TEST(ShaderResolve, UnknownImportReportsLine) {
  ResolvedSource source;
  std::string error;
  EXPECT_FALSE(resolve_source("test/uses_missing", {}, test_table_lookup(), source, error));
  EXPECT_EQ(error, "test/uses_missing.wgsl:1: unknown shader 'test/does_not_exist'");
  EXPECT_TRUE(source.text.empty());
}

TEST(ShaderResolve, DefinesSelectBranches) {
  const Sources sources{{"t",
                         "top\n"
                         "//#if X\n"
                         "x-on\n"
                         "  //#if !Y\n"
                         "x-not-y\n"
                         "  //#else\n"
                         "x-y\n"
                         "  //#endif\n"
                         "//#else\n"
                         "x-off\n"
                         "//#endif\n"
                         "end\n"}};
  constexpr std::array<std::string_view, 1> x{"X"};
  constexpr std::array<std::string_view, 1> y{"Y"};
  constexpr std::array<std::string_view, 2> xy{"X", "Y"};
  EXPECT_EQ(resolve(sources, "t", x).source.text, "top\nx-on\nx-not-y\nend\n");
  EXPECT_EQ(resolve(sources, "t", y).source.text, "top\nx-off\nend\n");
  EXPECT_EQ(resolve(sources, "t", xy).source.text, "top\nx-on\nx-y\nend\n");
  EXPECT_EQ(resolve(sources, "t").source.text, "top\nx-off\nend\n");
}

TEST(ShaderResolve, InactiveImportIsSkipped) {
  const Sources sources{
      {"a", "//#if NOPE\n//#import b\n//#endif\na1\n//#import b\na2\n"},
      {"b", "b-body\n"},
  };
  const Resolved result = resolve(sources, "a");
  ASSERT_TRUE(result.ok) << result.error;
  EXPECT_EQ(result.source.text, "a1\nb-body\na2\n");
  EXPECT_EQ(result.source.files, (std::vector<std::string>{"a", "b"}));
}

TEST(ShaderResolve, InactiveImportIsNotRead) {
  const Sources sources{{"a", "//#if NOPE\n//#import missing\n//#endif\nbody\n"}};
  const Resolved result = resolve(sources, "a");
  ASSERT_TRUE(result.ok) << result.error;
  EXPECT_EQ(result.source.files, (std::vector<std::string>{"a"}));
}

TEST(ShaderResolve, HoistsGlobalDirectives) {
  const Sources sources{
      {"a", "//#import b\n//#import c\n@diagnostic(off, derivative_uniformity) fn f() {}\nenable f16;\n"},
      {"b", "enable f16;\nconst B: u32 = 1u;\n"},
      {"c", "  enable f16;  \ndiagnostic(off, derivative_uniformity);\nconst C: u32 = 2u;\n"},
  };
  const Resolved result = resolve(sources, "a");
  ASSERT_TRUE(result.ok) << result.error;
  EXPECT_EQ(result.source.text,
            "enable f16;\n"
            "diagnostic(off, derivative_uniformity);\n"
            "const B: u32 = 1u;\n"
            "const C: u32 = 2u;\n"
            "@diagnostic(off, derivative_uniformity) fn f() {}\n");
  EXPECT_EQ(describe_line(result.source, 1, 1), "b.wgsl:1:1");
  EXPECT_EQ(describe_line(result.source, 2, 1), "c.wgsl:2:1");
  EXPECT_EQ(describe_line(result.source, 5, 1), "a.wgsl:3:1");
}

TEST(ShaderResolve, HoistsOnlyActiveDirectives) {
  const Sources sources{{"a", "const A: u32 = 1u;\n//#if AURORA_F16\nenable f16;\n//#endif\n"}};
  constexpr std::array<std::string_view, 1> f16{"AURORA_F16"};
  EXPECT_EQ(resolve(sources, "a", f16).source.text, "enable f16;\nconst A: u32 = 1u;\n");
  EXPECT_EQ(resolve(sources, "a").source.text, "const A: u32 = 1u;\n");
}

TEST(ShaderResolve, DirectiveErrors) {
  EXPECT_EQ(resolve_error({{"t", "//#foo\n"}}), "t.wgsl:1: unknown directive '//#foo'");
  EXPECT_EQ(resolve_error({{"t", "x\n//#include x\n"}}), "t.wgsl:2: unknown directive '//#include'");
  EXPECT_EQ(resolve_error({{"t", "//#else\n"}}), "t.wgsl:1: #else without #if");
  EXPECT_EQ(resolve_error({{"t", "//#if A\n//#else\n//#else\n//#endif\n"}}), "t.wgsl:3: duplicate #else");
  EXPECT_EQ(resolve_error({{"t", "//#endif\n"}}), "t.wgsl:1: #endif without #if");
  EXPECT_EQ(resolve_error({{"t", "x\n//#if A\ny\n"}}), "t.wgsl:2: unterminated #if");
  EXPECT_EQ(resolve_error({{"t", "//#if a\n//#endif\n"}}), "t.wgsl:1: invalid define 'a'");
  EXPECT_EQ(resolve_error({{"t", "//#if A B\n//#endif\n"}}), "t.wgsl:1: unexpected text after directive");
  EXPECT_EQ(resolve_error({{"t", "//#if A\n//#endif x\n"}}), "t.wgsl:2: unexpected text after directive");
  EXPECT_EQ(resolve_error({{"t", "//#import a b\n"}}), "t.wgsl:1: unexpected text after directive");
  EXPECT_EQ(resolve_error({{"t", "//#import Post/X\n"}}), "t.wgsl:1: invalid shader id 'Post/X'");
  EXPECT_EQ(resolve_error({{"t", "//#import ../x\n"}}), "t.wgsl:1: invalid shader id '../x'");
  EXPECT_EQ(resolve_error({{"t", "//#import x\n"}}), "t.wgsl:1: unknown shader 'x'");
  EXPECT_EQ(resolve_error({{"t", "//#import t\n"}}), "t.wgsl:1: import cycle: t -> t");

  std::string nested;
  for (int i = 0; i < 17; ++i) {
    nested += "//#if A\n";
  }
  EXPECT_EQ(resolve_error({{"t", nested}}), "t.wgsl:17: #if nested deeper than 16");

  Sources deep;
  for (int i = 0; i < 34; ++i) {
    deep.emplace("d" + std::to_string(i), i < 33 ? "//#import d" + std::to_string(i + 1) + "\n" : "end\n");
  }
  EXPECT_EQ(resolve_error(deep, "d0"), "d32.wgsl:1: imports nested deeper than 32");
  deep.erase("d33");
  deep["d32"] = "end\n";
  EXPECT_TRUE(resolve(deep, "d0").ok);

  constexpr std::array<std::string_view, 1> lower{"lower"};
  const Resolved badDefine = resolve({{"t", "x\n"}}, "t", lower);
  EXPECT_FALSE(badDefine.ok);
  EXPECT_EQ(badDefine.error, "t.wgsl: invalid define 'lower'");
  EXPECT_EQ(resolve_error({}, "missing"), "unknown shader 'missing'");
  EXPECT_EQ(resolve_error({}, "Bad"), "invalid shader id 'Bad'");
}

TEST(ShaderResolve, DirectivesAllowTrailingComments) {
  const Sources sources{
      {"t", "\t//#import b // shared\n//#if A // note\nyes\n//#else // other\nno\n//#endif // done\n"},
      {"b", "b\n"},
  };
  const Resolved result = resolve(sources, "t");
  ASSERT_TRUE(result.ok) << result.error;
  EXPECT_EQ(result.source.text, "b\nno\n");
}

TEST(ShaderResolve, SourceMapPointsToOrigin) {
  const Sources sources{
      {"post/a", "first\n//#import post/b\n//#if NOPE\nhidden\n//#endif\nlast\n"},
      {"post/b", "\n//#if !NOPE\nb-line\n//#endif\n"},
  };
  const Resolved result = resolve(sources, "post/a");
  ASSERT_TRUE(result.ok) << result.error;
  EXPECT_EQ(result.source.text, "first\n\nb-line\nlast\n");
  const std::array<std::string_view, 4> expected{"post/a.wgsl:1:3", "post/b.wgsl:1:3", "post/b.wgsl:3:3",
                                                 "post/a.wgsl:6:3"};
  ASSERT_EQ(result.source.lines.size(), expected.size());
  for (uint32_t line = 1; line <= expected.size(); ++line) {
    EXPECT_EQ(describe_line(result.source, line, 3), expected[line - 1]);
  }
  EXPECT_EQ(describe_line(result.source, 99, 5), "<generated>:99:5");
  EXPECT_EQ(describe_line(result.source, 0, 1), "<generated>:0:1");
}

TEST(ShaderResolve, OutputEndsWithOneNewline) {
  EXPECT_EQ(resolve({{"t", "a\nb"}}, "t").source.text, "a\nb\n");
  EXPECT_EQ(resolve({{"t", "a\r\n\r\n\n"}}, "t").source.text, "a\n");
  const Resolved result = resolve({{"t", "a\n\n"}}, "t");
  EXPECT_EQ(result.source.lines.size(), 1u);
}

TEST(ShaderResolve, NormalizeStripsCrAndBom) {
  EXPECT_EQ(normalize_source("\xEF\xBB\xBF"
                             "a\r\nb\r\n"),
            "a\nb\n");
  EXPECT_EQ(normalize_source("a\xEF\xBB\xBF"), "a\xEF\xBB\xBF");
}

TEST(ShaderResolve, DefineOrderDoesNotChangeText) {
  const Sources sources{{"t", "//#if A\na\n//#endif\n//#if B\nb\n//#endif\n"}};
  constexpr std::array<std::string_view, 3> shuffled{"B", "A", "A"};
  constexpr std::array<std::string_view, 2> sorted{"A", "B"};
  const Resolved first = resolve(sources, "t", shuffled);
  const Resolved second = resolve(sources, "t", sorted);
  ASSERT_TRUE(first.ok && second.ok);
  EXPECT_EQ(first.source.text, "a\nb\n");
  EXPECT_EQ(first.source.text, second.source.text);
}

TEST(ShaderResolve, ValidatesIdsAndDefines) {
  for (const std::string_view id : {"gx_fog", "post/composite", "a/b_2/c"}) {
    EXPECT_TRUE(is_valid_shader_id(id)) << id;
  }
  for (const std::string_view id : {"", "/a", "a/", "a//b", "Post/x", "a.b", "../a", "a b", "a-b"}) {
    EXPECT_FALSE(is_valid_shader_id(id)) << id;
  }
  for (const std::string_view name : {"A", "AURORA_F16", "X2_Y"}) {
    EXPECT_TRUE(is_valid_define(name)) << name;
  }
  for (const std::string_view name : {"", "a", "_A", "2A", "A-B", "Ab"}) {
    EXPECT_FALSE(is_valid_define(name)) << name;
  }
}

TEST(ShaderEmbed, TestTableIsClean) {
  const std::set<std::string> expected{
      "test/broken",         "test/enable_f16", "test/fill_pattern", "test/fullscreen",
      "test/pattern_common", "test/uses_frame", "test/uses_missing",
  };
  std::set<std::string> ids;
  for (const EmbeddedShader& shader : embedded_test_shaders()) {
    ids.emplace(shader.id);
    const std::string_view text = text_of(shader);
    EXPECT_EQ(text.find('\r'), std::string_view::npos) << shader.id;
    EXPECT_EQ(text.find('\0'), std::string_view::npos) << shader.id;
    EXPECT_FALSE(text.starts_with("\xEF\xBB\xBF")) << shader.id;
    EXPECT_EQ(shader.data[shader.size], 0) << shader.id;
    if (std::string_view{shader.id} == "test/fill_pattern") {
      EXPECT_TRUE(text.starts_with("//#import test/pattern_common\n"));
    }
  }
  EXPECT_EQ(ids, expected);
}

TEST(ShaderEmbed, LibraryIdsAreValid) {
  std::set<std::string_view> ids;
  for (const EmbeddedShader& shader : embedded_library()) {
    EXPECT_TRUE(is_valid_shader_id(shader.id)) << shader.id;
    EXPECT_TRUE(ids.emplace(shader.id).second) << shader.id;
    EXPECT_FALSE(std::string_view{shader.id}.starts_with("test/")) << shader.id;
  }
  EXPECT_TRUE(ids.contains("aurora/frame"));
  EXPECT_TRUE(ids.contains("aurora/indirect"));
  EXPECT_TRUE(ids.contains("scan"));
}

struct LogLine {
  AuroraLogLevel level;
  std::string message;
};

std::mutex g_logMutex;
std::vector<LogLine> g_logLines;

void capture_log(AuroraLogLevel level, const char*, const char* message, unsigned int len) {
  std::lock_guard lock{g_logMutex};
  g_logLines.push_back({level, std::string{message, len}});
}

std::atomic_bool g_quiesceOnWarning = false;

// Every warning before a finish runs with no lock held, so quiescing here lands between a slot's sweep and its finish
void quiesce_on_warning(AuroraLogLevel level, const char* module, const char* message, unsigned int len) {
  capture_log(level, module, message, len);
  if (level == LOG_WARNING && g_quiesceOnWarning.exchange(false)) {
    quiesce();
  }
}

size_t count_logs(AuroraLogLevel level, std::string_view text) {
  std::lock_guard lock{g_logMutex};
  return static_cast<size_t>(std::ranges::count_if(g_logLines, [&](const LogLine& line) {
    return line.level == level && line.message.find(text) != std::string::npos;
  }));
}

constexpr std::array<std::string_view, 1> Squares{"FILL_SQUARES"};
constexpr std::array<std::string_view, 2> SquaresTwice{"FILL_SQUARES", "FILL_SQUARES"};
const std::array<Constant, 1> Scale3{{{"scale", 3.0}}};
const std::array<Constant, 1> Scale4{{{"scale", 4.0}}};

void register_test_table() {
  for (const EmbeddedShader& shader : embedded_test_shaders()) {
    ASSERT_TRUE(register_source(shader.id, text_of(shader))) << shader.id;
  }
}

class RenderShaders : public ::testing::Test {
protected:
  void SetUp() override {
    {
      std::lock_guard lock{g_logMutex};
      g_logLines.clear();
    }
    g_config.logCallback = capture_log;
    start({}, 2);
  }

  void TearDown() override {
    stop();
    g_config.logCallback = nullptr;
  }

  void start(const test::TestDeviceOptions& options, uint32_t workers) {
    std::string error;
    m_device = test::create_test_device(options, error);
    if (!m_device) {
      GTEST_SKIP() << "no null device: " << error;
    }
    m_globals = std::make_unique<test::ScopedGlobalDevice>(*m_device, options);
    jobs::initialize(workers);
    initialize();
    register_test_table();
  }

  void stop() {
    if (!m_device) {
      return;
    }
    shutdown();
    jobs::shutdown();
    EXPECT_TRUE(test::take_uncaptured_errors(*m_device).empty());
    m_globals.reset();
    m_device.reset();
  }

  void restart(const test::TestDeviceOptions& options, uint32_t workers) {
    stop();
    start(options, workers);
  }

  std::unique_ptr<test::TestDevice> m_device;
  std::unique_ptr<test::ScopedGlobalDevice> m_globals;
};

TEST_F(RenderShaders, LibraryShadersCompile) {
  ASSERT_FALSE(embedded_library().empty());
  for (const EmbeddedShader& shader : embedded_library()) {
    std::string error;
    EXPECT_TRUE(validate_module(shader.id, {}, error)) << shader.id << ": " << error;
  }
}

TEST_F(RenderShaders, NeverReadyIsNeverReady) {
  EXPECT_TRUE(NeverReady);
  EXPECT_EQ(status(NeverReady), PipelineStatus::Failed);
  EXPECT_FALSE(ready(NeverReady));
  EXPECT_EQ(wait(NeverReady, WaitNs), PipelineStatus::Failed);
  EXPECT_FALSE(compute_pipeline(NeverReady));
  EXPECT_FALSE(render_pipeline(NeverReady));
  const std::array ids{NeverReady.value, 0u};
  EXPECT_FALSE(pipelines_ready(ids));
  const PipelineId real = request_compute({.shader = "test/fill_pattern"});
  ASSERT_EQ(wait(real, WaitNs), PipelineStatus::Ready);
  EXPECT_NE(real, NeverReady);
}

TEST_F(RenderShaders, SessionAdvancesAcrossShutdown) {
  const uint64_t running = session();
  EXPECT_NE(running, 0u);
  shutdown();
  const uint64_t stopped = session();
  EXPECT_GT(stopped, running);
  initialize();
  EXPECT_GT(session(), stopped);
  register_test_table();
}

TEST_F(RenderShaders, LibraryImportResolves) {
  std::string error;
  EXPECT_TRUE(validate_module("test/uses_frame", {}, error)) << error;
  EXPECT_FALSE(validate_module("test/broken", {}, error));
  EXPECT_NE(error.find("test/broken.wgsl:3:"), std::string::npos) << error;
}

TEST_F(RenderShaders, ComputePipelineBecomesReady) {
  const PipelineId id = request_compute({.shader = "test/fill_pattern", .defines = Squares, .constants = Scale3});
  ASSERT_TRUE(id);
  EXPECT_EQ(wait(id, WaitNs), PipelineStatus::Ready);
  EXPECT_TRUE(ready(id));
  EXPECT_TRUE(compute_pipeline(id));
  EXPECT_FALSE(render_pipeline(id));
  const PipelineStats counts = stats();
  EXPECT_EQ(counts.pending, 0u);
  EXPECT_EQ(counts.ready, 1u);
  EXPECT_EQ(counts.modules, 1u);
}

TEST_F(RenderShaders, DuplicateRequestReturnsSameId) {
  const PipelineId first = request_compute({.shader = "test/fill_pattern", .defines = Squares, .constants = Scale3});
  const PipelineId second =
      request_compute({.shader = "test/fill_pattern", .defines = SquaresTwice, .constants = Scale3});
  const PipelineId other = request_compute({.shader = "test/fill_pattern", .defines = Squares, .constants = Scale4});
  ASSERT_TRUE(first && other);
  EXPECT_EQ(first, second);
  EXPECT_NE(first, other);
  EXPECT_EQ(wait(other, WaitNs), PipelineStatus::Ready);
  EXPECT_EQ(wait(first, WaitNs), PipelineStatus::Ready);
  // Overrides are pipeline state, so both pipelines share one module
  EXPECT_EQ(stats().modules, 1u);
}

TEST_F(RenderShaders, BrokenShaderFailsGracefully) {
  const PipelineId id = request_compute({.shader = "test/broken"});
  ASSERT_TRUE(id);
  EXPECT_EQ(wait(id, WaitNs), PipelineStatus::Failed);
  EXPECT_FALSE(compute_pipeline(id));
  const std::vector<PipelineFailure> first = failures();
  ASSERT_FALSE(first.empty());
  EXPECT_NE(first.back().message.find("test/broken.wgsl:3:"), std::string::npos) << first.back().message;
  EXPECT_EQ(count_logs(LOG_WARNING, "test/broken.wgsl:3:"), 1u);
  const uint32_t modules = stats().modules;

  const PipelineId other = request_compute({.shader = "test/broken", .entryPoint = "other"});
  ASSERT_TRUE(other);
  EXPECT_NE(other, id);
  EXPECT_EQ(wait(other, WaitNs), PipelineStatus::Failed);
  const std::vector<PipelineFailure> second = failures();
  ASSERT_EQ(second.size(), first.size() + 1);
  EXPECT_NE(second.back().message.find("test/broken.wgsl:3:"), std::string::npos);
  EXPECT_EQ(count_logs(LOG_WARNING, "test/broken.wgsl:3:"), 1u);
  EXPECT_EQ(stats().modules, modules);
  EXPECT_EQ(stats().failed, 2u);
}

TEST_F(RenderShaders, ProductionTogglesStillReportTintErrors) {
  restart({.productionToggles = true}, 2);
  if (!m_device) {
    GTEST_SKIP() << "no device with production toggles";
  }
  const PipelineId id = request_compute({.shader = "test/broken"});
  EXPECT_EQ(wait(id, WaitNs), PipelineStatus::Failed);
  const std::vector<PipelineFailure> list = failures();
  ASSERT_FALSE(list.empty());
  EXPECT_NE(list.back().message.find("test/broken.wgsl:3:"), std::string::npos) << list.back().message;
}

TEST_F(RenderShaders, MissingEntryPointFails) {
  const PipelineId id = request_compute({.shader = "test/fill_pattern", .entryPoint = "nope"});
  EXPECT_EQ(wait(id, WaitNs), PipelineStatus::Failed);
  EXPECT_FALSE(compute_pipeline(id));
  EXPECT_EQ(count_logs(LOG_WARNING, "test/fill_pattern:nope"), 1u);
}

TEST_F(RenderShaders, UnknownShaderFails) {
  const PipelineId id = request_compute({.shader = "test/uses_missing"});
  EXPECT_EQ(wait(id, WaitNs), PipelineStatus::Failed);
  ASSERT_FALSE(failures().empty());
  EXPECT_EQ(failures().back().message, "test/uses_missing.wgsl:1: unknown shader 'test/does_not_exist'");
}

TEST_F(RenderShaders, RejectedRequestsFailOnce) {
  const PipelineId invalid = request_compute({.shader = "Not/Valid"});
  ASSERT_TRUE(invalid);
  EXPECT_EQ(status(invalid), PipelineStatus::Failed);
  EXPECT_EQ(request_compute({.shader = "Not/Valid"}), invalid);
  const PipelineId wide = request_render({.shader = "test/fullscreen", .colorCount = MaxColorTargets + 1});
  ASSERT_TRUE(wide);
  EXPECT_NE(wide, invalid);
  EXPECT_EQ(status(wide), PipelineStatus::Failed);
  EXPECT_EQ(request_render({.shader = "test/fullscreen", .colorCount = MaxColorTargets + 1}), wide);
  const std::vector<PipelineFailure> list = failures();
  ASSERT_EQ(list.size(), 2u);
  EXPECT_EQ(list[0].message, "invalid shader id 'Not/Valid'");
  EXPECT_EQ(list[1].message, "9 colour targets, more than 8");
  EXPECT_EQ(count_logs(LOG_WARNING, "'Not/Valid'"), 1u);
  EXPECT_EQ(count_logs(LOG_WARNING, "more than 8"), 1u);
  EXPECT_EQ(stats().failed, 2u);
  EXPECT_EQ(stats().pending, 0u);
  EXPECT_FALSE(pipelines_ready(std::array{invalid.value}));
}

TEST_F(RenderShaders, RenderPipelineBecomesReady) {
  const PipelineId id = request_render({
      .shader = "test/fullscreen",
      .colorCount = 1,
      .colors = {ColorTarget{.format = wgpu::TextureFormat::RGBA8Unorm}},
  });
  ASSERT_TRUE(id);
  EXPECT_EQ(wait(id, WaitNs), PipelineStatus::Ready);
  EXPECT_TRUE(render_pipeline(id));
  EXPECT_FALSE(compute_pipeline(id));

  const PipelineId blended = request_render({
      .shader = "test/fullscreen",
      .colorCount = 1,
      .colors = {ColorTarget{
          .format = wgpu::TextureFormat::RGBA8Unorm,
          .blend = true,
          .blendState = {.color = {.srcFactor = wgpu::BlendFactor::SrcAlpha,
                                   .dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha},
                         .alpha = {}},
      }},
      .depthFormat = wgpu::TextureFormat::Depth32Float,
      .depthWrite = true,
      .depthCompare = wgpu::CompareFunction::GreaterEqual,
  });
  EXPECT_NE(blended, id);
  EXPECT_EQ(wait(blended, WaitNs), PipelineStatus::Ready);
  const PipelineId depthOnly = request_render({
      .shader = "test/fullscreen",
      .fragmentEntry = "",
      .depthFormat = wgpu::TextureFormat::Depth32Float,
      .depthWrite = true,
      .depthCompare = wgpu::CompareFunction::GreaterEqual,
  });
  EXPECT_EQ(wait(depthOnly, WaitNs), PipelineStatus::Ready);
}

TEST_F(RenderShaders, ConstantsUsedByOneStageValidate) {
  ASSERT_TRUE(register_source("test/tinted", R"(override tint: f32 = 1.0;
@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
  return vec4f(f32(index), 0.0, 0.0, 1.0);
}
@fragment
fn fs_main() -> @location(0) vec4f {
  return vec4f(tint);
}
)"));
  const std::array<Constant, 1> tint{{{"tint", 0.5}}};
  const PipelineId id = request_render({
      .shader = "test/tinted",
      .constants = tint,
      .colorCount = 1,
      .colors = {ColorTarget{.format = wgpu::TextureFormat::RGBA8Unorm}},
  });
  EXPECT_EQ(wait(id, WaitNs), PipelineStatus::Ready);
}

TEST_F(RenderShaders, BlockingModeIsSynchronous) {
  gfx::set_blocking_pipelines(true);
  const PipelineId id = request_compute({.shader = "test/fill_pattern", .constants = Scale4});
  EXPECT_EQ(status(id), PipelineStatus::Ready);
  const PipelineId broken = request_compute({.shader = "test/broken"});
  EXPECT_EQ(status(broken), PipelineStatus::Failed);
  const PipelineId missing = request_compute({.shader = "test/fill_pattern", .entryPoint = "nope"});
  EXPECT_EQ(status(missing), PipelineStatus::Failed);
  gfx::set_blocking_pipelines(false);
}

TEST_F(RenderShaders, BlockingModeWaitsForPendingHit) {
  restart({}, 1);
  std::atomic_bool release = false;
  std::atomic_bool started = false;
  const jobs::JobHandle gate = jobs::submit([&](const jobs::JobContext& context) {
    started = true;
    while (!release && !context.cancelled()) {
      std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
  });
  while (!started) {
    std::this_thread::yield();
  }
  const ComputePipelineDesc desc{.shader = "test/fill_pattern", .constants = Scale4};
  const PipelineId id = request_compute(desc);
  ASSERT_EQ(status(id), PipelineStatus::Pending);
  gfx::set_blocking_pipelines(true);
  PipelineId again;
  std::atomic<PipelineStatus> seen = PipelineStatus::Pending;
  std::atomic_bool returned = false;
  std::thread requester([&] {
    again = request_compute(desc);
    seen = status(again);
    returned = true;
  });
  while (testing::waiters() == 0 && !returned) {
    std::this_thread::yield();
  }
  // The gate still holds the only worker, so the requester is parked in wait() until the release
  EXPECT_EQ(testing::waiters(), 1u);
  EXPECT_FALSE(returned);
  EXPECT_EQ(status(id), PipelineStatus::Pending);
  release = true;
  requester.join();
  gfx::set_blocking_pipelines(false);
  gate.wait();
  EXPECT_EQ(again, id);
  EXPECT_EQ(seen, PipelineStatus::Ready);
  EXPECT_EQ(testing::waiters(), 0u);
}

TEST_F(RenderShaders, ImplicitDefinesFollowDevice) {
  const auto has = [](std::string_view name) {
    return std::ranges::find(implicit_defines(), name) != implicit_defines().end();
  };
  const wgpu::Device& device = webgpu::g_device;
  EXPECT_EQ(has("AURORA_F16"), device.HasFeature(wgpu::FeatureName::ShaderF16));
  EXPECT_EQ(has("AURORA_SUBGROUPS"), device.HasFeature(wgpu::FeatureName::Subgroups));
  EXPECT_EQ(has("AURORA_CORE"), device.HasFeature(wgpu::FeatureName::CoreFeaturesAndLimits));
  std::string error;
  EXPECT_TRUE(validate_module("test/enable_f16", {}, error)) << error;
}

TEST_F(RenderShaders, RegisterSourceRejectsLibraryShadow) {
  EXPECT_FALSE(register_source("aurora/frame", "const X: u32 = 1u;\n"));
  EXPECT_FALSE(register_source("Test/Upper", "const X: u32 = 1u;\n"));
  EXPECT_FALSE(register_source("test/../x", "const X: u32 = 1u;\n"));
  EXPECT_TRUE(has_source("aurora/frame"));
  EXPECT_TRUE(has_source("test/broken"));
  EXPECT_FALSE(has_source("test/nothing"));
  EXPECT_TRUE(register_source("test/crlf", "\xEF\xBB\xBF@compute @workgroup_size(1)\r\nfn main() {}\r\n"));
  std::string error;
  EXPECT_TRUE(validate_module("test/crlf", {}, error)) << error;
}

constexpr std::string_view ValueV1 = R"(override value: f32 = 0.0;
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1)
fn main() {
  out[0] = value;
}
)";

constexpr std::string_view ValueV2 = R"(override value: f32 = 0.0;
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1)
fn main() {
  out[0] = value * 2.0;
}
)";

TEST_F(RenderShaders, SourceGenerationInvalidatesDedup) {
  ASSERT_TRUE(register_source("test/x", ValueV1));
  const PipelineId first = request_compute({.shader = "test/x"});
  EXPECT_EQ(wait(first, WaitNs), PipelineStatus::Ready);
  ASSERT_TRUE(register_source("test/x", ValueV2));
  const PipelineId second = request_compute({.shader = "test/x"});
  EXPECT_NE(first, second);
  EXPECT_EQ(wait(second, WaitNs), PipelineStatus::Ready);
  EXPECT_TRUE(compute_pipeline(first));
  EXPECT_EQ(stats().modules, 2u);
}

TEST_F(RenderShaders, ConcurrentRegisterKeepsSnapshot) {
  restart({}, 4);
  ASSERT_TRUE(register_source("test/x", ValueV1));
  constexpr int Count = 200;
  std::thread writer([] {
    for (int i = 0; i < Count; ++i) {
      register_source("test/x", i % 2 == 0 ? ValueV2 : ValueV1);
    }
  });
  std::vector<PipelineId> ids;
  for (int i = 0; i < Count; ++i) {
    const std::array<Constant, 1> value{{{"value", static_cast<double>(i)}}};
    ids.push_back(request_compute({.shader = "test/x", .constants = value}));
  }
  writer.join();
  for (const PipelineId id : ids) {
    ASSERT_TRUE(id);
    EXPECT_EQ(wait(id, WaitNs), PipelineStatus::Ready);
  }
  EXPECT_TRUE(wait_idle(WaitNs));
  EXPECT_EQ(stats().pending, 0u);
}

TEST_F(RenderShaders, PendingUntilCompileJobRuns) {
  restart({}, 1);
  std::atomic_bool release = false;
  std::atomic_bool started = false;
  const jobs::JobHandle gate = jobs::submit([&](const jobs::JobContext& context) {
    started = true;
    while (!release && !context.cancelled()) {
      std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
  });
  while (!started) {
    std::this_thread::yield();
  }
  const PipelineId id = request_compute({.shader = "test/fill_pattern"});
  EXPECT_EQ(status(id), PipelineStatus::Pending);
  EXPECT_FALSE(pipelines_ready(std::array{id.value, 0u}));
  EXPECT_TRUE(pipelines_ready(std::array{0u, 0u}));
  EXPECT_EQ(stats().pending, 1u);
  EXPECT_FALSE(wait_idle(1'000'000));
  release = true;
  EXPECT_EQ(wait(id, WaitNs), PipelineStatus::Ready);
  EXPECT_TRUE(pipelines_ready(std::array{id.value, 0u}));
  gate.wait();
}

TEST_F(RenderShaders, ShutdownFailsQueuedCompiles) {
  restart({}, 1);
  std::atomic_bool started = false;
  const jobs::JobHandle gate = jobs::submit([&](const jobs::JobContext& context) {
    started = true;
    while (!context.cancelled()) {
      std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
  });
  while (!started) {
    std::this_thread::yield();
  }
  const PipelineId id = request_compute({.shader = "test/fill_pattern"});
  EXPECT_EQ(status(id), PipelineStatus::Pending);
  // The gate only leaves through the pool's own shutdown, so the compile is still queued here
  shutdown();
  jobs::shutdown();
  // A queued compile that never ran must not hold shutdown for the full idle timeout
  EXPECT_EQ(count_logs(LOG_WARNING, "still compiling"), 0u);
  EXPECT_FALSE(request_compute({.shader = "test/fill_pattern"}));
  EXPECT_EQ(status(id), PipelineStatus::Failed);
  jobs::initialize(1);
  initialize();
}

TEST_F(RenderShaders, QuiesceFailsQueuedCompiles) {
  restart({}, 1);
  std::atomic_bool started = false;
  const jobs::JobHandle gate = jobs::submit([&](const jobs::JobContext& context) {
    started = true;
    while (!context.cancelled()) {
      std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
  });
  while (!started) {
    std::this_thread::yield();
  }
  const PipelineId ready = request_compute({.shader = "test/fill_pattern", .constants = Scale3});
  gfx::set_blocking_pipelines(true);
  ASSERT_EQ(status(request_compute({.shader = "test/fill_pattern", .constants = Scale4})), PipelineStatus::Ready);
  gfx::set_blocking_pipelines(false);
  ASSERT_EQ(status(ready), PipelineStatus::Pending);
  quiesce();
  EXPECT_EQ(status(ready), PipelineStatus::Failed);
  EXPECT_EQ(stats().pending, 0u);
  EXPECT_FALSE(request_compute({.shader = "test/fill_pattern"}));
  quiesce();
  shutdown();
  jobs::shutdown();
  jobs::initialize(1);
  initialize();
}

TEST_F(RenderShaders, SweptSlotFinishesOnce) {
  g_config.logCallback = quiesce_on_warning;
  gfx::set_blocking_pipelines(true);
  g_quiesceOnWarning = true;
  const PipelineId broken = request_compute({.shader = "test/broken"});
  gfx::set_blocking_pipelines(false);
  ASSERT_TRUE(broken);
  EXPECT_FALSE(g_quiesceOnWarning);
  EXPECT_EQ(status(broken), PipelineStatus::Failed);
  EXPECT_EQ(stats().pending, 0u);
  EXPECT_EQ(stats().failed, 1u);
  EXPECT_TRUE(failures().empty());

  shutdown();
  initialize();
  g_quiesceOnWarning = true;
  const PipelineId rejected = request_compute({.shader = "Not/Valid"});
  ASSERT_TRUE(rejected);
  EXPECT_FALSE(g_quiesceOnWarning);
  EXPECT_EQ(status(rejected), PipelineStatus::Failed);
  EXPECT_EQ(stats().pending, 0u);
  EXPECT_EQ(stats().failed, 1u);
  EXPECT_TRUE(failures().empty());
  EXPECT_EQ(count_logs(LOG_WARNING, "'Not/Valid'"), 1u);
  g_config.logCallback = capture_log;
}

TEST_F(RenderShaders, QuiesceWithoutCompilesChangesNothing) {
  const PipelineId id = request_compute({.shader = "test/fill_pattern"});
  ASSERT_EQ(wait(id, WaitNs), PipelineStatus::Ready);
  const PipelineStats before = stats();
  quiesce();
  EXPECT_EQ(status(id), PipelineStatus::Ready);
  EXPECT_TRUE(compute_pipeline(id));
  EXPECT_EQ(stats().ready, before.ready);
  EXPECT_EQ(stats().failed, before.failed);
  EXPECT_TRUE(failures().empty());
}

TEST_F(RenderShaders, RequestsBeforeInitializeAreInvalid) {
  shutdown();
  EXPECT_FALSE(request_compute({.shader = "test/fill_pattern"}));
  EXPECT_FALSE(request_compute({.shader = "Not/Valid"}));
  EXPECT_EQ(count_logs(LOG_WARNING, "'Not/Valid'"), 0u);
  EXPECT_EQ(status({}), PipelineStatus::Failed);
  EXPECT_FALSE(ready({}));
  EXPECT_TRUE(pipelines_ready({}));
  initialize();
}

TEST_F(RenderShaders, FrameStructMatchesUniformsSize) {
  const auto layout = [](uint64_t frameBytes) {
    const std::array frameEntries{wgpu::BindGroupLayoutEntry{
        .binding = 0,
        .visibility = wgpu::ShaderStage::Compute,
        .buffer = {.type = wgpu::BufferBindingType::Uniform, .minBindingSize = frameBytes},
    }};
    const std::array passEntries{wgpu::BindGroupLayoutEntry{
        .binding = 0,
        .visibility = wgpu::ShaderStage::Compute,
        .buffer = {.type = wgpu::BufferBindingType::Storage},
    }};
    const wgpu::BindGroupLayoutDescriptor frameDescriptor{.entryCount = frameEntries.size(),
                                                          .entries = frameEntries.data()};
    const wgpu::BindGroupLayoutDescriptor passDescriptor{.entryCount = passEntries.size(),
                                                         .entries = passEntries.data()};
    const std::array groups{webgpu::g_device.CreateBindGroupLayout(&frameDescriptor),
                            webgpu::g_device.CreateBindGroupLayout(&passDescriptor)};
    const wgpu::PipelineLayoutDescriptor descriptor{.bindGroupLayoutCount = groups.size(),
                                                    .bindGroupLayouts = groups.data()};
    return webgpu::g_device.CreatePipelineLayout(&descriptor);
  };
  static_assert(sizeof(detail::FrameUniforms) == 256);
  const PipelineId exact =
      request_compute({.shader = "test/uses_frame", .layout = layout(sizeof(detail::FrameUniforms)), .layoutKey = 1});
  EXPECT_EQ(wait(exact, WaitNs), PipelineStatus::Ready);
  // WGSL struct sizes are multiples of 16, so passing at 256 and failing at 240 pins the size
  const PipelineId smaller = request_compute(
      {.shader = "test/uses_frame", .layout = layout(sizeof(detail::FrameUniforms) - 16), .layoutKey = 2});
  EXPECT_EQ(wait(smaller, WaitNs), PipelineStatus::Failed);
}

struct GpuReadback {
  std::unique_ptr<test::TestDevice> device;
  std::unique_ptr<test::ScopedGlobalDevice> globals;

  bool start(bool productionToggles = false, bool core = true) {
    const test::TestDeviceOptions options{
        .backend = test::parse_test_backend(std::getenv("AURORA_TEST_BACKEND")).value_or(wgpu::BackendType::Undefined),
        .core = core,
        .allowCpu = true,
        .productionToggles = productionToggles,
    };
    std::string error;
    device = test::create_test_device(options, error);
    if (!device) {
      return false;
    }
    // Hardware runs pick their adapter from the environment, so the report names the one used
    ::testing::Test::RecordProperty("adapter", std::string{std::string_view{device->adapterInfo.device}});
    ::testing::Test::RecordProperty("backend", static_cast<int>(device->adapterInfo.backendType));
    globals = std::make_unique<test::ScopedGlobalDevice>(*device, options);
    jobs::initialize(2);
    initialize();
    register_test_table();
    return true;
  }

  ~GpuReadback() {
    if (!device) {
      return;
    }
    shutdown();
    jobs::shutdown();
    EXPECT_TRUE(test::take_uncaptured_errors(*device).empty());
  }

  [[nodiscard]] wgpu::Buffer buffer(uint64_t size, wgpu::BufferUsage usage) const {
    const wgpu::BufferDescriptor descriptor{.usage = usage, .size = size};
    return device->device.CreateBuffer(&descriptor);
  }

  [[nodiscard]] std::vector<uint32_t> run(const wgpu::ComputePipeline& pipeline,
                                          std::span<const wgpu::BindGroup> groups, uint32_t workgroups,
                                          const wgpu::Buffer& output, uint64_t size) const {
    const wgpu::CommandEncoder encoder = device->device.CreateCommandEncoder();
    const wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(pipeline);
    for (uint32_t i = 0; i < groups.size(); ++i) {
      pass.SetBindGroup(i, groups[i]);
    }
    pass.DispatchWorkgroups(workgroups);
    pass.End();
    const wgpu::CommandBuffer commands = encoder.Finish();
    device->queue.Submit(1, &commands);
    return test::read_buffer(*device, output, 0, size).value_or(std::vector<uint32_t>{});
  }

  [[nodiscard]] wgpu::BindGroup group(const wgpu::ComputePipeline& pipeline, uint32_t index,
                                      const wgpu::Buffer& resource, uint64_t size) const {
    const wgpu::BindGroupEntry entry{.binding = 0, .buffer = resource, .size = size};
    const wgpu::BindGroupDescriptor descriptor{
        .layout = pipeline.GetBindGroupLayout(index),
        .entryCount = 1,
        .entries = &entry,
    };
    return device->device.CreateBindGroup(&descriptor);
  }
};

TEST(RenderShadersGpu, ComputeReadback) {
  GpuReadback gpu;
  if (!gpu.start()) {
    GTEST_SKIP() << "no adapter";
  }
  const PipelineId id = request_compute({.shader = "test/fill_pattern", .defines = Squares, .constants = Scale3});
  ASSERT_EQ(wait(id, WaitNs), PipelineStatus::Ready);
  constexpr uint32_t Count = 256;
  constexpr uint64_t Size = Count * sizeof(uint32_t);
  const wgpu::Buffer output = gpu.buffer(Size, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc);
  const wgpu::ComputePipeline pipeline = compute_pipeline(id);
  const std::array groups{gpu.group(pipeline, 0, output, Size)};
  const std::vector<uint32_t> values = gpu.run(pipeline, groups, Count / 64, output, Size);
  ASSERT_EQ(values.size(), Count);
  for (uint32_t i = 0; i < Count; ++i) {
    EXPECT_EQ(values[i], i * i * 3 + 7) << i;
  }
}

constexpr std::string_view FrameFieldsWgsl = R"(//#import aurora/frame
@group(1) @binding(0) var<storage, read_write> out_words: array<u32>;
@compute @workgroup_size(1)
fn main() {
  out_words[0] = aurora_frame.efb_size.x;
  out_words[1] = aurora_frame.efb_size.y;
  out_words[2] = aurora_frame.output_size.x;
  out_words[3] = aurora_frame.output_size.y;
  out_words[4] = bitcast<u32>(aurora_frame.inv_efb_size.x);
  out_words[5] = bitcast<u32>(aurora_frame.inv_efb_size.y);
  out_words[6] = bitcast<u32>(aurora_frame.inv_output_size.x);
  out_words[7] = bitcast<u32>(aurora_frame.inv_output_size.y);
  out_words[8] = aurora_frame.frame_index;
  out_words[9] = aurora_frame.flags;
  out_words[10] = aurora_frame.msaa_samples;
  out_words[11] = aurora_frame.pad0;
  out_words[12] = aurora_frame.reserved[0].x;
  out_words[13] = aurora_frame.reserved[12].w;
}
)";

TEST(RenderShadersGpu, FrameUniformsMatchWgsl) {
  GpuReadback gpu;
  if (!gpu.start()) {
    GTEST_SKIP() << "no adapter";
  }
  ASSERT_TRUE(register_source("test/frame_fields", FrameFieldsWgsl));
  const PipelineId id = request_compute({.shader = "test/frame_fields"});
  ASSERT_EQ(wait(id, WaitNs), PipelineStatus::Ready);

  detail::FrameUniforms uniforms;
  detail::FrameConstants& constants = uniforms.constants;
  constants = {
      .efbSize = {640, 528},
      .outputSize = {1920, 1080},
      .invEfbSize = {1.0f / 640.0f, 1.0f / 528.0f},
      .invOutputSize = {1.0f / 1920.0f, 1.0f / 1080.0f},
      .frameIndex = 77,
      .flags = detail::FrameFlagReversedZ | detail::FrameFlagMultisampled,
      .msaaSamples = 4,
      .pad0 = 0xA5A5A5A5u,
  };
  uniforms.reserved.front() = 0x1234u;
  uniforms.reserved.back() = 0xBEEFu;
  std::array<uint32_t, detail::FrameUniformsSize / sizeof(uint32_t)> words{};
  std::memcpy(words.data(), &uniforms, sizeof(uniforms));

  const wgpu::Buffer frame = gpu.buffer(sizeof(uniforms), wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
  gpu.device->queue.WriteBuffer(frame, 0, &uniforms, sizeof(uniforms));
  constexpr uint64_t Size = 14 * sizeof(uint32_t);
  const wgpu::Buffer output = gpu.buffer(Size, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc);
  const wgpu::ComputePipeline pipeline = compute_pipeline(id);
  const std::array groups{gpu.group(pipeline, 0, frame, sizeof(uniforms)), gpu.group(pipeline, 1, output, Size)};
  const std::vector<uint32_t> values = gpu.run(pipeline, groups, 1, output, Size);
  ASSERT_EQ(values.size(), 14u);
  for (size_t i = 0; i < 12; ++i) {
    EXPECT_EQ(values[i], words[i]) << i;
  }
  EXPECT_EQ(values[12], 0x1234u);
  EXPECT_EQ(values[13], 0xBEEFu);
}

constexpr std::string_view IndirectDrawWgsl = R"(//#import aurora/indirect
@group(0) @binding(0) var<storage, read_write> records: array<AuroraDrawArgs>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  let first_vertex = select((i * 13u) % 520u, 0xffffffffu - i, i % 7u == 0u);
  let instance_capacity = select(1000u, AURORA_UNBOUNDED, i % 5u == 0u);
  records[i] = aurora_draw_args((i * 37u) % 600u, (i * 331u) % 1500u, first_vertex, 512u, instance_capacity);
}
)";

constexpr std::string_view IndirectDrawIndexedWgsl = R"(//#import aurora/indirect
@group(0) @binding(0) var<storage, read_write> records: array<AuroraDrawIndexedArgs>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  let first_index = select((i * 101u) % 2100u, 0xfffffff0u + i % 16u, i % 13u == 0u);
  let index_capacity = select(2048u, AURORA_UNBOUNDED, i % 11u == 0u);
  records[i] = aurora_draw_indexed_args((i * 7919u) % 4096u, (i * 331u) % 1500u, first_index, i32(i) - 128,
                                        index_capacity, 1000u);
}
)";

constexpr std::string_view IndirectDispatchWgsl = R"(//#import aurora/indirect
@group(0) @binding(0) var<storage, read_write> records: array<AuroraDispatchArgs>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  records[i] = aurora_dispatch_args((i * 977u) % 70000u, (i * 13u) % 70000u + 1u, select(1u, 65536u, i % 9u == 0u));
}
)";

template <typename Record>
void append_words(std::vector<uint32_t>& words, const Record& record) {
  const size_t first = words.size();
  words.resize(first + sizeof(Record) / sizeof(uint32_t));
  std::memcpy(words.data() + first, &record, sizeof(Record));
}

TEST(RenderShadersGpu, IndirectArgsMatchCpu) {
  constexpr uint32_t Count = 256;
  std::vector<uint32_t> draws;
  std::vector<uint32_t> indexed;
  std::vector<uint32_t> dispatches;
  for (uint32_t i = 0; i < Count; ++i) {
    const uint32_t firstVertex = i % 7 == 0 ? 0xffffffffu - i : (i * 13) % 520;
    const uint32_t instanceCapacity = i % 5 == 0 ? indirect::Unbounded : 1000;
    append_words(draws, indirect::make_draw((i * 37) % 600, (i * 331) % 1500, firstVertex, 512, instanceCapacity));
    const uint32_t firstIndex = i % 13 == 0 ? 0xfffffff0u + i % 16 : (i * 101) % 2100;
    const uint32_t indexCapacity = i % 11 == 0 ? indirect::Unbounded : 2048;
    append_words(indexed, indirect::make_draw_indexed((i * 7919) % 4096, (i * 331) % 1500, firstIndex,
                                                      static_cast<int32_t>(i) - 128, indexCapacity, 1000));
    append_words(dispatches, indirect::make_dispatch((i * 977) % 70000, (i * 13) % 70000 + 1, i % 9 == 0 ? 65536 : 1));
  }
  struct Kernel {
    std::string_view id;
    std::string_view source;
    const std::vector<uint32_t>& expected;
  };
  const std::array<Kernel, 3> kernels{{
      {"test/indirect_draw", IndirectDrawWgsl, draws},
      {"test/indirect_draw_indexed", IndirectDrawIndexedWgsl, indexed},
      {"test/indirect_dispatch", IndirectDispatchWgsl, dispatches},
  }};
  for (const auto [productionToggles, core] :
       {std::pair{false, true}, std::pair{true, true}, std::pair{false, false}}) {
    SCOPED_TRACE(std::string{core ? "core" : "compat"} + (productionToggles ? ", production toggles" : ""));
    GpuReadback gpu;
    if (!gpu.start(productionToggles, core)) {
      GTEST_SKIP() << "no adapter";
    }
    if (gpu.device->adapterInfo.backendType == wgpu::BackendType::Null) {
      GTEST_SKIP() << "the null backend runs no shaders";
    }
    for (const Kernel& kernel : kernels) {
      SCOPED_TRACE(kernel.id);
      ASSERT_TRUE(register_source(kernel.id, kernel.source));
      const PipelineId id = request_compute({.shader = kernel.id});
      ASSERT_EQ(wait(id, WaitNs), PipelineStatus::Ready);
      const uint64_t size = kernel.expected.size() * sizeof(uint32_t);
      const wgpu::Buffer output = gpu.buffer(size, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc);
      const wgpu::ComputePipeline pipeline = compute_pipeline(id);
      const std::array groups{gpu.group(pipeline, 0, output, size)};
      EXPECT_EQ(gpu.run(pipeline, groups, Count / 64, output, size), kernel.expected);
    }
  }
}
} // namespace
} // namespace aurora::render::shaders
