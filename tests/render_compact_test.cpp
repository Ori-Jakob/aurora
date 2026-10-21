#include <gtest/gtest.h>

#include "render_test_caps.hpp"
#include "render_test_device.hpp"

#include "render/geometry/compact.hpp"
#include "render/jobs.hpp"
#include "render/shaders.hpp"
#include "webgpu/caps.hpp"
#include "webgpu/feature_profile.hpp"
#include "webgpu/gpu.hpp"

#include <aurora/render_caps.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <optional>
#include <ostream>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace aurora::render::compact {
namespace {
using aurora::test::TestDevice;
using aurora::test::TestDeviceOptions;

constexpr uint64_t WaitNs = 30'000'000'000;
constexpr uint32_t Poison = 0xA5A5A5A5u;
constexpr uint32_t Guard = 0xDEADBEEFu;
constexpr uint32_t Seed = 0x5EED1234u;
constexpr uint32_t Level2Size = 50000;
constexpr uint32_t Level3Size = 1048577;

Caps compute_limits(Caps caps) {
  DeviceLimits& limits = caps.limits;
  limits.maxComputeWorkgroupsPerDimension = 65535;
  if (caps.core) {
    limits.maxComputeInvocationsPerWorkgroup = 256;
    limits.maxComputeWorkgroupSizeX = 256;
    limits.maxComputeWorkgroupStorageSize = 32768;
  } else {
    limits.maxComputeInvocationsPerWorkgroup = 128;
    limits.maxComputeWorkgroupSizeX = 128;
    limits.maxComputeWorkgroupStorageSize = 16384;
  }
  return caps;
}

Caps full() { return compute_limits(render::test::full_caps()); }
Caps compat() { return compute_limits(render::test::compat_caps()); }
Caps metal32() { return compute_limits(render::test::metal32_caps()); }

struct Reference {
  std::vector<uint32_t> output;
  uint32_t total = 0;
  uint32_t live = 0;
};

Reference cpu_scan(std::span<const uint32_t> input, uint32_t capacity, uint32_t live) {
  const uint32_t n = std::min(live, capacity);
  Reference out{.output = std::vector<uint32_t>(capacity), .live = n};
  std::exclusive_scan(input.begin(), input.begin() + n, out.output.begin(), 0u);
  out.total = std::accumulate(input.begin(), input.begin() + n, 0u);
  std::fill(out.output.begin() + n, out.output.end(), out.total);
  return out;
}

Reference cpu_compact(std::span<const uint32_t> input, uint32_t capacity, uint32_t live, uint32_t mask) {
  const uint32_t n = std::min(live, capacity);
  Reference out{.live = n};
  for (uint32_t i = 0; i < n; ++i) {
    if ((input[i] & mask) != 0) {
      out.output.push_back(i);
    }
  }
  out.total = static_cast<uint32_t>(out.output.size());
  return out;
}

enum class Pattern : uint8_t { Zeros, Ones, Index, AllOnes, Random };
constexpr std::array AllPatterns{Pattern::Zeros, Pattern::Ones, Pattern::Index, Pattern::AllOnes, Pattern::Random};

std::string_view pattern_name(Pattern pattern) {
  switch (pattern) {
  case Pattern::Zeros:
    return "zeros";
  case Pattern::Ones:
    return "ones";
  case Pattern::Index:
    return "index";
  case Pattern::AllOnes:
    return "all ones";
  case Pattern::Random:
    return "random";
  }
  return "unknown";
}

std::vector<uint32_t> make_input(Pattern pattern, uint32_t count, uint32_t seed = Seed) {
  std::vector<uint32_t> out(count);
  std::mt19937 rng{seed};
  for (uint32_t i = 0; i < count; ++i) {
    switch (pattern) {
    case Pattern::Zeros:
      out[i] = 0;
      break;
    case Pattern::Ones:
      out[i] = 1;
      break;
    case Pattern::Index:
      out[i] = i;
      break;
    case Pattern::AllOnes:
      out[i] = UINT32_MAX;
      break;
    case Pattern::Random:
      out[i] = static_cast<uint32_t>(rng());
      break;
    }
  }
  return out;
}

std::string_view backend_name(wgpu::BackendType backend) {
  switch (backend) {
  case wgpu::BackendType::Null:
    return "Null";
  case wgpu::BackendType::D3D11:
    return "D3D11";
  case wgpu::BackendType::D3D12:
    return "D3D12";
  case wgpu::BackendType::Metal:
    return "Metal";
  case wgpu::BackendType::Vulkan:
    return "Vulkan";
  case wgpu::BackendType::OpenGL:
    return "OpenGL";
  case wgpu::BackendType::OpenGLES:
    return "OpenGLES";
  default:
    return "Undefined";
  }
}

void expect_words(std::span<const uint32_t> got, std::span<const uint32_t> want, std::string_view what) {
  ASSERT_EQ(got.size(), want.size()) << what;
  size_t bad = 0;
  size_t first = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    if (got[i] != want[i]) {
      first = bad == 0 ? i : first;
      ++bad;
    }
  }
  EXPECT_EQ(bad, 0u) << what << ": first mismatch at " << first << ", got " << got[first] << ", want " << want[first];
}

size_t count_not(std::span<const uint32_t> words, uint32_t value) {
  return static_cast<size_t>(std::ranges::count_if(words, [value](uint32_t word) { return word != value; }));
}

struct Config {
  const char* name;
  std::optional<AuroraFeatureProfile> profile;
  bool productionToggles;
};

void PrintTo(const Config& config, std::ostream* os) { *os << config.name; }

std::string config_name(const ::testing::TestParamInfo<Config>& info) { return info.param.name; }

const std::array NullConfigs{
    Config{"Default", std::nullopt, false},
    Config{"FullValidated", FEATURE_PROFILE_FULL, false},
    Config{"Compat", FEATURE_PROFILE_COMPAT, true},
    Config{"Metal", FEATURE_PROFILE_METAL, true},
};

const std::array GpuConfigs{
    Config{"FullRelease", FEATURE_PROFILE_FULL, true},
    Config{"FullValidated", FEATURE_PROFILE_FULL, false},
    Config{"Compat", FEATURE_PROFILE_COMPAT, true},
    Config{"Metal", FEATURE_PROFILE_METAL, true},
};

struct JobSetup {
  Op op = Op::Scan;
  uint32_t capacity = 0;
  std::vector<uint32_t> input;
  std::optional<uint32_t> live;
  uint32_t countWord = 0;
  uint32_t keepMask = UINT32_MAX;
  ArgsSpec args;
};

struct Bound {
  Job job;
  std::vector<uint32_t> input;
  wgpu::Buffer inputBuffer;
  wgpu::Buffer outputBuffer;
  wgpu::Buffer scratchBuffer;
  wgpu::Buffer countBuffer;
  wgpu::Buffer argsBuffer;

  [[nodiscard]] uint32_t capacity() const noexcept { return job.plan.capacity; }
};

class CompactFixture : public ::testing::Test {
protected:
  bool open(const Config& config, wgpu::BackendType backend) {
    m_options = {
        .backend = backend,
        .allowCpu = true,
        .productionToggles = config.productionToggles,
        .profile = config.profile,
    };
    m_device = aurora::test::create_test_device(m_options, m_error);
    if (!m_device) {
      return false;
    }
    m_globals = std::make_unique<aurora::test::ScopedGlobalDevice>(*m_device, m_options);
    jobs::initialize(2);
    shaders::initialize();
    initialize();
    testing::set_options({.releaseRejections = true});
    m_started = true;
    return true;
  }

  void TearDown() override {
    if (m_started) {
      shutdown();
      shaders::shutdown();
      jobs::shutdown();
    }
    if (m_device) {
      EXPECT_TRUE(aurora::test::take_uncaptured_errors(*m_device).empty());
    }
    m_globals.reset();
    m_device.reset();
  }

  [[nodiscard]] wgpu::Buffer make_buffer(uint64_t size, wgpu::BufferUsage extra = wgpu::BufferUsage::None) const {
    const wgpu::BufferDescriptor descriptor{
        .usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst | extra,
        .size = std::max<uint64_t>(size, 4),
    };
    return m_device->device.CreateBuffer(&descriptor);
  }

  void write(const wgpu::Buffer& buffer, uint64_t offset, std::span<const uint32_t> words) const {
    if (m_fill && !words.empty()) {
      m_device->queue.WriteBuffer(buffer, offset, words.data(), words.size_bytes());
    }
  }

  void fill(const wgpu::Buffer& buffer, uint64_t offset, uint64_t size, uint32_t word) const {
    if (m_fill && size != 0) {
      const std::vector<uint32_t> words(size / sizeof(uint32_t), word);
      write(buffer, offset, words);
    }
  }

  [[nodiscard]] std::vector<uint32_t> read(const wgpu::Buffer& buffer, uint64_t offset, uint64_t size) const {
    return aurora::test::read_buffer(*m_device, buffer, offset, size).value_or(std::vector<uint32_t>{});
  }

  [[nodiscard]] Bound bind(const JobSetup& setup) const {
    Bound out;
    const Plan plan = make_plan(setup.op, setup.capacity, caps());
    const uint64_t bytes = uint64_t{setup.capacity} * sizeof(uint32_t);
    out.input = setup.input;
    out.input.resize(setup.capacity, 0);
    out.inputBuffer = make_buffer(bytes);
    write(out.inputBuffer, 0, out.input);
    out.outputBuffer = make_buffer(bytes);
    fill(out.outputBuffer, 0, bytes, Poison);
    out.scratchBuffer = make_buffer(plan.scratchBytes);
    fill(out.scratchBuffer, 0, plan.scratchBytes, Poison);
    out.job = {
        .plan = plan,
        .input = {.buffer = out.inputBuffer, .size = bytes},
        .output = {.buffer = out.outputBuffer, .size = bytes},
        .scratch = {.buffer = out.scratchBuffer, .size = plan.scratchBytes},
        .countWord = setup.countWord,
        .keepMask = setup.keepMask,
        .args = setup.args,
    };
    if (setup.live) {
      const uint64_t size = (uint64_t{setup.countWord} + 1) * sizeof(uint32_t);
      out.countBuffer = make_buffer(size);
      fill(out.countBuffer, 0, size, Poison);
      const std::array word{*setup.live};
      write(out.countBuffer, uint64_t{setup.countWord} * sizeof(uint32_t), word);
      out.job.count = {.buffer = out.countBuffer, .size = size};
    }
    if (setup.args.kind != ArgsKind::None) {
      const uint64_t size = uint64_t{setup.args.offset} + args_record_words(setup.args.kind) * sizeof(uint32_t);
      out.argsBuffer = make_buffer(size, wgpu::BufferUsage::Indirect);
      fill(out.argsBuffer, 0, size, Poison);
      out.job.argsTarget = {.buffer = out.argsBuffer, .size = size};
    }
    return out;
  }

  static std::vector<Job> jobs_of(std::span<const Bound> bound) {
    std::vector<Job> out;
    for (const Bound& entry : bound) {
      out.push_back(entry.job);
    }
    return out;
  }

  bool run(std::span<const Job> jobs) const {
    const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
    const bool ok = encode(encoder, jobs, "compact test");
    const wgpu::CommandBuffer commands = encoder.Finish();
    EXPECT_TRUE(aurora::test::submit_and_wait(*m_device, commands));
    return ok;
  }

  bool run(const Job& job) const { return run(std::span{&job, 1}); }

  void push_scope() const { m_device->device.PushErrorScope(wgpu::ErrorFilter::Validation); }

  [[nodiscard]] std::optional<std::string> pop_scope() const { return aurora::test::pop_validation_error(*m_device); }

  void expect_header(const Bound& bound, uint32_t total, uint32_t live) const {
    const std::vector<uint32_t> header = read(bound.scratchBuffer, 0, HeaderWords * sizeof(uint32_t));
    ASSERT_EQ(header.size(), HeaderWords);
    EXPECT_EQ(header[TotalWord], total);
    EXPECT_EQ(header[LiveWord], live);
    EXPECT_EQ(count_not(std::span{header}.subspan(2), Poison), 0u) << "reserved header words were written";
  }

  void expect_scan(const Bound& bound, uint32_t live) const {
    const uint32_t capacity = bound.capacity();
    const Reference ref = cpu_scan(bound.input, capacity, live);
    const std::vector<uint32_t> out = read(bound.outputBuffer, 0, uint64_t{capacity} * sizeof(uint32_t));
    expect_words(out, ref.output, "scan output");
    expect_header(bound, ref.total, ref.live);
  }

  void expect_compact(const Bound& bound, uint32_t live) const {
    const uint32_t capacity = bound.capacity();
    const Reference ref = cpu_compact(bound.input, capacity, live, bound.job.keepMask);
    const std::vector<uint32_t> out = read(bound.outputBuffer, 0, uint64_t{capacity} * sizeof(uint32_t));
    ASSERT_EQ(out.size(), capacity);
    expect_words(std::span{out}.first(ref.total), ref.output, "compact list");
    EXPECT_EQ(count_not(std::span{out}.subspan(ref.total), Poison), 0u) << "compaction wrote past its total";
    expect_header(bound, ref.total, ref.live);
  }

  void expect_result(const Bound& bound, std::optional<uint32_t> live = std::nullopt) const {
    const uint32_t n = live.value_or(bound.capacity());
    if (bound.job.plan.op == Op::Scan) {
      expect_scan(bound, n);
    } else {
      expect_compact(bound, n);
    }
  }

  [[nodiscard]] bool cpu_adapter() const noexcept {
    return m_device->adapterInfo.adapterType == wgpu::AdapterType::CPU;
  }

  TestDeviceOptions m_options;
  std::unique_ptr<TestDevice> m_device;
  std::unique_ptr<aurora::test::ScopedGlobalDevice> m_globals;
  std::string m_error;
  bool m_started = false;
  bool m_fill = true;
};

TEST(CompactPlan, Shapes) {
  const Caps caps = full();
  struct Shape {
    uint32_t capacity;
    uint8_t levels;
    uint8_t dispatches;
    uint64_t scratch;
  };
  constexpr std::array<Shape, 6> Shapes{{
      {1, 1, 1, 64},
      {1024, 1, 1, 64},
      {1025, 2, 3, 80},
      {1u << 20, 2, 3, 4160},
      {(1u << 20) + 1, 3, 5, 4192},
      {MaxCapacity, 3, 5, 262464},
  }};
  for (const Shape& shape : Shapes) {
    for (const Op op : {Op::Scan, Op::Compact}) {
      const Plan plan = make_plan(op, shape.capacity, caps);
      EXPECT_TRUE(plan.valid()) << shape.capacity;
      EXPECT_EQ(plan.op, op);
      EXPECT_EQ(plan.capacity, shape.capacity);
      EXPECT_EQ(plan.levels, shape.levels) << shape.capacity;
      EXPECT_EQ(plan.dispatches, shape.dispatches) << shape.capacity;
      EXPECT_EQ(plan.scratchBytes, shape.scratch) << shape.capacity;
    }
  }
  EXPECT_EQ(max_capacity(caps), MaxCapacity);
  EXPECT_EQ(max_capacity(compat()), 33'554'432u);
  EXPECT_EQ(max_capacity(metal32()), MaxCapacity);
  EXPECT_TRUE(make_plan(Op::Compact, 33'554'432, compat()).valid());
}

TEST(CompactPlan, Rejects) {
  EXPECT_FALSE(make_plan(Op::Scan, 0, full()).valid());
  EXPECT_FALSE(make_plan(Op::Scan, MaxCapacity + 1, full()).valid());
  EXPECT_FALSE(make_plan(Op::Compact, 33'554'433, compat()).valid());
  EXPECT_FALSE(make_plan(Op::Scan, 1024, Caps{}).valid());
  EXPECT_EQ(max_capacity(Caps{}), 0u);
}

TEST(CompactPlan, Support) {
  EXPECT_TRUE(supported(full()));
  EXPECT_TRUE(supported(compat()));
  EXPECT_TRUE(supported(metal32()));
  EXPECT_FALSE(supported(Caps{}));
  EXPECT_FALSE(supported(render::test::full_caps())) << "the shared fixture leaves the workgroup size at 0";
  Caps immediates = full();
  immediates.limits.maxImmediateSize = 48;
  EXPECT_FALSE(supported(immediates));
  Caps storage = full();
  storage.limits.maxStorageBuffersPerShaderStage = 4;
  EXPECT_FALSE(supported(storage));
  Caps invocations = compat();
  invocations.limits.maxComputeInvocationsPerWorkgroup = 64;
  EXPECT_FALSE(supported(invocations));
  Caps workgroupStorage = compat();
  workgroupStorage.limits.maxComputeWorkgroupStorageSize = 256;
  EXPECT_FALSE(supported(workgroupStorage));
}

TEST(CompactPlan, ArgsWords) {
  EXPECT_EQ(args_record_words(ArgsKind::None), 0u);
  EXPECT_EQ(args_record_words(ArgsKind::Dispatch), 3u);
  EXPECT_EQ(args_record_words(ArgsKind::Draw), 4u);
  EXPECT_EQ(args_record_words(ArgsKind::DrawIndexed), 5u);
  EXPECT_EQ(reject_name(Reject::Overlap), "Overlap");
  EXPECT_EQ(reject_name(static_cast<Reject>(200)), "Unknown");
}

TEST(CompactPlan, ValidateRecomputesPlan) {
  const Caps caps = full();
  Job handBuilt{.plan = {.op = Op::Scan, .capacity = 5000, .levels = 1, .dispatches = 1, .scratchBytes = 64}};
  EXPECT_EQ(validate(handBuilt, caps), Reject::Plan);
  const Job big{.plan = make_plan(Op::Compact, 40'000'000, caps)};
  ASSERT_TRUE(big.plan.valid());
  EXPECT_EQ(validate(big, compat()), Reject::Plan);
  EXPECT_EQ(validate(Job{}, caps), Reject::Plan);
  EXPECT_EQ(validate(big, Caps{}), Reject::Unsupported);
  const Job valid{.plan = make_plan(Op::Scan, 5000, caps)};
  EXPECT_EQ(validate(valid, caps), Reject::Usage) << "a valid plan with no buffers fails on its views";
}

TEST(CompactPlan, ValidateArgsSpec) {
  const Caps caps = full();
  const auto job = [&](Op op, uint32_t capacity, ArgsSpec args, uint32_t keepMask = UINT32_MAX) {
    return Job{.plan = make_plan(op, capacity, caps), .keepMask = keepMask, .args = args};
  };
  EXPECT_EQ(validate(job(Op::Scan, 100, {}, 1), caps), Reject::KeepMask);
  EXPECT_EQ(validate(job(Op::Compact, 100, {.kind = ArgsKind::Draw, .scale = 0}), caps), Reject::Args);
  EXPECT_EQ(validate(job(Op::Scan, 100, {.kind = ArgsKind::Draw, .limit = 0}), caps), Reject::Args);
  EXPECT_EQ(validate(job(Op::Compact, 100, {.kind = ArgsKind::Draw, .offset = 2}), caps), Reject::Args);
  EXPECT_EQ(validate(job(Op::Compact, 1u << 20, {.kind = ArgsKind::Dispatch}), caps), Reject::Args);
  EXPECT_EQ(validate(job(Op::Compact, 1u << 20, {.kind = ArgsKind::Dispatch, .scale = 64}), caps), Reject::Usage);
  EXPECT_EQ(validate(job(Op::Compact, 1u << 20, {.kind = ArgsKind::Draw, .scale = 5000}), caps), Reject::Args);
  EXPECT_EQ(validate(job(Op::Compact, 100, {.kind = ArgsKind::Draw, .field = 2}), caps), Reject::Args);
  EXPECT_EQ(validate(job(Op::Compact, 100, {.kind = static_cast<ArgsKind>(9)}), caps), Reject::Args);
  Caps noFirstInstance = caps;
  noFirstInstance.indirectFirstInstance = false;
  const Job draw = job(Op::Compact, 100, {.kind = ArgsKind::Draw, .record = {0, 1, 0, 7, 0}});
  const Job indexed = job(Op::Compact, 100, {.kind = ArgsKind::DrawIndexed, .record = {0, 1, 0, 0, 7}});
  EXPECT_EQ(validate(draw, noFirstInstance), Reject::Args);
  EXPECT_EQ(validate(indexed, noFirstInstance), Reject::Args);
  EXPECT_EQ(validate(draw, caps), Reject::Usage);
  EXPECT_EQ(validate(indexed, caps), Reject::Usage);
}

class CompactNull : public CompactFixture, public ::testing::WithParamInterface<Config> {
protected:
  void SetUp() override {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    m_fill = false;
    if (!open(GetParam(), wgpu::BackendType::Null)) {
      GTEST_SKIP() << "no null device: " << m_error;
    }
    ASSERT_EQ(shaders::wait(pipeline(), WaitNs), shaders::PipelineStatus::Ready);
  }

  Bound scan(uint32_t capacity) const { return bind({.op = Op::Scan, .capacity = capacity}); }

  Bound compact(uint32_t capacity, ArgsSpec args = {}) const {
    return bind({.op = Op::Compact, .capacity = capacity, .keepMask = 1, .args = args});
  }
};

TEST_P(CompactNull, BecomesReady) {
  const shaders::PipelineId id = pipeline();
  EXPECT_TRUE(id);
  EXPECT_NE(id, shaders::NeverReady);
  EXPECT_TRUE(ready());
  EXPECT_EQ(pipeline(), id);
  EXPECT_TRUE(shaders::compute_pipeline(id));
}

TEST_P(CompactNull, EveryShapeEncodesCleanly) {
  for (const uint32_t capacity : {1000u, 5000u, Level3Size}) {
    for (const Op op : {Op::Scan, Op::Compact}) {
      for (const bool live : {false, true}) {
        for (const ArgsKind kind : {ArgsKind::None, ArgsKind::Dispatch, ArgsKind::Draw, ArgsKind::DrawIndexed}) {
          SCOPED_TRACE(::testing::Message() << capacity << " op " << static_cast<int>(op) << " live " << live
                                            << " args " << static_cast<int>(kind));
          const ArgsSpec args{
              .kind = kind,
              .field = 1,
              .scale = kind == ArgsKind::Dispatch ? 64u : 1u,
              .limit = op == Op::Scan ? capacity : 0,
              .record = {3, 0, 0, 0, 0},
          };
          const Bound bound = bind({
              .op = op,
              .capacity = capacity,
              .live = live ? std::optional<uint32_t>{capacity / 2} : std::nullopt,
              .countWord = live ? 3u : 0u,
              .keepMask = op == Op::Compact ? 1u : UINT32_MAX,
              .args = args,
          });
          const Stats before = stats();
          push_scope();
          EXPECT_TRUE(run(bound.job));
          EXPECT_EQ(pop_scope(), std::nullopt);
          const Stats after = stats();
          EXPECT_EQ(after.dispatches - before.dispatches, bound.job.plan.dispatches + (kind != ArgsKind::None));
          EXPECT_EQ(after.passes - before.passes, 1u);
          EXPECT_EQ(after.jobs - before.jobs, 1u);
          EXPECT_EQ(after.rejected, before.rejected);
        }
      }
    }
  }
}

TEST_P(CompactNull, BatchSharesOnePass) {
  const std::array bound{
      scan(1000), compact(5000),
      bind({.op = Op::Scan, .capacity = Level3Size, .args = {.kind = ArgsKind::Dispatch, .scale = 64, .limit = 4096}})};
  const std::vector<Job> jobs = jobs_of(bound);
  const Stats before = stats();
  push_scope();
  EXPECT_TRUE(run(jobs));
  EXPECT_EQ(pop_scope(), std::nullopt);
  const Stats after = stats();
  EXPECT_EQ(after.passes - before.passes, 1u);
  EXPECT_EQ(after.jobs - before.jobs, 3u);
  EXPECT_EQ(after.dispatches - before.dispatches, 1u + 3u + 5u + 1u);
}

TEST_P(CompactNull, InPassEncodeUsesCallerPass) {
  const std::array bound{scan(1000), compact(Level2Size)};
  const std::vector<Job> jobs = jobs_of(bound);
  const Stats before = stats();
  push_scope();
  const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
  const wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
  EXPECT_TRUE(encode(pass, jobs));
  pass.End();
  const wgpu::CommandBuffer commands = encoder.Finish();
  EXPECT_TRUE(aurora::test::submit_and_wait(*m_device, commands));
  EXPECT_EQ(pop_scope(), std::nullopt);
  const Stats after = stats();
  EXPECT_EQ(after.passes, before.passes);
  EXPECT_EQ(after.dispatches - before.dispatches, 1u + 3u);
  EXPECT_EQ(after.jobs - before.jobs, 2u);
}

TEST_P(CompactNull, RejectsMisuse) {
  struct Case {
    std::string_view name;
    Reject expected;
    Job job;
  };
  std::vector<Case> cases;
  const auto add = [&](std::string_view name, Reject expected, const Job& job) {
    cases.push_back({name, expected, job});
  };
  const Bound base = compact(5000, {.kind = ArgsKind::Draw, .field = 1, .record = {3, 0, 0, 0}});
  const uint64_t bytes = 5000 * sizeof(uint32_t);
  const uint64_t alignment = std::max<uint64_t>(caps().limits.minStorageBufferOffsetAlignment, 4);
  const wgpu::Buffer wide = make_buffer(bytes + 2 * alignment);
  const wgpu::BufferDescriptor uniformDescriptor{.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
                                                 .size = bytes};
  const wgpu::Buffer uniform = m_device->device.CreateBuffer(&uniformDescriptor);

  Job job = base.job;
  job.output = job.input;
  add("overlap", Reject::Overlap, job);
  job = base.job;
  job.input = {.buffer = wide, .offset = 4, .size = bytes};
  add("misaligned offset", Reject::Alignment, job);
  job = base.job;
  job.input = {.buffer = wide, .offset = 0, .size = bytes + 2};
  add("unaligned size", Reject::Alignment, job);
  job = base.job;
  job.input = {.buffer = wide, .offset = alignment * 2, .size = bytes + alignment};
  add("past the buffer", Reject::Range, job);
  job = base.job;
  job.input = {.buffer = uniform, .size = bytes};
  add("no storage usage", Reject::Usage, job);
  job = base.job;
  job.output.size = bytes - 4;
  add("output too small", Reject::Size, job);
  job = base.job;
  job.scratch.size -= 4;
  add("scratch too small", Reject::Size, job);
  job = base.job;
  job.plan.levels = 1;
  job.plan.dispatches = 1;
  add("hand-built plan", Reject::Plan, job);
  job = base.job;
  job.argsTarget = {};
  add("args without target", Reject::Usage, job);
  job = base.job;
  job.argsTarget.size -= 4;
  add("args target too small", Reject::Size, job);
  job = base.job;
  job.plan = make_plan(Op::Scan, 5000, caps());
  job.keepMask = UINT32_MAX;
  job.args.limit = 0;
  add("scan args without a limit", Reject::Args, job);
  job = base.job;
  job.args.scale = 1u << 20;
  add("draw count overflow", Reject::Args, job);
  job = base.job;
  job.args.field = 3;
  add("firstInstance field", Reject::Args, job);

  for (const Case& entry : cases) {
    SCOPED_TRACE(entry.name);
    EXPECT_EQ(validate(entry.job, caps()), entry.expected) << reject_name(validate(entry.job, caps()));
#ifndef NDEBUG
    testing::set_options({});
    EXPECT_DEATH(run(entry.job), "");
    testing::set_options({.releaseRejections = true});
#else
    const Stats before = stats();
    push_scope();
    EXPECT_FALSE(run(entry.job));
    EXPECT_EQ(pop_scope(), std::nullopt);
    EXPECT_EQ(stats().rejected - before.rejected, 1u);
    EXPECT_EQ(stats().jobs, before.jobs);
#endif
  }

  const Caps saved = caps();
  Caps noFirstInstance = saved;
  noFirstInstance.indirectFirstInstance = false;
  webgpu::set_caps(noFirstInstance);
  job = base.job;
  job.args.record[3] = 1;
  EXPECT_EQ(validate(job, caps()), Reject::Args);
#ifdef NDEBUG
  {
    const Stats before = stats();
    push_scope();
    EXPECT_FALSE(run(job));
    EXPECT_EQ(pop_scope(), std::nullopt);
    EXPECT_EQ(stats().rejected - before.rejected, 1u);
  }
#endif
  webgpu::set_caps(saved);

  testing::force_pipeline(shaders::NeverReady);
#ifndef NDEBUG
  testing::set_options({});
  EXPECT_DEATH(run(base.job), "");
  testing::set_options({.releaseRejections = true});
#else
  const Stats before = stats();
  push_scope();
  EXPECT_FALSE(run(base.job));
  EXPECT_EQ(pop_scope(), std::nullopt);
  EXPECT_EQ(stats().rejected - before.rejected, 1u);
#endif
  testing::force_pipeline({});
}

TEST_P(CompactNull, D3D11RejectsSharedBuffers) {
  const Bound base = compact(5000);
  const uint64_t bytes = 5000 * sizeof(uint32_t);
  const uint64_t alignment = std::max<uint64_t>(caps().limits.minStorageBufferOffsetAlignment, 256);
  const uint64_t stride = (bytes + alignment - 1) / alignment * alignment;
  const wgpu::Buffer shared = make_buffer(2 * stride);
  Job job = base.job;
  job.input = {.buffer = shared, .size = bytes};
  job.output = {.buffer = shared, .offset = stride, .size = bytes};
  EXPECT_EQ(validate(job, caps()), Reject::None);
  const Caps saved = caps();
  Caps d3d11 = saved;
  d3d11.backend = BACKEND_D3D11;
  webgpu::set_caps(d3d11);
  EXPECT_EQ(validate(job, caps()), Reject::Overlap);
  EXPECT_EQ(validate(base.job, caps()), Reject::None);
#ifdef NDEBUG
  const Stats before = stats();
  push_scope();
  EXPECT_FALSE(run(job));
  EXPECT_EQ(pop_scope(), std::nullopt);
  EXPECT_EQ(stats().rejected - before.rejected, 1u);
#endif
  webgpu::set_caps(saved);
}

TEST_P(CompactNull, RejectedJobBackstops) {
  Bound rejected =
      bind({.op = Op::Compact, .capacity = 5000, .keepMask = 1, .args = {.kind = ArgsKind::DrawIndexed, .field = 2}});
  const std::array bound{scan(1000), rejected, compact(Level2Size)};
  const std::vector<Job> jobs = jobs_of(bound);
  const Stats before = stats();
  push_scope();
  EXPECT_FALSE(run(jobs));
  EXPECT_EQ(pop_scope(), std::nullopt);
  const Stats after = stats();
  EXPECT_EQ(after.passes - before.passes, 1u);
  EXPECT_EQ(after.jobs - before.jobs, 2u);
  EXPECT_EQ(after.rejected - before.rejected, 1u);
  EXPECT_EQ(after.backstops - before.backstops, 1u);
  EXPECT_EQ(after.dispatches - before.dispatches, 1u + 1u + 3u);
}

TEST_P(CompactNull, ClearFallbackWhenNotReady) {
  const Bound bound = compact(5000, {.kind = ArgsKind::Dispatch, .scale = 64});
  testing::force_pipeline(shaders::NeverReady);
  EXPECT_EQ(pipeline(), shaders::NeverReady);
  EXPECT_FALSE(ready());
  const Stats before = stats();
  push_scope();
  EXPECT_FALSE(run(bound.job));
  EXPECT_EQ(pop_scope(), std::nullopt);
  const Stats after = stats();
  EXPECT_EQ(after.passes, before.passes);
  EXPECT_EQ(after.dispatches, before.dispatches);
  EXPECT_EQ(after.rejected - before.rejected, 1u);
  testing::force_pipeline({});
  EXPECT_TRUE(ready());
}

TEST_P(CompactNull, NullEncoderIsQuiet) {
  const Bound bound = scan(1000);
  const Stats before = stats();
  EXPECT_FALSE(encode(wgpu::CommandEncoder{}, bound.job, "compact test"));
  EXPECT_FALSE(encode(wgpu::ComputePassEncoder{}, std::span{&bound.job, 1}));
  const Stats after = stats();
  EXPECT_EQ(after.jobs, before.jobs);
  EXPECT_EQ(after.passes, before.passes);
  EXPECT_EQ(after.dispatches, before.dispatches);
  EXPECT_EQ(after.rejected, before.rejected);
  EXPECT_EQ(after.backstops, before.backstops);
}

TEST_P(CompactNull, UnsupportedGivesNeverReady) {
  const Caps saved = caps();
  Caps reduced = saved;
  reduced.limits.maxImmediateSize = 48;
  shutdown();
  webgpu::set_caps(reduced);
  initialize();
  testing::set_options({.releaseRejections = true});
  EXPECT_EQ(pipeline(), shaders::NeverReady);
  EXPECT_FALSE(ready());
  shutdown();
  webgpu::set_caps(saved);
  initialize();
  testing::set_options({.releaseRejections = true});
  EXPECT_NE(pipeline(), shaders::NeverReady);
  EXPECT_EQ(shaders::wait(pipeline(), WaitNs), shaders::PipelineStatus::Ready);
}

TEST_P(CompactNull, ShaderSessionCycleRerequests) {
  const Bound bound = scan(1000);
  EXPECT_TRUE(run(bound.job));
  const uint64_t session = shaders::session();
  shaders::shutdown();
  EXPECT_GT(shaders::session(), session);
  shaders::initialize();
  EXPECT_GT(shaders::session(), session);
  const shaders::PipelineId again = pipeline();
  EXPECT_NE(again, shaders::NeverReady);
  EXPECT_EQ(shaders::wait(again, WaitNs), shaders::PipelineStatus::Ready);
  push_scope();
  EXPECT_TRUE(run(bound.job));
  EXPECT_EQ(pop_scope(), std::nullopt);
}

TEST_P(CompactNull, DeviceChangeGivesNeverReady) {
  const Bound bound = scan(1000);
  const Caps saved = caps();
  std::string error;
  const std::unique_ptr<TestDevice> other = aurora::test::create_test_device(m_options, error);
  ASSERT_TRUE(other) << error;
  {
    const aurora::test::ScopedGlobalDevice swap{*other, m_options};
    EXPECT_EQ(pipeline(), shaders::NeverReady);
    EXPECT_FALSE(ready());
    const Stats before = stats();
    const wgpu::CommandEncoder encoder = other->device.CreateCommandEncoder();
    EXPECT_FALSE(encode(encoder, bound.job, "compact test"));
    const Stats after = stats();
    EXPECT_EQ(after.rejected - before.rejected, 1u);
    EXPECT_EQ(after.backstops, before.backstops);
    EXPECT_EQ(after.passes, before.passes);
    EXPECT_FALSE(make_binding(bound.job).valid());
  }
  webgpu::set_caps(saved);
  EXPECT_TRUE(aurora::test::take_uncaptured_errors(*other).empty());
  EXPECT_NE(pipeline(), shaders::NeverReady);
  EXPECT_TRUE(run(bound.job));
}

TEST_P(CompactNull, SteadyEncodeReusesBindGroup) {
  const Bound bound = compact(5000, {.kind = ArgsKind::Draw, .field = 1, .record = {3, 0, 0, 0}});
  const Stats before = stats();
  worker_frame_begin(1);
  EXPECT_TRUE(run(bound.job));
  worker_frame_begin(2);
  EXPECT_TRUE(run(bound.job));
  EXPECT_TRUE(run(bound.job));
  EXPECT_EQ(stats().bindGroupsCreated - before.bindGroupsCreated, 1u);
}

TEST_P(CompactNull, ManyJobsStayCached) {
  std::vector<Bound> bound;
  for (uint32_t i = 0; i < 100; ++i) {
    bound.push_back(scan(256));
  }
  const std::vector<Job> jobs = jobs_of(bound);
  const Stats before = stats();
  for (uint64_t frame = 1; frame <= 3; ++frame) {
    worker_frame_begin(frame);
    if (frame == 2) {
      EXPECT_EQ(stats().bindGroupsLastFrame, 100u);
    }
    if (frame == 3) {
      EXPECT_EQ(stats().bindGroupsLastFrame, 0u);
    }
    EXPECT_TRUE(run(jobs));
  }
  EXPECT_EQ(stats().bindGroupsCreated - before.bindGroupsCreated, 100u);
  worker_frame_begin(7);
  EXPECT_TRUE(run(jobs));
  EXPECT_EQ(stats().bindGroupsCreated - before.bindGroupsCreated, 100u) << "four idle frames keep the groups";
  worker_frame_begin(12);
  EXPECT_TRUE(run(jobs));
  EXPECT_EQ(stats().bindGroupsCreated - before.bindGroupsCreated, 200u) << "five idle frames evict them";
  std::vector<Bound> more;
  for (uint32_t i = 0; i < MaxCachedBindings + 10; ++i) {
    more.push_back(scan(256));
  }
  const std::vector<Job> manyJobs = jobs_of(more);
  push_scope();
  EXPECT_TRUE(run(manyJobs));
  EXPECT_EQ(pop_scope(), std::nullopt);
}

TEST_P(CompactNull, CallerBindingSkipsCache) {
  const Bound bound = compact(5000, {.kind = ArgsKind::Draw, .field = 1, .record = {3, 0, 0, 0}});
  const Binding binding = make_binding(bound.job);
  ASSERT_TRUE(binding.valid());
  Job job = bound.job;
  job.binding = &binding;
  const Stats before = stats();
  push_scope();
  EXPECT_TRUE(run(job));
  EXPECT_EQ(pop_scope(), std::nullopt);
  EXPECT_EQ(stats().bindGroupsCreated, before.bindGroupsCreated);
  EXPECT_EQ(stats().jobs - before.jobs, 1u);

  Job overlapping = bound.job;
  overlapping.output = overlapping.input;
  EXPECT_FALSE(make_binding(overlapping).valid());
#ifdef NDEBUG
  const Bound other = scan(1000);
  Job mismatched = other.job;
  mismatched.binding = &binding;
  push_scope();
  EXPECT_TRUE(run(mismatched));
  EXPECT_EQ(pop_scope(), std::nullopt);
  EXPECT_EQ(stats().bindGroupsCreated - before.bindGroupsCreated, 1u);
#endif
}

INSTANTIATE_TEST_SUITE_P(Configs, CompactNull, ::testing::ValuesIn(NullConfigs), config_name);

class CompactGpu : public CompactFixture, public ::testing::WithParamInterface<Config> {
protected:
  void SetUp() override {
    const wgpu::BackendType backend =
        aurora::test::parse_test_backend(std::getenv("AURORA_TEST_BACKEND")).value_or(wgpu::BackendType::Undefined);
    if (!open(GetParam(), backend)) {
      GTEST_SKIP() << "no adapter: " << m_error;
    }
    const wgpu::AdapterInfo& info = m_device->adapterInfo;
    if (info.backendType == wgpu::BackendType::Null) {
      GTEST_SKIP() << "the null backend runs no shaders";
    }
    const DeviceLimits& limits = caps().limits;
    RecordProperty("adapter", std::string{std::string_view{info.device}});
    RecordProperty("backend", std::string{backend_name(info.backendType)});
    RecordProperty("profile", std::string{webgpu::feature_profile_name(m_device->profile)});
    RecordProperty("core", m_device->core ? 1 : 0);
    RecordProperty("subgroupMinSize", static_cast<int>(info.subgroupMinSize));
    RecordProperty("subgroupMaxSize", static_cast<int>(info.subgroupMaxSize));
    RecordProperty("maxComputeInvocationsPerWorkgroup", static_cast<int>(limits.maxComputeInvocationsPerWorkgroup));
    RecordProperty("maxComputeWorkgroupSizeX", static_cast<int>(limits.maxComputeWorkgroupSizeX));
    RecordProperty("maxComputeWorkgroupStorageSize", static_cast<int>(limits.maxComputeWorkgroupStorageSize));
    RecordProperty("maxComputeWorkgroupsPerDimension", static_cast<int>(limits.maxComputeWorkgroupsPerDimension));
    RecordProperty("maxStorageBuffersPerShaderStage", static_cast<int>(limits.maxStorageBuffersPerShaderStage));
    RecordProperty("maxImmediateSize", static_cast<int>(limits.maxImmediateSize));
    const shaders::PipelineStatus status = shaders::wait(pipeline(), WaitNs);
    const bool clampable = info.backendType == wgpu::BackendType::D3D11 ||
                           info.backendType == wgpu::BackendType::OpenGL ||
                           info.backendType == wgpu::BackendType::OpenGLES;
    RecordProperty("clamped", status == shaders::PipelineStatus::Failed && clampable ? 1 : 0);
    if (status == shaders::PipelineStatus::Failed) {
      std::string failures;
      for (const shaders::PipelineFailure& failure : shaders::failures()) {
        failures += failure.label + ": " + failure.message + "\n";
      }
      if (clampable) {
        GTEST_SKIP() << "clamped: the scan kernel does not build here\n" << failures;
      }
      FAIL() << "the scan kernel failed to build\n" << failures;
    }
    ASSERT_EQ(status, shaders::PipelineStatus::Ready);
  }

  [[nodiscard]] std::vector<uint32_t> sizes() const {
    std::vector<uint32_t> out{1, 2, 127, 128, 129, 1023, 1024, 1025, 8191, 65553, 1048576, 1048577};
    if (cpu_adapter()) {
      std::erase(out, 1048576u);
    }
    return out;
  }
};

TEST_P(CompactGpu, ScanMatchesCpu) {
  for (const uint32_t size : sizes()) {
    SCOPED_TRACE(size);
    std::vector<Bound> bound;
    for (const Pattern pattern : AllPatterns) {
      bound.push_back(bind({.op = Op::Scan, .capacity = size, .input = make_input(pattern, size)}));
    }
    ASSERT_TRUE(run(jobs_of(bound)));
    for (size_t i = 0; i < bound.size(); ++i) {
      SCOPED_TRACE(pattern_name(AllPatterns[i]));
      expect_result(bound[i]);
    }
  }
}

TEST_P(CompactGpu, CompactMatchesCpu) {
  for (const uint32_t size : sizes()) {
    SCOPED_TRACE(size);
    std::vector<Bound> bound;
    std::vector<std::string> names;
    for (const Pattern pattern : AllPatterns) {
      const std::vector<uint32_t> input = make_input(pattern, size);
      for (const uint32_t mask : {1u, 0x80000000u}) {
        bound.push_back(bind({.op = Op::Compact, .capacity = size, .input = input, .keepMask = mask}));
        names.push_back(std::string{pattern_name(pattern)} + (mask == 1 ? ", bit 0" : ", bit 31"));
      }
    }
    ASSERT_TRUE(run(jobs_of(bound)));
    for (size_t i = 0; i < bound.size(); ++i) {
      SCOPED_TRACE(names[i]);
      expect_result(bound[i]);
    }
  }
}

TEST_P(CompactGpu, LiveCountClamps) {
  for (const uint32_t size : {1000u, Level2Size, Level3Size}) {
    const std::vector<uint32_t> input = make_input(Pattern::Random, size);
    std::vector<Bound> bound;
    std::vector<uint32_t> lives;
    for (const uint32_t live : {0u, 1u, size / 2, size, size + 5, UINT32_MAX}) {
      for (const Op op : {Op::Scan, Op::Compact}) {
        bound.push_back(bind({
            .op = op,
            .capacity = size,
            .input = input,
            .live = live,
            .countWord = 3,
            .keepMask = op == Op::Compact ? 1u : UINT32_MAX,
        }));
        lives.push_back(live);
      }
    }
    ASSERT_TRUE(run(jobs_of(bound)));
    for (size_t i = 0; i < bound.size(); ++i) {
      SCOPED_TRACE(::testing::Message() << size << " live " << lives[i] << " op "
                                        << static_cast<int>(bound[i].job.plan.op));
      expect_result(bound[i], lives[i]);
    }
  }
}

TEST_P(CompactGpu, ChainedLiveCount) {
  for (const uint32_t size : {1000u, Level2Size, Level3Size}) {
    SCOPED_TRACE(size);
    const Bound first =
        bind({.op = Op::Compact, .capacity = size, .input = make_input(Pattern::Random, size), .keepMask = 1});
    Bound second = bind({.op = Op::Scan, .capacity = size});
    second.job.input = first.job.output;
    second.job.count = first.job.scratch;
    second.job.countWord = TotalWord;
    const std::array jobs{first.job, second.job};
    ASSERT_TRUE(run(jobs));
    expect_result(first);
    const Reference list = cpu_compact(first.input, size, size, 1);
    second.input = list.output;
    second.input.resize(size, Poison);
    expect_scan(second, list.total);
  }
}

TEST_P(CompactGpu, ArgsFromTotal) {
  const uint32_t size = 5000;
  const std::vector<uint32_t> input = make_input(Pattern::Random, size);
  const uint32_t total = cpu_compact(input, size, size, 1).total;
  struct Case {
    Op op;
    ArgsSpec args;
    std::vector<uint32_t> expected;
  };
  const auto ceil = [](uint32_t value, uint32_t divisor) { return (value + divisor - 1) / divisor; };
  std::vector<Case> cases{
      {Op::Compact, {.kind = ArgsKind::Dispatch, .scale = 64}, {ceil(total, 64), 1, 1}},
      {Op::Compact, {.kind = ArgsKind::Dispatch, .scale = 64, .limit = 1000}, {ceil(1000, 64), 1, 1}},
      {Op::Compact, {.kind = ArgsKind::Dispatch, .scale = 1, .limit = 9000}, {total, 1, 1}},
      {Op::Compact, {.kind = ArgsKind::Draw, .field = 1, .record = {3, 0, 0, 0}}, {3, total, 0, 0}},
      {Op::Compact,
       {.kind = ArgsKind::Draw, .field = 0, .scale = 3, .limit = 100, .record = {0, 1, 9, 0}},
       {300, 1, 9, 0}},
      {Op::Compact,
       {.kind = ArgsKind::DrawIndexed, .offset = 8, .field = 0, .scale = 6, .record = {0, 1, 7, 0xFFFFFFF0u, 0}},
       {Poison, Poison, 6 * total, 1, 7, 0xFFFFFFF0u, 0}},
      {Op::Scan, {.kind = ArgsKind::Dispatch, .scale = 256, .limit = 4000}, {ceil(4000, 256), 1, 1}},
  };
  if (caps().indirectFirstInstance) {
    cases.push_back(
        {Op::Compact, {.kind = ArgsKind::DrawIndexed, .field = 1, .record = {36, 0, 0, 0, 5}}, {36, total, 0, 0, 5}});
  }
  std::vector<Bound> bound;
  for (const Case& entry : cases) {
    bound.push_back(bind({
        .op = entry.op,
        .capacity = size,
        .input = entry.op == Op::Scan ? make_input(Pattern::Ones, size) : input,
        .keepMask = entry.op == Op::Compact ? 1u : UINT32_MAX,
        .args = entry.args,
    }));
  }
  ASSERT_TRUE(run(jobs_of(bound)));
  for (size_t i = 0; i < bound.size(); ++i) {
    SCOPED_TRACE(i);
    const uint64_t words = bound[i].job.argsTarget.size / sizeof(uint32_t);
    const std::vector<uint32_t> got = read(bound[i].argsBuffer, 0, words * sizeof(uint32_t));
    EXPECT_EQ(got, cases[i].expected);
  }
}

TEST_P(CompactGpu, SharedBufferRanges) {
  const uint32_t size = 3000;
  const uint64_t alignment = std::max<uint64_t>(caps().limits.minStorageBufferOffsetAlignment, 256);
  const auto align = [alignment](uint64_t value) { return (value + alignment - 1) / alignment * alignment; };
  const bool oneViewPerBuffer = caps().backend == BACKEND_D3D11;
  RecordProperty("oneViewPerBuffer", oneViewPerBuffer ? 1 : 0);
  for (const Op op : {Op::Scan, Op::Compact}) {
    SCOPED_TRACE(static_cast<int>(op));
    const Plan plan = make_plan(op, size, caps());
    const ArgsSpec args{
        .kind = ArgsKind::DrawIndexed, .field = 1, .limit = op == Op::Scan ? 100000u : 0u, .record = {6, 0, 0, 0, 0}};
    const std::array<uint64_t, StorageBindings> sizes{size * 4ull, size * 4ull, plan.scratchBytes, 16, 20};
    std::array<uint64_t, StorageBindings> offsets{};
    uint64_t end = 0;
    for (size_t i = 0; i < sizes.size(); ++i) {
      offsets[i] = end;
      end = align(offsets[i] + sizes[i] + alignment);
    }
    const wgpu::Buffer shared = make_buffer(end, wgpu::BufferUsage::Indirect);
    fill(shared, 0, end, Guard);
    const std::vector<uint32_t> input = make_input(Pattern::Random, size);
    write(shared, offsets[0], input);
    fill(shared, offsets[1], sizes[1], Poison);
    fill(shared, offsets[2], sizes[2], Poison);
    fill(shared, offsets[4], sizes[4], Poison);
    const std::array live{0u, 0u, 2500u, 0u};
    write(shared, offsets[3], live);
    const Job job{
        .plan = plan,
        .input = {.buffer = shared, .offset = offsets[0], .size = sizes[0]},
        .output = {.buffer = shared, .offset = offsets[1], .size = sizes[1]},
        .scratch = {.buffer = shared, .offset = offsets[2], .size = sizes[2]},
        .count = {.buffer = shared, .offset = offsets[3], .size = sizes[3]},
        .countWord = 2,
        .keepMask = op == Op::Compact ? 1u : UINT32_MAX,
        .args = args,
        .argsTarget = {.buffer = shared, .offset = offsets[4], .size = sizes[4]},
    };
    ASSERT_EQ(validate(job, caps()), oneViewPerBuffer ? Reject::Overlap : Reject::None);
    EXPECT_EQ(run(job), !oneViewPerBuffer);
    const std::vector<uint32_t> words = read(shared, 0, end);
    ASSERT_EQ(words.size(), end / 4);
    for (size_t i = 0; i < sizes.size(); ++i) {
      const uint64_t from = (offsets[i] + sizes[i]) / 4;
      const uint64_t to = i + 1 < sizes.size() ? offsets[i + 1] / 4 : end / 4;
      EXPECT_EQ(count_not(std::span{words}.subspan(from, to - from), Guard), 0u) << "guard after view " << i;
    }
    const std::span<const uint32_t> output = std::span{words}.subspan(offsets[1] / 4, size);
    const std::span<const uint32_t> header = std::span{words}.subspan(offsets[2] / 4, HeaderWords);
    const std::span<const uint32_t> record = std::span{words}.subspan(offsets[4] / 4, 5);
    if (oneViewPerBuffer) {
      EXPECT_EQ(count_not(record, 0), 0u) << "the rejected record reads zero";
      EXPECT_EQ(count_not(output, Poison), 0u);
      EXPECT_EQ(count_not(header, Poison), 0u) << "a header sharing the record's buffer is left alone";
      continue;
    }
    if (op == Op::Scan) {
      const Reference ref = cpu_scan(input, size, 2500);
      expect_words(output, ref.output, "scan output");
      EXPECT_EQ(header[0], ref.total);
      EXPECT_EQ(record[1], std::min(ref.total, 100000u));
    } else {
      const Reference ref = cpu_compact(input, size, 2500, 1);
      expect_words(output.first(ref.total), ref.output, "compact list");
      EXPECT_EQ(count_not(output.subspan(ref.total), Poison), 0u);
      EXPECT_EQ(header[0], ref.total);
      EXPECT_EQ(record[1], ref.total);
    }
    EXPECT_EQ(header[1], 2500u);
    EXPECT_EQ(record[0], 6u);
  }
}

TEST_P(CompactGpu, KernelBoundsHoldUndersizedViews) {
  testing::set_options({.releaseRejections = true, .skipSizeChecks = true});
  const uint64_t alignment = std::max<uint64_t>(caps().limits.minStorageBufferOffsetAlignment, 256);
  const auto align = [alignment](uint64_t value) { return (value + alignment - 1) / alignment * alignment; };
  for (const uint32_t size : {1000u, Level2Size, Level3Size}) {
    for (const Op op : {Op::Scan, Op::Compact}) {
      SCOPED_TRACE(::testing::Message() << size << " op " << static_cast<int>(op));
      const Plan plan = make_plan(op, size, caps());
      const uint32_t half = size / 2;
      const std::array<uint64_t, StorageBindings> sizes{half * 4ull, half * 4ull, 32, 4, 8};
      std::array<wgpu::Buffer, StorageBindings> buffers;
      std::array<BufferView, StorageBindings> views;
      for (size_t i = 0; i < sizes.size(); ++i) {
        buffers[i] = make_buffer(align(sizes[i]) + 2 * alignment, wgpu::BufferUsage::Indirect);
        fill(buffers[i], 0, align(sizes[i]) + 2 * alignment, Guard);
        views[i] = {.buffer = buffers[i], .offset = alignment, .size = sizes[i]};
      }
      write(buffers[0], alignment, make_input(Pattern::Ones, half));
      const Job job{
          .plan = plan,
          .input = views[0],
          .output = views[1],
          .scratch = views[2],
          .count = views[3],
          .countWord = 5,
          .keepMask = op == Op::Compact ? 1u : UINT32_MAX,
          .args = {.kind = ArgsKind::DrawIndexed, .field = 0, .limit = op == Op::Scan ? 10u : 0u},
          .argsTarget = views[4],
      };
      ASSERT_EQ(validate(job, caps()), Reject::None);
      ASSERT_TRUE(run(job));
      for (size_t i = 0; i < sizes.size(); ++i) {
        const std::vector<uint32_t> words = read(buffers[i], 0, align(sizes[i]) + 2 * alignment);
        ASSERT_FALSE(words.empty());
        const std::span<const uint32_t> all{words};
        EXPECT_EQ(count_not(all.first(alignment / 4), Guard), 0u) << "guard before view " << i;
        EXPECT_EQ(count_not(all.subspan((alignment + sizes[i]) / 4), Guard), 0u) << "guard after view " << i;
        if (i == 1 && op == Op::Scan) {
          std::vector<uint32_t> expected(std::min(half, TileSize));
          std::iota(expected.begin(), expected.end(), 0u);
          expect_words(all.subspan(alignment / 4, expected.size()), expected, "first tile of the scan");
        }
      }
    }
  }
}

TEST_P(CompactGpu, RejectedJobZeroesOutputs) {
  const uint32_t size = Level2Size;
  const std::vector<uint32_t> input = make_input(Pattern::Random, size);
  const Bound before = bind({.op = Op::Compact,
                             .capacity = size,
                             .input = input,
                             .keepMask = 1,
                             .args = {.kind = ArgsKind::Draw, .field = 1, .record = {3, 0, 0, 0}}});
  const Bound rejected = bind({.op = Op::Compact,
                               .capacity = size,
                               .input = input,
                               .keepMask = 1,
                               .args = {.kind = ArgsKind::DrawIndexed, .field = 2, .record = {3, 1, 2, 3, 0}}});
  Bound unbindable = bind({.op = Op::Compact,
                           .capacity = size,
                           .input = input,
                           .keepMask = 1,
                           .args = {.kind = ArgsKind::Draw, .offset = 4, .field = 1}});
  unbindable.job.args.offset = 2;
  const Bound after = bind({.op = Op::Scan, .capacity = size, .input = input});
  const std::array jobs{before.job, rejected.job, unbindable.job, after.job};
  EXPECT_FALSE(run(jobs));
  expect_result(before);
  expect_result(after);
  EXPECT_EQ(count_not(read(rejected.argsBuffer, 0, 20), 0), 0u) << "the rejected record reads zero";
  EXPECT_EQ(count_not(read(unbindable.argsBuffer, 0, 20), Poison), 0u) << "an unbindable record is left alone";
  for (const Bound* bound : std::array<const Bound*, 2>{&rejected, &unbindable}) {
    const std::vector<uint32_t> header = read(bound->scratchBuffer, 0, HeaderWords * sizeof(uint32_t));
    ASSERT_EQ(header.size(), HeaderWords);
    EXPECT_EQ(header[TotalWord], 0u);
    EXPECT_EQ(header[LiveWord], 0u);
    EXPECT_EQ(count_not(std::span{header}.subspan(2), Poison), 0u);
    EXPECT_EQ(count_not(read(bound->outputBuffer, 0, uint64_t{size} * 4), Poison), 0u);
  }
}

TEST_P(CompactGpu, ClearFallbackZeroes) {
  const Bound bound = bind({.op = Op::Compact,
                            .capacity = 5000,
                            .input = make_input(Pattern::Ones, 5000),
                            .keepMask = 1,
                            .args = {.kind = ArgsKind::DrawIndexed, .offset = 4, .field = 1}});
  testing::force_pipeline(shaders::NeverReady);
  EXPECT_FALSE(run(bound.job));
  testing::force_pipeline({});
  const std::vector<uint32_t> args = read(bound.argsBuffer, 0, 24);
  EXPECT_EQ(args, (std::vector<uint32_t>{Poison, 0, 0, 0, 0, 0}));
  const std::vector<uint32_t> header = read(bound.scratchBuffer, 0, HeaderWords * sizeof(uint32_t));
  ASSERT_EQ(header.size(), HeaderWords);
  EXPECT_EQ(header[TotalWord], 0u);
  EXPECT_EQ(header[LiveWord], 0u);
  EXPECT_EQ(count_not(std::span{header}.subspan(2), Poison), 0u);
  EXPECT_EQ(count_not(read(bound.outputBuffer, 0, 5000 * 4), Poison), 0u);
}

TEST_P(CompactGpu, InPassMatchesCpu) {
  const std::vector<uint32_t> input = make_input(Pattern::Random, Level2Size);
  const std::array bound{
      bind({.op = Op::Scan, .capacity = Level2Size, .input = input}),
      bind({.op = Op::Compact, .capacity = Level2Size, .input = input, .keepMask = 0x80000000u}),
      bind({.op = Op::Compact, .capacity = 700, .input = input, .keepMask = 1}),
  };
  const std::vector<Job> jobs = jobs_of(bound);
  const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
  const wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
  EXPECT_TRUE(encode(pass, jobs));
  pass.End();
  const wgpu::CommandBuffer commands = encoder.Finish();
  ASSERT_TRUE(aurora::test::submit_and_wait(*m_device, commands));
  for (const Bound& entry : bound) {
    expect_result(entry);
  }
}

TEST_P(CompactGpu, RepeatRunsIdentical) {
  const uint32_t size = cpu_adapter() ? Level3Size : 1u << 20;
  const std::vector<uint32_t> input = make_input(Pattern::Random, size, Seed + 7);
  for (const Op op : {Op::Scan, Op::Compact}) {
    SCOPED_TRACE(static_cast<int>(op));
    const Bound bound =
        bind({.op = op, .capacity = size, .input = input, .keepMask = op == Op::Compact ? 1u : UINT32_MAX});
    std::vector<uint32_t> first;
    for (int round = 0; round < 3; ++round) {
      fill(bound.outputBuffer, 0, uint64_t{size} * 4, Poison);
      ASSERT_TRUE(run(bound.job));
      std::vector<uint32_t> words = read(bound.outputBuffer, 0, uint64_t{size} * 4);
      ASSERT_EQ(words.size(), size);
      if (round == 0) {
        expect_result(bound);
        first = std::move(words);
      } else {
        expect_words(words, first, "repeat run");
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Configs, CompactGpu, ::testing::ValuesIn(GpuConfigs), config_name);

class DISABLED_CompactGpuTiming : public CompactFixture {
protected:
  void SetUp() override {
    const wgpu::BackendType backend =
        aurora::test::parse_test_backend(std::getenv("AURORA_TEST_BACKEND")).value_or(wgpu::BackendType::Undefined);
    if (!open(GpuConfigs[0], backend)) {
      GTEST_SKIP() << "no adapter: " << m_error;
    }
    if (m_device->adapterInfo.backendType == wgpu::BackendType::Null) {
      GTEST_SKIP() << "the null backend runs no shaders";
    }
    if (shaders::wait(pipeline(), WaitNs) != shaders::PipelineStatus::Ready) {
      GTEST_SKIP() << "the scan kernel is not available here";
    }
    RecordProperty("adapter", std::string{std::string_view{m_device->adapterInfo.device}});
    RecordProperty("backend", std::string{backend_name(m_device->adapterInfo.backendType)});
  }

  double gpu_us(std::span<const Job> jobs, int rounds) const {
    const bool timestamps = m_device->device.HasFeature(wgpu::FeatureName::TimestampQuery);
    double best = 0;
    for (int round = 0; round < rounds; ++round) {
      const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
      wgpu::QuerySet queries;
      wgpu::Buffer resolved;
      wgpu::PassTimestampWrites writes;
      wgpu::ComputePassDescriptor descriptor;
      if (timestamps) {
        const wgpu::QuerySetDescriptor queryDescriptor{.type = wgpu::QueryType::Timestamp, .count = 2};
        queries = m_device->device.CreateQuerySet(&queryDescriptor);
        const wgpu::BufferDescriptor resolveDescriptor{
            .usage = wgpu::BufferUsage::QueryResolve | wgpu::BufferUsage::CopySrc, .size = 16};
        resolved = m_device->device.CreateBuffer(&resolveDescriptor);
        writes = {.querySet = queries, .beginningOfPassWriteIndex = 0, .endOfPassWriteIndex = 1};
        descriptor.timestampWrites = &writes;
      }
      const wgpu::ComputePassEncoder pass = encoder.BeginComputePass(&descriptor);
      encode(pass, jobs);
      pass.End();
      if (timestamps) {
        encoder.ResolveQuerySet(queries, 0, 2, resolved, 0);
      }
      const wgpu::CommandBuffer commands = encoder.Finish();
      const auto start = std::chrono::steady_clock::now();
      EXPECT_TRUE(aurora::test::submit_and_wait(*m_device, commands));
      double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
      if (timestamps) {
        const std::vector<uint32_t> ticks = read(resolved, 0, 16);
        if (ticks.size() == 4) {
          const uint64_t begin = ticks[0] | uint64_t{ticks[1]} << 32;
          const uint64_t endTick = ticks[2] | uint64_t{ticks[3]} << 32;
          us = static_cast<double>(endTick - begin) / 1000.0;
        }
      }
      best = round == 0 ? us : std::min(best, us);
    }
    return best;
  }

  double encode_us(std::span<const Job> jobs) const {
    const wgpu::CommandEncoder encoder = m_device->device.CreateCommandEncoder();
    const auto start = std::chrono::steady_clock::now();
    encode(encoder, jobs, "compact timing");
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
    const wgpu::CommandBuffer commands = encoder.Finish();
    EXPECT_TRUE(aurora::test::submit_and_wait(*m_device, commands));
    return us;
  }
};

TEST_F(DISABLED_CompactGpuTiming, Measure) {
  std::string line = "compact-perf:";
  for (const uint32_t size : {262144u, 1048576u, 4194304u}) {
    for (const Op op : {Op::Scan, Op::Compact}) {
      const Bound bound = bind({.op = op,
                                .capacity = size,
                                .input = make_input(Pattern::Random, size),
                                .keepMask = op == Op::Compact ? 1u : UINT32_MAX});
      const double us = gpu_us(std::span{&bound.job, 1}, 20);
      const std::string key = std::string{op == Op::Scan ? "scan" : "compact"} + std::to_string(size) + "Us";
      RecordProperty(key, std::to_string(us));
      line += " " + key + "=" + std::to_string(us);
    }
  }
  const std::vector<uint32_t> input = make_input(Pattern::Random, 262144);
  std::vector<Bound> buckets;
  for (uint32_t bit = 0; bit < 8; ++bit) {
    buckets.push_back(bind({.op = Op::Compact, .capacity = 262144, .input = input, .keepMask = 1u << bit}));
  }
  const std::vector<Job> bucketJobs = jobs_of(buckets);
  const double batchUs = gpu_us(bucketJobs, 20);
  RecordProperty("batch8x262144Us", std::to_string(batchUs));
  line += " batch8x262144Us=" + std::to_string(batchUs);

  const double coldUs = encode_us(bucketJobs);
  double warmUs = 0;
  for (int round = 0; round < 10; ++round) {
    const double us = encode_us(bucketJobs);
    warmUs = round == 0 ? us : std::min(warmUs, us);
  }
  std::vector<Binding> bindings;
  std::vector<Job> boundJobs = bucketJobs;
  for (const Job& job : bucketJobs) {
    bindings.push_back(make_binding(job));
  }
  for (size_t i = 0; i < boundJobs.size(); ++i) {
    boundJobs[i].binding = &bindings[i];
  }
  double bindingUs = 0;
  for (int round = 0; round < 10; ++round) {
    const double us = encode_us(boundJobs);
    bindingUs = round == 0 ? us : std::min(bindingUs, us);
  }
  RecordProperty("encodeColdUs", std::to_string(coldUs));
  RecordProperty("encodeWarmUs", std::to_string(warmUs));
  RecordProperty("encodeBindingUs", std::to_string(bindingUs));
  line += " encodeColdUs=" + std::to_string(coldUs) + " encodeWarmUs=" + std::to_string(warmUs) +
          " encodeBindingUs=" + std::to_string(bindingUs);
  std::printf("%s\n", line.c_str());
}
} // namespace
} // namespace aurora::render::compact
