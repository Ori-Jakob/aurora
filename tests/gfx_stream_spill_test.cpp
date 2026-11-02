#include <gtest/gtest.h>

#include "render_test_device.hpp"

#include "gfx/encoding.hpp"
#include "gfx/frame.hpp"
#include "gfx/frame_packet.hpp"
#include "gfx/recording.hpp"
#include "gfx/resources.hpp"
#include "gx/gx.hpp"
#include "internal.hpp"
#include "render/graph_bind.hpp"

#include <aurora/aurora.h>
#include <aurora/gfx.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace aurora::gfx {
namespace {
constexpr uint64_t MiB = 1024 * 1024;
constexpr uint64_t Capacity = MiB;
constexpr uint32_t CopyGroupSize = 256;
constexpr std::string_view OverrunWarning = "more than a staging spill can hold";
constexpr std::string_view UniformWarning = "Frame uniform data outgrew its 1 MiB staging";
constexpr std::string_view StorageWarning = "Frame storage data outgrew its 1 MiB staging";

constexpr const char* CopyWgsl = R"(
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x < arrayLength(&dst)) {
    dst[id.x] = src[id.x];
  }
}
)";

std::vector<uint8_t> pattern(size_t size, uint32_t seed) {
  std::vector<uint8_t> bytes(size);
  for (size_t i = 0; i < size; ++i) {
    bytes[i] = static_cast<uint8_t>((i * 2654435761u + seed) >> 13);
  }
  return bytes;
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

struct SeenSizes {
  uint64_t vertex = 0;
  uint64_t index = 0;
};

class FrameStreamSpill : public ::testing::Test {
protected:
  void SetUp() override {
    const test::TestDeviceOptions options{
        .backend = test::parse_test_backend(std::getenv("AURORA_TEST_BACKEND")).value_or(wgpu::BackendType::Undefined),
        .efbSize = {64, 64, 1},
        .createEfb = true,
    };
    m_device = test::create_test_device(options, m_error);
    if (!m_device) {
      GTEST_SKIP() << "no device: " << m_error;
    }
    m_globals = std::make_unique<test::ScopedGlobalDevice>(*m_device, options);
    m_null = m_device->adapterInfo.backendType == wgpu::BackendType::Null;
    g_warnings.clear();
    m_savedLog = std::exchange(g_config.logCallback, &capture_warning);
    detail::testing::reset_stream_warnings();
    // From zero, so the request shows what this frame alone asked for
    detail::testing::reset_streaming_request({.vertex = 0, .index = 0, .storage = 0, .uniform = 0});

    auto& res = detail::resources();
    m_savedVertex =
        std::exchange(res.vertexBuffer, buffer(Capacity, wgpu::BufferUsage::Storage | wgpu::BufferUsage::Vertex |
                                                             wgpu::BufferUsage::CopyDst));
    m_savedIndex =
        std::exchange(res.indexBuffer, buffer(Capacity, wgpu::BufferUsage::Index | wgpu::BufferUsage::Storage |
                                                            wgpu::BufferUsage::CopyDst));
    m_savedStorage =
        std::exchange(res.storageBuffer, buffer(Capacity, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst));
    m_savedUniform =
        std::exchange(res.uniformBuffer, buffer(Capacity, wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst));
    m_savedSizes = std::exchange(res.deviceSizes,
                                 {.vertex = Capacity, .index = Capacity, .storage = Capacity, .uniform = Capacity});
    m_savedStaticGroup = std::exchange(res.staticBindGroup, {});
    m_savedStaticLayout = std::exchange(res.staticBindGroupLayout, {});
    m_savedUniformGroup = std::exchange(res.uniformBindGroup, {});
    m_savedUniformLayout = std::exchange(res.uniformBindGroupLayout, {});

    // Mapped the way begin_frame maps a staging slot
    m_frame.sizes = res.deviceSizes;
    const detail::StagingLayout layout = detail::staging_layout(m_frame.sizes);
    const wgpu::BufferDescriptor stagingDescriptor{
        .usage = wgpu::BufferUsage::MapWrite | wgpu::BufferUsage::CopySrc,
        .size = layout.total,
        .mappedAtCreation = true,
    };
    m_staging = m_device->device.CreateBuffer(&stagingDescriptor);
    m_frame.verts = ByteBuffer{static_cast<uint8_t*>(m_staging.GetMappedRange(layout.vertex, Capacity)), Capacity};
    m_frame.uniforms = ByteBuffer{static_cast<uint8_t*>(m_staging.GetMappedRange(layout.uniform, Capacity)), Capacity};
    m_frame.indices = ByteBuffer{static_cast<uint8_t*>(m_staging.GetMappedRange(layout.index, Capacity)), Capacity};
    m_frame.storage = ByteBuffer{static_cast<uint8_t*>(m_staging.GetMappedRange(layout.storage, Capacity)), Capacity};
    detail::testing::set_staging_buffer(0, m_staging);
    m_frame.stagingBuffer = 0;

    const wgpu::ShaderSourceWGSL wgsl{wgpu::ShaderSourceWGSL::Init{.code = CopyWgsl}};
    const wgpu::ShaderModuleDescriptor moduleDescriptor{.nextInChain = &wgsl};
    const wgpu::ShaderModule module = m_device->device.CreateShaderModule(&moduleDescriptor);
    const wgpu::ComputePipelineDescriptor pipelineDescriptor{.compute = {.module = module, .entryPoint = "main"}};
    m_copy = m_device->device.CreateComputePipeline(&pipelineDescriptor);
    m_task = register_encoder_task_type({.label = "test.stream_copy", .callback = &copy_task, .userdata = this});
    m_draw = register_draw_type({.label = "test.counted_draw", .draw = &counted_draw, .userdata = this});

    detail::testing::suppress_render_worker(true);
    detail::begin_recording(m_frame, 0);
    m_recording = true;
    m_started = true;
  }

  void TearDown() override {
    if (!m_started) {
      return;
    }
    if (m_recording) {
      finish();
      detail::end_recording();
    }
    detail::shutdown_recording();
    detail::testing::suppress_render_worker(false);
    unregister_draw_type(m_draw);
    unregister_encoder_task_type(m_task);
    m_frame = {};
    detail::testing::set_staging_buffer(0, {});
    auto& res = detail::resources();
    res.vertexBuffer = std::move(m_savedVertex);
    res.indexBuffer = std::move(m_savedIndex);
    res.storageBuffer = std::move(m_savedStorage);
    res.uniformBuffer = std::move(m_savedUniform);
    res.deviceSizes = m_savedSizes;
    res.staticBindGroup = std::move(m_savedStaticGroup);
    res.staticBindGroupLayout = std::move(m_savedStaticLayout);
    res.uniformBindGroup = std::move(m_savedUniformGroup);
    res.uniformBindGroupLayout = std::move(m_savedUniformLayout);
    detail::testing::reset_streaming_request();
    g_config.logCallback = m_savedLog;
    EXPECT_TRUE(test::take_uncaptured_errors(*m_device).empty());
    m_globals.reset();
    m_device.reset();
  }

  [[nodiscard]] wgpu::Buffer buffer(uint64_t size, wgpu::BufferUsage usage) const {
    const wgpu::BufferDescriptor descriptor{.usage = usage, .size = size};
    return m_device->device.CreateBuffer(&descriptor);
  }

  // Inside the task, the streaming buffers hold what a draw at this point of the frame would read
  static void copy_task(const EncoderTaskContext& ctx, const wgpu::CommandEncoder& cmd, const void* payload, size_t,
                        void* userdata) {
    auto& self = *static_cast<FrameStreamSpill*>(userdata);
    uint32_t slot = 0;
    std::memcpy(&slot, payload, sizeof(slot));
    self.m_seen[slot] = {.vertex = ctx.vertexBuffer.GetSize(), .index = ctx.indexBuffer.GetSize()};
    if (slot == 1 && self.m_vertexOut) {
      self.copy_out(cmd, ctx.vertexBuffer, self.m_vertexOut);
      self.copy_out(cmd, ctx.indexBuffer, self.m_indexOut);
    }
  }

  static void counted_draw(const DrawContext&, const wgpu::RenderPassEncoder&, const void*, size_t, void* userdata) {
    ++static_cast<FrameStreamSpill*>(userdata)->m_draws;
  }

  void copy_out(const wgpu::CommandEncoder& cmd, const wgpu::Buffer& src, const wgpu::Buffer& dst) const {
    const std::array entries{
        wgpu::BindGroupEntry{.binding = 0, .buffer = src},
        wgpu::BindGroupEntry{.binding = 1, .buffer = dst},
    };
    const wgpu::BindGroupDescriptor descriptor{
        .layout = m_copy.GetBindGroupLayout(0),
        .entryCount = entries.size(),
        .entries = entries.data(),
    };
    const wgpu::BindGroup group = m_device->device.CreateBindGroup(&descriptor);
    const wgpu::ComputePassEncoder pass = cmd.BeginComputePass();
    pass.SetPipeline(m_copy);
    pass.SetBindGroup(0, group);
    pass.DispatchWorkgroups(static_cast<uint32_t>((dst.GetSize() / 4 + CopyGroupSize - 1) / CopyGroupSize));
    pass.End();
  }

  void push_task(uint32_t slot) { ASSERT_TRUE(push_encoder_task(m_task, &slot, sizeof(slot))); }

  // Encodes the ops the way the worker does, skipping passes that would need gx's bind groups
  void encode_and_submit(bool (*wanted)(const detail::FrameOp&)) {
    finish();
    detail::end_recording();
    m_recording = false;
    wgpu::CommandEncoder cmd = m_device->device.CreateCommandEncoder();
    for (const detail::FrameOp& op : m_frame.ops) {
      if (wanted(op)) {
        detail::encode_op(cmd, m_frame, op);
      }
    }
    m_staging.Unmap();
    for (const detail::StagingSpill& spill : m_frame.spills) {
      spill.buffer.Unmap();
    }
    const wgpu::CommandBuffer commands = cmd.Finish();
    EXPECT_TRUE(test::submit_and_wait(*m_device, commands));
  }

  [[nodiscard]] std::vector<uint8_t> read(const wgpu::Buffer& buffer) const {
    const auto words = test::read_buffer(*m_device, buffer, 0, buffer.GetSize());
    if (!words) {
      return {};
    }
    std::vector<uint8_t> bytes(words->size() * sizeof(uint32_t));
    std::memcpy(bytes.data(), words->data(), bytes.size());
    return bytes;
  }

  std::unique_ptr<test::TestDevice> m_device;
  std::unique_ptr<test::ScopedGlobalDevice> m_globals;
  std::string m_error;
  bool m_started = false;
  bool m_recording = false;
  bool m_null = false;
  AuroraLogCallback m_savedLog = nullptr;
  wgpu::Buffer m_savedVertex;
  wgpu::Buffer m_savedIndex;
  wgpu::Buffer m_savedStorage;
  wgpu::Buffer m_savedUniform;
  detail::StreamingSizes m_savedSizes;
  wgpu::BindGroup m_savedStaticGroup;
  wgpu::BindGroupLayout m_savedStaticLayout;
  wgpu::BindGroup m_savedUniformGroup;
  wgpu::BindGroupLayout m_savedUniformLayout;
  wgpu::Buffer m_staging;
  wgpu::ComputePipeline m_copy;
  wgpu::Buffer m_vertexOut;
  wgpu::Buffer m_indexOut;
  EncoderTaskId m_task = InvalidEncoderTask;
  DrawTypeId m_draw = InvalidDrawType;
  std::array<SeenSizes, 2> m_seen{};
  uint32_t m_draws = 0;
  detail::FramePacket m_frame;
};

bool is_task(const detail::FrameOp& op) { return op.type == detail::FrameOpType::EncoderTask; }

TEST_F(FrameStreamSpill, OutgrownStagingSpillsAndKeepsEveryRange) {
  const std::vector<uint8_t> v1 = pattern(640 * 1024, 1);
  const std::vector<uint8_t> v2 = pattern(Capacity, 2);
  const std::vector<uint8_t> v3 = pattern(2 * MiB, 3);
  const std::vector<uint8_t> i1 = pattern(512 * 1024, 4);
  const std::vector<uint8_t> i2 = pattern(768 * 1024, 5);

  EXPECT_EQ(push_verts(v1.data(), v1.size(), 4), (Range{0, 640 * 1024}));
  EXPECT_EQ(push_indices(i1.data(), i1.size(), 4), (Range{0, 512 * 1024}));
  ASSERT_NO_FATAL_FAILURE(push_task(0));
  EXPECT_TRUE(m_frame.spills.empty());

  // One frame jumps past its capacity: 3.6 MiB of vertices and 1.25 MiB of indices against 1 MiB each
  EXPECT_EQ(push_verts(v2.data(), v2.size(), 4), (Range{640 * 1024, static_cast<uint32_t>(Capacity)}));
  EXPECT_EQ(push_verts(v3.data(), v3.size(), 4), (Range{1664 * 1024, static_cast<uint32_t>(2 * MiB)}));
  EXPECT_EQ(push_indices(i2.data(), i2.size(), 4), (Range{512 * 1024, 768 * 1024}));
  ASSERT_EQ(m_frame.spills.size(), 3u);
  ASSERT_NE(m_frame.vertexSpill, nullptr);
  ASSERT_NE(m_frame.indexSpill, nullptr);
  EXPECT_EQ(m_frame.vertexSpill->size, 4 * MiB);
  EXPECT_EQ(m_frame.indexSpill->size, 2 * MiB);
  EXPECT_EQ(m_frame.vertexOverrun, 0u);
  EXPECT_EQ(m_frame.indexOverrun, 0u);

  std::vector<uint8_t> vertices = v1;
  vertices.insert(vertices.end(), v2.begin(), v2.end());
  vertices.insert(vertices.end(), v3.begin(), v3.end());
  std::vector<uint8_t> indices = i1;
  indices.insert(indices.end(), i2.begin(), i2.end());
  m_vertexOut = buffer(vertices.size(), wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc);
  m_indexOut = buffer(indices.size(), wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc);
  ASSERT_NO_FATAL_FAILURE(push_task(1));
  ASSERT_NO_FATAL_FAILURE(encode_and_submit(&is_task));

  EXPECT_EQ(m_frame.stats.lastVertSize, vertices.size());
  EXPECT_EQ(m_frame.stats.lastIndexSize, indices.size());
  // The next frame's staging is sized for this one
  EXPECT_GE(detail::testing::requested_streaming_sizes().vertex, vertices.size());
  EXPECT_GE(detail::testing::requested_streaming_sizes().index, indices.size());
  for (const auto& pass : m_frame.renderPasses) {
    EXPECT_FALSE(pass.dropDraws);
  }
  EXPECT_EQ(m_seen[0].vertex, Capacity);
  EXPECT_EQ(m_seen[0].index, Capacity);
  EXPECT_EQ(m_seen[1].vertex, 4 * MiB);
  EXPECT_EQ(m_seen[1].index, 2 * MiB);
  EXPECT_EQ(detail::resources().deviceSizes.vertex, 4 * MiB);
  EXPECT_EQ(detail::resources().deviceSizes.index, 2 * MiB);
  EXPECT_EQ(warnings_with(OverrunWarning), 0u);
  if (m_null) {
    return;
  }
  EXPECT_TRUE(read(m_vertexOut) == vertices);
  EXPECT_TRUE(read(m_indexOut) == indices);
}

TEST_F(FrameStreamSpill, PastTheLimitDropsLaterDraws) {
  // Vertices are bound whole as storage, so this binding size is their limit
  detail::resources().limits.maxStorageBufferBindingSize = 2 * MiB;
  const std::vector<uint8_t> v1 = pattern(640 * 1024, 1);
  const std::vector<uint8_t> v2 = pattern(Capacity, 2);
  const std::vector<uint8_t> v3 = pattern(Capacity, 3);
  const std::array<uint8_t, 16> v4{};

  EXPECT_EQ(push_verts(v1.data(), v1.size(), 4), (Range{0, 640 * 1024}));
  EXPECT_EQ(push_verts(v2.data(), v2.size(), 4), (Range{640 * 1024, static_cast<uint32_t>(Capacity)}));
  ASSERT_NE(m_frame.vertexSpill, nullptr);
  EXPECT_EQ(m_frame.vertexSpill->size, 2 * MiB);
  ASSERT_TRUE(push_custom_draw(m_draw, nullptr, 0));
  ASSERT_NO_FATAL_FAILURE(push_task(0));

  // Ranges past the limit still come out contiguous, so merged draws keep their checks
  EXPECT_EQ(push_verts(v3.data(), v3.size(), 4), (Range{1664 * 1024, static_cast<uint32_t>(Capacity)}));
  EXPECT_EQ(push_verts(v4.data(), v4.size(), 0), (Range{2688 * 1024, 16}));
  EXPECT_EQ(m_frame.verts.size(), 1664 * 1024);
  EXPECT_EQ(m_frame.vertexOverrun, Capacity + 16);
  EXPECT_EQ(m_frame.spills.size(), 1u);
  EXPECT_EQ(warnings_with(OverrunWarning), 1u);
  ASSERT_TRUE(push_custom_draw(m_draw, nullptr, 0));
  ASSERT_NO_FATAL_FAILURE(push_task(1));

  ASSERT_EQ(m_frame.renderPasses.size(), 3u);
  // The first pass needs gx's bind groups and the last the depth peek, so only the dropped middle pass is encoded
  ASSERT_NO_FATAL_FAILURE(encode_and_submit([](const detail::FrameOp& op) {
    return is_task(op) || (op.type == detail::FrameOpType::RenderPass && op.index == 1);
  }));
  EXPECT_FALSE(m_frame.renderPasses[0].dropDraws);
  EXPECT_TRUE(m_frame.renderPasses[1].dropDraws);
  EXPECT_TRUE(m_frame.renderPasses[2].dropDraws);
  EXPECT_EQ(m_draws, 0u);
  EXPECT_EQ(m_frame.stats.lastVertSize, 1664 * 1024);
  EXPECT_EQ(m_seen[1].vertex, 2 * MiB);
  EXPECT_EQ(warnings_with(OverrunWarning), 1u);
}

using FrameStreamOverrun = FrameStreamSpill;

bool is_task_or_middle_pass(const detail::FrameOp& op) {
  return is_task(op) || (op.type == detail::FrameOpType::RenderPass && op.index == 1);
}

TEST_F(FrameStreamOverrun, FullUniformsDropLaterDrawsAndGrow) {
  const std::vector<uint8_t> u1 = pattern(640 * 1024, 1);
  const std::vector<uint8_t> u2 = pattern(382 * 1024, 2);
  const std::vector<uint8_t> u3 = pattern(4096, 3);

  EXPECT_EQ(push_uniform(u1.data(), u1.size()), (Range{0, 640 * 1024}));
  ASSERT_TRUE(push_custom_draw(m_draw, nullptr, 0));
  ASSERT_NO_FATAL_FAILURE(push_task(0));
  EXPECT_GT(uniform_space_left(), 0u);

  // It fits the staging alone, but not with the gx window that finish pads behind the last push
  EXPECT_EQ(push_uniform(u2.data(), u2.size()), (Range{0, 382 * 1024}));
  EXPECT_EQ(push_uniform(u3.data(), u3.size()), (Range{0, 4096}));
  EXPECT_EQ(m_frame.uniforms.size(), 640 * 1024);
  EXPECT_EQ(m_frame.uniformOverrun, 386 * 1024);
  EXPECT_EQ(uniform_space_left(), 0u);
  EXPECT_EQ(warnings_with(UniformWarning), 1u);
  EXPECT_EQ(detail::testing::requested_streaming_sizes().uniform, 2 * MiB);
  ASSERT_TRUE(push_custom_draw(m_draw, nullptr, 0));
  ASSERT_NO_FATAL_FAILURE(push_task(1));

  ASSERT_EQ(m_frame.renderPasses.size(), 3u);
  ASSERT_NO_FATAL_FAILURE(encode_and_submit(&is_task_or_middle_pass));
  EXPECT_FALSE(m_frame.renderPasses[0].dropDraws);
  EXPECT_FALSE(m_frame.renderPasses[0].dropResolve);
  EXPECT_TRUE(m_frame.renderPasses[1].dropDraws);
  EXPECT_TRUE(m_frame.renderPasses[1].dropResolve);
  EXPECT_TRUE(m_frame.renderPasses[2].dropDraws);
  EXPECT_EQ(m_draws, 0u);
  EXPECT_EQ(m_frame.stats.lastUniformSize, 640 * 1024 + gx::MaxUniformSize);
  EXPECT_EQ(warnings_with(UniformWarning), 1u);
  EXPECT_EQ(warnings_with(StorageWarning), 0u);
}

TEST_F(FrameStreamOverrun, FullStorageRefusesAndDropsLaterDraws) {
  const std::vector<uint8_t> s1 = pattern(640 * 1024, 1);
  const std::vector<uint8_t> s2 = pattern(512 * 1024, 2);
  const std::vector<uint8_t> s3 = pattern(16, 3);

  EXPECT_EQ(push_storage(s1.data(), s1.size()), (Range{0, 640 * 1024}));
  ASSERT_TRUE(push_custom_draw(m_draw, nullptr, 0));
  ASSERT_NO_FATAL_FAILURE(push_task(0));

  // Refused rather than spilled, since the resident arena sits right after streaming storage
  EXPECT_EQ(push_storage(s2.data(), s2.size()), Range{});
  EXPECT_EQ(push_storage(s3.data(), s3.size()), Range{});
  EXPECT_EQ(m_frame.storage.size(), 640 * 1024);
  EXPECT_NE(m_frame.storageOverrun, 0u);
  EXPECT_EQ(warnings_with(StorageWarning), 1u);
  EXPECT_EQ(detail::testing::requested_streaming_sizes().storage, 2 * MiB);
  // Channel pushes check their own room, so GPU systems keep what still fits
  EXPECT_EQ(push_channel_storage(s3).size, s3.size());
  ASSERT_TRUE(push_custom_draw(m_draw, nullptr, 0));
  ASSERT_NO_FATAL_FAILURE(push_task(1));

  ASSERT_EQ(m_frame.renderPasses.size(), 3u);
  ASSERT_NO_FATAL_FAILURE(encode_and_submit(&is_task_or_middle_pass));
  EXPECT_FALSE(m_frame.renderPasses[0].dropDraws);
  EXPECT_TRUE(m_frame.renderPasses[1].dropDraws);
  EXPECT_FALSE(m_frame.renderPasses[1].dropResolve);
  EXPECT_TRUE(m_frame.renderPasses[2].dropDraws);
  EXPECT_EQ(m_draws, 0u);
  EXPECT_EQ(m_frame.stats.lastStorageSize, 640 * 1024 + s3.size());
  EXPECT_EQ(warnings_with(StorageWarning), 1u);
  EXPECT_EQ(warnings_with(UniformWarning), 0u);
}

TEST_F(FrameStreamOverrun, GrownUniformBufferIsRebound) {
  struct ScopedBindings {
    ScopedBindings() { render::detail::shutdown_bindings(); }
    ~ScopedBindings() { render::detail::shutdown_bindings(); }
  };
  const ScopedBindings bindings;
  auto& res = detail::resources();
  const wgpu::BindGroup frameGroup = render::detail::frame_group();
  ASSERT_TRUE(frameGroup);
  detail::grow_uniform_buffer(2 * MiB);
  EXPECT_EQ(res.uniformBuffer.GetSize(), 2 * MiB);
  EXPECT_EQ(res.deviceSizes.uniform, 2 * MiB);
  EXPECT_TRUE(res.uniformBindGroup);
  // The graph binds the uniform buffer in its own frame group, which has to follow the new buffer
  const wgpu::BindGroup grown = render::detail::frame_group();
  EXPECT_TRUE(grown);
  EXPECT_NE(grown.Get(), frameGroup.Get());
  EXPECT_EQ(render::detail::frame_group().Get(), grown.Get());
}
} // namespace
} // namespace aurora::gfx
