#pragma once

#include "render_test_device.hpp"

#include "gfx/pipeline_cache.hpp"
#include "gfx/resources.hpp"
#include "gx/hints.hpp"
#include "render/frame_state.hpp"
#include "render/gpu_heap.hpp"
#include "render/graph.hpp"
#include "render/graph_bind.hpp"
#include "render/graph_pool.hpp"
#include "render/gx_ext/records.hpp"
#include "render/hints.hpp"
#include "render/module.hpp"
#include "render/post/taa.hpp"
#include "render/post/temporal_resolve.hpp"
#include "render/post/upscale/compose.hpp"
#include "render/post/upscale/fsr/fsr_pass.hpp"
#include "render/post/upscale/upscale.hpp"
#include "render/scene/motion.hpp"
#include "render/scene_depth.hpp"
#include "render/settings.hpp"
#include "render/stage.hpp"
#include "render/stage_data.hpp"
#include "webgpu/caps.hpp"
#include "webgpu/gpu.hpp"

#include <aurora/render_caps.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace aurora::render::detail::fsr_rig {
inline Mat4x4<float> perspective() {
  Mat4x4<float> proj{};
  proj.m0[0] = 1.2990381f;
  proj.m1[1] = 1.7320508f;
  proj.m2[2] = -1.f / 999.f;
  proj.m2[3] = -1000.f / 999.f;
  proj.m3[2] = -1.f;
  return proj;
}

inline gx::hints::Camera main_camera(uint32_t cutEpoch) {
  const Mat4x4<float> proj = perspective();
  return {
      .role = AURORA_CAMERA_MAIN,
      .view = {{1.f, 0.f, 0.f, 0.f}, {0.f, 1.f, 0.f, 0.f}, {0.f, 0.f, 1.f, 0.f}},
      .proj = proj,
      .type = GX_PERSPECTIVE,
      .cutEpoch = cutEpoch,
      .gxProj = gx::hints::gx_projection(proj, GX_PERSPECTIVE),
  };
}

inline Settings fsr_settings(UpscalerQuality quality, float sharpness = 0.f) {
  Settings settings;
  settings.upscaler = Upscaler::Fsr;
  settings.upscalerQuality = quality;
  settings.taaSharpness = sharpness;
  return settings;
}

struct Storage {
  static inline std::vector<uint8_t> bytes;
  static inline bool refuse = false;

  static void reset() {
    bytes.clear();
    refuse = false;
  }

  static gfx::Range push(std::span<const uint8_t> data) noexcept {
    if (refuse) {
      return {};
    }
    const size_t begin = std::max<size_t>((bytes.size() + 15) & ~size_t{15}, 16);
    bytes.resize(begin);
    bytes.insert(bytes.end(), data.begin(), data.end());
    return {static_cast<uint32_t>(begin), static_cast<uint32_t>(data.size())};
  }
};

struct FrameOptions {
  uint32_t cutEpoch = 0;
  bool camera = true;
  bool invalidate = false;
  bool refuseStorage = false;
  bool compose = true;
  std::shared_ptr<const SceneDepth> sceneDepth;
};

class Rig {
public:
  Rig(test::TestDevice& device, const test::TestDeviceOptions& options) : m_device(device) {
    gpu_heap::shutdown();
    reset_module_state();
    register_gx_resolver(GxResource::Depth, &resolve_gx_depth);
    testing::set_hints_override(1);
    install_hints();
    Storage::reset();
    gx_ext::records::testing::set_push(&Storage::push);
    scene::motion::testing::set_push(&Storage::push);
    scene::motion::testing::set_jitter_env(scene::motion::testing::JitterEnv::Off);
    scene::motion::testing::set_check_range(0, 0);
    gfx::set_blocking_pipelines(true);
    const wgpu::TextureDescriptor depth{
        .label = "FSR rig EFB depth",
        .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding,
        .size = options.efbSize,
        .format = wgpu::TextureFormat::Depth32Float,
    };
    wgpu::Texture texture = device.device.CreateTexture(&depth);
    webgpu::g_depthBuffer = {.texture = texture,
                             .view = texture.CreateView(),
                             .size = depth.size,
                             .format = wgpu::TextureFormat::Depth32Float};
    taa::initialize();
    upscale::initialize();
    m_started = true;
    m_ok = webgpu::set_gbuffer(true);
  }

  ~Rig() {
    if (!m_started) {
      return;
    }
    m_state.reset();
    m_graph.reset();
    upscale::testing::reset();
    upscale::shutdown();
    webgpu::set_scene_render_size(0, 0);
    scene::motion::testing::reset();
    gx_ext::records::testing::reset();
    testing::reset_hints();
    webgpu::set_gbuffer(false);
    reset_module_state();
    gpu_heap::shutdown();
    Storage::reset();
    gfx::set_blocking_pipelines(false);
  }

  Rig(const Rig&) = delete;
  Rig& operator=(const Rig&) = delete;

  [[nodiscard]] bool ok() const noexcept { return m_ok; }

  static void reset_module_state() {
    taa::shutdown();
    taa::testing::reset();
    upscale::shutdown();
    shutdown_bindings();
    testing::reset_transient_pool(nullptr);
    transient_pool().set_release_listener(&bind_cache_on_release);
    testing::clear_passes();
    reset_frame_cull();
  }

  std::shared_ptr<CompiledGraph> compile(const Settings& settings) {
    m_settings = settings;
    m_env = gather_graph_environment();
    std::shared_ptr<CompiledGraph> graph =
        compile_graph(m_settings, caps(), m_env, {.sceneDepthSource = DepthSource::Direct});
    prepare_graph_bindings(
        *graph, {.settings = &m_settings, .caps = &caps(), .env = &m_env, .sceneDepth = DepthSource::Direct});
    return graph;
  }

  void latch() {
    taa::on_latch(m_settings, m_graph.get());
    upscale::latch_graph(m_settings, m_graph.get());
    latch_pass_params();
  }

  std::shared_ptr<CompiledGraph> use(std::shared_ptr<CompiledGraph> graph) {
    m_graph = std::move(graph);
    m_state = std::make_unique<FrameState>(FrameState{.graph = m_graph});
    latch();
    return m_graph;
  }

  std::shared_ptr<CompiledGraph> install(const Settings& settings) {
    upscale::on_latch(settings);
    return use(compile(settings));
  }

  [[nodiscard]] NodeId node(std::string_view name) const {
    return m_graph != nullptr ? m_graph->find_node(name) : InvalidNode;
  }

  StageData build(uint64_t frameId, const FrameOptions& options = {}) {
    Storage::refuse = options.refuseStorage;
    begin_hint_frame();
    scene::motion::latch(m_settings, true);
    latch();
    gx_ext::records::testing::set_active(true, true, false);
    if (options.camera) {
      const gx::hints::Sink& sink = *gx::hints::sink();
      sink.camera(sink.userData, main_camera(options.cutEpoch));
    }
    on_worker_frame_begin(frameId, m_state.get());
    StageData begin{.stage = Stage::SceneBegin, .frameId = frameId, .markerSeen = true, .frame = m_state.get()};
    build_stage_data(begin);
    if (options.invalidate) {
      invalidate_frame_histories();
    }
    StageData post{
        .stage = Stage::PrePost,
        .frameId = frameId,
        .frameIndex = static_cast<uint32_t>(frameId),
        .markerSeen = true,
        .frame = m_state.get(),
        .sceneDepth = options.sceneDepth,
    };
    build_stage_data(post);
    Storage::refuse = false;
    return post;
  }

  std::optional<std::string> encode(const StageData& post, bool compose) {
    m_device.device.PushErrorScope(wgpu::ErrorFilter::Validation);
    wgpu::CommandEncoder encoder = m_device.device.CreateCommandEncoder();
    if (post.any_enabled()) {
      run_stage(encoder, post);
    }
    if (compose && upscale::split_live()) {
      const GraphExtent scene = upscale::scene_size();
      const GraphExtent output = upscale::output_size();
      SceneComposeData data{
          .mode = ComposeMode::Split,
          .ratio = {upscale::ratio(), upscale::ratio()},
          .rcasStrength = upscale::rcas_strength(),
          .frameId = post.frameId,
      };
      if (post.sceneHints != nullptr) {
        data.jitterPx[0] = post.sceneHints->jitterPx[0];
        data.jitterPx[1] = post.sceneHints->jitterPx[1];
      }
      const upscale::compose::Params params = upscale::compose::make_params(data, scene, output);
      m_device.queue.WriteBuffer(gfx::detail::resources().uniformBuffer, 0, &params, sizeof(params));
      upscale::compose::encode(encoder, data);
    }
    const wgpu::CommandBuffer commands = encoder.Finish();
    const bool submitted = test::submit_and_wait(m_device, commands);
    std::optional<std::string> error = test::pop_validation_error(m_device);
    if (!submitted && !error) {
      error = "submit failed";
    }
    return error;
  }

  std::optional<std::string> run(uint64_t frameId, const FrameOptions& options = {}) {
    const StageData post = build(frameId, options);
    return encode(post, options.compose);
  }

  test::TestDevice& m_device;
  bool m_started = false;
  bool m_ok = false;
  Settings m_settings;
  GraphEnvironment m_env;
  std::shared_ptr<CompiledGraph> m_graph;
  std::unique_ptr<FrameState> m_state;
};
} // namespace aurora::render::detail::fsr_rig
