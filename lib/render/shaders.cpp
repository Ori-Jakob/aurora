#include "shaders.hpp"

#include "jobs.hpp"
#include "shader_resolve.hpp"

#include "../gfx/hash.hpp"
#include "../gfx/pipeline_cache.hpp"
#include "../internal.hpp"
#include "../webgpu/gpu.hpp"

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <magic_enum.hpp>

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <type_traits>
#include <utility>

namespace aurora::render::shaders {
namespace {
constexpr Module Log{"aurora::render::shaders"};
constexpr uint64_t DawnWaitNs = 5'000'000'000;
constexpr uint64_t ShutdownWaitNs = 2'000'000'000;
constexpr uint64_t MaxWaitNs = 86'400'000'000'000;
constexpr size_t MaxFailures = 64;

struct SlotState {
  std::atomic<PipelineStatus> status{PipelineStatus::Pending};
  wgpu::ComputePipeline compute;
  wgpu::RenderPipeline render;
  std::string label;
  std::string message;
  uint64_t session = 0;
  std::atomic_bool dispatched = false;
};

struct RenderState {
  wgpu::PrimitiveTopology topology = wgpu::PrimitiveTopology::TriangleList;
  wgpu::FrontFace frontFace = wgpu::FrontFace::CCW;
  wgpu::CullMode cullMode = wgpu::CullMode::None;
  uint32_t colorCount = 0;
  std::array<ColorTarget, MaxColorTargets> colors{};
  wgpu::TextureFormat depthFormat = wgpu::TextureFormat::Undefined;
  bool depthWrite = false;
  wgpu::CompareFunction depthCompare = wgpu::CompareFunction::Always;
  uint32_t sampleCount = 1;
};

// Owns copies of everything a desc views, because creation outlives the caller's arrays
struct Request {
  std::shared_ptr<SlotState> slot;
  bool render = false;
  std::string shader;
  std::string entry;
  std::string fragmentEntry;
  std::vector<std::string> defines;
  std::vector<std::pair<std::string, double>> constants;
  wgpu::PipelineLayout layout;
  RenderState state;
};

struct ModuleEntry {
  wgpu::ShaderModule module;
  std::string error;
};

struct LibraryShader {
  std::string_view id;
  std::string_view text;
};

using RuntimeSources = std::vector<std::pair<std::string, std::shared_ptr<const std::string>>>;

std::mutex g_mutex;
std::condition_variable g_cv;
// Dawn has no device lock without ImplicitDeviceSynchronization, so this file adds at most one creating thread
std::mutex g_deviceMutex;
bool g_initialized = false;
bool g_shuttingDown = false;
uint64_t g_session = 0;
std::atomic_uint64_t g_sourceGeneration = 0;
std::vector<std::string_view> g_implicitDefines;
std::map<std::string, std::shared_ptr<const std::string>, std::less<>> g_runtime;
std::vector<std::shared_ptr<SlotState>> g_slots;
absl::flat_hash_map<uint64_t, PipelineId> g_dedup;
absl::flat_hash_map<uint64_t, ModuleEntry> g_modules;
absl::flat_hash_set<std::string> g_warned;
std::deque<PipelineFailure> g_failures;
// Changed only under g_mutex, so waiters on g_cv see every update
std::atomic_uint32_t g_pendingCount = 0;
std::atomic_uint32_t g_readyCount = 0;
std::atomic_uint32_t g_failedCount = 0;
std::atomic_uint32_t g_moduleCount = 0;
std::atomic_uint32_t g_waiters = 0;
jobs::JobGroup g_compileGroup;

const std::vector<LibraryShader>& library() {
  static const std::vector<LibraryShader> shaders = [] {
    std::vector<LibraryShader> out;
    for (const EmbeddedShader& shader : embedded_library()) {
      out.push_back({shader.id, {reinterpret_cast<const char*>(shader.data), shader.size}});
    }
    // The generated table is in path order, and a leading shaders/ is stripped after sorting
    std::ranges::sort(out, {}, &LibraryShader::id);
    return out;
  }();
  return shaders;
}

std::optional<std::string_view> library_source(std::string_view id) {
  const auto& shaders = library();
  const auto it = std::ranges::lower_bound(shaders, id, {}, &LibraryShader::id);
  return it != shaders.end() && it->id == id ? std::optional{it->text} : std::nullopt;
}

RuntimeSources runtime_sources() {
  std::lock_guard lock{g_mutex};
  return {g_runtime.begin(), g_runtime.end()};
}

bool resolve(std::string_view shader, std::span<const std::string_view> defines, ResolvedSource& out,
             std::string& error) {
  // Owning copies keep a text alive while a concurrent register_source replaces it
  const RuntimeSources runtime = runtime_sources();
  const SourceLookup lookup = [&runtime](std::string_view id) -> std::optional<std::string_view> {
    if (const auto text = library_source(id)) {
      return text;
    }
    const auto it =
        std::ranges::lower_bound(runtime, id, {}, [](const auto& entry) -> std::string_view { return entry.first; });
    return it != runtime.end() && it->first == id ? std::optional{std::string_view{*it->second}} : std::nullopt;
  };
  std::vector<std::string_view> all{g_implicitDefines.begin(), g_implicitDefines.end()};
  all.insert(all.end(), defines.begin(), defines.end());
  return resolve_source(shader, all, lookup, out, error);
}

bool warn_once(std::string_view key) {
  std::lock_guard lock{g_mutex};
  return g_warned.emplace(key).second;
}

struct WaitState {
  std::mutex mutex;
  std::optional<std::string> error;
  std::vector<std::pair<uint64_t, uint64_t>> positions;
  std::vector<std::string> messages;
};

// Requires g_deviceMutex held. State is shared with the callback, which Dawn may still run after a timeout
std::optional<std::string> pop_error_scope() {
  auto state = std::make_shared<WaitState>();
  const wgpu::Future future = webgpu::g_device.PopErrorScope(
      wgpu::CallbackMode::WaitAnyOnly,
      [state](wgpu::PopErrorScopeStatus status, wgpu::ErrorType type, wgpu::StringView message) {
        std::lock_guard lock{state->mutex};
        if (status != wgpu::PopErrorScopeStatus::Success) {
          state->error = "error scope lost";
        } else if (type != wgpu::ErrorType::NoError) {
          state->error = std::string{std::string_view{message}};
        }
      });
  if (webgpu::g_instance.WaitAny(future, DawnWaitNs) != wgpu::WaitStatus::Success) {
    return "timed out";
  }
  std::lock_guard lock{state->mutex};
  return state->error;
}

std::optional<std::string> pop_error_scopes() {
  const std::optional<std::string> internal = pop_error_scope();
  const std::optional<std::string> validation = pop_error_scope();
  if (internal && validation) {
    return *validation + "\n" + *internal;
  }
  return validation ? validation : internal;
}

void push_error_scopes() {
  webgpu::g_device.PushErrorScope(wgpu::ErrorFilter::Validation);
  webgpu::g_device.PushErrorScope(wgpu::ErrorFilter::Internal);
}

// Requires g_deviceMutex held.
std::string compilation_errors(const wgpu::ShaderModule& module, const ResolvedSource& source) {
  auto state = std::make_shared<WaitState>();
  const wgpu::Future future =
      module.GetCompilationInfo(wgpu::CallbackMode::WaitAnyOnly,
                                [state](wgpu::CompilationInfoRequestStatus status, const wgpu::CompilationInfo* info) {
                                  if (status != wgpu::CompilationInfoRequestStatus::Success || info == nullptr) {
                                    return;
                                  }
                                  std::lock_guard lock{state->mutex};
                                  for (size_t i = 0; i < info->messageCount; ++i) {
                                    const wgpu::CompilationMessage& message = info->messages[i];
                                    if (message.type == wgpu::CompilationMessageType::Error) {
                                      state->positions.emplace_back(message.lineNum, message.linePos);
                                      state->messages.emplace_back(std::string_view{message.message});
                                    }
                                  }
                                });
  if (webgpu::g_instance.WaitAny(future, DawnWaitNs) != wgpu::WaitStatus::Success) {
    return {};
  }
  std::lock_guard lock{state->mutex};
  std::string out;
  for (size_t i = 0; i < state->messages.size(); ++i) {
    const auto [line, column] = state->positions[i];
    if (!out.empty()) {
      out.push_back('\n');
    }
    out += fmt::format("{}: {}", describe_line(source, static_cast<uint32_t>(line), static_cast<uint32_t>(column)),
                       state->messages[i]);
  }
  return out;
}

// Requires g_deviceMutex held.
wgpu::ShaderModule create_module_locked(const std::string& label, const ResolvedSource& source, std::string& error) {
  const wgpu::ShaderSourceWGSL wgsl{wgpu::ShaderSourceWGSL::Init{.code = source.text.c_str()}};
  const wgpu::ShaderModuleDescriptor descriptor{.nextInChain = &wgsl, .label = label.c_str()};
  push_error_scopes();
  wgpu::ShaderModule module = webgpu::g_device.CreateShaderModule(&descriptor);
  const std::optional<std::string> scopeError = pop_error_scopes();
  if (!scopeError) {
    return module;
  }
  error = compilation_errors(module, source);
  if (error.empty()) {
    error = *scopeError;
  }
  return {};
}

bool find_module(uint64_t key, wgpu::ShaderModule& module, std::string& error) {
  std::lock_guard lock{g_mutex};
  const auto it = g_modules.find(key);
  if (it == g_modules.end()) {
    return false;
  }
  module = it->second.module;
  error = it->second.error;
  return true;
}

wgpu::ShaderModule obtain_module(const Request& request, const std::string& label, std::string& error) {
  const std::vector<std::string_view> defines{request.defines.begin(), request.defines.end()};
  ResolvedSource source;
  if (!resolve(request.shader, defines, source, error)) {
    if (warn_once(error)) {
      Log.warn("Shader {} failed: {}", label, error);
    }
    return {};
  }
  const uint64_t key = XXH3_64bits(source.text.data(), source.text.size());
  wgpu::ShaderModule module;
  if (find_module(key, module, error)) {
    return module;
  }
  {
    std::lock_guard device{g_deviceMutex};
    // Another job may have built the same text while this one waited for the device
    if (find_module(key, module, error)) {
      return module;
    }
    module = create_module_locked(request.shader, source, error);
    std::lock_guard lock{g_mutex};
    g_modules.emplace(key, ModuleEntry{.module = module, .error = error});
    if (module) {
      g_moduleCount.fetch_add(1, std::memory_order_relaxed);
    }
  }
  if (!module) {
    // A failure only disables the passes using this text, so it stays a warning
    Log.warn("Shader {} failed:\n{}", label, error);
  }
  return module;
}

void finish_slot(SlotState& slot, PipelineStatus status, std::string message, bool warn) {
  const bool failed = status == PipelineStatus::Failed;
  bool current = false;
  {
    std::lock_guard lock{g_mutex};
    // An orphan from before a shutdown, or a slot a quiesce already failed, must not count again
    current = slot.session == g_session && slot.status.load(std::memory_order_relaxed) == PipelineStatus::Pending;
  }
  // Warns before publishing, so a waiter that sees Failed also sees the warning
  if (warn && current && failed) {
    Log.warn("Pipeline {} failed: {}", slot.label, message);
  }
  {
    std::lock_guard lock{g_mutex};
    if (slot.status.load(std::memory_order_relaxed) != PipelineStatus::Pending) {
      return;
    }
    if (failed) {
      slot.message = std::move(message);
    }
    slot.status.store(status, std::memory_order_release);
    if (slot.session == g_session) {
      g_pendingCount.fetch_sub(1, std::memory_order_relaxed);
      (failed ? g_failedCount : g_readyCount).fetch_add(1, std::memory_order_relaxed);
      if (failed) {
        g_failures.push_back({slot.label, slot.message});
        if (g_failures.size() > MaxFailures) {
          g_failures.pop_front();
        }
      }
    }
  }
  g_cv.notify_all();
}

template <typename Pipeline>
void complete(SlotState& slot, wgpu::CreatePipelineAsyncStatus status, Pipeline pipeline, wgpu::StringView message) {
  if (status != wgpu::CreatePipelineAsyncStatus::Success) {
    finish_slot(slot, PipelineStatus::Failed,
                fmt::format("{}: {}", magic_enum::enum_name(status), std::string_view{message}), true);
    return;
  }
  if constexpr (std::is_same_v<Pipeline, wgpu::ComputePipeline>) {
    slot.compute = std::move(pipeline);
  } else {
    slot.render = std::move(pipeline);
  }
  finish_slot(slot, PipelineStatus::Ready, {}, false);
}

std::vector<wgpu::ConstantEntry> constant_entries(const Request& request) {
  std::vector<wgpu::ConstantEntry> out;
  out.reserve(request.constants.size());
  for (const auto& [name, value] : request.constants) {
    out.push_back({.key = std::string_view{name}, .value = value});
  }
  return out;
}

template <typename Descriptor, typename Create, typename CreateAsync>
void create_pipeline(const Request& request, const Descriptor& descriptor, bool blocking, Create create,
                     CreateAsync createAsync) {
  SlotState& slot = *request.slot;
  if (!blocking) {
    std::lock_guard device{g_deviceMutex};
    slot.dispatched.store(true, std::memory_order_release);
    createAsync(descriptor);
    return;
  }
  std::optional<std::string> error;
  {
    std::lock_guard device{g_deviceMutex};
    push_error_scopes();
    auto pipeline = create(descriptor);
    error = pop_error_scopes();
    if (!error) {
      complete(slot, wgpu::CreatePipelineAsyncStatus::Success, std::move(pipeline), {});
      return;
    }
  }
  finish_slot(slot, PipelineStatus::Failed, std::move(*error), true);
}

void create_compute(const Request& request, const wgpu::ShaderModule& module, bool blocking) {
  const std::vector<wgpu::ConstantEntry> constants = constant_entries(request);
  const wgpu::ComputePipelineDescriptor descriptor{
      .label = std::string_view{request.slot->label},
      .layout = request.layout,
      .compute =
          {
              .module = module,
              .entryPoint = std::string_view{request.entry},
              .constantCount = constants.size(),
              .constants = constants.data(),
          },
  };
  create_pipeline(
      request, descriptor, blocking,
      [](const wgpu::ComputePipelineDescriptor& d) { return webgpu::g_device.CreateComputePipeline(&d); },
      [slot = request.slot](const wgpu::ComputePipelineDescriptor& d) {
        webgpu::g_device.CreateComputePipelineAsync(
            &d, wgpu::CallbackMode::AllowSpontaneous,
            [slot](wgpu::CreatePipelineAsyncStatus status, wgpu::ComputePipeline pipeline, wgpu::StringView message) {
              complete(*slot, status, std::move(pipeline), message);
            });
      });
}

void create_render(const Request& request, const wgpu::ShaderModule& module, bool blocking) {
  const RenderState& state = request.state;
  const std::vector<wgpu::ConstantEntry> constants = constant_entries(request);
  std::array<wgpu::BlendState, MaxColorTargets> blends{};
  std::array<wgpu::ColorTargetState, MaxColorTargets> targets{};
  for (uint32_t i = 0; i < state.colorCount; ++i) {
    const ColorTarget& color = state.colors[i];
    blends[i] = color.blendState;
    targets[i] = {.format = color.format, .blend = color.blend ? &blends[i] : nullptr, .writeMask = color.writeMask};
  }
  const wgpu::FragmentState fragment{
      .module = module,
      .entryPoint = std::string_view{request.fragmentEntry},
      .constantCount = constants.size(),
      .constants = constants.data(),
      .targetCount = state.colorCount,
      .targets = targets.data(),
  };
  const wgpu::DepthStencilState depth{
      .format = state.depthFormat,
      .depthWriteEnabled = state.depthWrite ? wgpu::OptionalBool::True : wgpu::OptionalBool::False,
      .depthCompare = state.depthCompare,
  };
  const wgpu::RenderPipelineDescriptor descriptor{
      .label = std::string_view{request.slot->label},
      .layout = request.layout,
      .vertex =
          {
              .module = module,
              .entryPoint = std::string_view{request.entry},
              .constantCount = constants.size(),
              .constants = constants.data(),
          },
      .primitive = {.topology = state.topology, .frontFace = state.frontFace, .cullMode = state.cullMode},
      .depthStencil = state.depthFormat != wgpu::TextureFormat::Undefined ? &depth : nullptr,
      .multisample = {.count = state.sampleCount},
      .fragment = request.fragmentEntry.empty() ? nullptr : &fragment,
  };
  create_pipeline(
      request, descriptor, blocking,
      [](const wgpu::RenderPipelineDescriptor& d) { return webgpu::g_device.CreateRenderPipeline(&d); },
      [slot = request.slot](const wgpu::RenderPipelineDescriptor& d) {
        webgpu::g_device.CreateRenderPipelineAsync(
            &d, wgpu::CallbackMode::AllowSpontaneous,
            [slot](wgpu::CreatePipelineAsyncStatus status, wgpu::RenderPipeline pipeline, wgpu::StringView message) {
              complete(*slot, status, std::move(pipeline), message);
            });
      });
}

void create(const Request& request, bool blocking, const jobs::JobContext* context) {
  SlotState& slot = *request.slot;
  const auto cancelled = [&] { return context != nullptr && context->cancelled(); };
  if (cancelled()) {
    finish_slot(slot, PipelineStatus::Failed, "cancelled", false);
    return;
  }
  std::string error;
  const wgpu::ShaderModule module = obtain_module(request, slot.label, error);
  if (!module) {
    finish_slot(slot, PipelineStatus::Failed, std::move(error), false);
    return;
  }
  if (cancelled()) {
    finish_slot(slot, PipelineStatus::Failed, "cancelled", false);
    return;
  }
  if (request.render) {
    create_render(request, module, blocking);
  } else {
    create_compute(request, module, blocking);
  }
}

void hash_text(Hasher& hasher, std::string_view text) {
  hasher.update(static_cast<uint32_t>(text.size()));
  hasher.update(text.data(), text.size());
}

std::vector<std::string> sorted_defines(std::span<const std::string_view> defines) {
  std::vector<std::string> out{defines.begin(), defines.end()};
  std::ranges::sort(out);
  const auto duplicates = std::ranges::unique(out);
  out.erase(duplicates.begin(), duplicates.end());
  return out;
}

std::vector<std::pair<std::string, double>> sorted_constants(std::span<const Constant> constants) {
  std::vector<std::pair<std::string, double>> out;
  out.reserve(constants.size());
  for (const Constant& constant : constants) {
    out.emplace_back(constant.name, constant.value);
  }
  std::ranges::stable_sort(out, {}, &std::pair<std::string, double>::first);
  return out;
}

// Hashes fields one at a time, since wgpu structs carry chain pointers and padding
uint64_t request_key(const Request& request, uint64_t layoutKey, uint64_t generation) {
  Hasher hasher{0};
  hasher.update(static_cast<uint8_t>(request.render ? 2 : 1));
  hasher.update(generation);
  hash_text(hasher, request.shader);
  hash_text(hasher, request.entry);
  if (request.render) {
    hash_text(hasher, request.fragmentEntry);
  }
  hasher.update(static_cast<uint32_t>(request.defines.size()));
  for (const std::string& define : request.defines) {
    hasher.update(define.data(), define.size() + 1);
  }
  hasher.update(static_cast<uint32_t>(request.constants.size()));
  for (const auto& [name, value] : request.constants) {
    hasher.update(name.data(), name.size() + 1);
    hasher.update(std::bit_cast<uint64_t>(value));
  }
  hasher.update(layoutKey);
  if (request.render) {
    const RenderState& state = request.state;
    hasher.update(state.topology);
    hasher.update(state.frontFace);
    hasher.update(state.cullMode);
    hasher.update(state.colorCount);
    for (uint32_t i = 0; i < std::min(state.colorCount, MaxColorTargets); ++i) {
      const ColorTarget& color = state.colors[i];
      hasher.update(color.format);
      hasher.update(static_cast<uint8_t>(color.blend));
      for (const wgpu::BlendComponent* component : {&color.blendState.color, &color.blendState.alpha}) {
        hasher.update(component->operation);
        hasher.update(component->srcFactor);
        hasher.update(component->dstFactor);
      }
      hasher.update(static_cast<uint64_t>(color.writeMask));
    }
    hasher.update(state.depthFormat);
    hasher.update(static_cast<uint8_t>(state.depthWrite));
    hasher.update(state.depthCompare);
    hasher.update(state.sampleCount);
  }
  return hasher.digest();
}

std::string rejection(std::string_view shader, const wgpu::PipelineLayout& layout, uint64_t layoutKey,
                      uint32_t colorCount) {
  CHECK(!layout || layoutKey != 0, "shaders: a pipeline layout needs a content key");
  CHECK(layout || layoutKey == 0, "shaders: an auto layout takes no layout key");
  if (!is_valid_shader_id(shader)) {
    return fmt::format("invalid shader id '{}'", shader);
  }
  if (colorCount > MaxColorTargets) {
    return fmt::format("{} colour targets, more than {}", colorCount, MaxColorTargets);
  }
  return {};
}

PipelineId submit_request(std::shared_ptr<Request> request, std::string_view label, uint64_t layoutKey,
                          std::string rejected) {
  const uint64_t key = request_key(*request, layoutKey, g_sourceGeneration.load(std::memory_order_acquire));
  PipelineId id;
  bool reused = false;
  {
    std::lock_guard lock{g_mutex};
    if (!g_initialized || g_shuttingDown) {
      return {};
    }
    if (const auto it = g_dedup.find(key); it != g_dedup.end()) {
      id = it->second;
      reused = true;
    } else {
      auto slot = std::make_shared<SlotState>();
      slot->label = label.empty() ? fmt::format("{}:{}", request->shader, request->entry) : std::string{label};
      slot->session = g_session;
      g_slots.push_back(slot);
      id = {static_cast<uint32_t>(g_slots.size())};
      g_dedup.emplace(key, id);
      g_pendingCount.fetch_add(1, std::memory_order_relaxed);
      request->slot = std::move(slot);
    }
  }
  if (reused) {
    // The hit may still be compiling from before blocking mode was turned on
    if (gfx::blocking_pipelines()) {
      wait(id);
    }
    return id;
  }
  if (!rejected.empty()) {
    // A failed slot keeps the pass gated off, where a zero id would read as unused and bind a null pipeline
    Log.warn("Pipeline {} rejected: {}", request->slot->label, rejected);
    finish_slot(*request->slot, PipelineStatus::Failed, std::move(rejected), false);
    return id;
  }
  // Replay and capture need the pipeline before the frame that requested it
  if (gfx::blocking_pipelines()) {
    create(*request, true, nullptr);
    return id;
  }
  std::shared_ptr<const Request> shared = std::move(request);
  const std::string jobLabel = "shader " + shared->shader;
  g_compileGroup.submit([shared](const jobs::JobContext& context) { create(*shared, false, &context); },
                        jobs::Priority::High, jobLabel);
  return id;
}

std::shared_ptr<SlotState> find_slot(PipelineId id) {
  std::lock_guard lock{g_mutex};
  return id.value != 0 && id.value <= g_slots.size() ? g_slots[id.value - 1] : nullptr;
}

template <typename Predicate>
bool block_until(uint64_t timeoutNs, Predicate predicate) {
  std::unique_lock lock{g_mutex};
  if (timeoutNs == UINT64_MAX) {
    g_cv.wait(lock, predicate);
    return true;
  }
  return g_cv.wait_for(lock, std::chrono::nanoseconds{std::min(timeoutNs, MaxWaitNs)}, predicate);
}

bool stop_requests() {
  std::lock_guard lock{g_mutex};
  g_shuttingDown = g_initialized;
  return g_initialized;
}

// Leaves no compile job running; Dawn's own async work and its callbacks may still be in flight
void drain_compiles() {
  g_compileGroup.cancel();
  g_compileGroup.wait();
  {
    // A cancelled job never ran, so nothing else will ever finish its slot
    std::lock_guard lock{g_mutex};
    for (const auto& slot : g_slots) {
      if (slot->status.load(std::memory_order_acquire) == PipelineStatus::Pending &&
          !slot->dispatched.load(std::memory_order_acquire)) {
        slot->message = "cancelled";
        slot->status.store(PipelineStatus::Failed, std::memory_order_release);
        g_pendingCount.fetch_sub(1, std::memory_order_relaxed);
        g_failedCount.fetch_add(1, std::memory_order_relaxed);
      }
    }
  }
  g_cv.notify_all();
}
} // namespace

void initialize() {
  library();
  std::lock_guard lock{g_mutex};
  g_implicitDefines.clear();
  if (const wgpu::Device& device = webgpu::g_device) {
    if (device.HasFeature(wgpu::FeatureName::CoreFeaturesAndLimits)) {
      g_implicitDefines.emplace_back("AURORA_CORE");
    }
    if (device.HasFeature(wgpu::FeatureName::ShaderF16)) {
      g_implicitDefines.emplace_back("AURORA_F16");
    }
    if (device.HasFeature(wgpu::FeatureName::Subgroups)) {
      g_implicitDefines.emplace_back("AURORA_SUBGROUPS");
    }
  }
  ++g_session;
  g_sourceGeneration.store(1, std::memory_order_release);
  g_initialized = true;
  g_shuttingDown = false;
}

void quiesce() {
  if (stop_requests()) {
    drain_compiles();
  }
}

void shutdown() {
  if (!stop_requests()) {
    return;
  }
  drain_compiles();
  if (!wait_idle(ShutdownWaitNs)) {
    Log.warn("{} pipelines were still compiling at shutdown", g_pendingCount.load(std::memory_order_relaxed));
  }
  std::lock_guard lock{g_mutex};
  g_slots.clear();
  g_dedup.clear();
  g_modules.clear();
  g_runtime.clear();
  g_warned.clear();
  g_failures.clear();
  g_implicitDefines.clear();
  g_pendingCount.store(0, std::memory_order_relaxed);
  g_readyCount.store(0, std::memory_order_relaxed);
  g_failedCount.store(0, std::memory_order_relaxed);
  g_moduleCount.store(0, std::memory_order_relaxed);
  ++g_session;
  g_sourceGeneration.store(0, std::memory_order_release);
  g_initialized = false;
  g_shuttingDown = false;
}

bool register_source(std::string_view id, std::string_view source) {
  if (!is_valid_shader_id(id)) {
    Log.error("register_source: invalid shader id '{}'", id);
    return false;
  }
  if (library_source(id)) {
    Log.error("register_source: '{}' would shadow a library shader", id);
    return false;
  }
  auto text = std::make_shared<const std::string>(normalize_source(source));
  std::lock_guard lock{g_mutex};
  g_runtime.insert_or_assign(std::string{id}, std::move(text));
  g_sourceGeneration.fetch_add(1, std::memory_order_acq_rel);
  return true;
}

bool has_source(std::string_view id) {
  if (library_source(id)) {
    return true;
  }
  std::lock_guard lock{g_mutex};
  return g_runtime.contains(id);
}

std::span<const std::string_view> implicit_defines() noexcept { return g_implicitDefines; }

PipelineId request_compute(const ComputePipelineDesc& desc) {
  std::string rejected = rejection(desc.shader, desc.layout, desc.layoutKey, 0);
  auto request = std::make_shared<Request>(Request{
      .shader = std::string{desc.shader},
      .entry = std::string{desc.entryPoint},
      .defines = sorted_defines(desc.defines),
      .constants = sorted_constants(desc.constants),
      .layout = desc.layout,
  });
  return submit_request(std::move(request), desc.label, desc.layoutKey, std::move(rejected));
}

PipelineId request_render(const RenderPipelineDesc& desc) {
  std::string rejected = rejection(desc.shader, desc.layout, desc.layoutKey, desc.colorCount);
  auto request = std::make_shared<Request>(Request{
      .render = true,
      .shader = std::string{desc.shader},
      .entry = std::string{desc.vertexEntry},
      .fragmentEntry = std::string{desc.fragmentEntry},
      .defines = sorted_defines(desc.defines),
      .constants = sorted_constants(desc.constants),
      .layout = desc.layout,
      .state =
          {
              .topology = desc.topology,
              .frontFace = desc.frontFace,
              .cullMode = desc.cullMode,
              .colorCount = desc.colorCount,
              .colors = desc.colors,
              .depthFormat = desc.depthFormat,
              .depthWrite = desc.depthWrite,
              .depthCompare = desc.depthCompare,
              .sampleCount = desc.sampleCount,
          },
  });
  return submit_request(std::move(request), desc.label, desc.layoutKey, std::move(rejected));
}

PipelineStatus status(PipelineId id) noexcept {
  const std::shared_ptr<SlotState> slot = find_slot(id);
  return slot != nullptr ? slot->status.load(std::memory_order_acquire) : PipelineStatus::Failed;
}

bool ready(PipelineId id) noexcept { return status(id) == PipelineStatus::Ready; }

bool pipelines_ready(std::span<const uint32_t> ids) noexcept {
  if (std::ranges::all_of(ids, [](uint32_t id) { return id == 0; })) {
    return true;
  }
  std::lock_guard lock{g_mutex};
  return std::ranges::all_of(ids, [](uint32_t id) {
    return id == 0 ||
           (id <= g_slots.size() && g_slots[id - 1]->status.load(std::memory_order_acquire) == PipelineStatus::Ready);
  });
}

wgpu::ComputePipeline compute_pipeline(PipelineId id) noexcept {
  const std::shared_ptr<SlotState> slot = find_slot(id);
  return slot != nullptr && slot->status.load(std::memory_order_acquire) == PipelineStatus::Ready
             ? slot->compute
             : wgpu::ComputePipeline{};
}

wgpu::RenderPipeline render_pipeline(PipelineId id) noexcept {
  const std::shared_ptr<SlotState> slot = find_slot(id);
  return slot != nullptr && slot->status.load(std::memory_order_acquire) == PipelineStatus::Ready
             ? slot->render
             : wgpu::RenderPipeline{};
}

PipelineStatus wait(PipelineId id, uint64_t timeoutNs) {
  const std::shared_ptr<SlotState> slot = find_slot(id);
  if (slot == nullptr) {
    return PipelineStatus::Failed;
  }
  g_waiters.fetch_add(1, std::memory_order_relaxed);
  block_until(timeoutNs, [&] { return slot->status.load(std::memory_order_acquire) != PipelineStatus::Pending; });
  g_waiters.fetch_sub(1, std::memory_order_relaxed);
  return slot->status.load(std::memory_order_acquire);
}

bool wait_idle(uint64_t timeoutNs) {
  return block_until(timeoutNs, [] { return g_pendingCount.load(std::memory_order_relaxed) == 0; });
}

PipelineStats stats() noexcept {
  return {
      .pending = g_pendingCount.load(std::memory_order_relaxed),
      .ready = g_readyCount.load(std::memory_order_relaxed),
      .failed = g_failedCount.load(std::memory_order_relaxed),
      .modules = g_moduleCount.load(std::memory_order_relaxed),
  };
}

std::vector<PipelineFailure> failures() {
  std::lock_guard lock{g_mutex};
  return {g_failures.begin(), g_failures.end()};
}

bool validate_module(std::string_view id, std::span<const std::string_view> defines, std::string& error) {
  error.clear();
  if (!webgpu::g_device) {
    error = "no device";
    return false;
  }
  ResolvedSource source;
  if (!resolve(id, defines, source, error)) {
    return false;
  }
  std::lock_guard device{g_deviceMutex};
  return static_cast<bool>(create_module_locked(std::string{id}, source, error));
}

namespace testing {
uint32_t waiters() noexcept { return g_waiters.load(std::memory_order_relaxed); }
} // namespace testing
} // namespace aurora::render::shaders
