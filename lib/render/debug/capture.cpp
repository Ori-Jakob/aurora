#include "capture.hpp"

#include "../frame_state.hpp"
#include "../graph_bind.hpp"
#include "../jobs.hpp"
#include "../json.hpp"
#include "../pass_context.hpp"
#include "../../gfx/pipeline_cache.hpp"
#include "../../gfx/png_io.hpp"
#include "../../internal.hpp"
#include "../../io.hpp"
#include "../../logging.hpp"
#include "../../webgpu/gpu.hpp"

#include <SDL3/SDL_stdinc.h>
#include <fmt/format.h>
#include <magic_enum.hpp>

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <mutex>
#include <thread>
#include <utility>

namespace aurora::render::debug::capture {
namespace {
constexpr Module Log{"aurora::render::capture"};
constexpr TexRef Source{0};
constexpr uint32_t CopyRowAlignment = 256;
constexpr auto ShutdownWait = std::chrono::seconds{2};
constexpr auto ShutdownPoll = std::chrono::milliseconds{1};
constexpr std::array<size_t, MaxRequests> Indices{0, 1, 2, 3};
constexpr std::array<std::string_view, 8> StatusNames{
    "Idle", "Waiting", "Armed", "Written", "Unavailable", "MissingInput", "TimedOut", "Failed",
};

enum class SlotState : uint8_t { Available, CopySubmitted, MapPending };

struct Readback {
  wgpu::Buffer buffer;
  uint64_t size = 0;
  SlotState state = SlotState::Available;
  uint64_t frame = 0;
  CopyLayout layout;
  wgpu::TextureFormat format = wgpu::TextureFormat::Undefined;
};

struct Entry {
  Request request;
  Status status = Status::Idle;
  std::string message;
  uint64_t generation = 0;
  uint64_t dropped = 0;
  std::array<Readback, ReadbackSlots> slots;
  size_t nextSlot = 0;
  std::optional<Capture> latest;
  bool done = false;
  Placement placement;
  detail::PassId pass = detail::InvalidPassId;
  uint32_t resolves = 0;
  uint32_t latches = 0;
};

std::mutex g_mutex;
std::array<Entry, MaxRequests> g_entries;
std::array<std::atomic_bool, MaxRequests> g_armed{};
uint64_t g_nextGeneration = 1;
uint32_t g_latch = 0;

constexpr bool final_status(Status status) noexcept {
  return status == Status::Written || status == Status::Unavailable || status == Status::MissingInput ||
         status == Status::TimedOut || status == Status::Failed;
}

std::string label(const Request& request) { return request.resource + "@" + request.anchor; }

char sanitized(char c) noexcept {
  const bool keep = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' ||
                    c == '_' || c == '-' || c == '@';
  return keep ? c : '_';
}

float half_to_float(uint16_t half) noexcept {
  const uint32_t sign = uint32_t{half & 0x8000u} << 16;
  const uint32_t exponent = (half >> 10) & 0x1Fu;
  const uint32_t mantissa = half & 0x3FFu;
  if (exponent == 0) {
    const float value = std::ldexp(static_cast<float>(mantissa), -24);
    return sign != 0 ? -value : value;
  }
  const uint32_t bits =
      exponent == 0x1F ? sign | 0x7F800000u | (mantissa << 13) : sign | ((exponent + 112) << 23) | (mantissa << 13);
  return std::bit_cast<float>(bits);
}

uint32_t load_u32(const uint8_t* bytes) noexcept {
  uint32_t value = 0;
  std::memcpy(&value, bytes, sizeof(value));
  return value;
}

uint16_t load_u16(const uint8_t* bytes) noexcept {
  uint16_t value = 0;
  std::memcpy(&value, bytes, sizeof(value));
  return value;
}

uint32_t bytes_per_pixel(wgpu::TextureFormat format) noexcept {
  switch (format) {
  case wgpu::TextureFormat::R8Unorm:
    return 1;
  case wgpu::TextureFormat::BGRA8Unorm:
  case wgpu::TextureFormat::RGBA8Unorm:
  case wgpu::TextureFormat::RGB10A2Unorm:
  case wgpu::TextureFormat::RG16Float:
  case wgpu::TextureFormat::RGBA8Uint:
  case wgpu::TextureFormat::R32Uint:
  case wgpu::TextureFormat::R32Float:
    return 4;
  case wgpu::TextureFormat::RGBA16Float:
    return 8;
  case wgpu::TextureFormat::RGBA32Uint:
    return 16;
  default:
    return 0;
  }
}

uint8_t unorm8(float value) noexcept {
  return static_cast<uint8_t>(std::lround(std::clamp(std::isfinite(value) ? value : 0.f, 0.f, 1.f) * 255.f));
}

std::array<uint8_t, 3> motion_color(float x, float y, uint32_t width, uint32_t height) noexcept {
  if (!std::isfinite(x) || !std::isfinite(y)) {
    return {255, 0, 255};
  }
  const float px = x * 0.5f * static_cast<float>(width);
  const float py = -y * 0.5f * static_cast<float>(height);
  return {unorm8(0.5f + px / 32.f), unorm8(0.5f + py / 32.f), 128};
}

double rounded(double value) noexcept { return std::round(value * 1000.0) / 1000.0; }

std::filesystem::path directory(const Request& request) {
  return io::fs_path_from_string(request.dir.empty() ? default_dir() : request.dir);
}

void unregister(Entry& entry, size_t index) noexcept {
  if (entry.pass != detail::InvalidPassId) {
    detail::unregister_pass(entry.pass);
    entry.pass = detail::InvalidPassId;
  }
  g_armed[index].store(false, std::memory_order_relaxed);
}

void set_status(size_t index, Status status, std::string message) {
  std::lock_guard lock{g_mutex};
  g_entries[index].status = status;
  g_entries[index].message = std::move(message);
}

void accept(size_t index, uint64_t generation, Capture capture) {
  Request request;
  {
    std::lock_guard lock{g_mutex};
    Entry& entry = g_entries[index];
    if (entry.generation != generation || entry.done) {
      return;
    }
    if (entry.request.latest) {
      if (!entry.latest || capture.frame >= entry.latest->frame) {
        entry.latest = std::move(capture);
      }
      return;
    }
    entry.done = true;
    request = entry.request;
  }
  g_armed[index].store(false, std::memory_order_relaxed);
  const std::string stem = file_stem(request.resource, request.anchor, capture.frame);
  jobs::submit(
      [index, generation, request = std::move(request), stem, capture = std::move(capture)](const jobs::JobContext&) {
        const std::filesystem::path dir = directory(request);
        const bool written = write_capture(dir, stem, capture);
        const std::string path = io::fs_path_to_string(dir / stem);
        std::lock_guard lock{g_mutex};
        Entry& entry = g_entries[index];
        if (entry.generation != generation) {
          return;
        }
        entry.status = written ? Status::Written : Status::Failed;
        entry.message = written ? fmt::format("wrote frame {} to {}.*", capture.frame, path)
                                : fmt::format("failed to write {}.*", path);
      });
}

std::vector<uint8_t> unpad(const uint8_t* mapped, const CopyLayout& layout) {
  std::vector<uint8_t> bytes(uint64_t{layout.rowBytes} * layout.height);
  for (uint32_t y = 0; y < layout.height; ++y) {
    std::memcpy(bytes.data() + uint64_t{y} * layout.rowBytes, mapped + uint64_t{y} * layout.paddedRowBytes,
                layout.rowBytes);
  }
  return bytes;
}

void complete(size_t index, size_t slotIndex, uint64_t generation, const wgpu::Buffer& buffer,
              wgpu::MapAsyncStatus status) {
  std::optional<Capture> capture;
  CopyLayout layout;
  {
    std::lock_guard lock{g_mutex};
    const Entry& entry = g_entries[index];
    const Readback& slot = entry.slots[slotIndex];
    if (entry.generation == generation && slot.buffer.Get() == buffer.Get()) {
      layout = slot.layout;
      capture = Capture{.resource = entry.request.resource,
                        .anchor = entry.request.anchor,
                        .frame = slot.frame,
                        .format = slot.format,
                        .width = layout.width,
                        .height = layout.height};
    }
  }
  if (status == wgpu::MapAsyncStatus::Success) {
    if (capture) {
      if (const auto* mapped = static_cast<const uint8_t*>(buffer.GetConstMappedRange(0, layout.bufferSize))) {
        capture->bytes = unpad(mapped, layout);
      }
    }
    buffer.Unmap();
  }
  {
    std::lock_guard lock{g_mutex};
    Readback& slot = g_entries[index].slots[slotIndex];
    if (slot.buffer.Get() == buffer.Get()) {
      slot.state = SlotState::Available;
    }
  }
  if (capture && !capture->bytes.empty()) {
    accept(index, generation, std::move(*capture));
  }
}

void map_slot(size_t index, size_t slotIndex, uint64_t generation, const wgpu::Buffer& buffer, uint64_t size) {
  buffer.MapAsync(wgpu::MapMode::Read, 0, size, wgpu::CallbackMode::AllowSpontaneous,
                  [index, slotIndex, generation, buffer](wgpu::MapAsyncStatus status, wgpu::StringView) {
                    complete(index, slotIndex, generation, buffer, status);
                  });
}

void declare_capture(detail::InternalPassBuilder& builder, const detail::PassSetup&, const void* state) {
  const size_t index = *static_cast<const size_t*>(state);
  const Entry& entry = g_entries[index];
  std::memcpy(builder.local().data(), &entry.generation, sizeof(entry.generation));
  builder.after(entry.placement.after);
  if (!entry.placement.before.empty()) {
    builder.before(entry.placement.before);
  }
  builder.use(entry.request.resource, Access::CopySrc, Missing::DisablePass);
  builder.side_effect();
}

size_t free_slot(const Entry& entry) noexcept {
  for (size_t i = 0; i < ReadbackSlots; ++i) {
    const size_t candidate = (entry.nextSlot + i) % ReadbackSlots;
    if (entry.slots[candidate].state == SlotState::Available) {
      return candidate;
    }
  }
  return ReadbackSlots;
}

void wait_for_slot(size_t index) {
  const auto start = std::chrono::steady_clock::now();
  while (std::chrono::steady_clock::now() - start < ShutdownWait) {
    webgpu::g_instance.ProcessEvents();
    {
      std::lock_guard lock{g_mutex};
      if (free_slot(g_entries[index]) != ReadbackSlots) {
        return;
      }
    }
    std::this_thread::sleep_for(ShutdownPoll);
  }
}

bool capture_enabled(const detail::StageData&, const void* state) {
  return g_armed[*static_cast<const size_t*>(state)].load(std::memory_order_relaxed);
}

void encode_capture(detail::PassContext& ctx, const void* state) {
  const size_t index = *static_cast<const size_t*>(state);
  uint64_t generation = 0;
  std::memcpy(&generation, ctx.local().data(), sizeof(generation));
  const wgpu::Texture texture = ctx.texture(Source);
  if (!texture) {
    return;
  }
  const wgpu::TextureFormat format = texture.GetFormat();
  const std::optional<CopyLayout> layout = copy_layout(format, texture.GetWidth(), texture.GetHeight());
  size_t slotIndex = ReadbackSlots;
  wgpu::Buffer buffer;
  {
    std::unique_lock lock{g_mutex};
    Entry& entry = g_entries[index];
    if (entry.generation != generation || entry.done) {
      return;
    }
    if (!layout) {
      entry.status = Status::Failed;
      entry.message = fmt::format("{} cannot be read back", magic_enum::enum_name(format));
      entry.done = true;
      return;
    }
    slotIndex = free_slot(entry);
    if (slotIndex == ReadbackSlots && gfx::blocking_pipelines()) {
      lock.unlock();
      wait_for_slot(index);
      lock.lock();
      if (entry.generation != generation || entry.done) {
        return;
      }
      slotIndex = free_slot(entry);
    }
    if (slotIndex == ReadbackSlots) {
      ++entry.dropped;
      return;
    }
    Readback& slot = entry.slots[slotIndex];
    if (!slot.buffer || slot.size != layout->bufferSize) {
      const wgpu::BufferDescriptor descriptor{
          .label = "Render capture readback",
          .usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst,
          .size = layout->bufferSize,
      };
      slot.buffer = ctx.device().CreateBuffer(&descriptor);
      slot.size = layout->bufferSize;
    }
    slot.state = SlotState::CopySubmitted;
    slot.frame = ctx.frame_id();
    slot.layout = *layout;
    slot.format = format;
    buffer = slot.buffer;
    entry.nextSlot = (slotIndex + 1) % ReadbackSlots;
  }
  const wgpu::TexelCopyTextureInfo source{.texture = texture};
  const wgpu::TexelCopyBufferInfo destination{
      .layout = {.bytesPerRow = layout->paddedRowBytes, .rowsPerImage = layout->height},
      .buffer = buffer,
  };
  const wgpu::Extent3D extent{layout->width, layout->height, 1};
  ctx.encoder().CopyTextureToBuffer(&source, &destination, &extent);
  detail::queue_after_frame([index, slotIndex, generation, buffer, size = layout->bufferSize] {
    {
      std::lock_guard lock{g_mutex};
      Readback& slot = g_entries[index].slots[slotIndex];
      if (slot.buffer.Get() != buffer.Get() || slot.state != SlotState::CopySubmitted) {
        return;
      }
      slot.state = SlotState::MapPending;
    }
    map_slot(index, slotIndex, generation, buffer, size);
  });
}

detail::PassDesc capture_pass_desc(size_t index, const Placement& placement) {
  return {
      .name = std::string{PassNames[index]},
      .stage = placement.stage,
      .kind = PassKind::Copy,
      .cls = detail::PassClass::Optional,
      .order = placement.order,
      .declare = declare_capture,
      .enabled = capture_enabled,
      .encode = encode_capture,
      .state = &Indices[index],
  };
}

std::string cull_message(const detail::CompiledGraph& graph, detail::NodeId node) {
  for (const detail::Diagnostic& diagnostic : graph.diagnostics) {
    if (diagnostic.node == node && !diagnostic.message.empty()) {
      return diagnostic.message;
    }
  }
  return std::string{detail::cull_reason_name(graph.nodes[node].status)};
}

void resolve_entry(size_t index, const detail::CompiledGraph* graph) {
  Entry& entry = g_entries[index];
  Status status;
  Request request;
  bool done = false;
  {
    std::lock_guard lock{g_mutex};
    status = entry.status;
    request = entry.request;
    done = entry.done;
  }
  if (status == Status::Idle) {
    return;
  }
  if (final_status(status)) {
    if (entry.pass != detail::InvalidPassId) {
      unregister(entry, index);
    }
    return;
  }
  if (entry.pass == detail::InvalidPassId) {
    ++entry.resolves;
    Placement placement;
    switch (place(request.anchor, graph, placement)) {
    case PlaceResult::Unavailable:
      set_status(index, Status::Unavailable, fmt::format("nothing runs after {}", request.anchor));
      return;
    case PlaceResult::Wait:
      if (entry.resolves >= ResolveLimit) {
        set_status(index, Status::Unavailable, fmt::format("pass '{}' never appeared in the graph", request.anchor));
      } else {
        set_status(index, Status::Waiting,
                   graph == nullptr ? std::string{"waiting for a compiled graph"}
                                    : fmt::format("waiting for pass '{}'", request.anchor));
      }
      return;
    case PlaceResult::Placed:
      break;
    }
    entry.placement = std::move(placement);
    entry.latches = 0;
    entry.pass = detail::register_pass(capture_pass_desc(index, entry.placement));
    Log.info("Capture {}: registered in {} at latch {}", label(request), stage_name(entry.placement.stage), g_latch);
    if (request.latest) {
      g_armed[index].store(true, std::memory_order_relaxed);
      set_status(index, Status::Armed, "capturing every frame; the latest is written at shutdown");
    } else {
      set_status(index, Status::Waiting, fmt::format("arming in {} frames", ArmDelay));
    }
    return;
  }
  if (graph != nullptr) {
    const detail::NodeId node = graph->find_node(PassNames[index]);
    if (node != detail::InvalidNode && graph->nodes[node].id == entry.pass &&
        graph->nodes[node].status != detail::CullReason::None) {
      const detail::CullReason reason = graph->nodes[node].status;
      const bool missing = reason == detail::CullReason::MissingInput || reason == detail::CullReason::MissingTarget;
      unregister(entry, index);
      set_status(index, missing ? Status::MissingInput : Status::Unavailable, cull_message(*graph, node));
      return;
    }
  }
  ++entry.latches;
  if (request.latest || done) {
    return;
  }
  if (status == Status::Waiting) {
    // Registering changes the graph hash, which resets histories, so temporal output needs a few frames back
    if (entry.latches >= ArmDelay) {
      entry.latches = 0;
      g_armed[index].store(true, std::memory_order_relaxed);
      set_status(index, Status::Armed, "waiting for the next frame");
    } else {
      set_status(index, Status::Waiting, fmt::format("arming in {} frames", ArmDelay - entry.latches));
    }
    return;
  }
  if (entry.latches >= ArmedLimit) {
    unregister(entry, index);
    set_status(index, Status::TimedOut, fmt::format("no frame reached {} in {} frames", request.anchor, ArmedLimit));
  }
}

bool slots_busy() {
  std::lock_guard lock{g_mutex};
  return std::ranges::any_of(g_entries, [](const Entry& entry) {
    return std::ranges::any_of(entry.slots, [](const Readback& slot) { return slot.state != SlotState::Available; });
  });
}

void drain_readbacks() {
  if (!webgpu::g_device || !webgpu::g_instance) {
    return;
  }
  for (size_t index = 0; index < MaxRequests; ++index) {
    for (size_t slotIndex = 0; slotIndex < ReadbackSlots; ++slotIndex) {
      wgpu::Buffer buffer;
      uint64_t generation = 0;
      uint64_t size = 0;
      {
        std::lock_guard lock{g_mutex};
        Readback& slot = g_entries[index].slots[slotIndex];
        if (slot.state != SlotState::CopySubmitted) {
          continue;
        }
        slot.state = SlotState::MapPending;
        buffer = slot.buffer;
        generation = g_entries[index].generation;
        size = slot.size;
      }
      map_slot(index, slotIndex, generation, buffer, size);
    }
  }
  const auto start = std::chrono::steady_clock::now();
  while (slots_busy() && std::chrono::steady_clock::now() - start < ShutdownWait) {
    webgpu::g_instance.ProcessEvents();
    std::this_thread::sleep_for(ShutdownPoll);
  }
}

void write_latest() {
  for (Entry& entry : g_entries) {
    std::optional<Capture> capture;
    Request request;
    Status status;
    {
      std::lock_guard lock{g_mutex};
      if (!entry.request.latest || entry.status == Status::Idle) {
        continue;
      }
      capture = std::move(entry.latest);
      request = entry.request;
      status = entry.status;
    }
    if (!capture) {
      Log.warn("Capture {}: no frame was captured ({})", label(request), status_name(status));
      continue;
    }
    const std::filesystem::path dir = directory(request);
    const std::string stem = file_stem(request.resource, request.anchor, std::nullopt);
    if (write_capture(dir, stem, *capture)) {
      Log.info("Capture {}: wrote frame {} to {}.*", label(request), capture->frame, io::fs_path_to_string(dir / stem));
    } else {
      Log.warn("Capture {}: failed to write {}.*", label(request), io::fs_path_to_string(dir / stem));
    }
  }
}
} // namespace

std::string_view status_name(Status status) noexcept {
  const auto index = static_cast<size_t>(status);
  return index < StatusNames.size() ? StatusNames[index] : "Unknown";
}

std::vector<Request> parse_env(std::string_view text, std::string_view dir) {
  std::vector<Request> requests;
  while (!text.empty()) {
    const size_t end = text.find(';');
    const std::string_view item = text.substr(0, end);
    text = end == std::string_view::npos ? std::string_view{} : text.substr(end + 1);
    const size_t at = item.find('@');
    if (at == std::string_view::npos || at == 0 || at + 1 == item.size()) {
      if (!item.empty()) {
        Log.warn("AURORA_RENDER_CAPTURE: '{}' is not <resource>@<anchor>; ignored", item);
      }
      continue;
    }
    requests.push_back({
        .resource = std::string{item.substr(0, at)},
        .anchor = std::string{item.substr(at + 1)},
        .latest = true,
        .dir = std::string{dir},
    });
  }
  return requests;
}

std::optional<CopyLayout> copy_layout(wgpu::TextureFormat format, uint32_t width, uint32_t height) noexcept {
  const uint32_t bpp = bytes_per_pixel(format);
  if (bpp == 0 || width == 0 || height == 0) {
    return std::nullopt;
  }
  const uint32_t rowBytes = width * bpp;
  const uint32_t padded = (rowBytes + CopyRowAlignment - 1) / CopyRowAlignment * CopyRowAlignment;
  return CopyLayout{
      .width = width,
      .height = height,
      .bytesPerPixel = bpp,
      .rowBytes = rowBytes,
      .paddedRowBytes = padded,
      .bufferSize = uint64_t{padded} * height,
  };
}

FogSummary summarize_fog(std::span<const uint8_t> bytes) {
  FogSummary summary;
  for (size_t i = 0; i + 4 <= bytes.size(); i += 4) {
    ++summary.records;
    const uint32_t rgb = uint32_t{bytes[i]} << 16 | uint32_t{bytes[i + 1]} << 8 | bytes[i + 2];
    const uint8_t a = bytes[i + 3];
    if (rgb == 0 && a == 0) {
      continue;
    }
    ++summary.nonZero;
    const uint32_t q = a >> 1;
    ++summary.qHistogram[q];
    summary.withFog += q != 0 ? 1 : 0;
    summary.locked += (a & 1u) != 0 ? 1 : 0;
    if (std::ranges::find(summary.distinct, rgb) == summary.distinct.end()) {
      if (summary.distinct.size() < MaxDistinctColors) {
        summary.distinct.push_back(rgb);
      } else {
        summary.distinctOverflow = true;
      }
    }
  }
  return summary;
}

MotionSummary summarize_motion(std::span<const uint8_t> bytes, uint32_t width, uint32_t height) {
  MotionSummary summary;
  uint64_t counted = 0;
  double total = 0.0;
  for (size_t i = 0; i + 4 <= bytes.size(); i += 4) {
    const uint16_t hx = load_u16(&bytes[i]);
    const uint16_t hy = load_u16(&bytes[i + 2]);
    if ((hx & 0x7FFFu) == 0 && (hy & 0x7FFFu) == 0) {
      continue;
    }
    ++summary.nonZero;
    const float x = half_to_float(hx);
    const float y = half_to_float(hy);
    if (!std::isfinite(x) || !std::isfinite(y) || std::abs(x) >= 2.f || std::abs(y) >= 2.f) {
      continue;
    }
    const double px = x * 0.5 * width;
    const double py = y * 0.5 * height;
    const double length = std::sqrt(px * px + py * py);
    summary.maxPx = std::max(summary.maxPx, length);
    total += length;
    ++counted;
  }
  summary.maxPx = rounded(summary.maxPx);
  summary.meanPx = counted != 0 ? rounded(total / static_cast<double>(counted)) : 0.0;
  return summary;
}

NormalSummary summarize_normal(std::span<const uint8_t> bytes) {
  NormalSummary summary;
  for (size_t i = 0; i + 4 <= bytes.size(); i += 4) {
    ++summary.codeCounts[load_u32(&bytes[i]) >> 30];
  }
  return summary;
}

std::string header_json(const Capture& capture, std::string_view binName) {
  const json::MutDoc doc{yyjson_mut_doc_new(nullptr)};
  if (!doc) {
    return {};
  }
  yyjson_mut_doc* d = doc.get();
  yyjson_mut_val* root = yyjson_mut_obj(d);
  yyjson_mut_doc_set_root(d, root);
  const auto add_str = [d](yyjson_mut_val* obj, const char* key, std::string_view value) {
    yyjson_mut_obj_add_strncpy(d, obj, key, value.data(), value.size());
  };
  const uint32_t bpp = bytes_per_pixel(capture.format);
  add_str(root, "schema", Schema);
  yyjson_mut_obj_add_uint(d, root, "version", 1);
  add_str(root, "resource", capture.resource);
  add_str(root, "anchor", capture.anchor);
  yyjson_mut_obj_add_uint(d, root, "frame", capture.frame);
  add_str(root, "format", magic_enum::enum_name(capture.format));
  yyjson_mut_obj_add_uint(d, root, "width", capture.width);
  yyjson_mut_obj_add_uint(d, root, "height", capture.height);
  yyjson_mut_obj_add_uint(d, root, "bytesPerPixel", bpp);
  yyjson_mut_obj_add_uint(d, root, "rowBytes", uint64_t{capture.width} * bpp);
  add_str(root, "bin", binName);
  switch (capture.format) {
  case wgpu::TextureFormat::RGBA8Uint: {
    const FogSummary fog = summarize_fog(capture.bytes);
    yyjson_mut_val* obj = yyjson_mut_obj_add_obj(d, root, "fog");
    yyjson_mut_obj_add_uint(d, obj, "records", fog.records);
    yyjson_mut_obj_add_uint(d, obj, "nonZero", fog.nonZero);
    yyjson_mut_obj_add_uint(d, obj, "withFog", fog.withFog);
    yyjson_mut_obj_add_uint(d, obj, "locked", fog.locked);
    yyjson_mut_val* distinct = yyjson_mut_obj_add_arr(d, obj, "distinct");
    for (const uint32_t rgb : fog.distinct) {
      const std::string text = fmt::format("#{:06x}", rgb);
      yyjson_mut_arr_add_strncpy(d, distinct, text.data(), text.size());
    }
    yyjson_mut_obj_add_bool(d, obj, "distinctOverflow", fog.distinctOverflow);
    yyjson_mut_val* histogram = yyjson_mut_obj_add_arr(d, obj, "qHistogram");
    for (const uint64_t count : fog.qHistogram) {
      yyjson_mut_arr_add_uint(d, histogram, count);
    }
    break;
  }
  case wgpu::TextureFormat::RG16Float: {
    const MotionSummary motion = summarize_motion(capture.bytes, capture.width, capture.height);
    yyjson_mut_val* obj = yyjson_mut_obj_add_obj(d, root, "motion");
    yyjson_mut_obj_add_uint(d, obj, "nonZero", motion.nonZero);
    yyjson_mut_obj_add_real(d, obj, "maxPx", motion.maxPx);
    yyjson_mut_obj_add_real(d, obj, "meanPx", motion.meanPx);
    break;
  }
  case wgpu::TextureFormat::RGB10A2Unorm: {
    const NormalSummary normal = summarize_normal(capture.bytes);
    yyjson_mut_val* obj = yyjson_mut_obj_add_obj(d, root, "normal");
    yyjson_mut_val* counts = yyjson_mut_obj_add_arr(d, obj, "codeCounts");
    for (const uint64_t count : normal.codeCounts) {
      yyjson_mut_arr_add_uint(d, counts, count);
    }
    break;
  }
  default:
    break;
  }
  return json::write(d, true);
}

std::vector<uint8_t> preview_rgba(const Capture& capture) {
  const uint32_t bpp = bytes_per_pixel(capture.format);
  const uint64_t pixels = uint64_t{capture.width} * capture.height;
  if (bpp == 0 || capture.bytes.size() < pixels * bpp) {
    return {};
  }
  std::vector<uint8_t> out(pixels * 4, 255);
  for (uint64_t i = 0; i < pixels; ++i) {
    const uint8_t* in = &capture.bytes[i * bpp];
    uint8_t* px = &out[i * 4];
    std::array<uint8_t, 3> rgb{};
    switch (capture.format) {
    case wgpu::TextureFormat::RGBA8Unorm:
      rgb = {in[0], in[1], in[2]};
      break;
    case wgpu::TextureFormat::BGRA8Unorm:
      rgb = {in[2], in[1], in[0]};
      break;
    case wgpu::TextureFormat::RGB10A2Unorm: {
      const uint32_t v = load_u32(in);
      rgb = {static_cast<uint8_t>((v >> 2) & 0xFF), static_cast<uint8_t>((v >> 12) & 0xFF),
             static_cast<uint8_t>((v >> 22) & 0xFF)};
      break;
    }
    case wgpu::TextureFormat::RG16Float:
      rgb = motion_color(half_to_float(load_u16(in)), half_to_float(load_u16(in + 2)), capture.width, capture.height);
      break;
    case wgpu::TextureFormat::R8Unorm:
      rgb = {in[0], in[0], in[0]};
      break;
    case wgpu::TextureFormat::RGBA8Uint:
      rgb = in[0] == 0 && in[1] == 0 && in[2] == 0 && in[3] == 0 ? std::array<uint8_t, 3>{0, 0, 64}
                                                                 : std::array<uint8_t, 3>{in[0], in[1], in[2]};
      break;
    case wgpu::TextureFormat::R32Uint:
    case wgpu::TextureFormat::RGBA32Uint:
      rgb = {in[0], in[bpp == 4 ? 1 : 4], in[bpp == 4 ? 2 : 8]};
      break;
    case wgpu::TextureFormat::R32Float: {
      const uint8_t v = unorm8(std::bit_cast<float>(load_u32(in)));
      rgb = {v, v, v};
      break;
    }
    case wgpu::TextureFormat::RGBA16Float:
      rgb = {unorm8(half_to_float(load_u16(in))), unorm8(half_to_float(load_u16(in + 2))),
             unorm8(half_to_float(load_u16(in + 4)))};
      break;
    default:
      break;
    }
    px[0] = rgb[0];
    px[1] = rgb[1];
    px[2] = rgb[2];
  }
  return out;
}

std::string file_stem(std::string_view resource, std::string_view anchor, std::optional<uint64_t> frame) {
  std::string stem = std::string{resource} + "@" + std::string{anchor};
  std::ranges::transform(stem, stem.begin(), sanitized);
  if (frame) {
    stem += fmt::format("_{}", *frame);
  }
  return stem;
}

bool write_capture(const std::filesystem::path& dir, std::string_view stem, const Capture& capture) {
  if (!io::create_directories(dir)) {
    return false;
  }
  const std::string base{stem};
  const std::string binName = base + ".bin";
  const std::string header = header_json(capture, binName);
  const std::vector<uint8_t> preview = preview_rgba(capture);
  const bool bin = io::write_file_atomic(dir / binName, capture.bytes);
  const bool json =
      !header.empty() &&
      io::write_file_atomic(dir / (base + ".json"), {reinterpret_cast<const uint8_t*>(header.data()), header.size()});
  const bool png =
      !preview.empty() && gfx::png::write_rgba8_png(dir / (base + ".png"), capture.width, capture.height, preview);
  return bin && json && png;
}

PlaceResult place(std::string_view anchor, const detail::CompiledGraph* graph, Placement& out) {
  if (const std::optional<Stage> stage = detail::chain_node_stage(anchor)) {
    if (*stage == Stage::AfterFrame) {
      return PlaceResult::Unavailable;
    }
    out = {.stage = *stage, .after = std::string{anchor}, .before = {}, .order = ChainOrder};
    return PlaceResult::Placed;
  }
  // The first latch runs before any graph exists, so pass anchors wait for one
  if (graph == nullptr) {
    return PlaceResult::Wait;
  }
  const detail::NodeId node = graph->find_node(anchor);
  if (node == detail::InvalidNode || graph->nodes[node].kind != detail::NodeKind::Pass) {
    return PlaceResult::Wait;
  }
  const detail::CompiledNode& pass = graph->nodes[node];
  if (pass.stage == Stage::AfterFrame) {
    return PlaceResult::Unavailable;
  }
  out = {.stage = pass.stage, .after = std::string{anchor}, .before = {}, .order = pass.order};
  for (const detail::Edge& edge : graph->edges) {
    if (edge.from == node && edge.kind == detail::EdgeKind::Bound) {
      out.before = graph->nodes[edge.to].name;
      break;
    }
  }
  return PlaceResult::Placed;
}

std::string default_dir() {
  if (const char* dir = SDL_getenv("AURORA_RENDER_CAPTURE_DIR"); dir != nullptr && *dir != '\0') {
    return dir;
  }
  const auto base = io::fs_path_from_string(g_config.cachePath != nullptr ? g_config.cachePath : ".");
  return io::fs_path_to_string(base / "captures");
}

std::optional<size_t> request(Request request) {
  std::lock_guard lock{g_mutex};
  for (size_t index = 0; index < MaxRequests; ++index) {
    Entry& entry = g_entries[index];
    if ((entry.status != Status::Idle && !final_status(entry.status)) || entry.pass != detail::InvalidPassId) {
      continue;
    }
    entry.request = std::move(request);
    entry.status = Status::Waiting;
    entry.message = "waiting for the next latch";
    entry.generation = g_nextGeneration++;
    entry.dropped = 0;
    entry.latest.reset();
    entry.done = false;
    entry.resolves = 0;
    entry.latches = 0;
    return index;
  }
  return std::nullopt;
}

std::vector<EntryInfo> entries() {
  std::vector<EntryInfo> out;
  std::lock_guard lock{g_mutex};
  for (const Entry& entry : g_entries) {
    if (entry.status != Status::Idle) {
      out.push_back(
          {.request = entry.request, .status = entry.status, .message = entry.message, .dropped = entry.dropped});
    }
  }
  return out;
}

void initialize() {
  const char* value = SDL_getenv("AURORA_RENDER_CAPTURE");
  if (value == nullptr || *value == '\0') {
    return;
  }
  const std::string dir = default_dir();
  std::vector<Request> requests = parse_env(value, dir);
  Log.warn("AURORA_RENDER_CAPTURE: capturing {} target(s) into {}", requests.size(), dir);
  for (Request& entry : requests) {
    const std::string name = label(entry);
    if (!request(std::move(entry))) {
      Log.warn("AURORA_RENDER_CAPTURE: '{}' is past the {} capture limit; ignored", name, MaxRequests);
    }
  }
}

void shutdown() noexcept {
  for (size_t index = 0; index < MaxRequests; ++index) {
    unregister(g_entries[index], index);
  }
  drain_readbacks();
  write_latest();
  testing::reset();
}

void on_latch() {
  ++g_latch;
  const std::shared_ptr<const detail::FrameState>& frame = detail::current_frame_state();
  resolve(frame != nullptr ? frame->graph.get() : nullptr);
}

void resolve(const detail::CompiledGraph* graph) {
  for (size_t index = 0; index < MaxRequests; ++index) {
    resolve_entry(index, graph);
  }
}

namespace testing {
detail::PassId pass_id(size_t index) noexcept { return index < MaxRequests ? g_entries[index].pass : 0; }

void deliver(size_t index, Capture capture) {
  uint64_t generation = 0;
  {
    std::lock_guard lock{g_mutex};
    generation = g_entries[index].generation;
  }
  accept(index, generation, std::move(capture));
}

std::optional<Capture> read_texture(const wgpu::Texture& texture, uint64_t frame) {
  const std::optional<CopyLayout> layout = copy_layout(texture.GetFormat(), texture.GetWidth(), texture.GetHeight());
  if (!layout || !webgpu::g_device) {
    return std::nullopt;
  }
  const wgpu::BufferDescriptor descriptor{
      .label = "Render capture readback",
      .usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst,
      .size = layout->bufferSize,
  };
  const wgpu::Buffer buffer = webgpu::g_device.CreateBuffer(&descriptor);
  const wgpu::CommandEncoder encoder = webgpu::g_device.CreateCommandEncoder();
  const wgpu::TexelCopyTextureInfo source{.texture = texture};
  const wgpu::TexelCopyBufferInfo destination{
      .layout = {.bytesPerRow = layout->paddedRowBytes, .rowsPerImage = layout->height},
      .buffer = buffer,
  };
  const wgpu::Extent3D extent{layout->width, layout->height, 1};
  encoder.CopyTextureToBuffer(&source, &destination, &extent);
  const wgpu::CommandBuffer commands = encoder.Finish();
  webgpu::g_queue.Submit(1, &commands);
  std::atomic_bool finished{false};
  std::optional<Capture> capture;
  buffer.MapAsync(wgpu::MapMode::Read, 0, layout->bufferSize, wgpu::CallbackMode::AllowSpontaneous,
                  [&](wgpu::MapAsyncStatus status, wgpu::StringView) {
                    if (status == wgpu::MapAsyncStatus::Success) {
                      const auto* mapped =
                          static_cast<const uint8_t*>(buffer.GetConstMappedRange(0, layout->bufferSize));
                      if (mapped != nullptr) {
                        capture = Capture{.frame = frame,
                                          .format = texture.GetFormat(),
                                          .width = layout->width,
                                          .height = layout->height,
                                          .bytes = unpad(mapped, *layout)};
                      }
                      buffer.Unmap();
                    }
                    finished.store(true, std::memory_order_release);
                  });
  const auto start = std::chrono::steady_clock::now();
  while (!finished.load(std::memory_order_acquire) && std::chrono::steady_clock::now() - start < ShutdownWait) {
    webgpu::g_instance.ProcessEvents();
    std::this_thread::sleep_for(ShutdownPoll);
  }
  return finished.load(std::memory_order_acquire) ? std::move(capture) : std::nullopt;
}

void reset() noexcept {
  std::lock_guard lock{g_mutex};
  for (size_t index = 0; index < MaxRequests; ++index) {
    Entry& entry = g_entries[index];
    if (entry.pass != detail::InvalidPassId) {
      detail::unregister_pass(entry.pass);
    }
    entry = {};
    g_armed[index].store(false, std::memory_order_relaxed);
  }
  g_latch = 0;
}
} // namespace testing
} // namespace aurora::render::debug::capture
