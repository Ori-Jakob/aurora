#include "companion_cache.hpp"

#include "../internal.hpp"
#include "../webgpu/gpu.hpp"
#include "texture.hpp"

#include <absl/container/flat_hash_map.h>
#include <fmt/format.h>
#include <tracy/Tracy.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

namespace aurora::gfx::companion_cache {
namespace {
constexpr Module Log{"aurora::gfx::companions"};

using texture_replacement::CompanionLoadError;
using texture_replacement::CompanionRole;
using texture_replacement::CompanionRoleCount;

enum class RoleState : uint8_t { Absent, Pending, Loaded, Failed };

struct RoleSlot {
  RoleState state = RoleState::Absent;
  TextureHandle handle;
  uint64_t bytes = 0;
};

struct SetEntry {
  std::array<RoleSlot, CompanionRoleCount> roles;
  uint64_t lastSeenFrame = 0;
};

struct Completion {
  uint32_t set = 0;
  CompanionRole role = CompanionRole::Albedo;
  uint64_t generation = 0;
  uint64_t session = 0;
  std::optional<ConvertedTexture> texture;
  CompanionLoadError error = CompanionLoadError::None;
};

std::mutex s_completionMutex;
std::condition_variable s_completionCv;
std::vector<Completion> s_completions;
uint32_t s_inFlight = 0;

absl::flat_hash_map<uint32_t, SetEntry> s_sets;
uint64_t s_generation = 0;
uint64_t s_frame = 0;
std::atomic<uint64_t> s_session{1};
std::atomic<uint64_t> s_epoch{0};
testing::Stats s_stats;
bool s_deviceless = false;

void drop_all() noexcept {
  if (!s_sets.empty()) {
    s_sets.clear();
    s_epoch.fetch_add(1, std::memory_order_release);
  }
}

void sync_generation() noexcept {
  const uint64_t generation = texture_replacement::companion_generation();
  if (generation != s_generation) {
    s_generation = generation;
    drop_all();
  }
}

TextureHandle upload(CompanionRole role, const ConvertedTexture& texture) noexcept {
  const wgpu::Extent3D size{.width = texture.width, .height = texture.height, .depthOrArrayLayers = 1};
  if (s_deviceless) {
    return std::make_shared<TextureRef>(wgpu::Texture{}, wgpu::TextureView{}, wgpu::TextureView{}, size, texture.format,
                                        texture.mips, InvalidTextureFormat);
  }
  const auto label = fmt::format("Companion {}", texture_replacement::companion_role_name(role));
  const wgpu::TextureDescriptor descriptor{
      .label = label.c_str(),
      .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst,
      .dimension = wgpu::TextureDimension::e2D,
      .size = size,
      .format = texture.format,
      .mipLevelCount = texture.mips,
      .sampleCount = 1,
  };
  auto gpuTexture = webgpu::g_device.CreateTexture(&descriptor);
  const auto viewLabel = fmt::format("{} view", label);
  const wgpu::TextureViewDescriptor viewDescriptor{
      .label = viewLabel.c_str(),
      .format = texture.format,
      .dimension = wgpu::TextureViewDimension::e2D,
      .mipLevelCount = texture.mips,
  };
  auto view = gpuTexture.CreateView(&viewDescriptor);
  auto handle = std::make_shared<TextureRef>(std::move(gpuTexture), std::move(view), wgpu::TextureView{}, size,
                                             texture.format, texture.mips, InvalidTextureFormat);
  handle->isReplacement = true;
  write_texture(*handle, texture.data);
  return handle;
}

bool finish(RoleSlot& slot, CompanionRole role, std::optional<ConvertedTexture> texture,
            CompanionLoadError error) noexcept {
  if (!texture.has_value()) {
    slot.state = RoleState::Failed;
    ++s_stats.rejected;
    s_stats.rejectedFormat += error == CompanionLoadError::Format ? 1 : 0;
    s_stats.rejectedBc += error == CompanionLoadError::NoBc ? 1 : 0;
    return false;
  }
  slot.handle = upload(role, *texture);
  slot.bytes = calc_texture_size(texture->format, texture->width, texture->height, texture->mips);
  slot.state = RoleState::Loaded;
  ++s_stats.loaded;
  s_stats.loadedBytes += slot.bytes;
  s_stats.loadedEmit += role == CompanionRole::Emit ? 1 : 0;
  s_stats.loadedDelight += role == CompanionRole::Delight ? 1 : 0;
  return true;
}

void load_now(uint32_t set, CompanionRole role, RoleSlot& slot) noexcept {
  const auto file = texture_replacement::companion_file(set, role);
  if (!file.has_value()) {
    slot.state = RoleState::Failed;
    return;
  }
  CompanionLoadError error = CompanionLoadError::None;
  auto texture = texture_replacement::load_companion(role, file->path, &error);
  ++s_stats.syncLoads;
  finish(slot, role, std::move(texture), error);
}

void queue_load(uint32_t set, CompanionRole role, RoleSlot& slot) noexcept {
  auto file = texture_replacement::companion_file(set, role);
  if (!file.has_value()) {
    slot.state = RoleState::Failed;
    return;
  }
  slot.state = RoleState::Pending;
  {
    std::lock_guard lock{s_completionMutex};
    ++s_inFlight;
  }
  texture_replacement::queue_companion_task([set, role, path = std::move(file->path), generation = s_generation,
                                             session = s_session.load(std::memory_order_relaxed)] {
    CompanionLoadError error = CompanionLoadError::None;
    auto texture = texture_replacement::load_companion(role, path, &error);
    {
      std::lock_guard lock{s_completionMutex};
      s_completions.push_back({
          .set = set,
          .role = role,
          .generation = generation,
          .session = session,
          .texture = std::move(texture),
          .error = error,
      });
      if (s_inFlight != 0) {
        --s_inFlight;
      }
    }
    s_completionCv.notify_all();
  });
}
} // namespace

uint8_t views(uint32_t set, uint8_t wanted, std::array<WGPUTextureView, ViewSlotCount>& out) noexcept {
  ZoneScoped;
  out.fill(nullptr);
  if (set == 0 || wanted == 0) {
    return 0;
  }
  sync_generation();
  if ((wanted & role_bit(CompanionRole::Nrm)) != 0) {
    wanted &= static_cast<uint8_t>(~role_bit(CompanionRole::Mat));
  }
  if ((wanted & role_bit(CompanionRole::Albedo)) != 0) {
    wanted &= static_cast<uint8_t>(~role_bit(CompanionRole::Delight));
  }
  const auto [it, inserted] = s_sets.try_emplace(set);
  s_stats.sets += inserted ? 1 : 0;
  auto& entry = it->second;
  entry.lastSeenFrame = s_frame;
  const bool synchronous = texture_replacement::synchronous_loads();
  uint8_t loaded = 0;
  for (uint32_t index = 0; index < CompanionRoleCount; ++index) {
    const auto role = static_cast<CompanionRole>(index);
    const int32_t slotIndex = view_slot(role);
    if ((wanted & role_bit(role)) == 0 || slotIndex < 0) {
      continue;
    }
    auto& slot = entry.roles[index];
    if (slot.state == RoleState::Absent || (synchronous && slot.state == RoleState::Pending)) {
      if (synchronous) {
        // Replays need companion pixels on first use, like base replacements
        load_now(set, role, slot);
      } else {
        queue_load(set, role, slot);
      }
    }
    if (slot.state == RoleState::Loaded) {
      out[static_cast<size_t>(slotIndex)] = slot.handle->sampleTextureView.Get();
      loaded |= role_bit(role);
    }
  }
  return loaded;
}

void process() noexcept {
  ZoneScoped;
  sync_generation();
  ++s_frame;
  std::vector<Completion> done;
  {
    std::lock_guard lock{s_completionMutex};
    done.swap(s_completions);
  }
  const uint64_t session = s_session.load(std::memory_order_relaxed);
  bool landed = false;
  for (auto& completion : done) {
    if (completion.session != session || completion.generation != s_generation) {
      continue;
    }
    const auto it = s_sets.find(completion.set);
    if (it == s_sets.end()) {
      continue;
    }
    auto& slot = it->second.roles[static_cast<size_t>(completion.role)];
    if (slot.state != RoleState::Pending) {
      continue;
    }
    landed |= finish(slot, completion.role, std::move(completion.texture), completion.error);
  }
  if (landed) {
    s_epoch.fetch_add(1, std::memory_order_release);
  }
  if (s_frame > EvictAfterFrames) {
    const uint64_t oldest = s_frame - EvictAfterFrames;
    s_stats.evicted +=
        absl::erase_if(s_sets, [oldest](const auto& item) { return item.second.lastSeenFrame < oldest; });
  }
}

uint64_t epoch() noexcept { return s_epoch.load(std::memory_order_acquire); }

void shutdown() noexcept {
  if (s_stats.sets != 0) {
    Log.info(
        "Companions: sets {}, loaded {} ({:.1f} MB; emit {}, delight {}), rejected {} (D7 {}, BC {}), evicted {}, "
        "sync loads {}",
        s_stats.sets, s_stats.loaded, static_cast<double>(s_stats.loadedBytes) / (1024.0 * 1024.0), s_stats.loadedEmit,
        s_stats.loadedDelight, s_stats.rejected, s_stats.rejectedFormat, s_stats.rejectedBc, s_stats.evicted,
        s_stats.syncLoads);
  }
  s_session.fetch_add(1, std::memory_order_relaxed);
  {
    std::lock_guard lock{s_completionMutex};
    s_completions.clear();
    s_inFlight = 0;
  }
  s_completionCv.notify_all();
  s_sets.clear();
  s_generation = texture_replacement::companion_generation();
  s_frame = 0;
  s_stats = {};
}

namespace testing {
void set_deviceless(bool deviceless) noexcept { s_deviceless = deviceless; }

bool wait_idle(uint32_t timeoutMs) noexcept {
  std::unique_lock lock{s_completionMutex};
  return s_completionCv.wait_for(lock, std::chrono::milliseconds{timeoutMs}, [] { return s_inFlight == 0; });
}

Stats stats() noexcept { return s_stats; }

size_t resident_sets() noexcept { return s_sets.size(); }
} // namespace testing
} // namespace aurora::gfx::companion_cache
