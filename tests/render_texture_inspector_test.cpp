#include <gtest/gtest.h>

#include "gx/guest_memory.hpp"
#include "gx/gx.hpp"
#include "gx/pipeline.hpp"
#include "render/debug/debug.hpp"
#include "render/debug/pick.hpp"
#include "render/debug/textures.hpp"
#include "render/fakeshadow/texture_gate.hpp"
#include "render/frame_state.hpp"
#include "render/hints.hpp"
#include "render/material/gx_lookup.hpp"
#include "render/material/registry.hpp"
#include "render/scene/classify.hpp"
#include "render/settings.hpp"
#include "gfx/frame_packet.hpp"
#include "gfx/pipeline_cache.hpp"

#include <dolphin/gx/GXStruct.h>
#include <aurora/texture.hpp>

#include <fmt/format.h>

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace aurora::render::debug::textures {
namespace {
constexpr uint32_t Side = 8;
constexpr size_t I8Bytes = Side * Side;

std::vector<uint8_t> pattern(uint8_t seed) {
  std::vector<uint8_t> bytes(I8Bytes);
  for (size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<uint8_t>(seed * 31 + i * 7);
  }
  return bytes;
}

texture::TextureSourceKey key_of(const std::vector<uint8_t>& bytes) {
  return *texture::compute_source_key(GX_TF_I8, Side, Side, bytes);
}

scene::DrawClass span_draw(uint32_t index) {
  scene::DrawClass cls;
  cls.key.role = scene::Role::Main;
  cls.key.window = scene::Window::Opaque;
  cls.key.mainCamera = true;
  cls.mainClass = true;
  cls.index = index;
  return cls;
}

gx::ShaderInfo sampling(std::initializer_list<uint32_t> slots) {
  gx::ShaderInfo info;
  for (const uint32_t slot : slots) {
    info.sampledTextures.set(slot);
  }
  return info;
}

const void* g_pointerData = nullptr;

bool pointer_probe(const GXTexObj_& obj) noexcept { return obj.data == g_pointerData; }

std::span<const uint8_t> resolve_same(void*, gx::guest_memory::Kind, uint64_t address, size_t size) noexcept {
  return {reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(address)), size};
}

gx::GXBindGroups neutral_stub(const gx::ShaderInfo&, uint8_t whiteMask, uint8_t clearMask, bool) noexcept {
  return {.textureBindGroup = 0x200000u + whiteMask * 0x100u + clearMask};
}

class TextureInspector : public ::testing::Test {
protected:
  void SetUp() override { reset_all(); }
  void TearDown() override { reset_all(); }

  static void reset_all() {
    testing::reset();
    debug::testing::reset();
    scene::testing::reset();
    texture::clear_material_manifests();
    material::latch();
    gx::guest_memory::set_hooks(nullptr);
    gx::g_gxState = {};
    g_pointerData = nullptr;
  }

  static void load(uint32_t slot, const std::vector<uint8_t>& bytes, uint32_t id) {
    GXTexObj_& obj = gx::g_gxState.loadedTextures[slot];
    obj = {};
    obj.data = bytes.data();
    obj.mWidth = Side;
    obj.mHeight = Side;
    obj.mFormat = GX_TF_I8;
    obj.texObjId = id;
    obj.texDataVersion = 1;
  }

  static gx::DrawData draw_data(uint32_t instances = 1) {
    gx::DrawData draw{};
    draw.instanceCount = instances;
    return draw;
  }

  void observe_draw(const gx::ShaderInfo& info, const scene::DrawClass& cls, uint32_t instances = 1) {
    gx::DrawData draw = draw_data(instances);
    observe(draw, m_config, info, cls);
    m_lastPad = draw.immediateData._pad;
  }

  static const KeyRow* find_row(const TexturesReport& report, const texture::TextureSourceKey& key) {
    for (const KeyRow& row : report.keys) {
      if (row.state == KeyState::Ready && row.key == key) {
        return &row;
      }
    }
    return nullptr;
  }

  gx::PipelineConfig m_config{};
  uint32_t m_lastPad = 0;
};

TEST_F(TextureInspector, RecordsKeysPerFrame) {
  testing::force_visible(1);
  on_latch();
  const auto a = pattern(1);
  const auto b = pattern(2);
  load(0, a, 10);
  load(1, b, 11);
  observe_draw(sampling({0}), span_draw(0));
  observe_draw(sampling({0, 1}), span_draw(1));
  on_latch();
  const auto first = latest_report();
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(first->draws.size(), 2u);
  EXPECT_EQ(first->slots.size(), 3u);
  ASSERT_EQ(first->keys.size(), 2u);
  const KeyRow* rowA = find_row(*first, key_of(a));
  const KeyRow* rowB = find_row(*first, key_of(b));
  ASSERT_NE(rowA, nullptr);
  ASSERT_NE(rowB, nullptr);
  EXPECT_EQ(rowA->draws, 2u);
  EXPECT_EQ(rowA->mainDraws, 2u);
  EXPECT_EQ(rowB->draws, 1u);
  EXPECT_EQ(first->draws[1].slotCount, 2u);

  observe_draw(sampling({1}), span_draw(0));
  on_latch();
  const auto second = latest_report();
  ASSERT_NE(second, nullptr);
  EXPECT_EQ(second->sequence, first->sequence + 1);
  ASSERT_EQ(second->keys.size(), 1u);
  EXPECT_EQ(second->keys[0].key, key_of(b));
  EXPECT_EQ(first->keys.size(), 2u);
}

TEST_F(TextureInspector, InspectorKeyMatchesBindMaterial) {
  const auto bytes = pattern(3);
  const texture::TextureSourceKey key = key_of(bytes);
  const std::string text = fmt::format(R"({{"version":1,"materials":{{"{}":{{"flags":["stochastic"]}}}}}})",
                                       texture::format_material_key(key));
  ASSERT_EQ(texture::register_material_manifest(text, {.layer = texture::MaterialLayer::User}).status,
            texture::MaterialManifestStatus::Accepted);
  material::latch();
  testing::force_visible(1);
  on_latch();
  load(2, bytes, 20);
  // The core stores exactly what the lookup returns for the draw path's key
  gx::g_gxState.textures[2].material = material::testing::gx_find(key);
  ASSERT_NE(gx::g_gxState.textures[2].material.id, 0u);
  observe_draw(sampling({2}), span_draw(0));
  on_latch();
  const auto report = latest_report();
  ASSERT_NE(report, nullptr);
  ASSERT_EQ(report->slots.size(), 1u);
  const SlotRecord& slot = report->slots[0];
  EXPECT_EQ(slot.material, gx::g_gxState.textures[2].material);
  ASSERT_LT(slot.row, report->keys.size());
  const KeyRow& row = report->keys[slot.row];
  ASSERT_EQ(row.state, KeyState::Ready);
  EXPECT_EQ(material::latched_snapshot()->find(row.key), slot.material.id);
  EXPECT_EQ(row.material, slot.material);
}

TEST_F(TextureInspector, PointerReplacementSlotsAreNotEditable) {
  const auto plain = pattern(4);
  const auto pointer = pattern(5);
  g_pointerData = pointer.data();
  testing::set_pointer_probe(&pointer_probe);
  testing::force_visible(1);
  on_latch();
  load(0, plain, 30);
  load(1, pointer, 31);
  observe_draw(sampling({0, 1}), span_draw(0));
  on_latch();
  const auto report = latest_report();
  ASSERT_NE(report, nullptr);
  ASSERT_EQ(report->slots.size(), 2u);
  EXPECT_FALSE(report->slots[0].pointerReplacement);
  EXPECT_TRUE(report->slots[1].pointerReplacement);
  const KeyRow* row = find_row(*report, key_of(pointer));
  ASSERT_NE(row, nullptr);
  EXPECT_TRUE(row->pointerReplacement);
  EXPECT_FALSE(find_row(*report, key_of(plain))->pointerReplacement);
}

TEST_F(TextureInspector, KeyMemoHashesOncePerObject) {
  testing::force_visible(1);
  on_latch();
  const auto bytes = pattern(6);
  load(0, bytes, 40);
  observe_draw(sampling({0}), span_draw(0));
  observe_draw(sampling({0}), span_draw(1));
  on_latch();
  ASSERT_NE(latest_report(), nullptr);
  EXPECT_EQ(latest_report()->hashedBytes, I8Bytes);
  EXPECT_EQ(testing::memo_size(), 1u);
  observe_draw(sampling({0}), span_draw(0));
  on_latch();
  const auto report = latest_report();
  EXPECT_EQ(report->hashedBytes, 0u);
  ASSERT_EQ(report->keys.size(), 1u);
  EXPECT_EQ(report->keys[0].state, KeyState::Ready);
  EXPECT_EQ(report->keys[0].key, key_of(bytes));
  EXPECT_EQ(testing::memo_size(), 1u);
}

TEST_F(TextureInspector, KeysPauseWhileGuestMemoryHooked) {
  static constexpr gx::guest_memory::Hooks Hooks{.resolve = &resolve_same};
  testing::force_visible(1);
  on_latch();
  const auto bytes = pattern(7);
  load(0, bytes, 50);
  gx::guest_memory::set_hooks(&Hooks);
  observe_draw(sampling({0}), span_draw(0));
  on_latch();
  const auto paused = latest_report();
  ASSERT_EQ(paused->keys.size(), 1u);
  EXPECT_EQ(paused->keys[0].state, KeyState::Paused);
  EXPECT_EQ(paused->pausedKeys, 1u);
  EXPECT_EQ(paused->hashedBytes, 0u);
  EXPECT_EQ(testing::memo_size(), 0u);

  gx::guest_memory::set_hooks(nullptr);
  observe_draw(sampling({0}), span_draw(0));
  on_latch();
  const auto resumed = latest_report();
  ASSERT_EQ(resumed->keys.size(), 1u);
  EXPECT_EQ(resumed->keys[0].state, KeyState::Ready);
  EXPECT_EQ(resumed->pausedKeys, 0u);
}

TEST_F(TextureInspector, HashBudgetDefersKeys) {
  testing::force_visible(1);
  testing::set_hash_budget(I8Bytes);
  on_latch();
  const auto a = pattern(8);
  const auto b = pattern(9);
  load(0, a, 60);
  load(1, b, 61);
  observe_draw(sampling({0, 1}), span_draw(0));
  on_latch();
  const auto first = latest_report();
  ASSERT_EQ(first->keys.size(), 2u);
  EXPECT_EQ(first->keys[0].state, KeyState::Ready);
  EXPECT_EQ(first->keys[1].state, KeyState::Pending);
  EXPECT_EQ(first->pendingKeys, 1u);
  EXPECT_EQ(first->hashedBytes, I8Bytes);

  observe_draw(sampling({0, 1}), span_draw(0));
  on_latch();
  const auto second = latest_report();
  ASSERT_EQ(second->keys.size(), 2u);
  EXPECT_EQ(second->pendingKeys, 0u);
  EXPECT_NE(find_row(*second, key_of(a)), nullptr);
  EXPECT_NE(find_row(*second, key_of(b)), nullptr);
}

TEST_F(TextureInspector, SealsBeforeReserve) {
  testing::force_visible(1);
  on_latch();
  for (uint32_t i = 0; i < 3; ++i) {
    observe_draw(sampling({}), span_draw(i));
  }
  on_latch();
  const auto first = latest_report();
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(first->draws.size(), 3u);
  const testing::Capacities capacities = testing::live_capacities();
  EXPECT_GE(capacities.draws, MaxDraws);
  EXPECT_GE(capacities.slots, MaxSlots);
  EXPECT_GE(capacities.keys, MaxKeyRows);
  observe_draw(sampling({}), span_draw(0));
  on_latch();
  EXPECT_EQ(first->draws.size(), 3u);
  EXPECT_EQ(latest_report()->draws.size(), 1u);
  EXPECT_NE(latest_report(), first);
}

TEST_F(TextureInspector, PickIdEncodesFrameAndDraw) {
  testing::force_visible(1);
  on_latch();
  testing::set_arm_state(pick::ArmState::Armed);
  const uint64_t sequence = testing::sequence();
  observe_draw(sampling({}), span_draw(0));
  EXPECT_EQ(m_lastPad, pick::encode_id(sequence, 0));
  observe_draw(sampling({}), span_draw(1));
  const uint32_t id = m_lastPad;
  EXPECT_EQ(id, pick::encode_id(sequence, 1));
  scene::DrawClass offscreen = span_draw(2);
  offscreen.offscreen = true;
  observe_draw(sampling({}), offscreen);
  EXPECT_EQ(m_lastPad, 0u);
  const auto decoded = pick::decode_id(id);
  ASSERT_TRUE(decoded.has_value());
  EXPECT_EQ(decoded->sequence, static_cast<uint8_t>(sequence));
  EXPECT_EQ(decoded->record, 1u);
  EXPECT_FALSE(pick::decode_id(0).has_value());
  testing::set_arm_state(pick::ArmState::Off);
  on_latch();
  const auto report = report_for(decoded->sequence);
  ASSERT_NE(report, nullptr);
  ASSERT_GT(report->draws.size(), decoded->record);
  EXPECT_EQ(report->draws[decoded->record].index, 1u);
  EXPECT_TRUE(report->draws[decoded->record].inSpan);
  EXPECT_FALSE(report->draws[2].inSpan);
}

TEST_F(TextureInspector, EmptyDrawIsNeverStamped) {
  testing::force_visible(1);
  on_latch();
  testing::set_arm_state(pick::ArmState::Armed);
  observe_draw(sampling({}), span_draw(0), 0);
  EXPECT_EQ(m_lastPad, 0u);
  observe_draw(sampling({}), span_draw(1), 1);
  EXPECT_NE(m_lastPad, 0u);
  testing::set_arm_state(pick::ArmState::Off);
  on_latch();
  EXPECT_EQ(latest_report()->draws.size(), 2u);
}

TEST_F(TextureInspector, HiddenTabInstallsNothing) {
  set_pick_enabled(true);
  testing::force_visible(0);
  on_latch();
  EXPECT_FALSE(testing::observer_installed());
  EXPECT_EQ(arm_state(), pick::ArmState::Off);
  EXPECT_FALSE(pick::pass_registered());
  EXPECT_EQ(testing::memo_size(), 0u);
  EXPECT_EQ(testing::live_capacities().draws, 0u);
  EXPECT_EQ(latest_report(), nullptr);

  testing::force_visible(-1);
  on_latch();
  EXPECT_FALSE(testing::observer_installed());

  testing::force_visible(1);
  on_latch();
  EXPECT_TRUE(testing::observer_installed());
  // No device in this target, so the pick stays off even with the tab shown
  EXPECT_EQ(arm_state(), pick::ArmState::Off);
  testing::force_visible(0);
  on_latch();
  EXPECT_FALSE(testing::observer_installed());
  EXPECT_EQ(testing::live_capacities().draws, 0u);
}

TEST_F(TextureInspector, WarmArmsAt98PercentOrFiveSeconds) {
  // 49 GX configs plus 1 Clear config of 51 known is 96%
  EXPECT_FALSE(pick::warm_complete({.ready = 49, .total = 51, .seconds = 1.0, .observedFrames = 1}));
  EXPECT_TRUE(pick::warm_complete({.ready = 50, .total = 51, .seconds = 1.0, .observedFrames = 1}));
  EXPECT_TRUE(pick::warm_complete({.ready = 51, .total = 51, .seconds = 0.0, .observedFrames = 1}));
  EXPECT_FALSE(pick::warm_complete({.ready = 0, .total = 51, .seconds = 4.9, .observedFrames = 1}));
  EXPECT_TRUE(pick::warm_complete({.ready = 0, .total = 51, .seconds = 5.0, .observedFrames = 1}));
  EXPECT_FALSE(pick::warm_complete({.ready = 4, .total = 4, .seconds = 9.0, .observedFrames = 0}));
  EXPECT_FALSE(pick::warm_complete({.ready = 0, .total = 0, .seconds = 1.0, .observedFrames = 3}));

  using pick::ArmState;
  EXPECT_EQ(pick::next_arm_state(ArmState::Off, {.wanted = true}), ArmState::Warming);
  EXPECT_EQ(pick::next_arm_state(ArmState::Warming, {.wanted = true}), ArmState::Warming);
  EXPECT_EQ(pick::next_arm_state(ArmState::Warming, {.wanted = true, .warmComplete = true}), ArmState::Armed);
  EXPECT_EQ(pick::next_arm_state(ArmState::Warming, {.wanted = false, .warmComplete = true}), ArmState::Off);
}

TEST_F(TextureInspector, BaseLayoutChangeWhileArmedRewarms) {
  using pick::ArmState;
  EXPECT_EQ(pick::next_arm_state(ArmState::Armed, {.wanted = true}), ArmState::Armed);
  EXPECT_EQ(pick::next_arm_state(ArmState::Armed, {.wanted = true, .baseLayoutChanged = true}), ArmState::Warming);
  EXPECT_EQ(pick::next_arm_state(ArmState::Armed, {.wanted = false}), ArmState::Off);

  gfx::RenderTargetLayout live{
      .colorAttachmentCount = 1,
      .colorAttachments = {{{gfx::ColorAttachmentSemantic::SceneColor, wgpu::TextureFormat::BGRA8Unorm, 64, 32}}},
      .depthStencilFormat = wgpu::TextureFormat::Depth32Float,
  };
  gfx::detail::finalize_render_target_layout(live);
  const gfx::RenderTargetLayout armed = pick::armed_layout(live);
  ASSERT_EQ(armed.colorAttachmentCount, 2u);
  EXPECT_EQ(armed.colorAttachments[1].semantic, gfx::ColorAttachmentSemantic::DebugId);
  EXPECT_EQ(armed.colorAttachments[1].format, wgpu::TextureFormat::R32Uint);
  EXPECT_EQ(gfx::pipeline_base_layout(armed).key, live.key);
  EXPECT_EQ(pick::armed_layout(armed).key, armed.key);
  gfx::RenderTargetLayout withNormal = live;
  withNormal.colorAttachments[withNormal.colorAttachmentCount++] = {gfx::ColorAttachmentSemantic::Normal,
                                                                    wgpu::TextureFormat::RGB10A2Unorm, 64, 32};
  gfx::detail::finalize_render_target_layout(withNormal);
  EXPECT_NE(pick::armed_layout(withNormal).key, armed.key);
  EXPECT_NE(gfx::pipeline_base_layout(pick::armed_layout(withNormal)).key, gfx::pipeline_base_layout(armed).key);
}

TEST_F(TextureInspector, CapabilityReasons) {
  const pick::CapabilityInputs good{
      .device = true, .core = true, .msaaSamples = 1, .hints = true, .colorBytes = 12, .colorBudget = 32};
  EXPECT_TRUE(pick::pick_capability(good).available);
  EXPECT_TRUE(pick::pick_capability(good).reason.empty());
  auto inputs = good;
  inputs.device = false;
  EXPECT_EQ(pick::pick_capability(inputs).reason, "no device");
  inputs = good;
  inputs.core = false;
  EXPECT_EQ(pick::pick_capability(inputs).reason, "needs core features");
  inputs = good;
  inputs.msaaSamples = 4;
  EXPECT_EQ(pick::pick_capability(inputs).reason, "needs MSAA off");
  inputs = good;
  inputs.hints = false;
  EXPECT_EQ(pick::pick_capability(inputs).reason, "render hints are off");
  inputs = good;
  inputs.colorBytes = 36;
  EXPECT_FALSE(pick::pick_capability(inputs).available);
  EXPECT_EQ(pick::pick_capability(inputs).reason, "over the colour attachment budget");
  inputs.colorBytes = 32;
  EXPECT_TRUE(pick::pick_capability(inputs).available);
}

TEST_F(TextureInspector, EfbPixelInvertsPresentViewport) {
  // 640x480 on 1920x1080 letterboxes to 1440x1080 at x 240
  const Vec2<uint32_t> surface{1920, 1080};
  const Vec2<uint32_t> efb{640, 480};
  EXPECT_EQ(pick::efb_pixel_at({960.f, 540.f}, {1.f, 1.f}, surface, efb), (pick::PixelPos{320, 240}));
  EXPECT_EQ(pick::efb_pixel_at({240.f, 0.f}, {1.f, 1.f}, surface, efb), (pick::PixelPos{0, 0}));
  EXPECT_EQ(pick::efb_pixel_at({1679.f, 1079.f}, {1.f, 1.f}, surface, efb), (pick::PixelPos{639, 479}));
  EXPECT_FALSE(pick::efb_pixel_at({239.f, 540.f}, {1.f, 1.f}, surface, efb).has_value());
  EXPECT_FALSE(pick::efb_pixel_at({1680.f, 540.f}, {1.f, 1.f}, surface, efb).has_value());
  EXPECT_EQ(pick::efb_pixel_at({480.f, 270.f}, {2.f, 2.f}, surface, efb), (pick::PixelPos{320, 240}));
  EXPECT_FALSE(pick::efb_pixel_at({10.f, 10.f}, {1.f, 1.f}, {0, 0}, efb).has_value());
}

TEST_F(TextureInspector, OverflowCountsWithoutAllocating) {
  testing::force_visible(1);
  on_latch();
  const testing::Capacities before = testing::live_capacities();
  std::array<std::vector<uint8_t>, gx::MaxTextures> bytes;
  for (uint32_t i = 0; i < gx::MaxTextures; ++i) {
    bytes[i] = pattern(static_cast<uint8_t>(20 + i));
    load(i, bytes[i], 100 + i);
  }
  const gx::ShaderInfo all = sampling({0, 1, 2, 3, 4, 5, 6, 7});
  for (uint32_t i = 0; i < MaxDraws + 5; ++i) {
    observe_draw(all, span_draw(i));
  }
  const testing::Capacities after = testing::live_capacities();
  EXPECT_EQ(after.draws, before.draws);
  EXPECT_EQ(after.slots, before.slots);
  EXPECT_EQ(after.keys, before.keys);
  on_latch();
  const auto report = latest_report();
  EXPECT_EQ(report->draws.size(), MaxDraws);
  EXPECT_EQ(report->overflow.draws, 5u);
  EXPECT_EQ(report->slots.size(), MaxSlots);
  EXPECT_EQ(report->overflow.slots, MaxDraws * gx::MaxTextures - MaxSlots);
  EXPECT_EQ(report->keys.size(), gx::MaxTextures);

  // Objects with no data get one row each, so distinct ids run past the row cap
  gx::g_gxState.loadedTextures = {};
  for (uint32_t i = 0; i < MaxKeyRows + 3; ++i) {
    gx::g_gxState.loadedTextures[0].texObjId = 1000 + i;
    observe_draw(sampling({0}), span_draw(i));
  }
  EXPECT_EQ(testing::live_capacities().keys, before.keys);
  on_latch();
  EXPECT_EQ(latest_report()->keys.size(), MaxKeyRows);
  EXPECT_EQ(latest_report()->overflow.keys, 3u);
}

TEST_F(TextureInspector, NeutralColumnFollowsGateDecision) {
  struct Hints {
    Hints() {
      detail::testing::set_hints_override(1);
      detail::install_hints();
    }
    ~Hints() {
      detail::remove_hints();
      detail::testing::reset_hints();
      detail::testing::reset();
    }
  } hints;
  constexpr uint8_t Light = 1u << static_cast<uint32_t>(fakeshadow::Gate::Light);
  const std::array rules{
      fakeshadow::Rule{.materialId = 7,
                       .neutral = fakeshadow::Neutral::Clear,
                       .gateBit = Light,
                       .scopeMask = scene::category::Terrain,
                       .scoped = true},
      fakeshadow::Rule{.materialId = 8,
                       .neutral = fakeshadow::Neutral::Skip,
                       .gateBit = Light,
                       .scopeMask = scene::category::Water,
                       .scoped = true},
  };
  fakeshadow::testing::set_rules(rules);
  fakeshadow::testing::set_neutral_builder(&neutral_stub);
  fakeshadow::set_force(fakeshadow::Gate::Light, fakeshadow::Force::On);
  fakeshadow::begin_frame(detail::FrameState{});
  testing::force_visible(1);
  on_latch();
  testing::set_arm_state(pick::ArmState::Armed);
  m_config.blendMode = GX_BM_BLEND;
  m_config.blendFacSrc = GX_BL_SRCALPHA;
  m_config.blendFacDst = GX_BL_INVSRCALPHA;
  const auto shade = pattern(10);
  const auto plain = pattern(11);
  load(0, shade, 70);
  load(1, plain, 71);
  const auto bind = [](uint32_t slot, uint32_t id) {
    gfx::TextureBind& texture = gx::g_gxState.textures[slot];
    texture.ref =
        std::make_shared<gfx::TextureRef>(wgpu::Texture{}, wgpu::TextureView{}, wgpu::TextureView{},
                                          wgpu::Extent3D{Side, Side, 1}, wgpu::TextureFormat::RGBA8Unorm, 1, GX_TF_I8);
    texture.material = {id, 0};
  };
  const auto push = [this](const gx::ShaderInfo& info, uint32_t index, uint32_t categories) {
    scene::DrawClass cls = span_draw(index);
    cls.key.categories = categories;
    gx::DrawData draw = draw_data();
    fakeshadow::push(draw, m_config, info, cls);
    observe(draw, m_config, info, cls);
    return draw;
  };
  bind(0, 7);
  bind(1, 0);
  const gx::DrawData cleared = push(sampling({0, 1}), 0, scene::category::Terrain);
  EXPECT_EQ(cleared.bindGroups.textureBindGroup, 0x200001u);
  EXPECT_NE(cleared.immediateData._pad, 0u);
  push(sampling({0, 1}), 1, scene::category::Water);
  bind(0, 8);
  const gx::DrawData skipped = push(sampling({0}), 2, scene::category::Water);
  EXPECT_EQ(skipped.instanceCount, 0u);
  EXPECT_EQ(skipped.immediateData._pad, 0u);
  testing::set_arm_state(pick::ArmState::Off);
  on_latch();

  const auto report = latest_report();
  ASSERT_NE(report, nullptr);
  ASSERT_EQ(report->draws.size(), 3u);
  ASSERT_EQ(report->slots.size(), 5u);
  EXPECT_FALSE(report->draws[0].skipped);
  EXPECT_FALSE(report->draws[1].skipped);
  EXPECT_TRUE(report->draws[2].skipped);
  EXPECT_TRUE(report->slots[0].neutral);
  EXPECT_FALSE(report->slots[1].neutral);
  EXPECT_FALSE(report->slots[2].neutral);
  EXPECT_FALSE(report->slots[3].neutral);
  EXPECT_TRUE(report->slots[4].neutral);
  const KeyRow* shadeRow = find_row(*report, key_of(shade));
  const KeyRow* plainRow = find_row(*report, key_of(plain));
  ASSERT_NE(shadeRow, nullptr);
  ASSERT_NE(plainRow, nullptr);
  EXPECT_EQ(shadeRow->draws, 3u);
  EXPECT_EQ(shadeRow->neutralDraws, 2u);
  EXPECT_EQ(plainRow->draws, 2u);
  EXPECT_EQ(plainRow->neutralDraws, 0u);

  fakeshadow::set_force(fakeshadow::Gate::Light, fakeshadow::Force::Off);
  fakeshadow::begin_frame(detail::FrameState{});
  bind(0, 7);
  const gx::DrawData kept = push(sampling({0, 1}), 0, scene::category::Terrain);
  EXPECT_EQ(kept.bindGroups.textureBindGroup, 0u);
  on_latch();
  const auto off = latest_report();
  ASSERT_NE(off, nullptr);
  ASSERT_EQ(off->slots.size(), 2u);
  EXPECT_FALSE(off->slots[0].neutral);
  EXPECT_EQ(find_row(*off, key_of(shade))->neutralDraws, 0u);
}

TEST_F(TextureInspector, FlaggedSeenCountsIntersection) {
  material::Snapshot snapshot;
  snapshot.records.resize(8);
  for (const uint16_t id : {1, 3, 5}) {
    snapshot.records[id].id = id;
  }
  SeenSet seen;
  EXPECT_EQ(flagged_seen(seen, &snapshot), 0u);
  seen.set(1);
  seen.set(2);
  seen.set(5);
  seen.set(40);
  EXPECT_EQ(flagged_seen(seen, &snapshot), 2u);
  seen.set(3);
  EXPECT_EQ(flagged_seen(seen, &snapshot), 3u);
  EXPECT_EQ(flagged_seen(seen, nullptr), 0u);
}
} // namespace
} // namespace aurora::render::debug::textures
