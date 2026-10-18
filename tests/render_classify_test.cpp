#include <gtest/gtest.h>

#include "render/scene/classifier.hpp"

#include <array>
#include <bit>
#include <cstdint>
#include <set>
#include <string_view>

namespace aurora::render::scene {
namespace {
// Passes every main-class test except the role, so each test only needs a camera and a window
DrawInputs main_draw() {
  return {
      .mainCamera = true,
      .projectionMatch = true,
      .perspective = true,
      .depthWrite = true,
      .colorWrite = true,
      .vertices = 3,
  };
}

ObjectHint object(uint64_t id, uint32_t subId = 0, uint32_t categories = 0, uint32_t flags = 0) {
  return {.id = id, .subId = subId, .generation = 1, .categories = categories, .flags = flags};
}

// The scope declaration hash as tools recompute it: FNV-1a over depth, role, categories and flags, little-endian
uint32_t fnv(uint32_t depth, uint8_t role, uint32_t categories, uint32_t flags) {
  uint32_t hash = 2166136261u;
  const auto mix = [&hash](uint32_t value, int bytes) {
    for (int i = 0; i < bytes; ++i) {
      hash = (hash ^ ((value >> (8 * i)) & 0xFF)) * 16777619u;
    }
  };
  mix(depth, 4);
  mix(role, 1);
  mix(categories, 4);
  mix(flags, 4);
  return hash;
}

class RenderClassify : public ::testing::Test {
protected:
  void SetUp() override {
    classifier.reserve();
    classifier.begin_frame();
  }

  // Root MAIN camera and SCENE_BEGIN, the minimum for a main-class draw
  void main_scene() {
    classifier.camera(Role::Main);
    ASSERT_TRUE(classifier.stage(Stage::SceneBegin));
  }

  const FrameStats& stats() const { return classifier.stats(); }

  Classifier classifier;
};

TEST_F(RenderClassify, NoHintsIsForeignAndKeyStable) {
  for (int i = 0; i < 3; ++i) {
    const DrawClass& cls = classifier.route(main_draw());
    EXPECT_EQ(cls.key.role, Role::None);
    EXPECT_EQ(cls.check, MainCheck::Role);
    EXPECT_EQ(cls.kind, Kind::Foreign);
    EXPECT_FALSE(cls.keyChanged);
    EXPECT_FALSE(cls.mainClass);
    EXPECT_EQ(cls.index, static_cast<uint32_t>(i));
  }
  classifier.end_frame();
  EXPECT_EQ(stats().hints.errors(), 0u);
  EXPECT_EQ(stats().unclassified, 3u);
  EXPECT_EQ(stats().draws, 3u);
  EXPECT_EQ(stats().roles[static_cast<size_t>(Role::None)], 3u);
  EXPECT_EQ(stats().windows[static_cast<size_t>(Window::PreScene)], 3u);
}

TEST_F(RenderClassify, MainCameraAndWindowMakeMainClass) {
  classifier.camera(Role::Main);
  EXPECT_EQ(classifier.route(main_draw()).check, MainCheck::Window);
  ASSERT_TRUE(classifier.stage(Stage::SceneBegin));
  const DrawClass& opaque = classifier.route(main_draw());
  EXPECT_EQ(opaque.key.role, Role::Main);
  EXPECT_EQ(opaque.key.window, Window::Opaque);
  EXPECT_EQ(opaque.check, MainCheck::Passed);
  EXPECT_TRUE(opaque.mainClass);
  EXPECT_EQ(opaque.kind, Kind::MainOpaque);
  EXPECT_TRUE(would_capture(opaque));

  DrawInputs blendedDraw = main_draw();
  blendedDraw.blended = true;
  EXPECT_EQ(classifier.route(blendedDraw).kind, Kind::MainBlended);
  DrawInputs noDepth = main_draw();
  noDepth.depthWrite = false;
  EXPECT_EQ(classifier.route(noDepth).kind, Kind::MainBlended);

  ASSERT_TRUE(classifier.stage(Stage::OpaqueDone));
  const DrawClass& translucent = classifier.route(main_draw());
  EXPECT_EQ(translucent.key.window, Window::Translucent);
  EXPECT_TRUE(translucent.mainClass);

  ASSERT_TRUE(classifier.stage(Stage::PrePost));
  const DrawClass& post = classifier.route(main_draw());
  EXPECT_EQ(post.key.window, Window::Post);
  EXPECT_EQ(post.check, MainCheck::Window);
  EXPECT_FALSE(post.mainClass);
  EXPECT_EQ(post.kind, Kind::Foreign);

  EXPECT_EQ(stats().kinds[static_cast<size_t>(Kind::MainOpaque)], 2u);
  EXPECT_EQ(stats().kinds[static_cast<size_t>(Kind::MainBlended)], 2u);
  EXPECT_EQ(stats().checks[static_cast<size_t>(MainCheck::Window)], 2u);
  EXPECT_EQ(stats().captureCandidates, 2u);
}

TEST_F(RenderClassify, ColourlessMainDrawIsDepthOnly) {
  main_scene();
  DrawInputs depthOnly = main_draw();
  depthOnly.colorWrite = false;
  const DrawClass& cls = classifier.route(depthOnly);
  EXPECT_TRUE(cls.mainClass);
  EXPECT_EQ(cls.kind, Kind::MainDepthOnly);
  EXPECT_FALSE(cls.colorWrite);
  EXPECT_FALSE(would_capture(cls));
  EXPECT_EQ(stats().captureCandidates, 0u);
  EXPECT_EQ(stats().kinds[static_cast<size_t>(Kind::MainDepthOnly)], 1u);
}

TEST_F(RenderClassify, ChecksReportFirstFailure) {
  main_scene();
  const auto check = [this](DrawInputs inputs) { return classifier.route(inputs).check; };
  DrawInputs offscreen = main_draw();
  offscreen.offscreen = true;
  EXPECT_EQ(check(offscreen), MainCheck::Target);
  DrawInputs ortho = main_draw();
  ortho.perspective = false;
  EXPECT_EQ(check(ortho), MainCheck::Ortho);
  DrawInputs noCamera = main_draw();
  noCamera.mainCamera = false;
  noCamera.projectionMatch = false;
  EXPECT_EQ(check(noCamera), MainCheck::NoCamera);
  DrawInputs projection = main_draw();
  projection.projectionMatch = false;
  EXPECT_EQ(check(projection), MainCheck::Projection);

  classifier.scope_push({.role = static_cast<uint8_t>(Role::Shadow)});
  const DrawClass& both = classifier.route(offscreen);
  EXPECT_EQ(both.check, MainCheck::Role);
  EXPECT_FALSE(both.roleOnly);
  EXPECT_TRUE(both.offscreen);
}

TEST_F(RenderClassify, OverlayOnlyInOpaqueWindow) {
  main_scene();
  classifier.scope_push({.categories = category::Overlay});
  EXPECT_EQ(classifier.route(main_draw()).kind, Kind::Overlay);
  ASSERT_TRUE(classifier.stage(Stage::OpaqueDone));
  EXPECT_EQ(classifier.route(main_draw()).kind, Kind::MainOpaque);
  DrawInputs blendedDraw = main_draw();
  blendedDraw.blended = true;
  EXPECT_EQ(classifier.route(blendedDraw).kind, Kind::MainBlended);
}

TEST_F(RenderClassify, SkyScopeIsSky) {
  classifier.camera(Role::Main);
  classifier.scope_push({.categories = category::Sky, .role = static_cast<uint8_t>(Role::Sky)});
  classifier.scope_push({});
  classifier.object_begin(object(5));
  for (int i = 0; i < 2; ++i) {
    const DrawClass& cls = classifier.route(main_draw());
    EXPECT_EQ(cls.key.role, Role::Sky);
    EXPECT_EQ(cls.kind, Kind::Sky);
    EXPECT_EQ(cls.check, MainCheck::Role);
  }
  const RoleScopeCounts& sky = stats().underScope[static_cast<size_t>(Role::Sky)];
  EXPECT_EQ(sky.draws, 2u);
  EXPECT_EQ(sky.asRole, 2u);
  EXPECT_EQ(sky.skyKind, 2u);
  EXPECT_EQ(sky.main, 0u);
  EXPECT_EQ(stats().skyMisses, 0u);
  EXPECT_EQ(stats().kinds[static_cast<size_t>(Kind::Sky)], 2u);
}

TEST_F(RenderClassify, NestedMainScopeInsideSkyIsCaught) {
  classifier.camera(Role::Main);
  classifier.scope_push({.role = static_cast<uint8_t>(Role::Sky)});
  EXPECT_EQ(classifier.route(main_draw()).kind, Kind::Sky);
  classifier.scope_push({.role = static_cast<uint8_t>(Role::Main)});
  EXPECT_EQ(classifier.route(main_draw()).key.role, Role::Main);
  const RoleScopeCounts& sky = stats().underScope[static_cast<size_t>(Role::Sky)];
  EXPECT_EQ(sky.draws, 2u);
  EXPECT_EQ(sky.asRole, 1u);
  EXPECT_LT(sky.asRole, sky.draws);
  EXPECT_EQ(sky.skyKind, 1u);
  EXPECT_EQ(stats().underScope[static_cast<size_t>(Role::Main)].draws, 1u);
}

TEST_F(RenderClassify, SkyMissCounted) {
  classifier.route(main_draw());
  EXPECT_EQ(stats().skyMisses, 0u);
  classifier.camera(Role::Main);
  struct Change {
    std::string_view name;
    void (*apply)(DrawInputs&);
  };
  const std::array<Change, 5> changes{{
      {"offscreen", [](DrawInputs& in) { in.offscreen = true; }},
      {"orthographic", [](DrawInputs& in) { in.perspective = false; }},
      {"projection mismatch", [](DrawInputs& in) { in.projectionMatch = false; }},
      {"no usable camera", [](DrawInputs& in) { in.mainCamera = false; }},
      {"colour off", [](DrawInputs& in) { in.colorWrite = false; }},
  }};
  for (const Change& change : changes) {
    DrawInputs inputs = main_draw();
    change.apply(inputs);
    classifier.route(inputs);
    EXPECT_EQ(stats().skyMisses, 0u) << change.name;
  }
  classifier.object_begin(object(2, 0, 0, object_flag::CaptureOnly));
  EXPECT_TRUE(classifier.route(main_draw()).suppressed);
  classifier.object_end();
  EXPECT_EQ(stats().skyMisses, 0u);
  classifier.route(main_draw());
  EXPECT_EQ(stats().skyMisses, 1u);
  classifier.scope_push({.role = static_cast<uint8_t>(Role::Sky)});
  classifier.route(main_draw());
  classifier.scope_pop();
  EXPECT_EQ(stats().skyMisses, 1u);
  ASSERT_TRUE(classifier.stage(Stage::SceneBegin));
  classifier.route(main_draw());
  EXPECT_EQ(stats().skyMisses, 1u);
}

TEST_F(RenderClassify, ReflectionScopeBlocksMainClass) {
  main_scene();
  classifier.scope_push({.role = static_cast<uint8_t>(Role::Reflection)});
  for (int i = 0; i < 3; ++i) {
    const DrawClass& cls = classifier.route(main_draw());
    EXPECT_EQ(cls.kind, Kind::Foreign);
    EXPECT_EQ(cls.check, MainCheck::Role);
    EXPECT_TRUE(cls.roleOnly);
    EXPECT_FALSE(would_capture(cls));
  }
  const RoleScopeCounts& reflection = stats().underScope[static_cast<size_t>(Role::Reflection)];
  EXPECT_EQ(reflection.draws, 3u);
  EXPECT_EQ(reflection.asRole, 3u);
  EXPECT_EQ(reflection.main, 0u);
  EXPECT_EQ(reflection.captures, 0u);
  EXPECT_EQ(reflection.roleOnly, 3u);
  EXPECT_EQ(stats().captureCandidates, 0u);
}

TEST_F(RenderClassify, CameraAttachesToInnermostEntry) {
  classifier.camera(Role::Main);
  EXPECT_EQ(classifier.route(main_draw()).key.role, Role::Main);
  classifier.scope_push({});
  classifier.camera(Role::Reflection);
  EXPECT_EQ(classifier.route(main_draw()).key.role, Role::Reflection);
  classifier.object_begin(object(1));
  EXPECT_EQ(classifier.route(main_draw()).key.role, Role::Reflection);
  classifier.object_end();
  classifier.scope_pop();
  EXPECT_EQ(classifier.route(main_draw()).key.role, Role::Main);
  EXPECT_EQ(stats().hints.cameras, 2u);
  EXPECT_EQ(stats().hints.mainCameras, 1u);
}

TEST_F(RenderClassify, ScopeInsideObjectOwnsItsAttachments) {
  classifier.camera(Role::Main);
  classifier.object_begin(object(1));
  classifier.scope_push({});
  classifier.camera(Role::Reflection);
  classifier.material(5);
  const DrawClass& inside = classifier.route(main_draw());
  EXPECT_EQ(inside.key.role, Role::Reflection);
  EXPECT_EQ(inside.key.materialClass, 5u);
  classifier.scope_pop();
  const DrawClass& base = classifier.route(main_draw());
  EXPECT_EQ(base.key.role, Role::Main);
  EXPECT_EQ(base.key.materialClass, 0u);
  EXPECT_TRUE(base.key.object.valid);
  classifier.object_end();

  classifier.object_begin(object(2));
  classifier.camera(Role::Offscreen);
  classifier.scope_push({});
  classifier.camera(Role::Reflection);
  classifier.material(5);
  EXPECT_EQ(classifier.route(main_draw()).key.role, Role::Reflection);
  classifier.scope_pop();
  const DrawClass& objectCamera = classifier.route(main_draw());
  EXPECT_EQ(objectCamera.key.role, Role::Offscreen);
  EXPECT_EQ(objectCamera.key.materialClass, 0u);
  classifier.object_end();
  EXPECT_EQ(classifier.route(main_draw()).key.role, Role::Main);
}

TEST_F(RenderClassify, ScopeRoleOverridesInnerCamera) {
  classifier.scope_push({.role = static_cast<uint8_t>(Role::Shadow)});
  classifier.scope_push({});
  classifier.camera(Role::Main);
  const DrawClass& cls = classifier.route(main_draw());
  EXPECT_EQ(cls.key.role, Role::Shadow);
  EXPECT_EQ(cls.key.roleScopes, 1u << static_cast<uint32_t>(Role::Shadow));
}

TEST_F(RenderClassify, CategoriesAndFlagsAccumulate) {
  classifier.scope_push({.categories = category::Decal});
  classifier.object_begin(object(1, 0, category::Static, object_flag::Billboard));
  const DrawClass& cls = classifier.route(main_draw());
  EXPECT_EQ(cls.key.categories, category::Decal | category::Static);
  EXPECT_EQ(cls.key.flags, object_flag::Billboard);
  EXPECT_EQ(stats().categories[0], 1u);
  EXPECT_EQ(stats().categories[8], 1u);
  EXPECT_EQ(stats().categories[3], 0u);
  EXPECT_EQ(stats().unclassified, 0u);
}

TEST_F(RenderClassify, MaterialAndTevPolicyFollowInnermost) {
  classifier.material(7);
  classifier.object_begin(object(1));
  classifier.tev_policy(0x10);
  classifier.material(0x1234);
  const DrawClass& inside = classifier.route(main_draw());
  EXPECT_EQ(inside.key.tevPolicy, 0x10u);
  EXPECT_EQ(inside.key.materialClass, 0x34u);
  classifier.object_end();
  const DrawClass& after = classifier.route(main_draw());
  EXPECT_EQ(after.key.tevPolicy, 0u);
  EXPECT_EQ(after.key.materialClass, 7u);
  EXPECT_EQ(stats().tevPolicyDraws, 1u);
}

TEST_F(RenderClassify, NestedObjectBeginEndsOuter) {
  classifier.object_begin(object(1));
  classifier.material(9);
  classifier.object_begin(object(2));
  EXPECT_EQ(stats().hints.nestedObjects, 1u);
  const DrawClass& inner = classifier.route(main_draw());
  EXPECT_EQ(inner.key.object.id, 2u);
  EXPECT_EQ(inner.key.materialClass, 0u);
  classifier.object_end();
  EXPECT_FALSE(classifier.route(main_draw()).key.object.valid);
  classifier.object_end();
  EXPECT_EQ(stats().hints.objectUnderflow, 0u);
  classifier.object_end();
  EXPECT_EQ(stats().hints.objectUnderflow, 1u);
  classifier.end_frame();
  EXPECT_EQ(stats().hints.openObjects, 0u);
  EXPECT_EQ(stats().hints.errors(), 2u);
  EXPECT_EQ(stats().hints.objectBegins, 2u);
  EXPECT_EQ(stats().hints.objectEnds, 3u);
}

TEST_F(RenderClassify, KeyChangesAtObjectBoundary) {
  classifier.object_begin(object(1));
  EXPECT_FALSE(classifier.route(main_draw()).keyChanged);
  EXPECT_FALSE(classifier.route(main_draw()).keyChanged);
  classifier.object_end();
  classifier.object_begin(object(2));
  EXPECT_TRUE(classifier.route(main_draw()).keyChanged);
  classifier.scope_push({.categories = category::Decal});
  classifier.scope_pop();
  EXPECT_FALSE(classifier.route(main_draw()).keyChanged);
  ASSERT_TRUE(classifier.stage(Stage::SceneBegin));
  EXPECT_TRUE(classifier.route(main_draw()).keyChanged);
}

TEST_F(RenderClassify, ObjectOnlyKeyIgnoresSubId) {
  classifier.set_merge_key(MergeKey::ObjectOnly);
  classifier.object_begin(object(7, 0));
  classifier.route(main_draw());
  classifier.object_end();
  classifier.object_begin({.id = 7, .subId = 1, .generation = 3});
  const DrawClass& sameObject = classifier.route(main_draw());
  EXPECT_FALSE(sameObject.keyChanged);
  EXPECT_EQ(sameObject.key.object.subId, 1u);
  EXPECT_EQ(sameObject.key.object.generation, 3u);
  classifier.object_end();
  classifier.object_begin(object(8, 0));
  EXPECT_TRUE(classifier.route(main_draw()).keyChanged);

  classifier.set_merge_key(MergeKey::Full);
  classifier.object_end();
  classifier.object_begin(object(8, 1));
  EXPECT_TRUE(classifier.route(main_draw()).keyChanged);
}

TEST_F(RenderClassify, SuppressedCaptureOnly) {
  classifier.set_objects(true);
  classifier.route(main_draw());
  classifier.object_begin(object(3, 0, 0, object_flag::CaptureOnly));
  const DrawClass& hidden = classifier.route(main_draw());
  EXPECT_TRUE(hidden.suppressed);
  EXPECT_TRUE(hidden.keyChanged);
  classifier.object_end();
  EXPECT_FALSE(classifier.route(main_draw()).keyChanged);
  EXPECT_EQ(stats().draws, 3u);
  EXPECT_EQ(stats().suppressed, 1u);
  EXPECT_EQ(stats().roles[static_cast<size_t>(Role::None)], 2u);
  EXPECT_EQ(stats().kinds[static_cast<size_t>(Kind::Foreign)], 2u);
  EXPECT_EQ(stats().unclassified, 2u);
  EXPECT_TRUE(classifier.objects().empty());

  classifier.scope_push({.flags = object_flag::CaptureOnly});
  EXPECT_TRUE(classifier.route(main_draw()).suppressed);
  EXPECT_EQ(stats().suppressed, 2u);
}

TEST_F(RenderClassify, FakeShadowOverlayCounted) {
  classifier.scope_push({.categories = category::FakeShadow | category::Overlay});
  classifier.route(main_draw());
  classifier.scope_pop();
  classifier.scope_push({.categories = category::FakeShadow});
  classifier.route(main_draw());
  EXPECT_EQ(stats().fakeShadow, 2u);
  EXPECT_EQ(stats().fakeShadowOverlay, 1u);
}

TEST_F(RenderClassify, ScopeOverflowAndUnderflow) {
  for (uint32_t i = 0; i <= MaxScopeDepth; ++i) {
    classifier.scope_push({.categories = 1u << (i % 12)});
  }
  EXPECT_EQ(stats().hints.scopeOverflow, 1u);
  EXPECT_EQ(classifier.route(main_draw()).key.scope, fnv(MaxScopeDepth, InheritRole, 1u << 3, 0));
  for (uint32_t i = 0; i <= MaxScopeDepth; ++i) {
    classifier.scope_pop();
  }
  EXPECT_EQ(stats().hints.scopeUnderflow, 0u);
  const DrawClass& empty = classifier.route(main_draw());
  EXPECT_EQ(empty.key.scope, 0u);
  EXPECT_EQ(empty.key.categories, 0u);
  EXPECT_EQ(stats().unclassified, 1u);
  classifier.scope_pop();
  EXPECT_EQ(stats().hints.scopeUnderflow, 1u);
  classifier.end_frame();
  EXPECT_EQ(stats().hints.openScopes, 0u);
  EXPECT_EQ(stats().hints.errors(), 2u);
}

TEST_F(RenderClassify, ScopeHashFollowsDeclaration) {
  // Computed by hand from the 13-byte declaration, so the helper and the classifier cannot drift together
  constexpr uint32_t EmissiveHash = 0x62d7d819;
  constexpr uint32_t FoliageCpuDeformedHash = 0xd6f3de95;
  EXPECT_EQ(fnv(1, InheritRole, category::Emissive, 0), EmissiveHash);
  EXPECT_EQ(fnv(1, InheritRole, category::Foliage, object_flag::CpuDeformed), FoliageCpuDeformedHash);

  classifier.scope_push({.categories = category::Emissive});
  EXPECT_EQ(classifier.route(main_draw()).key.scope, EmissiveHash);
  classifier.scope_push(
      {.categories = category::Foliage, .flags = object_flag::CpuDeformed, .role = static_cast<uint8_t>(Role::Sky)});
  EXPECT_EQ(classifier.route(main_draw()).key.scope,
            fnv(2, static_cast<uint8_t>(Role::Sky), category::Foliage, object_flag::CpuDeformed));
  classifier.scope_pop();
  EXPECT_EQ(classifier.route(main_draw()).key.scope, EmissiveHash);
  classifier.scope_pop();
  classifier.scope_push({.categories = category::Foliage, .flags = object_flag::CpuDeformed});
  EXPECT_EQ(classifier.route(main_draw()).key.scope, FoliageCpuDeformedHash);
}

TEST_F(RenderClassify, OpenEntriesCountedAtEndFrame) {
  classifier.scope_push({});
  classifier.scope_push({});
  classifier.object_begin(object(1));
  classifier.end_frame();
  EXPECT_EQ(stats().hints.openScopes, 2u);
  EXPECT_EQ(stats().hints.openObjects, 1u);
  classifier.scope_pop();
  classifier.end_frame();
  EXPECT_EQ(stats().hints.openScopes, 2u);
  EXPECT_EQ(stats().hints.errors(), 3u);
}

TEST_F(RenderClassify, MarkersAreMonotonic) {
  const std::array<Stage, 6> sequence{Stage::SceneBegin, Stage::OpaqueDone, Stage::OpaqueDone,
                                      Stage::SceneBegin, Stage::PrePost,    Stage::AfterFrame};
  const std::array<bool, 6> accepted{true, true, false, false, true, false};
  const std::array<Window, 6> windows{Window::Opaque,      Window::Translucent, Window::Translucent,
                                      Window::Translucent, Window::Post,        Window::Post};
  for (size_t i = 0; i < sequence.size(); ++i) {
    EXPECT_EQ(classifier.stage(sequence[i]), accepted[i]) << i;
    EXPECT_EQ(classifier.route(main_draw()).key.window, windows[i]) << i;
  }
  EXPECT_EQ(stats().hints.markers, 3u);
  EXPECT_EQ(stats().hints.staleMarkers, 3u);
  EXPECT_EQ(stats().markerDraw[static_cast<size_t>(Stage::SceneBegin)], 0);
  EXPECT_EQ(stats().markerDraw[static_cast<size_t>(Stage::OpaqueDone)], 1);
  EXPECT_EQ(stats().markerDraw[static_cast<size_t>(Stage::PrePost)], 4);
  EXPECT_EQ(stats().markerDraw[static_cast<size_t>(Stage::SceneEnd)], -1);
  EXPECT_EQ(stats().markerDraw[static_cast<size_t>(Stage::AfterFrame)], -1);
  EXPECT_EQ(stats().windows[static_cast<size_t>(Window::Translucent)], 3u);

  classifier.begin_frame();
  ASSERT_TRUE(classifier.stage(Stage::FrameEnd));
  EXPECT_EQ(classifier.route(main_draw()).key.window, Window::Final);
  EXPECT_FALSE(classifier.stage(Stage::BeforeHud));
}

TEST_F(RenderClassify, ObjectTableAggregates) {
  classifier.set_objects(true);
  main_scene();
  classifier.object_begin(object(1, 0, category::Static));
  classifier.route(main_draw());
  classifier.object_end();
  classifier.scope_push({.categories = category::Decal, .role = static_cast<uint8_t>(Role::Reflection)});
  classifier.object_begin({.id = 1, .subId = 0, .generation = 2, .flags = object_flag::Teleport});
  classifier.route(main_draw());
  classifier.object_end();
  classifier.scope_pop();
  classifier.object_begin(object(1, 1));
  classifier.route(main_draw());
  classifier.object_end();

  const auto objects = classifier.objects();
  ASSERT_EQ(objects.size(), 2u);
  EXPECT_EQ(objects[0].ref.id, 1u);
  EXPECT_EQ(objects[0].ref.subId, 0u);
  EXPECT_EQ(objects[0].ref.generation, 2u);
  EXPECT_EQ(objects[0].categories, category::Static | category::Decal);
  EXPECT_EQ(objects[0].flags, object_flag::Teleport);
  EXPECT_EQ(objects[0].roles,
            (1u << static_cast<uint32_t>(Role::Main)) | (1u << static_cast<uint32_t>(Role::Reflection)));
  EXPECT_EQ(objects[0].draws, 2u);
  EXPECT_EQ(objects[0].mainDraws, 1u);
  EXPECT_EQ(objects[1].ref.subId, 1u);
  EXPECT_EQ(objects[1].roles, 1u << static_cast<uint32_t>(Role::Main));
  EXPECT_EQ(objects[1].draws, 1u);
  EXPECT_EQ(objects[1].mainDraws, 1u);

  classifier.set_objects(false);
  classifier.object_begin(object(9));
  classifier.route(main_draw());
  EXPECT_EQ(classifier.objects().size(), 2u);
}

TEST_F(RenderClassify, ObjectTableOverflow) {
  classifier.set_objects(true);
  for (uint32_t i = 0; i <= MaxObjectEntries; ++i) {
    classifier.object_begin(object(i + 1));
    classifier.route(main_draw());
    classifier.route(main_draw());
    classifier.object_end();
  }
  EXPECT_EQ(classifier.objects().size(), MaxObjectEntries);
  EXPECT_EQ(stats().droppedObjects, 1u);
  EXPECT_EQ(stats().hints.errors(), 0u);
  const ObjectStats* storage = classifier.objects().data();

  classifier.begin_frame();
  EXPECT_TRUE(classifier.objects().empty());
  EXPECT_EQ(stats().droppedObjects, 0u);
  classifier.object_begin(object(MaxObjectEntries + 1));
  classifier.route(main_draw());
  ASSERT_EQ(classifier.objects().size(), 1u);
  EXPECT_EQ(classifier.objects().data(), storage);
  EXPECT_EQ(classifier.objects()[0].draws, 1u);
}

TEST_F(RenderClassify, RecordsOnlyWhenRequested) {
  classifier.route(main_draw());
  EXPECT_TRUE(classifier.draws().empty());

  classifier.set_record(true);
  classifier.begin_frame();
  DrawInputs indexed = main_draw();
  indexed.indexed = true;
  indexed.vertices = 7;
  classifier.route(indexed);
  ASSERT_EQ(classifier.draws().size(), 1u);
  EXPECT_TRUE(classifier.draws()[0].indexed);
  EXPECT_EQ(classifier.draws()[0].vertices, 7u);
  EXPECT_EQ(classifier.draws()[0].cls.index, 0u);
  const ClassifiedDraw* storage = classifier.draws().data();
  for (uint32_t i = 1; i <= MaxRecordedDraws; ++i) {
    classifier.route(main_draw());
  }
  EXPECT_EQ(classifier.draws().size(), MaxRecordedDraws);
  EXPECT_EQ(stats().droppedRecords, 1u);
  EXPECT_EQ(classifier.draws().data(), storage);
  EXPECT_EQ(classifier.draws().back().cls.index, MaxRecordedDraws - 1);
}

TEST_F(RenderClassify, BeginFrameResets) {
  classifier.set_record(true);
  classifier.set_merge_key(MergeKey::ObjectOnly);
  classifier.camera(Role::Main);
  classifier.scope_push({.role = static_cast<uint8_t>(Role::Sky)});
  classifier.object_begin(object(1));
  classifier.material(3);
  ASSERT_TRUE(classifier.stage(Stage::SceneBegin));
  classifier.route(main_draw());
  const uint64_t frame = stats().frame;

  classifier.begin_frame();
  EXPECT_EQ(stats().frame, frame + 1);
  EXPECT_EQ(stats().draws, 0u);
  EXPECT_EQ(stats().hints.cameras, 0u);
  EXPECT_TRUE(classifier.draws().empty());
  const DrawClass& cls = classifier.route(main_draw());
  EXPECT_EQ(cls.key.role, Role::None);
  EXPECT_EQ(cls.key.window, Window::PreScene);
  EXPECT_EQ(cls.key.materialClass, 0u);
  EXPECT_FALSE(cls.key.object.valid);
  EXPECT_FALSE(cls.keyChanged);
  EXPECT_EQ(stats().unclassified, 1u);
  EXPECT_EQ(stats().skyMisses, 0u);
  EXPECT_EQ(classifier.draws().size(), 1u);
  EXPECT_TRUE(classifier.stage(Stage::SceneBegin));

  classifier.reset();
  EXPECT_EQ(stats().frame, 0u);
  classifier.route(main_draw());
  EXPECT_TRUE(classifier.draws().empty());
  classifier.begin_frame();
  EXPECT_EQ(stats().frame, 1u);
}

TEST_F(RenderClassify, NamesCoverEveryValue) {
  const auto check_names = [](size_t count, auto name) {
    std::set<std::string_view> seen;
    for (size_t i = 0; i < count; ++i) {
      const std::string_view text = name(i);
      EXPECT_FALSE(text.empty()) << i;
      EXPECT_NE(text, "unknown") << i;
      EXPECT_TRUE(seen.insert(text).second) << text;
    }
    EXPECT_EQ(name(count), "unknown");
  };
  check_names(RoleCount, [](size_t i) { return role_name(static_cast<Role>(i)); });
  check_names(KindCount, [](size_t i) { return kind_name(static_cast<Kind>(i)); });
  check_names(MainCheckCount, [](size_t i) { return check_name(static_cast<MainCheck>(i)); });
  check_names(WindowCount, [](size_t i) { return window_name(static_cast<Window>(i)); });

  constexpr std::array<uint32_t, 22> Defined{category::Static,     category::Dynamic,      category::Skinned,
                                             category::Terrain,    category::Foliage,      category::Water,
                                             category::Sky,        category::Particle,     category::Decal,
                                             category::Ui,         category::Player,       category::Emissive,
                                             category::Overlay,    category::NoShadowCast, category::NoShadowReceive,
                                             category::NoPbr,      category::NoCapture,    category::NoRt,
                                             category::ScreenRead, category::FakeShadow,   category::CloudReceiver,
                                             category::NoAo};
  uint32_t mask = 0;
  std::set<std::string_view> seen;
  for (const uint32_t bit : Defined) {
    mask |= bit;
    const std::string_view text = category_name(static_cast<uint32_t>(std::countr_zero(bit)));
    EXPECT_FALSE(text.empty()) << bit;
    EXPECT_TRUE(seen.insert(text).second) << text;
  }
  for (uint32_t bit = 0; bit < 32; ++bit) {
    if ((mask & (1u << bit)) == 0) {
      EXPECT_TRUE(category_name(bit).empty()) << bit;
    }
  }
}
} // namespace
} // namespace aurora::render::scene
