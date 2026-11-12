#include <gtest/gtest.h>

#include "render/fakeshadow/gate.hpp"
#include "render/settings.hpp"
#include "render/shadows/vsm_pages.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace aurora::render::fakeshadow::gate {
namespace {
using Cover = CasterCover;

constexpr std::array<GameShadowCaster, 1> OneCaster{{{.id = 1}}};

GateState covering_state() {
  GateState s;
  s.consumer = true;
  s.prevResolveRan = true;
  s.sunDeclared = true;
  s.skyVisibility = 1.f;
  s.sunGeometric = true;
  s.coverageComplete = true;
  s.game = GameShadows::Auto;
  return s;
}

testing::Inputs inputs(const GateState& state, std::vector<std::pair<uint64_t, Cover>> casters = {{1, Cover::Full}}) {
  return {.state = state, .casters = std::move(casters)};
}

GameShadowEntry entry(uint64_t key, std::span<const GameShadowCaster> casters = OneCaster,
                      ShadowSource source = ShadowSource::SunMoon, ShadowClass cls = ShadowClass::Real) {
  return {.key = key, .source = source, .cls = cls, .casters = casters};
}

class GameShadowGate : public ::testing::Test {
protected:
  void SetUp() override { testing::reset(); }
  void TearDown() override { testing::reset(); }

  // One latch, then one decision per entry, as the game thread does each frame
  ShadowDecision step(const GameShadowEntry& e, const testing::Inputs& in) {
    testing::latch(in);
    return decide(e);
  }

  uint32_t frames_to_skip(const GameShadowEntry& e, const testing::Inputs& in, uint32_t limit = 100) {
    for (uint32_t frame = 1; frame <= limit; ++frame) {
      if (step(e, in) == ShadowDecision::Skip) {
        return frame;
      }
    }
    return 0;
  }
};

TEST_F(GameShadowGate, CoveredTable) {
  const GateState s = covering_state();
  struct Row {
    ShadowSource source;
    ShadowClass cls;
    bool want;
  };
  constexpr std::array<Row, 14> Rows{{
      {ShadowSource::SunMoon, ShadowClass::Real, true},
      {ShadowSource::SunMoon, ShadowClass::Bridge, true},
      {ShadowSource::RoomKey, ShadowClass::Real, false},
      {ShadowSource::PointLight, ShadowClass::Real, false},
      {ShadowSource::RoomLight, ShadowClass::Real, false},
      {ShadowSource::Lantern, ShadowClass::Real, false},
      {ShadowSource::EffectLight, ShadowClass::Real, false},
      {ShadowSource::Vertical, ShadowClass::Real, false},
      {ShadowSource::Unknown, ShadowClass::Real, false},
      {ShadowSource::SunMoon, ShadowClass::Telegraph, false},
      {ShadowSource::SunMoon, ShadowClass::GroundMark, false},
      {ShadowSource::SunMoon, ShadowClass::Blob, true},
      {ShadowSource::PointLight, ShadowClass::Blob, true},
      {ShadowSource::PointLight, ShadowClass::Cue, true},
  }};
  for (const Row& row : Rows) {
    EXPECT_EQ(covered(s, false, row.source, row.cls), row.want)
        << source_name(row.source) << " " << class_name(row.cls);
  }
  for (const auto& [field, label] :
       {std::pair{&GateState::sunDeclared, "undeclared"}, std::pair{&GateState::sunGeometric, "no plan"}}) {
    GateState off = s;
    off.*field = false;
    EXPECT_FALSE(covered(off, false, ShadowSource::SunMoon, ShadowClass::Real)) << label;
    EXPECT_FALSE(covered(off, false, ShadowSource::SunMoon, ShadowClass::Blob)) << label;
  }
  GateState interior = s;
  interior.interior = true;
  EXPECT_FALSE(covered(interior, false, ShadowSource::SunMoon, ShadowClass::Real));
  GateState dark = s;
  dark.skyVisibility = 0.f;
  EXPECT_FALSE(covered(dark, false, ShadowSource::SunMoon, ShadowClass::Real));
  EXPECT_FALSE(covered(dark, false, ShadowSource::SunMoon, ShadowClass::Blob)) << "a blob under a roof is a local's";
  dark.localCovered = true;
  EXPECT_TRUE(covered(dark, false, ShadowSource::SunMoon, ShadowClass::Cue));
  EXPECT_FALSE(covered(s, true, ShadowSource::SunMoon, ShadowClass::Real)) << "a KEY_ROOM plan light is no sun";
  GateState keyRoom = s;
  keyRoom.keyRoomDeclared = true;
  EXPECT_TRUE(covered(keyRoom, true, ShadowSource::RoomKey, ShadowClass::Real));
  EXPECT_FALSE(covered(keyRoom, false, ShadowSource::RoomKey, ShadowClass::Real)) << "M13 never shadows KEY_ROOM";
}

TEST_F(GameShadowGate, HoldThirtyFrames) {
  const testing::Inputs in = inputs(covering_state());
  for (uint32_t frame = 1; frame < HoldFrames; ++frame) {
    EXPECT_EQ(step(entry(7), in), ShadowDecision::KeepHold) << frame;
  }
  for (uint32_t frame = 0; frame < 5; ++frame) {
    EXPECT_EQ(step(entry(7), in), ShadowDecision::Skip) << frame;
  }
  const GameShadowStats stats = gate::stats(false);
  EXPECT_EQ(stats.total.decisions[static_cast<size_t>(ShadowDecision::KeepHold)], HoldFrames - 1);
  EXPECT_EQ(stats.total.decisions[static_cast<size_t>(ShadowDecision::Skip)], 5u);
  EXPECT_EQ(stats.total.transitions, 1u);
  EXPECT_EQ(stats.total.flipViolations, 0u);
  EXPECT_EQ(stats.current.entries, 1u) << "per-frame counts restart at each latch";
}

TEST_F(GameShadowGate, HysteresisToggle) {
  for (const uint32_t period : {1u, 15u, 29u, 30u, 31u, 60u}) {
    SCOPED_TRACE(period);
    testing::reset();
    uint64_t changes = 0;
    uint64_t violations = 0;
    uint64_t lastChange = 0;
    bool haveChange = false;
    bool wasSkipped = false;
    for (uint32_t frame = 0; frame < 600; ++frame) {
      GateState s = covering_state();
      const bool on = (frame / period) % 2 == 0;
      s.sunGeometric = on;
      s.frame = frame + 1;
      const ShadowDecision d = step(entry(9), inputs(s));
      const bool skipped = d == ShadowDecision::Skip;
      if (!on) {
        EXPECT_NE(d, ShadowDecision::Skip) << "an uncovered frame always keeps, frame " << frame;
      }
      if (skipped != wasSkipped) {
        ++changes;
        violations += haveChange && frame + 1 - lastChange < HoldFrames ? 1 : 0;
        haveChange = true;
        lastChange = frame + 1;
      }
      wasSkipped = skipped;
    }
    const GameShadowStats stats = gate::stats(false);
    EXPECT_EQ(stats.total.transitions, changes);
    EXPECT_EQ(stats.total.flipViolations, violations);
    if (period < HoldFrames) {
      EXPECT_EQ(changes, 0u) << "an entry covered for less than 30 frames never skips";
    }
    if (period == 30 || period == 31) {
      EXPECT_GT(violations, 0u);
      EXPECT_EQ(violations, stats.total.safetyReturns[static_cast<size_t>(ShadowDecision::KeepSource)]);
    }
    if (period == 60) {
      EXPECT_GT(changes, 0u);
      EXPECT_EQ(violations, 0u);
    }
  }
}

TEST_F(GameShadowGate, CollidingKeysSurviveNeighbourExpiry) {
  const uint64_t a = 1000;
  uint64_t b = a + 1;
  while (testing::table_slot(b) != testing::table_slot(a)) {
    ++b;
  }
  const testing::Inputs in = inputs(covering_state());
  for (uint32_t frame = 1; frame <= 10; ++frame) {
    testing::latch(in);
    EXPECT_EQ(decide(entry(a)), ShadowDecision::KeepHold);
    EXPECT_EQ(decide(entry(b)), ShadowDecision::KeepHold);
  }
  for (uint32_t frame = 11; frame < HoldFrames; ++frame) {
    EXPECT_EQ(step(entry(b), in), ShadowDecision::KeepHold) << frame;
  }
  EXPECT_EQ(step(entry(b), in), ShadowDecision::Skip) << "b keeps its hold after a, ahead of it, expired";
}

TEST_F(GameShadowGate, UnseenKeyResets) {
  const testing::Inputs in = inputs(covering_state());
  for (uint32_t frame = 1; frame <= 20; ++frame) {
    (void)step(entry(3), in);
  }
  testing::latch(in);
  EXPECT_EQ(frames_to_skip(entry(3), in), HoldFrames) << "a frame without the entry restarts its hold";
}

TEST_F(GameShadowGate, NewCasterDoesNotReset) {
  const testing::Inputs in = inputs(covering_state(), {{1, Cover::Full}, {2, Cover::Full}});
  for (uint32_t frame = 1; frame <= 15; ++frame) {
    EXPECT_EQ(step(entry(4), in), ShadowDecision::KeepHold);
  }
  constexpr std::array<GameShadowCaster, 2> Two{{{.id = 1}, {.id = 2}}};
  EXPECT_EQ(frames_to_skip(entry(4, Two), in), HoldFrames - 15);
}

TEST_F(GameShadowGate, ProvisionalFailureCountsMissing) {
  const testing::Inputs in = inputs(covering_state());
  for (uint32_t frame = 1; frame < HoldFrames; ++frame) {
    (void)step(entry(5), in);
  }
  constexpr std::array<GameShadowCaster, 2> WithNew{{{.id = 1}, {.id = 8, .enteredNow = true}}};
  EXPECT_EQ(step(entry(5, WithNew), in), ShadowDecision::Skip) << "a model entering this frame passes provisionally";
  EXPECT_EQ(gate::stats(false).current.provisional, 1u);
  testing::latch(in);
  GameShadowStats stats = gate::stats(false);
  EXPECT_EQ(stats.total.missingCasterLatch, 1u) << "the new model never reached the next caster set";
  EXPECT_EQ(stats.total.provisionalFailed, 1u);
  (void)decide(entry(5, WithNew));
  testing::latch(inputs(covering_state(), {{1, Cover::Full}, {8, Cover::Full}}));
  stats = gate::stats(false);
  EXPECT_EQ(stats.total.missingCasterLatch, 1u);
  EXPECT_EQ(stats.total.provisionalFailed, 1u);
}

TEST_F(GameShadowGate, EveryCasterRequired) {
  constexpr std::array<GameShadowCaster, 2> Casters{{{.id = 1}, {.id = 3}}};
  const testing::Inputs in = inputs(covering_state());
  for (uint32_t frame = 1; frame <= 40; ++frame) {
    EXPECT_EQ(step(entry(6, Casters), in), ShadowDecision::KeepCaster) << frame;
  }
}

TEST_F(GameShadowGate, PartialAndExcludedOnlyDoNotBlock) {
  constexpr std::array<GameShadowCaster, 3> Casters{{{.id = 1}, {.id = 2}, {.id = 4}}};
  const testing::Inputs in =
      inputs(covering_state(), {{1, Cover::Full}, {2, Cover::Partial}, {4, Cover::ExcludedOnly}});
  EXPECT_EQ(frames_to_skip(entry(6, Casters), in), HoldFrames);
  EXPECT_EQ(gate::stats(false).total.partialCaster, HoldFrames);
}

TEST_F(GameShadowGate, ExcludedOnlyAloneKeeps) {
  constexpr std::array<GameShadowCaster, 1> Casters{{{.id = 4}}};
  const testing::Inputs in = inputs(covering_state(), {{4, Cover::ExcludedOnly}});
  for (uint32_t frame = 1; frame <= 40; ++frame) {
    EXPECT_EQ(step(entry(6, Casters), in), ShadowDecision::KeepCaster) << frame;
  }
}

TEST_F(GameShadowGate, TruncatedIsUnknown) {
  const testing::Inputs in = inputs(covering_state());
  GameShadowEntry truncated = entry(6);
  truncated.castersTruncated = true;
  EXPECT_EQ(step(truncated, in), ShadowDecision::KeepCasterUnknown);
  EXPECT_EQ(step(entry(7, {}), in), ShadowDecision::KeepCasterUnknown);
}

TEST_F(GameShadowGate, CoverageIncompleteFreezes) {
  const testing::Inputs in = inputs(covering_state());
  for (uint32_t frame = 1; frame <= 10; ++frame) {
    (void)step(entry(2), in);
  }
  GateState loading = covering_state();
  loading.coverageComplete = false;
  for (uint32_t frame = 0; frame < 5; ++frame) {
    EXPECT_EQ(step(entry(2), inputs(loading)), ShadowDecision::KeepCoverage);
  }
  EXPECT_EQ(frames_to_skip(entry(2), in), HoldFrames - 10) << "the hold resumes where the gap froze it";
  for (uint32_t frame = 0; frame < 5; ++frame) {
    EXPECT_EQ(step(entry(2), inputs(loading)), ShadowDecision::Skip) << "a skipped entry stays skipped";
  }
  EXPECT_EQ(gate::stats(false).total.transitions, 1u);
}

TEST_F(GameShadowGate, PlanStopKeepsSameFrame) {
  ASSERT_EQ(frames_to_skip(entry(1), inputs(covering_state())), HoldFrames);
  GateState stopped = covering_state();
  stopped.sunGeometric = false;
  EXPECT_EQ(step(entry(1), inputs(stopped)), ShadowDecision::KeepSource);
  const GameShadowStats stats = gate::stats(false);
  EXPECT_EQ(stats.total.safetyReturns[static_cast<size_t>(ShadowDecision::KeepSource)], 1u);
  EXPECT_EQ(stats.total.safetyReturnsPerKeyMax, 1u);
  GateState noConsumer = covering_state();
  noConsumer.consumer = false;
  ASSERT_EQ(frames_to_skip(entry(1), inputs(covering_state())), HoldFrames);
  EXPECT_EQ(step(entry(1), inputs(noConsumer)), ShadowDecision::KeepNoConsumer);
  EXPECT_EQ(gate::stats(false).total.safetyReturnsPerKeyMax, 2u);
}

TEST_F(GameShadowGate, DecisionsArePure) {
  const auto run = [&] {
    testing::reset();
    std::vector<ShadowDecision> out;
    constexpr std::array<GameShadowCaster, 2> Casters{{{.id = 1}, {.id = 2, .enteredNow = true}}};
    for (uint32_t frame = 0; frame < 120; ++frame) {
      GateState s = covering_state();
      s.sunGeometric = frame % 50 < 40;
      s.coverageComplete = frame % 17 != 0;
      testing::latch(inputs(s, {{1, Cover::Full}, {2, frame % 3 == 0 ? Cover::Missing : Cover::Partial}}));
      for (uint64_t key = 1; key <= 4; ++key) {
        const ShadowDecision first = decide(entry(key, key % 2 == 0 ? std::span<const GameShadowCaster>{Casters}
                                                                    : std::span<const GameShadowCaster>{OneCaster}));
        EXPECT_EQ(decide(entry(key)), first) << "a second decision in one frame repeats the first";
        out.push_back(first);
      }
    }
    return out;
  };
  const std::vector<ShadowDecision> a = run();
  const std::vector<ShadowDecision> b = run();
  EXPECT_EQ(a, b);
  EXPECT_EQ(gate::stats(false).total.entries, 120u * 4u) << "a repeated decision is counted once";
}

TEST_F(GameShadowGate, DoubleDarkCountsRev3Classes) {
  const testing::Inputs in = inputs(covering_state(), {{1, Cover::Full}});
  testing::latch(in);
  constexpr std::array<GameShadowCaster, 2> OneMissing{{{.id = 1}, {.id = 3}}};
  EXPECT_EQ(decide(entry(1, OneMissing)), ShadowDecision::KeepCaster);
  EXPECT_EQ(gate::stats(false).current.doubleDarkEntries, 1u) << "the light is covered and a caster casts";
  EXPECT_EQ(decide(entry(2)), ShadowDecision::KeepHold);
  EXPECT_EQ(decide(entry(3, OneCaster, ShadowSource::Vertical)), ShadowDecision::KeepClass);
  EXPECT_EQ(decide(entry(4, OneCaster, ShadowSource::SunMoon, ShadowClass::GroundMark)), ShadowDecision::KeepClass);
  EXPECT_EQ(decide(entry(5, OneCaster, ShadowSource::PointLight)), ShadowDecision::KeepSource);
  EXPECT_EQ(gate::stats(false).current.doubleDarkEntries, 1u);
  GateState interior = covering_state();
  interior.interior = true;
  testing::latch(inputs(interior));
  EXPECT_EQ(decide(entry(1, OneMissing)), ShadowDecision::KeepInterior);
  GateState kept = covering_state();
  kept.game = GameShadows::Keep;
  testing::latch(inputs(kept));
  EXPECT_EQ(decide(entry(1, OneMissing)), ShadowDecision::KeepSetting);
  const GameShadowStats stats = gate::stats(false);
  EXPECT_EQ(stats.current.userKept, 1u);
  EXPECT_EQ(stats.total.doubleDarkEntries, 1u);
}

TEST_F(GameShadowGate, CastBitsMissingCounts) {
  constexpr std::array<GameShadowCaster, 2> Casters{{{.id = 1}, {.id = 2}}};
  testing::Inputs in = inputs(covering_state(), {{1, Cover::Full}, {2, Cover::Full}});
  in.slots = {{1, {5}}, {2, {9}}};
  for (uint32_t frame = 1; frame < HoldFrames; ++frame) {
    (void)step(entry(1, Casters), in);
  }
  const uint64_t skipFrame = decision_frame() + 1;
  EXPECT_EQ(step(entry(1, Casters), in), ShadowDecision::Skip);
  EXPECT_EQ(step(entry(1, Casters), in), ShadowDecision::Skip);
  GameShadowStats stats = gate::stats(true);
  EXPECT_EQ(stats.total.missingCasterCast, 0u);
  EXPECT_EQ(stats.current.castBitsPending, 2u) << "bits not read back yet never count as missing";
  std::array<uint32_t, CastBitWords> words{};
  words[0] = 1u << 5;
  testing::deliver_cast_bits(skipFrame, words);
  stats = gate::stats(true);
  EXPECT_EQ(stats.total.missingCasterCast, 1u) << "caster 2 drew nothing into the pages";
  EXPECT_EQ(stats.total.missingCasterLatch, 0u);
  EXPECT_EQ(stats.current.castBitsPending, 0u);
}

TEST_F(GameShadowGate, RemoveKeepsMarkers) {
  GateState s;
  s.game = GameShadows::Remove;
  testing::latch({.state = s, .available = false});
  for (const ShadowClass cls : {ShadowClass::Real, ShadowClass::Blob, ShadowClass::Cue, ShadowClass::Bridge}) {
    EXPECT_EQ(decide(entry(static_cast<uint64_t>(cls) + 1, {}, ShadowSource::Unknown, cls)), ShadowDecision::Skip)
        << class_name(cls);
  }
  EXPECT_EQ(decide(entry(20, {}, ShadowSource::Unknown, ShadowClass::GroundMark)), ShadowDecision::KeepClass);
  EXPECT_EQ(decide(entry(21, {}, ShadowSource::Unknown, ShadowClass::Telegraph)), ShadowDecision::KeepClass);
  EXPECT_EQ(decide(entry(22)), ShadowDecision::Skip);
  testing::latch({.state = s, .available = false});
  EXPECT_EQ(gate::stats(true).total.missingCasterLatch, 0u) << "Remove owes no caster for what it skips";
}

TEST_F(GameShadowGate, RemoveWithRenderOff) {
  EXPECT_EQ(setting_decision(GameShadows::Remove, ShadowClass::Real), ShadowDecision::Skip);
  EXPECT_EQ(setting_decision(GameShadows::Remove, ShadowClass::GroundMark), ShadowDecision::KeepClass);
  EXPECT_EQ(setting_decision(GameShadows::Remove, ShadowClass::Telegraph), ShadowDecision::KeepClass);
  EXPECT_EQ(setting_decision(GameShadows::Keep, ShadowClass::Real), ShadowDecision::KeepSetting);
  EXPECT_EQ(setting_decision(GameShadows::Auto, ShadowClass::Real), std::nullopt);
  EXPECT_EQ(decide(entry(1)), ShadowDecision::KeepSetting) << "nothing latched reads as Keep";
  GateState s = covering_state();
  testing::latch({.state = s, .available = false});
  EXPECT_EQ(decide(entry(1)), ShadowDecision::KeepNoConsumer) << "Auto without render or hints keeps";
}

TEST_F(GameShadowGate, KeepCuesKeepsCue) {
  GateState s = covering_state();
  s.keepCues = true;
  EXPECT_EQ(step(entry(1, OneCaster, ShadowSource::SunMoon, ShadowClass::Cue), inputs(s)), ShadowDecision::KeepClass);
  testing::reset();
  EXPECT_EQ(frames_to_skip(entry(1, OneCaster, ShadowSource::SunMoon, ShadowClass::Cue), inputs(covering_state())),
            HoldFrames);
}

TEST_F(GameShadowGate, OverflowKeeps) {
  testing::latch(inputs(covering_state()));
  for (uint64_t key = 1; key <= TableSlots; ++key) {
    EXPECT_EQ(decide(entry(key)), ShadowDecision::KeepHold);
  }
  EXPECT_EQ(decide(entry(TableSlots + 1)), ShadowDecision::KeepOverflow);
  const GameShadowStats stats = gate::stats(false);
  EXPECT_EQ(stats.current.overflow, 1u);
  EXPECT_EQ(stats.current.doubleDarkEntries, 1u);
  EXPECT_EQ(entry_rows().size(), TableSlots);
}

TEST_F(GameShadowGate, CastBitsCopyFollowsConsumer) {
  const detail::PassDesc desc = shadows::vsm::cast_bits_pass_desc();
  EXPECT_EQ(desc.name, CastBitsPass);
  EXPECT_EQ(desc.order, CastBitsOrder);
  EXPECT_EQ(desc.kind, PassKind::Copy);
  EXPECT_EQ(desc.stage, Stage::OpaqueDone);
  EXPECT_FALSE(cast_bits_wanted());
  GateState s = covering_state();
  s.consumer = false;
  testing::latch(inputs(s));
  EXPECT_FALSE(cast_bits_wanted()) << "without a consumer nothing can skip, so nothing needs its bits";
  testing::latch(inputs(covering_state()));
  EXPECT_TRUE(cast_bits_wanted());
  testing::latch({.state = covering_state(), .available = false});
  EXPECT_FALSE(cast_bits_wanted());
}

TEST_F(GameShadowGate, EnvAndTagsParse) {
  using detail::GameShadowsEnv;
  EXPECT_EQ(detail::parse_game_shadows_env("keep"), GameShadowsEnv::Keep);
  EXPECT_EQ(detail::parse_game_shadows_env("auto"), GameShadowsEnv::Auto);
  EXPECT_EQ(detail::parse_game_shadows_env("remove"), GameShadowsEnv::Remove);
  EXPECT_EQ(detail::parse_game_shadows_env("drop"), GameShadowsEnv::Drop);
  EXPECT_EQ(detail::parse_game_shadows_env("count"), GameShadowsEnv::Count);
  EXPECT_EQ(detail::parse_game_shadows_env("count=all"), GameShadowsEnv::CountAll);
  EXPECT_EQ(detail::parse_game_shadows_env("on"), std::nullopt);
  const uint32_t subId =
      2u | (4u << TagClassShift) | (static_cast<uint32_t>(ShadowDecision::KeepHold) << TagDecisionShift);
  EXPECT_EQ(tag_class(subId), ShadowClass::GroundMark);
  EXPECT_EQ(tag_decision(subId), ShadowDecision::KeepHold);
  EXPECT_EQ(tag_class(0), ShadowClass::Real);
  EXPECT_EQ(tag_decision(0), ShadowDecision::None);
  EXPECT_EQ(decision_name(ShadowDecision::KeepOverflow), "keepOverflow");
  EXPECT_EQ(ShadowDecisionCount, 12u);
}
} // namespace
} // namespace aurora::render::fakeshadow::gate
