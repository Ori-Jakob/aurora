#pragma once

#include "../graph.hpp"
#include "../shaders.hpp"

#include <aurora/render.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

namespace aurora::render {
struct Caps;
} // namespace aurora::render

namespace aurora::render::detail {
struct SceneHints;
struct StageData;
} // namespace aurora::render::detail

namespace aurora::render::shadows::vsm {
inline constexpr uint32_t PageTexels = 128;
inline constexpr uint32_t WindowPages = 32;
inline constexpr uint32_t WindowTexels = PageTexels * WindowPages;
inline constexpr uint32_t PagesPerView = WindowPages * WindowPages;
inline constexpr uint32_t SunLevels = 12;
inline constexpr uint32_t MaxViews = 16;
inline constexpr uint32_t MaxVirtualPages = MaxViews * PagesPerView;
inline constexpr uint32_t MinPoolPages = 256;
inline constexpr uint32_t MaxPoolPages = 4096;
inline constexpr uint32_t MinPageBudget = 8;
inline constexpr uint32_t MaxPageBudget = 256;
inline constexpr double Texel0Metres = 0.0025;
inline constexpr double LatchTexels = 4.0;
inline constexpr double LatchHeightMetres = 5.0;
inline constexpr double MaxLatchDegrees = 1.0;
inline constexpr double LowSunOnDegrees = 3.0;
inline constexpr double LowSunOffDegrees = 2.5;
inline constexpr uint32_t FilterMarginTexels = 4;
inline constexpr uint32_t MaxPassStorageBuffers = 8;
inline constexpr uint32_t RasterVertexStorage = 6;
inline constexpr double MinDepthRange = 32768.0;
inline constexpr double MaxDepthRange = 1048576.0;
inline constexpr double DepthRangePerRadius = 64.0;
inline constexpr double AnchorSnap = 1024.0;
inline constexpr double AnchorReach = 1048576.0;
inline constexpr double DefaultUnitsPerMetre = 100.0;
inline constexpr uint32_t MaxPairs = 1u << 19;
inline constexpr uint32_t PhysWords = 2;
inline constexpr uint32_t StatCount = 128;
inline constexpr uint32_t UpdateWorkgroup = 64;
inline constexpr uint32_t MaxConsumerTerms = 8;

inline constexpr std::string_view UpdatePass = "vsm.update";
inline constexpr std::string_view CheckPass = "vsm.check";
inline constexpr std::string_view Table = "vsm.pt";
inline constexpr std::string_view Phys = "vsm.phys";
inline constexpr std::string_view ViewsName = "vsm.views";
inline constexpr std::string_view StatsName = "vsm.stats";
inline constexpr std::string_view Pool = "vsm.pool";
inline constexpr std::string_view Shader = "shadows/vsm_pages";

namespace entry {
inline constexpr uint32_t PhysMask = 0xFFFu;
inline constexpr uint32_t Fresh = 1u << 12;
inline constexpr uint32_t Allocated = 1u << 13;
inline constexpr uint32_t Cached = 1u << 14;
inline constexpr uint32_t Dirty = 1u << 15;
inline constexpr uint32_t Priority = 1u << 16;
inline constexpr uint32_t RenderNow = 1u << 17;
inline constexpr uint32_t RenderIndexShift = 18;
inline constexpr uint32_t RenderIndexMask = 0xFFFu;
inline constexpr uint32_t Skip = 1u << 30;
inline constexpr uint32_t Check = 1u << 31;
} // namespace entry

namespace view_flag {
inline constexpr uint32_t Active = 1u << 0;
inline constexpr uint32_t Wrap = 1u << 1;
inline constexpr uint32_t Freed = 1u << 2;
inline constexpr uint32_t Recentred = 1u << 3;
inline constexpr uint32_t Mask = 0xFFu;
inline constexpr uint32_t KindShift = 8;
inline constexpr uint32_t TableShift = 16;
} // namespace view_flag

enum class ViewKind : uint8_t { None, SunLevel, Spot, CubeFace };

namespace header_flag {
inline constexpr uint32_t Producing = 1u << 0;
inline constexpr uint32_t Full = 1u << 1;
inline constexpr uint32_t Exempt = 1u << 2;
inline constexpr uint32_t Relatch = 1u << 3;
inline constexpr uint32_t Held = 1u << 4;
inline constexpr uint32_t Check = 1u << 5;
} // namespace header_flag

namespace full_reason {
inline constexpr uint32_t First = 1u << 0;
inline constexpr uint32_t Gap = 1u << 1;
inline constexpr uint32_t ChainIncomplete = 1u << 2;
inline constexpr uint32_t PoolGeneration = 1u << 3;
inline constexpr uint32_t KeyLight = 1u << 4;
inline constexpr uint32_t Anchor = 1u << 5;
inline constexpr uint32_t Check = 1u << 6;
inline constexpr uint32_t RetainedReset = 1u << 7;
inline constexpr uint32_t ClearInstances = 1u << 8;
inline constexpr uint32_t Unapplied = 1u << 9;
inline constexpr uint32_t SceneEpoch = 1u << 10;
inline constexpr uint32_t Count = 11;
} // namespace full_reason

namespace stat {
inline constexpr uint32_t FreedWrap = 0;
inline constexpr uint32_t FreedRelatch = 1;
inline constexpr uint32_t FreedRecentre = 2;
inline constexpr uint32_t FreedFull = 3;
inline constexpr uint32_t ScanValid = 4;
inline constexpr uint32_t ScanCasters = 5;
inline constexpr uint32_t Changed = 6;
inline constexpr uint32_t ChangedStatic = 7;
inline constexpr uint32_t ChangedImmobile = 8;
inline constexpr uint32_t ChangedPlayer = 9;
inline constexpr uint32_t ChangedPalette = 10;
inline constexpr uint32_t PendingStream = 11;
inline constexpr uint32_t PendingCasters = 12;
inline constexpr uint32_t ChunkNeed = 13;
inline constexpr uint32_t ReasonFirst = 14;
inline constexpr uint32_t NonPlayerOther = 23;
inline constexpr uint32_t FootprintPages = 24;
inline constexpr uint32_t FootprintPriority = 25;
inline constexpr uint32_t DirtiedPages = 26;
inline constexpr uint32_t DirtiedPriority = 27;
inline constexpr uint32_t Requested = 28;
inline constexpr uint32_t NeedPriority = 29;
inline constexpr uint32_t NeedNew = 30;
inline constexpr uint32_t NeedReuse = 31;
inline constexpr uint32_t Hits = 32;
inline constexpr uint32_t FreedDirty = 33;
inline constexpr uint32_t Evictable = 34;
inline constexpr uint32_t FreePages = 35;
inline constexpr uint32_t Rendered = 36;
inline constexpr uint32_t DeferredNew = 37;
inline constexpr uint32_t DeferredReuse = 38;
inline constexpr uint32_t OverflowPool = 39;
inline constexpr uint32_t PlayerDeferred = 40;
inline constexpr uint32_t Evicted = 41;
inline constexpr uint32_t Committed = 42;
inline constexpr uint32_t CommitSkipped = 43;
inline constexpr uint32_t SunUnshadowed = 64;
inline constexpr uint32_t SunSamples = 65;
inline constexpr uint32_t SunSampledUnshadowed = 66;
inline constexpr uint32_t SunSampledFallback = 67;
inline constexpr uint32_t SunSampledStale = 68;
inline constexpr uint32_t SunSampledOutOfRange = 69;
} // namespace stat

namespace chain_bit {
inline constexpr uint32_t Update = 1u << 0;
inline constexpr uint32_t Scan = 1u << 1;
inline constexpr uint32_t ScanList = 1u << 2;
inline constexpr uint32_t Dirty = 1u << 3;
inline constexpr uint32_t Mark = 1u << 4;
inline constexpr uint32_t Classify = 1u << 5;
inline constexpr uint32_t AllocLists = 1u << 6;
inline constexpr uint32_t Assign = 1u << 7;
inline constexpr uint32_t Clear = 1u << 8;
inline constexpr uint32_t Commit = 1u << 9;
inline constexpr uint32_t ChunkBounds = 1u << 10;
inline constexpr uint32_t CullCandidates = 1u << 11;
inline constexpr uint32_t CullList = 1u << 12;
inline constexpr uint32_t CullCount = 1u << 13;
inline constexpr uint32_t CullScan = 1u << 14;
inline constexpr uint32_t CullSkip = 1u << 15;
inline constexpr uint32_t CullEmit = 1u << 16;
inline constexpr uint32_t Raster = 1u << 17;
inline constexpr uint32_t CheckCopy = 1u << 18;
inline constexpr uint32_t CheckCompare = 1u << 19;
} // namespace chain_bit

struct ViewsHeader {
  uint32_t frameIndex = 0;
  uint32_t clock = 0;
  uint32_t flags = 0;
  uint32_t fullReason = 0;
  uint32_t viewCount = 0;
  float anchorHi[3]{};
  float anchorLo[3]{};
  uint32_t budget = 0;
  uint32_t poolPages = 0;
  uint32_t poolPagesX = 0;
  float levelBias = 0.f;
  float texel0 = 0.f;
  float sunDir[3]{};
  uint32_t lightId = 0;
  uint32_t maxPairs = 0;
  uint32_t pairsHeader[3]{};
  uint32_t generation = 0;
  uint32_t reserved[39]{};
};
static_assert(sizeof(ViewsHeader) == 64 * 4);

struct ViewRecord {
  float camRel[12]{};
  float anchorRel[12]{};
  int32_t origin[2]{};
  int32_t prevOrigin[2]{};
  float texel = 0.f;
  float depthScale = 0.f;
  uint32_t flags = 0;
  uint32_t pages = 0;

  [[nodiscard]] uint32_t view_flags() const noexcept { return flags & view_flag::Mask; }
  [[nodiscard]] ViewKind kind() const noexcept {
    return static_cast<ViewKind>((flags >> view_flag::KindShift) & 0xFFu);
  }
  [[nodiscard]] uint32_t table_offset() const noexcept { return flags >> view_flag::TableShift; }
  [[nodiscard]] uint32_t pages_x() const noexcept { return pages & 0xFFFFu; }
  [[nodiscard]] uint32_t pages_y() const noexcept { return pages >> 16; }
  [[nodiscard]] std::array<float, 3> direction() const noexcept;
};
static_assert(sizeof(ViewRecord) == 32 * 4);

struct Views {
  ViewsHeader header;
  std::array<ViewRecord, MaxViews> views{};
};
static_assert(sizeof(Views) == 2304);
inline constexpr uint64_t ViewsBytes = sizeof(Views);
inline constexpr uint64_t TableBytes = uint64_t{MaxVirtualPages} * 4;
inline constexpr uint64_t PhysBytes = uint64_t{MaxPoolPages} * PhysWords * 4;
inline constexpr uint64_t StatsBytes = uint64_t{StatCount} * 4;

enum class PoolFormat : uint8_t { DepthAtlas, U32Buffer };
[[nodiscard]] PoolFormat pool_format() noexcept;

struct PoolExtent {
  uint32_t pages = 0;
  uint32_t pagesX = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  bool fits = false;
  bool reduced = false;
};
[[nodiscard]] PoolExtent pool_extent(uint32_t pages, uint32_t maxDimension) noexcept;
[[nodiscard]] TexDesc pool_desc() noexcept;

struct SunSelection {
  bool found = false;
  uint32_t lightId = 0;
  uint32_t hintIndex = 0;
  uint32_t flags = 0;
  std::array<double, 3> dirWorld{};
  double elevationDeg = 0.0;
};
[[nodiscard]] SunSelection select_sun(const detail::SceneHints& hints) noexcept;

enum class Idle : uint8_t { None, Inactive, Pipelines, NoCamera, NoSun, SunDisabled, LowSun };
inline constexpr size_t IdleCount = 7;
[[nodiscard]] std::string_view idle_name(Idle reason) noexcept;

struct SunPlan {
  bool producing = false;
  uint32_t lightId = 0;
  uint32_t lightFlags = 0;
  std::array<double, 3> dirWorld{};
  double elevationDeg = 0.0;
  uint64_t frame = 0;
  Idle idle = Idle::Inactive;
};

struct PlanInputs {
  bool active = false;
  bool chainReady = false;
  bool mainCamera = false;
  bool noSun = false;
  SunSelection sun;
  uint64_t frame = 0;
};
[[nodiscard]] SunPlan decide_plan(const PlanInputs& in, bool& lowSunOn) noexcept;
[[nodiscard]] const SunPlan& plan() noexcept;

[[nodiscard]] constexpr uint32_t level_view(uint32_t level) noexcept { return SunLevels - 1 - level; }
[[nodiscard]] constexpr uint32_t page_index(uint32_t tableOffset, int32_t px, int32_t py) noexcept {
  return tableOffset + (static_cast<uint32_t>(px) & (WindowPages - 1)) +
         WindowPages * (static_cast<uint32_t>(py) & (WindowPages - 1));
}
[[nodiscard]] constexpr int32_t abs_page(int32_t origin, uint32_t slot) noexcept {
  return origin + static_cast<int32_t>((slot - static_cast<uint32_t>(origin)) & (WindowPages - 1));
}
[[nodiscard]] int32_t window_origin(double texel) noexcept;
[[nodiscard]] bool kept_after_shift(uint32_t sx, uint32_t sy, const std::array<int32_t, 2>& prev,
                                    const std::array<int32_t, 2>& next) noexcept;
[[nodiscard]] double level_texel(double texel0, uint32_t level) noexcept;
[[nodiscard]] double depth_range(double texel0, uint32_t level) noexcept;
[[nodiscard]] double latch_threshold(uint32_t level) noexcept;
[[nodiscard]] std::array<float, 3> view_coords(const ViewRecord& view, const std::array<float, 3>& pRel) noexcept;
[[nodiscard]] std::array<float, 3> view_coords_anchor(const ViewRecord& view,
                                                      const std::array<float, 3>& pAnchor) noexcept;
[[nodiscard]] uint32_t select_level(const Views& views, float footprint, const std::array<float, 3>& pRel,
                                    float bias) noexcept;

struct LevelLatch {
  bool valid = false;
  std::array<double, 3> dir{};
  double zc = 0.0;
};

struct LatchDecision {
  bool all = false;
  bool lagged = false;
  int32_t level = -1;
  bool recentre = false;
};
[[nodiscard]] LatchDecision latch_schedule(std::span<const LevelLatch, SunLevels> levels,
                                           const std::array<double, 3>& dir, const std::array<double, 3>& eyeRel,
                                           double texel0, bool forceAll) noexcept;

class SunClipmap {
public:
  struct Input {
    std::array<double, 3> eye{};
    std::array<double, 3> dir{};
    uint32_t lightId = 0;
    double texel0 = 0.0;
    bool full = false;
    bool sceneEpochChanged = false;
  };
  struct Result {
    bool relatchAll = false;
    bool lagged = false;
    bool keyLight = false;
    bool anchorMoved = false;
    int32_t level = -1;
    bool recentre = false;
  };

  Result update(const Input& in) noexcept;
  void fill(Views& views) const noexcept;
  void reset() noexcept { *this = {}; }
  [[nodiscard]] bool valid() const noexcept { return m_valid; }
  [[nodiscard]] const std::array<double, 3>& anchor() const noexcept { return m_anchor; }
  [[nodiscard]] const std::array<LevelLatch, SunLevels>& latches() const noexcept { return m_latch; }
  [[nodiscard]] std::array<int32_t, 2> origin(uint32_t level) const noexcept { return m_level[level].origin; }

private:
  struct Level {
    std::array<double, 3> bx{};
    std::array<double, 3> by{};
    std::array<int32_t, 2> origin{};
    std::array<int32_t, 2> prevOrigin{};
    uint32_t flags = 0;
    uint32_t latchSerial = 0;
  };
  void relatch(uint32_t level, const std::array<double, 3>& dir) noexcept;

  bool m_valid = false;
  uint32_t m_lightId = 0;
  double m_texel0 = 0.0;
  std::array<double, 3> m_anchor{};
  std::array<double, 3> m_eye{};
  std::array<LevelLatch, SunLevels> m_latch{};
  std::array<Level, SunLevels> m_level{};
  uint32_t m_serial = 0;
};

struct FrameFlags {
  bool full = false;
  bool sceneEpochChanged = false;
  bool cut = false;
  bool coverageComplete = false;
};
[[nodiscard]] bool exempt_frame(const SunClipmap::Result& result, const FrameFlags& flags) noexcept;
[[nodiscard]] std::array<double, 3> frame_direction(const detail::SceneHints* hints, uint32_t lightId,
                                                    const std::array<double, 3>& planDir) noexcept;

struct FrameSignals {
  bool retainedReset = false;
  bool clearInstances = false;
  bool unapplied = false;
  bool sceneEpoch = false;
};

struct FrameInput {
  uint32_t frameIndex = 0;
  uint64_t hintFrame = 0;
  const detail::SceneHints* hints = nullptr;
  std::array<double, 3> eye{};
  std::array<double, 3> planDir{};
  uint32_t lightId = 0;
  double texel0 = 0.0;
  bool producing = false;
  bool coverageComplete = false;
  uint32_t pages = 0;
  uint32_t pagesX = 0;
  uint32_t budget = 0;
  float levelBias = 0.f;
  uint32_t generation = 0;
  uint32_t chainMask = 0;
  FrameSignals signals;
};

struct FrameDecision {
  Views views{};
  uint32_t reasons = 0;
  bool full = false;
  bool exempt = false;
  bool held = false;
  bool cut = false;
  SunClipmap::Result clipmap;
};

// Worker: the CPU half of vsm.update, which also restarts the frame's chain mask
[[nodiscard]] FrameDecision decide_frame(const FrameInput& in) noexcept;
// Worker: an unapplied stream-out plan holds the scan until a plan lands, so one break costs one Full
[[nodiscard]] bool frame_held() noexcept;

struct UpdateStats {
  uint32_t freedWrap = 0;
  uint32_t freedRelatch = 0;
  uint32_t freedRecentre = 0;
  uint32_t freedFull = 0;
};
UpdateStats apply_update(const Views& views, std::span<uint32_t> table, std::span<uint32_t> phys) noexcept;

using ConsumerTerm = bool (*)(const detail::StageData& stage);

void initialize();
void shutdown();
void latch_hints() noexcept;
void latch_demand(const Settings& settings) noexcept;
void latch_gate(const Settings& settings, const Caps& caps) noexcept;
void latch_resources(bool active) noexcept;
void on_latch(const detail::PassSetup& setup, const detail::CompiledGraph* graph);
[[nodiscard]] bool active() noexcept;
[[nodiscard]] bool chain_ready_latched() noexcept;
// Applying... waits for the pool and every chain pipeline; an inactive VSM has nothing to wait for
[[nodiscard]] bool ready() noexcept;
void register_chain_pipeline(shaders::PipelineId id) noexcept;
void register_consumer_term(ConsumerTerm term) noexcept;

[[nodiscard]] bool frame_producing(const detail::StageData& stage) noexcept;
[[nodiscard]] bool consumed(const detail::StageData& stage) noexcept;
[[nodiscard]] bool sun_term_consumes() noexcept;
[[nodiscard]] bool shadow_view_selected() noexcept;
[[nodiscard]] bool key_room_only() noexcept;
[[nodiscard]] uint32_t pool_pages() noexcept;
[[nodiscard]] uint32_t pool_generation() noexcept;
[[nodiscard]] bool chain_enabled(const detail::StageData& stage) noexcept;

void mark_encoded(uint32_t bit) noexcept;
[[nodiscard]] uint32_t encoded_mask() noexcept;
[[nodiscard]] uint32_t expected_chain() noexcept;

[[nodiscard]] detail::PersistentHandle handle(std::string_view name) noexcept;
inline constexpr size_t PassCount = 21;
[[nodiscard]] std::array<detail::PassDesc, PassCount> pass_descs();
[[nodiscard]] detail::PassDesc cast_bits_pass_desc();

namespace testing {
struct WorkerCounts {
  uint64_t frames = 0;
  uint64_t fulls = 0;
  uint64_t held = 0;
  uint64_t exempt = 0;
  uint64_t exemptCut = 0;
  uint64_t exemptCoverage = 0;
  uint64_t exemptEpoch = 0;
  std::array<uint64_t, full_reason::Count> reasons{};
  std::array<uint64_t, SunLevels> relatches{};
  std::array<uint64_t, SunLevels> recentres{};
};
void reset() noexcept;
void force_consumed(std::optional<bool> force) noexcept;
void set_active(std::optional<bool> active) noexcept;
void set_plan(const SunPlan& plan) noexcept;
void set_pipeline_query(bool (*query)(uint32_t id) noexcept) noexcept;
void set_compact_ready(std::optional<bool> ready) noexcept;
void set_sun_term(std::optional<bool> applied) noexcept;
void set_shadow_view(std::optional<bool> selected) noexcept;
[[nodiscard]] std::span<const uint32_t> chain_pipelines() noexcept;
[[nodiscard]] WorkerCounts worker_counts() noexcept;
} // namespace testing
} // namespace aurora::render::shadows::vsm
