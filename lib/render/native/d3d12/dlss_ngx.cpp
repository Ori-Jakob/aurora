#include "dlss.hpp"

#include "../native.hpp"
#include "../../../logging.hpp"

#include <nvsdk_ngx_helpers.h>

#include <atomic>
#include <cstdint>
#include <string_view>

namespace aurora::render::detail::native::d3d12::dlss {
namespace {
constexpr Module Log{"aurora::render::native"};
constexpr unsigned int NodeMask = 1;

NVSDK_NGX_Parameter* g_parameters = nullptr;
NVSDK_NGX_FeatureCommonInfo g_info{};
std::atomic_bool g_logAttached{false};
bool g_initialized = false;

uint32_t result_code(NVSDK_NGX_Result result) noexcept { return static_cast<uint32_t>(result); }

void NVSDK_CONV forward_log(const char* message, NVSDK_NGX_Logging_Level, NVSDK_NGX_Feature) {
  if (message == nullptr || !g_logAttached.load(std::memory_order_acquire)) {
    return;
  }
  std::string_view text{message};
  while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) {
    text.remove_suffix(1);
  }
  Log.debug("NGX: {}", text);
}

const NVSDK_NGX_FeatureCommonInfo* common_info() noexcept {
  g_info = {};
  g_info.LoggingInfo.LoggingCallback = &forward_log;
  g_info.LoggingInfo.MinimumLoggingLevel = NVSDK_NGX_LOGGING_LEVEL_ON;
  // NGX would otherwise also write its own log files into the data folder
  g_info.LoggingInfo.DisableOtherLoggingSinks = true;
  return &g_info;
}

NVSDK_NGX_PerfQuality_Value perf_quality(UpscalerQuality quality) noexcept {
  switch (quality) {
  case UpscalerQuality::NativeAa:
    return NVSDK_NGX_PerfQuality_Value_DLAA;
  case UpscalerQuality::Quality:
    return NVSDK_NGX_PerfQuality_Value_MaxQuality;
  case UpscalerQuality::Balanced:
    return NVSDK_NGX_PerfQuality_Value_Balanced;
  case UpscalerQuality::Performance:
    return NVSDK_NGX_PerfQuality_Value_MaxPerf;
  case UpscalerQuality::UltraPerformance:
    return NVSDK_NGX_PerfQuality_Value_UltraPerformance;
  }
  return NVSDK_NGX_PerfQuality_Value_MaxQuality;
}

int feature_bits(const FeatureFlags& flags) noexcept {
  int bits = NVSDK_NGX_DLSS_Feature_Flags_None;
  bits |= flags.hdr ? NVSDK_NGX_DLSS_Feature_Flags_IsHDR : 0;
  bits |= flags.mvLowRes ? NVSDK_NGX_DLSS_Feature_Flags_MVLowRes : 0;
  bits |= flags.mvJittered ? NVSDK_NGX_DLSS_Feature_Flags_MVJittered : 0;
  bits |= flags.depthInverted ? NVSDK_NGX_DLSS_Feature_Flags_DepthInverted : 0;
  bits |= flags.autoExposure ? NVSDK_NGX_DLSS_Feature_Flags_AutoExposure : 0;
  return bits;
}

InitResult init_result(int value) noexcept {
  const auto result = static_cast<NVSDK_NGX_Result>(value);
  if (result == NVSDK_NGX_Result_FAIL_Denied) {
    return InitResult::Denied;
  }
  if (result == NVSDK_NGX_Result_FAIL_OutOfDate) {
    return InitResult::OutOfDate;
  }
  return NVSDK_NGX_FAILED(result) ? InitResult::Failed : InitResult::Ok;
}

std::optional<Requirements> ngx_requirements(IDXGIAdapter* adapter, const Identity& identity,
                                             const std::filesystem::path& data) noexcept {
  native::assert_worker();
  if (adapter == nullptr) {
    return std::nullopt;
  }
  NVSDK_NGX_FeatureDiscoveryInfo discovery{};
  discovery.SDKVersion = NVSDK_NGX_Version_API;
  discovery.FeatureID = NVSDK_NGX_Feature_SuperSampling;
  discovery.Identifier.IdentifierType = NVSDK_NGX_Application_Identifier_Type_Project_Id;
  discovery.Identifier.v.ProjectDesc.ProjectId = identity.projectId.c_str();
  discovery.Identifier.v.ProjectDesc.EngineType = NVSDK_NGX_ENGINE_TYPE_CUSTOM;
  discovery.Identifier.v.ProjectDesc.EngineVersion = identity.engineVersion.c_str();
  discovery.ApplicationDataPath = data.c_str();
  discovery.FeatureInfo = common_info();
  NVSDK_NGX_FeatureRequirement requirement{};
  g_logAttached.store(true, std::memory_order_release);
  const NVSDK_NGX_Result result = NVSDK_NGX_D3D12_GetFeatureRequirements(adapter, &discovery, &requirement);
  g_logAttached.store(g_initialized, std::memory_order_release);
  if (NVSDK_NGX_FAILED(result)) {
    Log.info("DLSS: NGX reported no feature requirements (result {:#010x})", result_code(result));
    return std::nullopt;
  }
  const auto bits = static_cast<uint32_t>(requirement.FeatureSupported);
  const auto has = [bits](NVSDK_NGX_Feature_Support_Result flag) { return (bits & static_cast<uint32_t>(flag)) != 0; };
  return Requirements{
      .driverTooOld = has(NVSDK_NGX_FeatureSupportResult_DriverVersionUnsupported),
      .adapterUnsupported =
          has(NVSDK_NGX_FeatureSupportResult_AdapterUnsupported) || has(NVSDK_NGX_FeatureSupportResult_NotImplemented),
      .osTooOld = has(NVSDK_NGX_FeatureSupportResult_OSVersionBelowMinimumSupported),
  };
}

bool ngx_init(ID3D12Device* device, const Identity& identity, const std::filesystem::path& data) noexcept {
  native::assert_worker();
  g_logAttached.store(true, std::memory_order_release);
  const NVSDK_NGX_Result result =
      NVSDK_NGX_D3D12_Init_with_ProjectID(identity.projectId.c_str(), NVSDK_NGX_ENGINE_TYPE_CUSTOM,
                                          identity.engineVersion.c_str(), data.c_str(), device, common_info());
  if (NVSDK_NGX_FAILED(result)) {
    g_logAttached.store(false, std::memory_order_release);
    Log.warn("DLSS: NGX init failed (result {:#010x})", result_code(result));
    return false;
  }
  g_initialized = true;
  return true;
}

NVSDK_NGX_Parameter* parameters() noexcept {
  if (g_parameters == nullptr) {
    const NVSDK_NGX_Result result = NVSDK_NGX_D3D12_GetCapabilityParameters(&g_parameters);
    if (NVSDK_NGX_FAILED(result) || g_parameters == nullptr) {
      g_parameters = nullptr;
      Log.warn("DLSS: NGX capability parameters unavailable (result {:#010x})", result_code(result));
    }
  }
  return g_parameters;
}

std::optional<Capability> ngx_capability() noexcept {
  native::assert_worker();
  if (parameters() == nullptr) {
    return std::nullopt;
  }
  int available = 0;
  int needsUpdatedDriver = 0;
  unsigned int minMajor = 0;
  unsigned int minMinor = 0;
  int initResult = static_cast<int>(NVSDK_NGX_Result_Success);
  NVSDK_NGX_Parameter_GetI(g_parameters, NVSDK_NGX_Parameter_SuperSampling_Available, &available);
  NVSDK_NGX_Parameter_GetI(g_parameters, NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver, &needsUpdatedDriver);
  NVSDK_NGX_Parameter_GetUI(g_parameters, NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMajor, &minMajor);
  NVSDK_NGX_Parameter_GetUI(g_parameters, NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMinor, &minMinor);
  if (available == 0) {
    NVSDK_NGX_Parameter_GetI(g_parameters, NVSDK_NGX_Parameter_SuperSampling_FeatureInitResult, &initResult);
  }
  return Capability{
      .available = available != 0,
      .needsUpdatedDriver = needsUpdatedDriver != 0,
      .minDriverMajor = minMajor,
      .minDriverMinor = minMinor,
      .initResult = init_result(initResult),
  };
}

std::optional<ModeSize> ngx_optimal(GraphExtent output, UpscalerQuality quality) noexcept {
  native::assert_worker();
  if (parameters() == nullptr) {
    return std::nullopt;
  }
  ModeSize size;
  float sharpness = 0.f;
  const NVSDK_NGX_Result result = NGX_DLSS_GET_OPTIMAL_SETTINGS(
      g_parameters, output.width, output.height, perf_quality(quality), &size.optimal.width, &size.optimal.height,
      &size.maximum.width, &size.maximum.height, &size.minimum.width, &size.minimum.height, &sharpness);
  if (NVSDK_NGX_FAILED(result)) {
    return std::nullopt;
  }
  return size;
}

void* ngx_create(ID3D12GraphicsCommandList* list, const FeatureDesc& desc) noexcept {
  native::assert_worker();
  if (list == nullptr || parameters() == nullptr) {
    return nullptr;
  }
  NVSDK_NGX_DLSS_Create_Params create{};
  create.Feature.InWidth = desc.render.width;
  create.Feature.InHeight = desc.render.height;
  create.Feature.InTargetWidth = desc.output.width;
  create.Feature.InTargetHeight = desc.output.height;
  create.Feature.InPerfQualityValue = perf_quality(desc.quality);
  create.InFeatureCreateFlags = feature_bits(desc.flags);
  NVSDK_NGX_Handle* handle = nullptr;
  const NVSDK_NGX_Result result = NGX_D3D12_CREATE_DLSS_EXT(list, NodeMask, NodeMask, &handle, g_parameters, &create);
  if (NVSDK_NGX_FAILED(result) || handle == nullptr) {
    Log.warn("DLSS: NGX feature creation failed (result {:#010x})", result_code(result));
    return nullptr;
  }
  return handle;
}

bool ngx_evaluate(ID3D12GraphicsCommandList* list, void* feature, const Evaluation& evaluation) noexcept {
  native::assert_worker();
  if (list == nullptr || feature == nullptr || g_parameters == nullptr) {
    return false;
  }
  NVSDK_NGX_D3D12_DLSS_Eval_Params params{};
  params.Feature.pInColor = evaluation.color;
  params.Feature.pInOutput = evaluation.output;
  params.pInDepth = evaluation.depth;
  params.pInMotionVectors = evaluation.motion;
  params.InJitterOffsetX = evaluation.jitter[0];
  params.InJitterOffsetY = evaluation.jitter[1];
  params.InRenderSubrectDimensions = {evaluation.render.width, evaluation.render.height};
  params.InReset = evaluation.reset ? 1 : 0;
  params.InMVScaleX = evaluation.motionScale[0];
  params.InMVScaleY = evaluation.motionScale[1];
  params.InFrameTimeDeltaInMsec = evaluation.frameTimeMs;
  const NVSDK_NGX_Result result =
      NGX_D3D12_EVALUATE_DLSS_EXT(list, static_cast<NVSDK_NGX_Handle*>(feature), g_parameters, &params);
  if (NVSDK_NGX_FAILED(result)) {
    Log.warn("DLSS: NGX evaluation failed (result {:#010x})", result_code(result));
    return false;
  }
  return true;
}

void ngx_release(void* feature) noexcept {
  native::assert_worker();
  const NVSDK_NGX_Result result = NVSDK_NGX_D3D12_ReleaseFeature(static_cast<NVSDK_NGX_Handle*>(feature));
  if (NVSDK_NGX_FAILED(result)) {
    Log.warn("DLSS: NGX feature release failed (result {:#010x})", result_code(result));
  }
}

std::optional<uint64_t> ngx_vram() noexcept {
  native::assert_worker();
  if (parameters() == nullptr) {
    return std::nullopt;
  }
  unsigned long long bytes = 0;
  if (NVSDK_NGX_FAILED(NGX_DLSS_GET_STATS(g_parameters, &bytes))) {
    return std::nullopt;
  }
  return bytes;
}

void ngx_shutdown(ID3D12Device* device) noexcept {
  native::assert_worker();
  if (g_parameters != nullptr) {
    NVSDK_NGX_D3D12_DestroyParameters(g_parameters);
    g_parameters = nullptr;
  }
  g_logAttached.store(false, std::memory_order_release);
  g_initialized = false;
  const NVSDK_NGX_Result result = NVSDK_NGX_D3D12_Shutdown1(device);
  if (NVSDK_NGX_FAILED(result)) {
    Log.warn("DLSS: NGX shutdown failed (result {:#010x})", result_code(result));
  }
}
} // namespace

const Api& ngx_api() noexcept {
  static constexpr Api api{
      .requirements = &ngx_requirements,
      .init = &ngx_init,
      .capability = &ngx_capability,
      .optimal = &ngx_optimal,
      .shutdown = &ngx_shutdown,
      .create = &ngx_create,
      .evaluate = &ngx_evaluate,
      .release = &ngx_release,
      .vram = &ngx_vram,
  };
  return api;
}
} // namespace aurora::render::detail::native::d3d12::dlss
