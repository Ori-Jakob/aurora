#ifndef DOLPHIN_GXAURORA_H
#define DOLPHIN_GXAURORA_H

#include <dolphin/types.h>
#include <dolphin/gx/GXEnum.h>

#if __cplusplus
extern "C" {
#endif

//
// Subcommands for GX_AURORA.
//

/**
 * Sets the actual render viewport in native framebuffer coordinates.
 * Must be followed by six f32 values: left, top, width, height, nearz, farz.
 */
#define GX_AURORA_LOAD_VIEWPORT_RENDER 0x0001

/**
 * Sets the actual render scissor in native framebuffer coordinates.
 * Must be followed by four u32 values: left, top, width, height.
 */
#define GX_AURORA_LOAD_SCISSOR_RENDER 0x0002

/**
 * Loads a full 4x4 projection matrix, bypassing GXSetProjection's 6-parameter
 * hardware encoding. Must be followed by sixteen f32 values in row-major order.
 */
#define GX_AURORA_LOAD_PROJECTION_FULL 0x0003

/**
 * Aurora equivalent of CP_REG_ARRAYBASE_ID: sets the base address and size of a vertex array.
 * This command must be followed by a 64-bit memory address, 32-bit size, and 1-byte little-endian flag.
 * The index of the vertex array is given by the lowest 4 bits of the command ID,
 * e.g. writing GX_AURORA_LOAD_ARRAYBASE + 5 will set the vertex array for the sixth vertex attribute.
 * To set strides, use the normal CP_REG_ARRAYSTRIDE_ID register.
 */
#define GX_AURORA_LOAD_ARRAYBASE 0x0010

/**
 * Pushes a debug group to the backend graphics API. These may show in debugging tools such as RenderDoc.
 * Must be followed by a u16 string length and that many UTF-8 characters (no null terminator required).
 * It is considered an error to have unpopped debug groups at the end of the frame. They will be automatically cleared.
 */
#define GX_AURORA_DEBUG_GROUP_PUSH 0x0020

/**
 * Pops a previously pushed debug group.
 * Followed by nothing.
 */
#define GX_AURORA_DEBUG_GROUP_POP 0x0021

/**
 * Sends a debug marker to the backend graphics API.
 * Must be followed by a u16 string length and that many UTF-8 characters (no null terminator required).
 */
#define GX_AURORA_DEBUG_MARKER_INSERT 0x0022

#define GX_AURORA_LOAD_TEXOBJ 0x0030

#define GX_AURORA_LOAD_TLUT 0x0031

#define GX_AURORA_DESTROY_TEXOBJ 0x0032

#define GX_AURORA_DESTROY_TLUT 0x0033

#define GX_AURORA_DESTROY_COPY_TEX 0x0034

#define GX_AURORA_LOAD_COPY_SRC 0x0035

#define GX_AURORA_LOAD_COPY_DST 0x0036

#define GX_AURORA_LOAD_COPY_DEST 0x0037

#define GX_AURORA_REQUEST_DEPTH_SNAPSHOT 0x0038

#define GX_AURORA_BEGIN_OFFSCREEN 0x0039

#define GX_AURORA_END_OFFSCREEN 0x003A

/**
 * Draw primitives with the vertex count derived from a byte length, as written by
 * GXBegin(prim, fmt, GX_AUTO). Must be followed by a u8 draw opcode (vtxfmt|prim),
 * a u32 vertex data byte length, then that many bytes of vertex data. The byte length
 * must be a whole multiple of the current vertex size or zero (no draw).
 */
#define GX_AURORA_DRAW_SIZED 0x0040

/**
 * Draw pre-merged triangles with a prebuilt index buffer, as written by the display
 * list optimizer (aurora::gx::dl::optimize). Must be followed by a u8 draw opcode
 * (vtxfmt | GX_TRIANGLES), a u16 vertex count, a u32 index count, that many u16
 * indices, then vertex count * vertex size bytes of packed vertex data. Index data
 * is always host-endian regardless of stream endianness.
 */
#define GX_AURORA_DRAW_INDEXED 0x0041

// IDs in this range carry a big-endian u32 payload size so decoders can skip ones they don't know.
#define GX_AURORA_HINT_CORE_FIRST 0x0050
#define GX_AURORA_HINT_CORE_LAST 0x00FF
#define GX_AURORA_HINT_MODULE_FIRST 0x0100
#define GX_AURORA_HINT_MODULE_LAST 0x01FF
#define GX_AURORA_HINT_FIRST GX_AURORA_HINT_CORE_FIRST
#define GX_AURORA_HINT_LAST GX_AURORA_HINT_MODULE_LAST
#define GX_AURORA_HINT_HEADER_SIZE 7

#define GX_AURORA_SET_ARRAY_IMMUTABLE 0x0050
#define GX_AURORA_RELEASE_ARRAY_IMMUTABLE 0x0051
#define GX_AURORA_RENDER_CAMERA 0x0100
#define GX_AURORA_RENDER_OBJECT_BEGIN 0x0101
#define GX_AURORA_RENDER_OBJECT_END 0x0102
#define GX_AURORA_RENDER_SCOPE_PUSH 0x0103
#define GX_AURORA_RENDER_SCOPE_POP 0x0104
#define GX_AURORA_RENDER_STAGE 0x0105
#define GX_AURORA_RENDER_LIGHTS 0x0106
#define GX_AURORA_RENDER_ENVIRONMENT 0x0107
#define GX_AURORA_RENDER_MATERIAL 0x0108
#define GX_AURORA_RENDER_SKELETON 0x0109
#define GX_AURORA_RENDER_SCENE_EPOCH 0x010A
#define GX_AURORA_RENDER_TEV_POLICY 0x010B
#define GX_AURORA_RENDER_INFLUENCERS 0x010C
#define GX_AURORA_RENDER_WIND_SOURCES 0x010D
#define GX_AURORA_RENDER_GRASS_PATCHES 0x010E
#define GX_AURORA_RENDER_SKY 0x010F
#define GX_AURORA_RENDER_FOG_VOLUMES 0x0110
// Draws depend on GPU-system commands, so they get producers only together with their decoder
#define GX_AURORA_GPU_UPLOAD 0x0120
#define GX_AURORA_GPU_RESERVE 0x0121
#define GX_AURORA_GPU_DISPATCH 0x0122
#define GX_AURORA_LOAD_ARRAY_GPU 0x0123
#define GX_AURORA_GPU_UPDATE_PERSISTENT 0x0124
#define GX_AURORA_GPU_CREATE_SYSTEM 0x0125
#define GX_AURORA_GPU_DESTROY_SYSTEM 0x0126

#define AURORA_RENDER_HINT_VERSION 2
#define AURORA_HINT_SET_ARRAY_IMMUTABLE_VERSION 1
#define AURORA_HINT_RELEASE_ARRAY_IMMUTABLE_VERSION 1
#define AURORA_HINT_CAMERA_VERSION 1
#define AURORA_HINT_OBJECT_BEGIN_VERSION 2
#define AURORA_HINT_OBJECT_END_VERSION 1
#define AURORA_HINT_SCOPE_PUSH_VERSION 1
#define AURORA_HINT_SCOPE_POP_VERSION 1
#define AURORA_HINT_STAGE_VERSION 1
#define AURORA_HINT_LIGHTS_VERSION 2
#define AURORA_HINT_ENVIRONMENT_VERSION 2
#define AURORA_HINT_MATERIAL_VERSION 1
#define AURORA_HINT_SKELETON_VERSION 1
#define AURORA_HINT_SCENE_EPOCH_VERSION 1
#define AURORA_HINT_TEV_POLICY_VERSION 1
#define AURORA_HINT_INFLUENCERS_VERSION 1
#define AURORA_HINT_WIND_SOURCES_VERSION 1
#define AURORA_HINT_GRASS_PATCHES_VERSION 1
#define AURORA_HINT_SKY_VERSION 1
#define AURORA_HINT_FOG_VOLUMES_VERSION 1

#define AURORA_MAX_LIGHTS 256
#define AURORA_MAX_SKELETON_JOINTS 256
#define AURORA_MAX_INFLUENCERS 128
#define AURORA_MAX_WIND_SOURCES 32
#define AURORA_MAX_GRASS_LIGHT_SETS 255
#define AURORA_MAX_FOG_VOLUMES 64

#define GX2_SET_POLYGON_OFFSET 0x1000


/*
 * Debug marker stuff
 */

/**
 * Pushes a debug group to the backend graphics API. These may show in debugging tools such as RenderDoc.
 * It is considered an error to have unpopped debug groups at the end of the frame. They will be automatically cleared.
 */
void GXPushDebugGroup(const char* label);

/**
 * Pop a debug group previously pushed via GXPushDebugGroup().
 */
void GXPopDebugGroup();

/**
 * Sends a debug marker to the backend graphics API. These may show in debugging tools such as RenderDoc.
 */
void GXInsertDebugMarker(const char* label);

typedef enum _AuroraViewportPolicy {
  AURORA_VIEWPORT_FIT = 0,     // Preserve logical aspect in the content framebuffer
  AURORA_VIEWPORT_STRETCH = 1, // Match content framebuffer aspect to the native surface
  AURORA_VIEWPORT_NATIVE = 2,  // Use active framebuffer pixels directly
} AuroraViewportPolicy;

/**
 * Configures content framebuffer sizing and how GXSetViewport/GXSetScissor parameters are applied to rendering.
 * When AURORA_VIEWPORT_NATIVE is used, GXSetTexCopySrc/GXSetTexCopyDst will use native framebuffer resolution.
 */
void AuroraSetViewportPolicy(AuroraViewportPolicy policy);

/**
 * Retrieves the current content framebuffer size.
 */
void AuroraGetRenderSize(u32* width, u32* height);

/**
 * Flush pending GX state and wait for FIFO processing without signaling a draw-done callback.
 */
void AuroraGXSync(void);

/**
 * Sets the actual render viewport in native framebuffer coordinates.
 * Overrides the automatically scaled values set by the logical GXSetViewport.
 */
void GXSetViewportRender(f32 left, f32 top, f32 wd, f32 ht, f32 nearz, f32 farz);

/**
 * Sets the actual render scissor in native framebuffer coordinates.
 * Overrides the automatically scaled values set by the logical GXSetScissor.
 */
void GXSetScissorRender(u32 left, u32 top, u32 wd, u32 ht);

void GX2SetPolygonOffset(f32 mFrontOffset, f32 mFrontScale, f32 mBackOffset, f32 mBackScale, f32 mClamp);

/**
 * Load an arbitrary 4x4 projection matrix, avoiding the 6-parameter hardware encoding.
 */
void GXSetProjectionFull(const void* mtx);

/**
 * Create an offscreen framebuffer and switch rendering to it.
 * All subsequent GX rendering will target this framebuffer until GXRestoreFrameBuffer() is called.
 * Use GXCopyTex to resolve the offscreen content into a texture.
 */
void GXCreateFrameBuffer(u32 width, u32 height);

/**
 * Restore rendering to the main EFB framebuffer.
 * Must be called after GXCreateFrameBuffer() to resume normal rendering.
 */
void GXRestoreFrameBuffer(void);

typedef enum AuroraCameraRole {
  AURORA_CAMERA_MAIN = 0,
  AURORA_CAMERA_SKY = 1,
  AURORA_CAMERA_REFLECTION = 2,
  AURORA_CAMERA_SHADOW = 3,
  AURORA_CAMERA_OFFSCREEN = 4,
  AURORA_CAMERA_UI = 5,
  AURORA_CAMERA_INHERIT = 0xFF,
} AuroraCameraRole;

typedef enum AuroraRenderStage {
  AURORA_STAGE_SCENE_BEGIN = 1,
  AURORA_STAGE_OPAQUE_DONE = 2,
  AURORA_STAGE_PRE_POST = 3,
  AURORA_STAGE_SCENE_END = 4,
  AURORA_STAGE_BEFORE_HUD = 5,
  AURORA_STAGE_FRAME_END = 6,
} AuroraRenderStage;

enum {
  AURORA_CAT_STATIC = 1u << 0,
  AURORA_CAT_DYNAMIC = 1u << 1,
  AURORA_CAT_SKINNED = 1u << 2,
  AURORA_CAT_TERRAIN = 1u << 3,
  AURORA_CAT_FOLIAGE = 1u << 4,
  AURORA_CAT_WATER = 1u << 5,
  AURORA_CAT_SKY = 1u << 6,
  AURORA_CAT_PARTICLE = 1u << 7,
  AURORA_CAT_DECAL = 1u << 8,
  AURORA_CAT_UI = 1u << 9,
  AURORA_CAT_PLAYER = 1u << 10,
  AURORA_CAT_EMISSIVE = 1u << 11,
  AURORA_CAT_OVERLAY = 1u << 12,
  AURORA_CAT_NO_SHADOW_CAST = 1u << 16,
  AURORA_CAT_NO_SHADOW_RECEIVE = 1u << 17,
  AURORA_CAT_NO_PBR = 1u << 18,
  AURORA_CAT_NO_CAPTURE = 1u << 19,
  AURORA_CAT_NO_RT = 1u << 20,
  AURORA_CAT_SCREEN_READ = 1u << 21,
  AURORA_CAT_FAKE_SHADOW = 1u << 22,
  AURORA_CAT_CLOUD_RECEIVER = 1u << 23,
  AURORA_CAT_NO_AO = 1u << 24,
};

// Shared with SCOPE_PUSH flags, so a scope can mark everything inside it as capture-only
enum {
  AURORA_OBJ_CAPTURE_ONLY = 1u << 0,
  AURORA_OBJ_TELEPORT = 1u << 1,
  AURORA_OBJ_CPU_DEFORMED = 1u << 2,
  AURORA_OBJ_BAKED_VERTEX_LIGHT = 1u << 3,
  AURORA_OBJ_DOUBLE_SIDED = 1u << 4,
  AURORA_OBJ_BILLBOARD = 1u << 5,
  AURORA_OBJ_IMMOBILE = 1u << 6,
};

typedef enum AuroraLightType {
  AURORA_LIGHT_DIRECTIONAL = 0,
  AURORA_LIGHT_POINT = 1,
  AURORA_LIGHT_SPOT = 2,
} AuroraLightType;

enum {
  AURORA_LIGHT_NO_SHADOW = 1u << 0,
  AURORA_LIGHT_NO_SPECULAR = 1u << 1,
  AURORA_LIGHT_FLICKER = 1u << 2,
  AURORA_LIGHT_CAMERA_FILL = 1u << 3,
  AURORA_LIGHT_AFFECTS_TRANSLUCENT = 1u << 4,
  AURORA_LIGHT_CLOUD_SHADOWED = 1u << 5,
  AURORA_LIGHT_KEY_ROOM = 1u << 6,
  AURORA_LIGHT_LANTERN = 1u << 7,
  AURORA_LIGHT_NO_VOLUMETRIC = 1u << 8,
};

enum {
  AURORA_ENV_INTERIOR = 1u << 0,
  AURORA_ENV_TWILIGHT = 1u << 1,
  AURORA_ENV_UNDERWATER = 1u << 2,
  AURORA_ENV_SENSES = 1u << 3,
  AURORA_ENV_RAIN = 1u << 4,
  AURORA_ENV_SNOW = 1u << 5,
  AURORA_ENV_NO_SUN = 1u << 6,
  AURORA_ENV_NO_VOLUMETRIC = 1u << 7,
  AURORA_ENV_ROOM = 1u << 8,
};

typedef enum AuroraTevRegister {
  AURORA_TEV_REG_C0 = 0,
  AURORA_TEV_REG_C1 = 1,
  AURORA_TEV_REG_C2 = 2,
  AURORA_TEV_REG_K0 = 3,
  AURORA_TEV_REG_K1 = 4,
  AURORA_TEV_REG_K2 = 5,
  AURORA_TEV_REG_K3 = 6,
} AuroraTevRegister;

typedef enum AuroraTevChannel {
  AURORA_TEV_CH_R = 0,
  AURORA_TEV_CH_G = 1,
  AURORA_TEV_CH_B = 2,
  AURORA_TEV_CH_A = 3,
} AuroraTevChannel;

// Per channel, because one register often mixes lighting and non-lighting data (K1 = fog density, ..., alpha)
#define AURORA_TEV_POLICY_BIT(reg, channel) (1u << ((reg) * 4u + (channel)))

// The upper bytes stay free for game-defined influencer bits
enum {
  AURORA_INFLUENCER_PUSH = 0,
  AURORA_INFLUENCER_FLATTEN = 1,
  AURORA_INFLUENCER_CLEAR = 2,
  AURORA_INFLUENCER_CUT_SWEEP = 3,
  AURORA_INFLUENCER_TERRAIN_FLATTEN = 4,
  AURORA_INFLUENCER_KIND_MASK = 0xFF,
};

enum {
  AURORA_GRASS_CUT = 1u << 0,
  AURORA_GRASS_SNAPPED = 1u << 1,
  AURORA_GRASS_GAME_CLIPPED = 1u << 2,
  AURORA_GRASS_ATTENUATOR = 1u << 3,
  AURORA_GRASS_NO_INTERACT = 1u << 4,
};
enum { AURORA_GRASS_LIGHTS_1_4_ONLY = 1u << 0 };
enum { AURORA_SKY_VISIBLE = 1u << 0, AURORA_SKY_KEEP_GAME_CLOUDS = 1u << 1 };
enum { AURORA_CLOUD_LAYER_NO_SHADOW = 1u << 0, AURORA_CLOUD_LAYER_SEA = 1u << 1 };
enum { AURORA_FOG_SHAPE_ELLIPSOID = 0, AURORA_FOG_SHAPE_BOX = 1 };
enum { AURORA_FOG_MODE_ADD = 0, AURORA_FOG_MODE_CARVE = 1 };

typedef struct AuroraLight {
  u32 id;
  u8 type;
  u8 shadowPriority;
  u16 flags;
  u32 categoryMask;
  f32 position[3];
  f32 direction[3];
  f32 color[3];
  f32 radius;
  f32 sourceRadius;
  f32 spotInnerCos;
  f32 spotOuterCos;
  f32 falloffExponent;
  f32 volumetricScale;
} AuroraLight;

typedef struct AuroraEnvironment {
  f32 skyVisibility;
  f32 ambientColor[3];
  f32 fogColor[3];
  f32 windDir[3];
  f32 windStrength;
  f32 exposureBias;
  f32 timeOfDay;
  u32 flags;
  f32 unitsPerMeter;
  f32 windPhase[2];
  f32 windPhasePrev[2];
  f32 flutterPhase[3];
  u32 fogType;
  f32 fogStart;
  f32 fogEnd;
  f32 viewNear;
  f32 viewFar;
  u32 fogRangeAdj;
  f32 shadowSun;
  f32 shadowLocal;
  f32 shadowThunderAdd;
  f32 grassDensityScale;
  f32 sunShaftGate;
  f32 hazeAtFogEnd;
  f32 hazeDensity;
  f32 heightFogDensity;
  f32 heightFogBase;
  f32 heightFogFalloff;
  f32 mediaAlbedo[3];
  f32 mediaEmissive[3];
  f32 phaseG;
  f32 phaseG2;
  f32 phaseBlend;
  f32 noiseScale;
  f32 noiseAmount;
  f32 relightStrength;
  f32 relightAmbient;
  f32 sunGlow;
  f32 volumeFar;
  u16 fogRangeTable[10];
  u16 fogRangeReserved[2];
} AuroraEnvironment;

typedef struct AuroraTevPolicy {
  u32 mask;
  u8 neutral[7][4];
} AuroraTevPolicy;

typedef struct AuroraInfluencer {
  f32 pos[3];
  f32 radius;
  f32 vel[3];
  u32 kind;
} AuroraInfluencer;

typedef struct AuroraWindSource {
  f32 pos[3];
  f32 radius;
  f32 dir[3];
  f32 power;
} AuroraWindSource;

// state = type | flags << 8 | regrow << 16 | lightSet << 24, so one u32 load unpacks a tuft
typedef struct AuroraGrassPatch {
  f32 pos[3];
  u32 seed;
  u32 state;
  u32 tint;
  s16 bendPrev[2];
  s16 bendCur[2];
} AuroraGrassPatch;

// 0xFF in lightIdx means no light, so a light set can reference lights 0-254 only
typedef struct AuroraGrassLightSet {
  u8 reg1Rgb[3];
  u8 shine;
  u8 matRgba[4];
  u8 lightIdx[7];
  u8 flags;
  f32 grassLightRatio;
  f32 fog[5];
  u8 fogRgb[3];
  u8 fogType;
  u32 reserved;
} AuroraGrassLightSet;

typedef struct AuroraCloudLayer {
  f32 base;
  f32 thickness;
  f32 coverage;
  f32 density;
  u32 type;
  f32 windScale;
  u32 flags;
} AuroraCloudLayer;

typedef struct AuroraSky {
  u32 flags;
  f32 horizonY;
  u32 cloudTicks;
  f32 cloudTickFrac;
  f32 tickSeconds;
  u32 seed;
  f32 cloudSpeedScale;
  f32 cloudShadowDensity;
  f32 rain;
  f32 thunderFlash;
  f32 weatherAmount;
  f32 skyColor[3];
  f32 cloudTop[3];
  f32 cloudBottom[3];
  f32 cloudShadow[3];
  f32 hazeOuter[3];
  u32 layerCount;
  AuroraCloudLayer layers[2];
} AuroraSky;

typedef struct AuroraFogVolume {
  f32 center[3];
  f32 radius[3];
  f32 rotationY;
  u8 shape;
  u8 mode;
  u16 flags;
  f32 density;
  f32 edgeFalloff;
  f32 albedo[3];
  f32 emissive[3];
} AuroraFogVolume;

void GXAuroraSetArrayImmutable(const void* data, u32 size, GXBool littleEndian, u32 generation);
void GXAuroraReleaseArrayImmutable(const void* data, u32 generation);
void GXAuroraSetCamera(AuroraCameraRole role, const f32 view[3][4], const f32 proj[4][4], GXProjectionType type,
                       u32 cutEpoch);
void GXAuroraBeginObject(u64 objectId, u32 subId, u32 generation, u32 categories, u32 flags);
void GXAuroraBeginObjectWorld(u64 objectId, u32 subId, u32 generation, u32 categories, u32 flags,
                              const f32 world[3][4]);
void GXAuroraEndObject(void);
void GXAuroraPushScope(u32 categories, u32 flags, AuroraCameraRole role);
void GXAuroraPopScope(void);
void GXAuroraStage(AuroraRenderStage stage);
void GXAuroraSetLights(const AuroraLight* lights, u16 count);
void GXAuroraSetEnvironment(const AuroraEnvironment* environment);
void GXAuroraSetMaterialClass(u32 materialClass);
void GXAuroraSetSkeleton(u64 objectId, const f32 (*joints)[3][4], u16 count);
void GXAuroraSetSceneEpoch(u32 epoch);
void GXAuroraSetTevPolicy(const AuroraTevPolicy* policy);
void GXAuroraSetInfluencers(const AuroraInfluencer* influencers, u16 count);
void GXAuroraSetWindSources(const AuroraWindSource* sources, u16 count);
void GXAuroraSetGrassPatches(u32 gpuHandle, u32 count, f32 interpStep, const AuroraGrassLightSet* sets, u8 setCount);
void GXAuroraSetSky(const AuroraSky* sky);
void GXAuroraSetFogVolumes(const AuroraFogVolume* volumes, u16 count);

#if __cplusplus
}
#endif

#endif
