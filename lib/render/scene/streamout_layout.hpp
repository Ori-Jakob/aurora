#pragma once

#include <cstdint>

namespace aurora::render::scene::streamout {
inline constexpr uint32_t GeneratorRevision = 3;
inline constexpr uint32_t WorkgroupSize = 64;
inline constexpr uint32_t ImmediateSize = 16;
inline constexpr uint32_t ParamsSize = 16;
inline constexpr uint32_t StorageBindings = 8;
inline constexpr uint32_t RigidMatrix = 0xFFFFFFFF;
inline constexpr uint32_t EmptyMin = 0xFFFFFFFF;
inline constexpr uint32_t EmptyMax = 0;
inline constexpr uint32_t NoMaterial = 0xFFFF;
inline constexpr uint32_t NoCoord = 0xFF;
inline constexpr uint32_t NoGroup = 0xFFFFFFFF;
inline constexpr uint32_t NoBounds = 0xFFFFFFFF;

namespace binding {
inline constexpr uint32_t GxVertex = 0;
inline constexpr uint32_t GxStorage = 1;
inline constexpr uint32_t GxIndex = 2;
inline constexpr uint32_t Vertices = 3;
inline constexpr uint32_t Indices = 4;
inline constexpr uint32_t Instances = 5;
inline constexpr uint32_t Palettes = 6;
inline constexpr uint32_t Frame = 7;
} // namespace binding

namespace vertex {
inline constexpr uint32_t Words = 8;
inline constexpr uint32_t Bytes = Words * 4;
inline constexpr uint32_t Position = 0;
inline constexpr uint32_t Normal = 3;
inline constexpr uint32_t Uv0 = 4;
inline constexpr uint32_t AlphaUv = 6;
inline constexpr uint32_t Aux = 7;
inline constexpr uint32_t HasNormal = 1u << 24;
inline constexpr uint32_t HasColor = 1u << 25;
inline constexpr float AlphaUvScale = 4096.0f;
inline constexpr float AlphaUvLimit = 8.0f;
} // namespace vertex

namespace instance_flag {
inline constexpr uint32_t Valid = 1u << 0;
inline constexpr uint32_t Palette = 1u << 1;
inline constexpr uint32_t CpuDeformed = 1u << 2;
inline constexpr uint32_t GpuArrays = 1u << 3;
inline constexpr uint32_t Latched = 1u << 4;
inline constexpr uint32_t Billboard = 1u << 5;
inline constexpr uint32_t NoNormal = 1u << 6;
inline constexpr uint32_t UvNonTex = 1u << 7;
inline constexpr uint32_t UvNonAffine = 1u << 8;
inline constexpr uint32_t Water = 1u << 9;
inline constexpr uint32_t Late = 1u << 10;
inline constexpr uint32_t Blended = 1u << 11;
inline constexpr uint32_t PrevWorldValid = 1u << 12;
inline constexpr uint32_t CullShift = 13;
inline constexpr uint32_t CullMask = 3u << CullShift;
inline constexpr uint32_t Fallback = 1u << 15;
inline constexpr uint32_t NegativeDeterminant = 1u << 16;
inline constexpr uint32_t NoColor = 1u << 17;
inline constexpr uint32_t AlphaUvIsUv0 = 1u << 18;
} // namespace instance_flag

namespace record {
inline constexpr uint32_t Words = 96;
inline constexpr uint32_t Bytes = Words * 4;
inline constexpr uint32_t CpuWords = 80;
inline constexpr uint32_t Flags = 0;
inline constexpr uint32_t Categories = 1;
inline constexpr uint32_t ObjectFlags = 2;
inline constexpr uint32_t IdentityHash = 3;
inline constexpr uint32_t VertexFirst = 4;
inline constexpr uint32_t VertexCount = 5;
inline constexpr uint32_t IndexFirst = 6;
inline constexpr uint32_t IndexCount = 7;
inline constexpr uint32_t PaletteFirst = 8;
inline constexpr uint32_t Material = 9;
inline constexpr uint32_t WorldFrame = 10;
inline constexpr uint32_t Group = 11;
inline constexpr uint32_t World = 12;
inline constexpr uint32_t PrevWorld = 28;
inline constexpr uint32_t WorldWords = 16;
inline constexpr uint32_t Normal = 44;
inline constexpr uint32_t NormalWords = 12;
inline constexpr uint32_t UvBase = 56;
inline constexpr uint32_t UvAlpha = 62;
inline constexpr uint32_t UvWords = 6;
inline constexpr uint32_t UvFrame = 68;
inline constexpr uint32_t MaterialFrame = 69;
inline constexpr uint32_t DrawAttributes = 70;
inline constexpr uint32_t ClusterFirst = 73;
inline constexpr uint32_t ClusterCount = 74;
inline constexpr uint32_t BoundsOffset = 75;
inline constexpr uint32_t CurAabb = 80;
inline constexpr uint32_t PrevAabb = 86;
inline constexpr uint32_t StreamFrame = 92;
inline constexpr uint32_t PrevStreamFrame = 93;
inline constexpr uint32_t PaletteMask = 94;
inline constexpr uint32_t GpuFlags = 95;
inline constexpr uint32_t MovedFirst = WorldFrame;
inline constexpr uint32_t MovedLast = Normal + NormalWords - 1;
inline constexpr uint32_t UvFirst = UvBase;
inline constexpr uint32_t UvLast = UvFrame;
} // namespace record

namespace gpu_flag {
inline constexpr uint32_t AlphaUvClamped = 1u << 0;
} // namespace gpu_flag

namespace frame {
inline constexpr uint32_t Words = 64;
inline constexpr uint32_t Bytes = Words * 4;
inline constexpr uint32_t FrameIndex = 0;
inline constexpr uint32_t Flags = 1;
inline constexpr uint32_t Serial = 2;
inline constexpr uint32_t BreakCount = 3;
inline constexpr uint32_t CamHi = 4;
inline constexpr uint32_t CamLo = 8;
inline constexpr uint32_t View = 12;
inline constexpr uint32_t InvView = 24;
inline constexpr uint32_t PrevCamHi = 36;
inline constexpr uint32_t PrevCamLo = 40;
inline constexpr uint32_t PrevView = 44;
inline constexpr uint32_t LateFrame = 56;
inline constexpr uint32_t CameraValid = 1u << 0;
inline constexpr uint32_t Reset = 1u << 1;
} // namespace frame

namespace block {
inline constexpr uint32_t HeaderWords = 16;
inline constexpr uint32_t Frame = 0;
inline constexpr uint32_t PiecesWord = 1;
inline constexpr uint32_t VtxChunksWord = 2;
inline constexpr uint32_t IdxChunksWord = 3;
inline constexpr uint32_t WriteJobsWord = 4;
inline constexpr uint32_t WriteJobCount = 5;
inline constexpr uint32_t RollJobsWord = 6;
inline constexpr uint32_t MatricesWord = 7;
inline constexpr uint32_t PieceCount = 8;
inline constexpr uint32_t VtxChunkCount = 9;
inline constexpr uint32_t IdxChunkCount = 10;
inline constexpr uint32_t RollJobCount = 11;

inline constexpr uint32_t PieceWords = 32;
inline constexpr uint32_t PieceImmediates = 0;
inline constexpr uint32_t PieceInstance = 16;
inline constexpr uint32_t PieceVtxCount = 17;
inline constexpr uint32_t PieceOutVertexFirst = 18;
inline constexpr uint32_t PieceOutIndexFirst = 19;
inline constexpr uint32_t PieceIndexCount = 20;
inline constexpr uint32_t PiecePrim = 21;
inline constexpr uint32_t PieceIdxByte = 22;
inline constexpr uint32_t PieceMatrixWord = 23;
inline constexpr uint32_t PieceFlags = 24;

inline constexpr uint32_t PieceHasNormal = 1u << 0;
inline constexpr uint32_t PieceAlphaUvIsUv0 = 1u << 1;
inline constexpr uint32_t PieceIndexed = 1u << 2;
inline constexpr uint32_t PieceAlphaUv = 1u << 3;

inline constexpr uint32_t ChunkWords = 2;
inline constexpr uint32_t WriteJobWords = 4;
inline constexpr uint32_t RollJobWords = 8;

inline constexpr uint32_t TargetInstances = 0;
inline constexpr uint32_t TargetFrame = 1;

inline constexpr uint32_t RollFresh = 1u << 0;
inline constexpr uint32_t RollPalette = 1u << 1;
inline constexpr uint32_t RollInstance = 0;
inline constexpr uint32_t RollFlags = 1;
inline constexpr uint32_t RollPiece = 2;
inline constexpr uint32_t RollPaletteWord = 3;
inline constexpr uint32_t RollOriginDelta = 4;
} // namespace block

namespace palette {
inline constexpr uint32_t Matrices = 10;
inline constexpr uint32_t MatrixWords = 12;
inline constexpr uint32_t NormalWord = Matrices * MatrixWords;
inline constexpr uint32_t BankWords = 2 * NormalWord;
inline constexpr uint32_t PairWords = 2 * BankWords;
inline constexpr uint32_t LastSlot = Matrices - 1;
inline constexpr uint32_t MaskSlots = 20;
inline constexpr uint32_t MaskOverflow = 1u << 31;
inline constexpr double OriginGrid = 512.0;
inline constexpr double ReoriginDistance = 1536.0;
} // namespace palette

inline constexpr uint32_t MaxStreamVerticesPerFrame = 262144;
inline constexpr uint32_t MaxNewVerticesPerFrame = 65536;
inline constexpr uint32_t MaxPiecesPerFrame = 4096;
inline constexpr uint32_t MaxPaletteDrawsPerFrame = 1024;
inline constexpr uint32_t MaxVariants = 256;
inline constexpr uint32_t MaxBlockBytes = 2u << 20;
inline constexpr uint32_t MaxPieceVertices = 65535;
inline constexpr uint32_t MaxDispatchGroups = 65535;

inline constexpr uint64_t VertexPoolBytes = 32ull << 20;
inline constexpr uint64_t IndexPoolBytes = 16ull << 20;
inline constexpr uint32_t InstanceSlots = 16384;
inline constexpr uint64_t InstancePoolBytes = uint64_t{InstanceSlots} * record::Bytes;
inline constexpr uint32_t PaletteSlots = 2048;
inline constexpr uint64_t PaletteBytes = 960;
inline constexpr uint64_t PalettePoolBytes = PaletteSlots * PaletteBytes;
inline constexpr uint32_t PalettePairs = PaletteSlots / 2;
static_assert(PaletteBytes == palette::BankWords * 4);
inline constexpr uint64_t FramePoolBytes = frame::Bytes;
inline constexpr uint32_t PoolGranule = 256;

namespace push {
inline constexpr uint32_t BlockWord = 0;
inline constexpr uint32_t Count = 1;
inline constexpr uint32_t Aux = 2;
inline constexpr uint32_t Frame = 3;
} // namespace push
} // namespace aurora::render::scene::streamout
