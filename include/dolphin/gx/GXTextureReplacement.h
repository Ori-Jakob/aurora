#ifndef DOLPHIN_GX_TEXTURE_REPLACEMENT_H
#define DOLPHIN_GX_TEXTURE_REPLACEMENT_H

#include <dolphin/gx/GXStruct.h>
#include <dolphin/types.h>

#if __cplusplus
extern "C" {
#endif

typedef struct AuroraTextureReplacementEntryInfo {
  u32 original_width;
  u32 original_height;
  u32 original_format;
  u32 replacement_width;
  u32 replacement_height;
  u32 replacement_mips;
  GXBool has_tlut;
  GXBool has_replacement_info;
  GXBool has_arbitrary_mips;
} AuroraTextureReplacementEntryInfo;

u32 AuroraGetTexObjTextureReplacementPath(const GXTexObj* obj, char* out, u32 out_size);
u32 AuroraGetTexObjTextureReplacementName(const GXTexObj* obj, char* out, u32 out_size);
u32 AuroraGetTextureReplacementRootPath(char* out, u32 out_size);
u32 AuroraGetTextureReplacementEntryCount(void);
GXBool AuroraGetTextureReplacementEntryInfo(u32 index, AuroraTextureReplacementEntryInfo* out_info);
GXBool AuroraLoadTextureReplacementEntryInfo(u32 index, AuroraTextureReplacementEntryInfo* out_info);
u32 AuroraGetTextureReplacementEntryName(u32 index, char* out, u32 out_size);
u32 AuroraGetTextureReplacementEntryPath(u32 index, char* out, u32 out_size);

#if __cplusplus
}
#endif

#endif
