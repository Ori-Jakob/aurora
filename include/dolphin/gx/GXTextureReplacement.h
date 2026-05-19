#ifndef DOLPHIN_GX_TEXTURE_REPLACEMENT_H
#define DOLPHIN_GX_TEXTURE_REPLACEMENT_H

#include <dolphin/gx/GXStruct.h>
#include <dolphin/types.h>

#if __cplusplus
extern "C" {
#endif

u32 AuroraGetTexObjTextureReplacementPath(const GXTexObj* obj, char* out, u32 out_size);
u32 AuroraGetTexObjTextureReplacementName(const GXTexObj* obj, char* out, u32 out_size);

#if __cplusplus
}
#endif

#endif
