#ifndef DOLPHIN_GX_TEXTURE_SAMPLING_H
#define DOLPHIN_GX_TEXTURE_SAMPLING_H

#include <dolphin/gx/GXStruct.h>
#include <dolphin/types.h>

#if __cplusplus
extern "C" {
#endif

void AuroraSetStochasticSamplingEnabled(GXBool enabled);
GXBool AuroraGetStochasticSamplingEnabled(void);
void AuroraSetStochasticSamplingParams(f32 cell_scale, f32 jitter, f32 blend_width);
void AuroraSetTexObjStochasticSampling(GXTexObj* obj, GXBool enabled);
GXBool AuroraGetTexObjStochasticSampling(const GXTexObj* obj);
void AuroraSetTexObjStochasticSamplingParams(GXTexObj* obj, f32 cell_scale, f32 jitter, f32 blend_width);
void AuroraClearTexObjStochasticSamplingParams(GXTexObj* obj);
GXBool AuroraGetTexObjStochasticSamplingParams(const GXTexObj* obj, f32* cell_scale, f32* jitter,
                                               f32* blend_width);

#if __cplusplus
}
#endif

#endif
