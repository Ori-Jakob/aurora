#include "surface_wgsl.hpp"

#include <string_view>

namespace aurora::gx {
namespace {
// Generator copy of aurora/surface's pack; core cannot import render WGSL
constexpr std::string_view SurfacePackWgsl = R"(
fn gx_unorm_safe(v: vec4f) -> vec4f {
    return select(v, vec4f(0.0), (bitcast<vec4u>(v) & vec4u(0x7fffffffu)) > vec4u(0x7f800000u));
}
fn gx_ufloat_pack(x: f32, mantissa_bits: u32, max_code: u32) -> u32 {
    let bits = bitcast<u32>(x);
    if (bits > 0x7f800000u || bits < 0x38800000u) {
        return 0u;
    }
    let shift = 23u - mantissa_bits;
    let code = (((bits >> 23u) - 112u) << mantissa_bits) | ((bits >> shift) & ((1u << mantissa_bits) - 1u));
    return min(code + ((bits >> (shift - 1u)) & 1u), max_code);
}
fn gx_surface_pack(rgb: vec3f, ao: f32, rough: f32, metal: f32, cls: u32, flags: u32, ambient: vec3f, category: u32,
                   emissive: vec3f) -> vec4u {
    let x = pack4x8unorm(gx_unorm_safe(vec4f(rgb, ao)));
    let y = pack4x8unorm(gx_unorm_safe(vec4f(rough, metal, 0.0, 0.0))) | ((cls & 0xffu) << 16u) |
            ((flags & 0xffu) << 24u);
    let z = (pack4x8unorm(gx_unorm_safe(vec4f(ambient, 0.0))) & 0xffffffu) | ((category & 0xffu) << 24u);
    let w = gx_ufloat_pack(emissive.x, 6u, 0x7bfu) | (gx_ufloat_pack(emissive.y, 6u, 0x7bfu) << 11u) |
            (gx_ufloat_pack(emissive.z, 5u, 0x3dfu) << 22u);
    return vec4u(x, y, z, w);
})";
} // namespace

void append_surface_pack(std::string& out) { out += SurfacePackWgsl; }
} // namespace aurora::gx
