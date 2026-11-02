// Class 0 means no PBR, so a cleared or unwritten texel always falls back to legacy shading
const GBUF_SURFACE_VERSION = 1u;
const GBUF_SURFACE_CLASS_NONE = 0u;
const GBUF_SURFACE_CLASS_DEFAULT = 1u;
const GBUF_SURFACE_FLAG_CLOUD_RECEIVER = 1u;
const GBUF_SURFACE_FLAG_NO_SHADOW_RECEIVE = 2u;
const GBUF_SURFACE_FLAG_THIN_TRANSLUCENT = 4u;
const GBUF_SURFACE_FLAG_TWO_SIDED = 8u;
const GBUF_SURFACE_FLAG_NO_SSR = 16u;
const GBUF_SURFACE_FLAG_UNLIT = 32u;
const GBUF_SURFACE_FLAG_BAKED_LIGHT = 64u;
const GBUF_INV_255: f32 = bitcast<f32>(0x3b808081u);

struct GbufSurface {
  albedo: vec3f,
  ao: f32,
  roughness: f32,
  metalness: f32,
  class_id: u32,
  flags: u32,
  ambient: vec3f,
  category: u32,
  emissive: vec3f,
}

// Tested on the bits, since WGSL lets compilers assume no NaN and fold x == x away
fn gbuf_unorm_safe(v: vec4f) -> vec4f {
  return select(v, vec4f(0.0), (bitcast<vec4u>(v) & vec4u(0x7fffffffu)) > vec4u(0x7f800000u));
}

fn gbuf_ufloat_pack(x: f32, mantissa_bits: u32, max_code: u32) -> u32 {
  let bits = bitcast<u32>(x);
  if (bits > 0x7f800000u || bits < 0x38800000u) {
    return 0u;
  }
  let shift = 23u - mantissa_bits;
  let code = (((bits >> 23u) - 112u) << mantissa_bits) | ((bits >> shift) & ((1u << mantissa_bits) - 1u));
  return min(code + ((bits >> (shift - 1u)) & 1u), max_code);
}

fn gbuf_r11g11b10_pack(v: vec3f) -> u32 {
  return gbuf_ufloat_pack(v.x, 6u, 0x7bfu) | (gbuf_ufloat_pack(v.y, 6u, 0x7bfu) << 11u) |
         (gbuf_ufloat_pack(v.z, 5u, 0x3dfu) << 22u);
}

fn gbuf_surface_pack(s: GbufSurface) -> vec4u {
  let x = pack4x8unorm(gbuf_unorm_safe(vec4f(s.albedo, s.ao)));
  let y = pack4x8unorm(gbuf_unorm_safe(vec4f(s.roughness, s.metalness, 0.0, 0.0))) | ((s.class_id & 0xffu) << 16u) |
          ((s.flags & 0xffu) << 24u);
  let z = (pack4x8unorm(gbuf_unorm_safe(vec4f(s.ambient, 0.0))) & 0xffffffu) | ((s.category & 0xffu) << 24u);
  return vec4u(x, y, z, gbuf_r11g11b10_pack(s.emissive));
}

fn gbuf_unorm8(b: u32) -> f32 {
  return f32(b) * GBUF_INV_255;
}

fn gbuf_unorm8x4(w: u32) -> vec4f {
  let b = (vec4u(w) >> vec4u(0u, 8u, 16u, 24u)) & vec4u(0xffu);
  return vec4f(gbuf_unorm8(b.x), gbuf_unorm8(b.y), gbuf_unorm8(b.z), gbuf_unorm8(b.w));
}

fn gbuf_ufloat_unpack(code: u32, mantissa_bits: u32) -> f32 {
  let e = code >> mantissa_bits;
  let m = code & ((1u << mantissa_bits) - 1u);
  return select(bitcast<f32>(((e + 112u) << 23u) | (m << (23u - mantissa_bits))), 0.0, e == 0u);
}

fn gbuf_r11g11b10_unpack(w: u32) -> vec3f {
  return vec3f(gbuf_ufloat_unpack(w & 0x7ffu, 6u), gbuf_ufloat_unpack((w >> 11u) & 0x7ffu, 6u),
               gbuf_ufloat_unpack(w >> 22u, 5u));
}

fn gbuf_surface_class(rec: vec4u) -> u32 {
  return (rec.y >> 16u) & 0xffu;
}

fn gbuf_surface_is_pbr(rec: vec4u) -> bool {
  return gbuf_surface_class(rec) != GBUF_SURFACE_CLASS_NONE;
}

fn gbuf_surface_albedo(rec: vec4u) -> vec3f {
  return gbuf_unorm8x4(rec.x).xyz;
}

fn gbuf_surface_decode(rec: vec4u) -> GbufSurface {
  let x = gbuf_unorm8x4(rec.x);
  let y = gbuf_unorm8x4(rec.y);
  let z = gbuf_unorm8x4(rec.z);
  return GbufSurface(x.xyz, x.w, y.x, y.y, gbuf_surface_class(rec), rec.y >> 24u, z.xyz, rec.z >> 24u,
                     gbuf_r11g11b10_unpack(rec.w));
}

// Under the base tier readers get the 1x1 dummy, so loads clamp and the dummy reads class 0
fn gbuf_surface_fetch(t: texture_2d<u32>, p: vec2u) -> vec4u {
  return textureLoad(t, min(p, textureDimensions(t) - 1u), 0);
}
