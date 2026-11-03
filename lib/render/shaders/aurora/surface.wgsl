// Slot 0 means no PBR, so a cleared or unwritten texel always falls back to legacy shading
const GBUF_SURFACE_VERSION = 2u;
const GBUF_SURFACE_CLASS_NONE = 0u;
const GBUF_SURFACE_CLASS_DEFAULT = 1u;
const GBUF_SURFACE_FLAG_CLOUD_RECEIVER = 1u;
const GBUF_SURFACE_FLAG_NO_SHADOW_RECEIVE = 2u;
const GBUF_SURFACE_FLAG_THIN_TRANSLUCENT = 4u;
const GBUF_SURFACE_FLAG_TWO_SIDED = 8u;
const GBUF_SURFACE_FLAG_NO_SSR = 16u;
const GBUF_SURFACE_FLAG_UNLIT = 32u;
const GBUF_SURFACE_FLAG_BAKED_LIGHT = 64u;
const GBUF_SURFACE_EXT_AMBIENT = 0x1000000u;
const GBUF_SURFACE_EXT_EMISSIVE = 0x2000000u;
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

// v1 words, which only the debug expand pass writes for tools
fn gbuf_surface_v1_pack(s: GbufSurface) -> vec4u {
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

// RGBA16Uint writes cheaper than RG32Uint, so each 32-bit word spans two channels, low half first
fn gbuf_surface_words(rec: vec4u) -> vec2u {
  return vec2u((rec.x & 0xffffu) | (rec.y << 16u), (rec.z & 0xffffu) | (rec.w << 16u));
}

fn gbuf_surface_slot(rec: vec4u) -> u32 {
  return rec.w & 0xffffu;
}

fn gbuf_surface_is_pbr(rec: vec4u) -> bool {
  return gbuf_surface_slot(rec) != 0u;
}

fn gbuf_surface_albedo(rec: vec4u) -> vec3f {
  return gbuf_unorm8x4(gbuf_surface_words(rec).x).xyz;
}

fn gbuf_surface_ao(rec: vec4u) -> f32 {
  return gbuf_unorm8((rec.y >> 8u) & 0xffu);
}

fn gbuf_surface_roughness(rec: vec4u) -> f32 {
  return gbuf_unorm8(rec.z & 0xffu);
}

fn gbuf_surface_metalness(rec: vec4u) -> f32 {
  return gbuf_unorm8((rec.z >> 8u) & 0xffu);
}

// Readers pass the table's length; a slot past it reads entry 0, which is not PBR
fn gbuf_surface_entry_index(slot: u32, len: u32) -> u32 {
  return select(0u, slot, slot < len);
}

fn gbuf_surface_class(entry: vec4u) -> u32 {
  return entry.x & 0xffu;
}

fn gbuf_surface_flags(entry: vec4u) -> u32 {
  return (entry.x >> 8u) & 0xffu;
}

fn gbuf_surface_category(entry: vec4u) -> u32 {
  return (entry.x >> 16u) & 0xffu;
}

fn gbuf_surface_reads_ext(entry: vec4u) -> bool {
  return (entry.x & (GBUF_SURFACE_EXT_AMBIENT | GBUF_SURFACE_EXT_EMISSIVE)) != 0u;
}

// ext is only read where the entry asks for it, so callers may pass zero otherwise
fn gbuf_surface_decode(rec: vec4u, entry: vec4u, ext: vec4u) -> GbufSurface {
  let w = gbuf_surface_words(rec);
  let e = gbuf_surface_words(ext);
  let x = gbuf_unorm8x4(w.x);
  let y = gbuf_unorm8x4(w.y);
  let ambient = select(entry.y, e.x, (entry.x & GBUF_SURFACE_EXT_AMBIENT) != 0u);
  let emissive = select(entry.z, e.y, (entry.x & GBUF_SURFACE_EXT_EMISSIVE) != 0u);
  return GbufSurface(x.xyz, x.w, y.x, y.y, gbuf_surface_class(entry), gbuf_surface_flags(entry),
                     gbuf_unorm8x4(ambient).xyz, gbuf_surface_category(entry), gbuf_r11g11b10_unpack(emissive));
}

// v1 words as the expand pass writes them; binding-free, so tools and views decode without the table
fn gbuf_surface_v1_decode(words: vec4u) -> GbufSurface {
  let x = gbuf_unorm8x4(words.x);
  let y = gbuf_unorm8x4(words.y);
  let z = gbuf_unorm8x4(words.z);
  return GbufSurface(x.xyz, x.w, y.x, y.y, (words.y >> 16u) & 0xffu, words.y >> 24u, z.xyz, words.z >> 24u,
                     gbuf_r11g11b10_unpack(words.w));
}

fn gbuf_surface_v1_expand(rec: vec4u, entry: vec4u, ext: vec4u) -> vec4u {
  let w = gbuf_surface_words(rec);
  let e = gbuf_surface_words(ext);
  let ambient = select(entry.y, e.x, (entry.x & GBUF_SURFACE_EXT_AMBIENT) != 0u);
  let emissive = select(entry.z, e.y, (entry.x & GBUF_SURFACE_EXT_EMISSIVE) != 0u);
  return vec4u(w.x, (w.y & 0xffffu) | ((entry.x & 0xffffu) << 16u),
               (ambient & 0xffffffu) | (((entry.x >> 16u) & 0xffu) << 24u), emissive);
}

// Under the base tier readers get the 1x1 dummy, so loads clamp and the dummy reads slot 0
fn gbuf_surface_fetch(t: texture_2d<u32>, p: vec2u) -> vec4u {
  return textureLoad(t, min(p, textureDimensions(t) - 1u), 0);
}
