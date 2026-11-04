// The table layout, lookups and LTC data follow adobe/openpbr-bsdf at c91aad1d1ce1693e803f039d7c92c2965c4eb013,
// which carries this notice:
//
// Copyright 2026 Adobe
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// The LTC table is reproduced there from tizian/ltc-sheen by Tizian Zeltner, Brent Burley and Matt Jen-Yuan Chiang,
// also licensed under the Apache License, Version 2.0.

// Mirrors material::classes::GpuRow; M18 and M23 import this, so fields only append
struct ClassParams {
  specular: vec4f,
  coat: vec4f,
  fuzz: vec4f,
  subsurface: vec4f,
  emission: vec4f,
  ior: vec4f,
  base: vec4f,
  misc: vec4f,
}

struct OpenPbrSurface {
  base_color: vec3f,
  roughness: f32,
  metalness: f32,
}

struct OpenPbrPrepared {
  n: vec3f,
  v: vec3f,
  mu_v: f32,
  alpha: f32,
  g1_v: f32,
  alpha_index: f32,
  eta: f32,
  ior_index: f32,
  spec_tint: vec3f,
  metal_f0: vec3f,
  metal_tint: vec3f,
  metal_weight: f32,
  metal_ms: vec3f,
  ecm_v: f32,
  ecm_avg: f32,
  diffuse: vec3f,
  transmit: vec3f,
  sigma: f32,
  ec_v: f32,
  inv_ec_avg: f32,
  wrap: f32,
  thin: bool,
  coat_weight: f32,
  coat_eta: f32,
  coat_alpha: f32,
  coat_tint: vec3f,
  coat_base: vec3f,
  fuzz_weight: f32,
  fuzz_colour: vec3f,
  fuzz_ltc: vec3f,
  fuzz_base: f32,
  t: vec3f,
  b: vec3f,
  ambient_diffuse: vec3f,
  ambient_specular: vec3f,
}

const OPENPBR_PI: f32 = 3.14159265358979;
const OPENPBR_TABLE_SIZE: f32 = 32.0;
const OPENPBR_TABLE_MAX: f32 = 31.0;
const OPENPBR_IOR_MAX: f32 = 2.5;
const OPENPBR_MIN_ROUGHNESS: f32 = 0.045;
const OPENPBR_MIN_MU_V: f32 = 1e-4;
const OPENPBR_MIN_ALPHA_ENERGY_LOSS: f32 = 0.0016;
const OPENPBR_MIN_AVERAGE: f32 = 1e-4;
const OPENPBR_ROW_THIN: u32 = 1u;
const OPENPBR_ROW_TWO_SIDED: u32 = 2u;
const OPENPBR_ROW_NO_SSR: u32 = 4u;
const OPENPBR_ROW_SUBSURFACE_TINT: u32 = 8u;
const OPENPBR_LAYER_AVERAGES: u32 = 0u;
const OPENPBR_LAYER_METAL: u32 = 1u;
const OPENPBR_LAYER_LTC: u32 = 2u;

// Texel-centre remap: index i lands on texel i, and the clamp makes the edge texels hold
fn openpbr_coord(index: f32) -> f32 {
  return clamp((index + 0.5) / OPENPBR_TABLE_SIZE, 0.5 / OPENPBR_TABLE_SIZE, 1.0 - 0.5 / OPENPBR_TABLE_SIZE);
}

fn openpbr_ior_index(ior: f32) -> f32 {
  if (ior < 1.0) {
    return 15.0 - (1.0 / ior - 1.0) / (OPENPBR_IOR_MAX - 1.0) * 15.0;
  }
  return 16.0 + (ior - 1.0) / (OPENPBR_IOR_MAX - 1.0) * 15.0;
}

fn openpbr_alpha_index(alpha: f32) -> f32 {
  return sqrt(max(alpha, 0.0)) * OPENPBR_TABLE_MAX;
}

fn openpbr_lut3d(lut: texture_3d<f32>, smp: sampler, mu: f32, alpha_index: f32, ior_index: f32) -> vec2f {
  let uvw = vec3f(openpbr_coord(mu * OPENPBR_TABLE_MAX), openpbr_coord(alpha_index), openpbr_coord(ior_index));
  return textureSampleLevel(lut, smp, uvw, 0.0).rg;
}

fn openpbr_lut2d(lut: texture_2d_array<f32>, smp: sampler, x: f32, y: f32, layer: u32) -> vec4f {
  return textureSampleLevel(lut, smp, vec2f(openpbr_coord(x), openpbr_coord(y)), layer, 0.0);
}

fn openpbr_f0(eta: f32) -> f32 {
  let r = (eta - 1.0) / (eta + 1.0);
  return r * r;
}

fn openpbr_apply_specular_weight(eta: f32, weight: f32) -> f32 {
  let s = sqrt(min(openpbr_f0(eta) * weight, 0.9999));
  let ior = (1.0 + s) / (1.0 - s);
  return select(ior, 1.0 / ior, eta < 1.0);
}

fn openpbr_fresnel(eta: f32, cos_in: f32) -> f32 {
  if (eta == 1.0) {
    return 0.0;
  }
  let c = saturate(cos_in);
  let st2 = (1.0 - c * c) / (eta * eta);
  if (st2 >= 1.0) {
    return 1.0;
  }
  let ct = sqrt(1.0 - st2);
  let rs = (c - eta * ct) / (c + eta * ct);
  let rp = (ct - eta * c) / (ct + eta * c);
  return 0.5 * (rs * rs + rp * rp);
}

fn openpbr_average_fresnel(eta: f32) -> f32 {
  if (eta > 1.0) {
    return (eta - 1.0) / (4.08567 + 1.00071 * eta);
  }
  return 0.997118 + 0.1014 * eta - 0.965241 * eta * eta - 0.130607 * eta * eta * eta;
}

fn openpbr_f82_b(f0: vec3f, tint: vec3f) -> vec3f {
  let one = 1.0 - 1.0 / 7.0;
  let one5 = one * one * one * one * one;
  return (f0 + (1.0 - f0) * one5) * (1.0 - tint) / ((1.0 / 7.0) * one5 * one);
}

fn openpbr_f82(f0: vec3f, tint: vec3f, cos_in: f32) -> vec3f {
  let c = saturate(cos_in);
  let m = 1.0 - c;
  let m5 = m * m * m * m * m;
  return saturate(f0 + ((1.0 - f0) - openpbr_f82_b(f0, tint) * c * m) * m5);
}

fn openpbr_f82_average(f0: vec3f, tint: vec3f) -> vec3f {
  return saturate(f0 + (1.0 - f0) / 21.0 - openpbr_f82_b(f0, tint) / 126.0);
}

// White F0 and tint keep the EC table's albedo exactly, so the metal furnace does not move with the DFG's sampling
fn openpbr_metal_single_scatter(f0: vec3f, tint: vec3f, metal: vec4f) -> vec3f {
  let white = 1.0 - metal.r;
  if (all(f0 >= vec3f(1.0)) && all(tint >= vec3f(1.0))) {
    return vec3f(white);
  }
  let tinted = f0 * metal.g + metal.b - openpbr_f82_b(f0, tint) * metal.a;
  return white * saturate(tinted / max(metal.g + metal.b, OPENPBR_MIN_AVERAGE));
}

// Opaque EC tables stop at IOR 2.5; beyond it OpenPBR scales toward zero with F0
fn openpbr_extrapolate(value: f32, eta: f32) -> f32 {
  if (eta > OPENPBR_IOR_MAX || eta < 1.0 / OPENPBR_IOR_MAX) {
    let f0_max = openpbr_f0(OPENPBR_IOR_MAX);
    return (1.0 - (openpbr_f0(eta) - f0_max) / (1.0 - f0_max)) * value;
  }
  return value;
}

fn openpbr_ggx_d(nh: f32, alpha: f32) -> f32 {
  let a2 = alpha * alpha;
  let t = nh * nh * (a2 - 1.0) + 1.0;
  return a2 / (OPENPBR_PI * t * t);
}

// Separable Smith, as the EC tables were integrated; a correlated G2 would gain energy against them
fn openpbr_smith_g1(mu: f32, alpha: f32) -> f32 {
  let a2 = alpha * alpha;
  return 2.0 * mu / (mu + sqrt(a2 + (1.0 - a2) * mu * mu));
}

fn openpbr_fon_albedo(mu: f32, sigma: f32) -> f32 {
  let a = 0.5 - 2.0 / (3.0 * OPENPBR_PI);
  let c = 1.0 - mu;
  let g = c * (0.0571085289 + c * (0.491881867 + c * (-0.332181442 + c * 0.0714429953)));
  return (1.0 + sigma * g) / (1.0 + a * sigma);
}

// Pi times EON without the cosine; sigma 0 is exactly Lambert, OpenPBR's default diffuse roughness
fn openpbr_eon_pi(rho: vec3f, sigma: f32, mu_v: f32, mu_l: f32, vl: f32) -> vec3f {
  if (sigma <= 0.0) {
    return rho;
  }
  let a = 0.5 - 2.0 / (3.0 * OPENPBR_PI);
  let b = 2.0 / 3.0 - 28.0 / (15.0 * OPENPBR_PI);
  let eps = 1e-7;
  let s = vl - mu_v * mu_l;
  let st = select(s, s / max(mu_v, mu_l), s > 0.0);
  let af = 1.0 / (1.0 + a * sigma);
  let avg = af * (1.0 + b * sigma);
  let ms = max(eps, 1.0 - openpbr_fon_albedo(mu_l, sigma)) * max(eps, 1.0 - openpbr_fon_albedo(mu_v, sigma)) /
           max(eps, 1.0 - avg);
  let rho_ms = rho * rho * avg / (1.0 - rho * (1.0 - avg));
  return rho * af * (1.0 + sigma * st) + rho_ms * ms;
}

fn openpbr_eon_albedo(rho: vec3f, sigma: f32, mu: f32) -> vec3f {
  if (sigma <= 0.0) {
    return rho;
  }
  let a = 0.5 - 2.0 / (3.0 * OPENPBR_PI);
  let b = 2.0 / 3.0 - 28.0 / (15.0 * OPENPBR_PI);
  let af = 1.0 / (1.0 + a * sigma);
  let avg = af * (1.0 + b * sigma);
  let e = openpbr_fon_albedo(mu, sigma);
  let rho_ms = rho * rho * avg / (1.0 - rho * (1.0 - avg));
  return rho * e + rho_ms * (1.0 - e);
}

fn openpbr_coat_passage(p: OpenPbrPrepared, c: f32) -> vec3f {
  if (c <= 0.0 || min(min(p.coat_tint.x, p.coat_tint.y), p.coat_tint.z) >= 1.0) {
    return vec3f(1.0);
  }
  let refracted = sqrt(max(0.0, 1.0 - (1.0 - c * c) / (p.coat_eta * p.coat_eta)));
  let scale = 1.0 / max(refracted, 1e-6);
  return 1.0 + (pow(sqrt(max(p.coat_tint, vec3f(0.0))), vec3f(scale)) - 1.0) * p.coat_weight;
}

fn openpbr_opaque_ec(lut3d: texture_3d<f32>, smp: sampler, mu: f32, alpha: f32, eta: f32) -> f32 {
  return openpbr_extrapolate(openpbr_lut3d(lut3d, smp, mu, openpbr_alpha_index(alpha), openpbr_ior_index(eta)).g, eta);
}

fn openpbr_prepare(s: OpenPbrSurface, cls: ClassParams, n: vec3f, v: vec3f, lut3d: texture_3d<f32>,
                   lut2d: texture_2d_array<f32>, smp: sampler) -> OpenPbrPrepared {
  var p: OpenPbrPrepared;
  p.n = n;
  p.v = v;
  p.mu_v = max(dot(n, v), OPENPBR_MIN_MU_V);
  let roughness = clamp(s.roughness, OPENPBR_MIN_ROUGHNESS, 1.0);
  p.alpha = roughness * roughness;
  p.g1_v = openpbr_smith_g1(p.mu_v, p.alpha);
  let spec_colour = cls.specular.rgb;
  let spec_weight = cls.specular.w;
  let sss_weight = cls.subsurface.w;
  let metal = saturate(s.metalness);
  let dielectric = 1.0 - metal;
  let base = s.base_color * cls.base.w;
  p.eta = openpbr_apply_specular_weight(cls.ior.x, spec_weight);
  p.alpha_index = openpbr_alpha_index(p.alpha);
  p.ior_index = openpbr_ior_index(p.eta);
  p.ec_v = openpbr_extrapolate(openpbr_lut3d(lut3d, smp, p.mu_v, p.alpha_index, p.ior_index).g, p.eta);
  let averages = openpbr_lut2d(lut2d, smp, p.alpha_index, p.ior_index, OPENPBR_LAYER_AVERAGES);
  p.inv_ec_avg = 1.0 / max(openpbr_extrapolate(averages.b, p.eta), OPENPBR_MIN_AVERAGE);
  p.spec_tint = spec_colour * dielectric;
  p.metal_weight = metal * spec_weight;
  p.metal_f0 = base;
  p.metal_tint = spec_colour;
  let favg = openpbr_f82_average(base, spec_colour);
  p.metal_ms = favg * favg * p.metal_weight;
  p.ecm_avg = 1.0;
  var metal_ss = vec3f(0.0);
  if (p.metal_weight > 0.0) {
    let metal = openpbr_lut2d(lut2d, smp, p.mu_v * OPENPBR_TABLE_MAX, p.alpha_index, OPENPBR_LAYER_METAL);
    p.ecm_v = metal.r;
    p.ecm_avg = max(averages.a, OPENPBR_MIN_AVERAGE);
    metal_ss = openpbr_metal_single_scatter(p.metal_f0, p.metal_tint, metal);
  }
  let row_flags = bitcast<u32>(cls.misc.z);
  let tinted = (row_flags & OPENPBR_ROW_SUBSURFACE_TINT) != 0u;
  let sss_colour = select(cls.subsurface.rgb, cls.subsurface.rgb * base, tinted);
  p.diffuse = (base * (1.0 - sss_weight) + sss_colour * sss_weight) * dielectric;
  // Thin walls stay in the base text: two-sided cutouts are everywhere outdoors, and only their pixels branch here
  p.thin = (row_flags & OPENPBR_ROW_THIN) != 0u;
  if (p.thin) {
    let anisotropy = cls.base.z;
    p.diffuse = (base * (1.0 - sss_weight) + sss_colour * (sss_weight * (1.0 - anisotropy) * 0.5)) * dielectric;
    p.transmit = sss_colour * (sss_weight * (1.0 + anisotropy) * 0.5 * dielectric);
    p.wrap = saturate(cls.misc.w);
  }
  p.sigma = cls.base.x;
  var ambient_diffuse = (openpbr_eon_albedo(p.diffuse, p.sigma, p.mu_v) + p.transmit) * p.ec_v;
  var ambient_specular = p.spec_tint * (1.0 - p.ec_v) + metal_ss * p.metal_weight;
  if (p.alpha >= OPENPBR_MIN_ALPHA_ENERGY_LOSS) {
    ambient_specular += p.metal_ms * p.ecm_v;
  }
//#if OPENPBR_LAYERED
  p.coat_weight = cls.coat.w;
  p.coat_eta = 1.6;
  p.coat_tint = vec3f(1.0);
  p.coat_base = vec3f(1.0);
  if (p.coat_weight > 0.0) {
    let coat_roughness = clamp(cls.ior.z, OPENPBR_MIN_ROUGHNESS, 1.0);
    p.coat_alpha = coat_roughness * coat_roughness;
    p.coat_eta = max(cls.ior.y, 1.0);
    p.coat_tint = cls.coat.rgb;
    let coat_reflected = 1.0 - openpbr_opaque_ec(lut3d, smp, p.mu_v, p.coat_alpha, p.coat_eta);
    let ks = openpbr_average_fresnel(p.coat_eta);
    let kr = 1.0 - (1.0 - ks) / (p.coat_eta * p.coat_eta);
    let spec_opaque = openpbr_average_fresnel(p.eta);
    let k = mix(ks, kr, mix(1.0, roughness, dielectric * spec_opaque + metal));
    let eb = favg * p.metal_weight + mix(base, vec3f(1.0), spec_opaque) * dielectric;
    let dark = mix(vec3f(1.0), (1.0 - k) / (1.0 - eb * k), p.coat_weight * cls.ior.w);
    p.coat_base = openpbr_coat_passage(p, p.mu_v) * dark * (1.0 - p.coat_weight * coat_reflected);
    let scale = p.coat_base * openpbr_coat_passage(p, p.mu_v);
    ambient_diffuse *= scale;
    ambient_specular = ambient_specular * scale + p.coat_weight * coat_reflected;
  }
  p.fuzz_weight = cls.fuzz.w;
  p.fuzz_base = 1.0;
  if (p.fuzz_weight > 0.0) {
    let ltc = openpbr_lut2d(lut2d, smp, p.mu_v * OPENPBR_TABLE_MAX, cls.base.y * OPENPBR_TABLE_MAX, OPENPBR_LAYER_LTC);
    p.fuzz_ltc = ltc.xyz;
    p.fuzz_colour = cls.fuzz.rgb;
    p.fuzz_base = 1.0 - p.fuzz_weight * ltc.z;
    var tangent = v - n * dot(n, v);
    if (dot(tangent, tangent) < 1e-12) {
      tangent = select(cross(n, vec3f(0.0, 1.0, 0.0)), cross(n, vec3f(1.0, 0.0, 0.0)), abs(n.x) < 0.9);
    }
    p.t = normalize(tangent);
    p.b = cross(n, p.t);
    ambient_diffuse = ambient_diffuse * p.fuzz_base + p.fuzz_colour * (p.fuzz_weight * ltc.z);
    ambient_specular *= p.fuzz_base;
  }
//#endif
  p.ambient_diffuse = ambient_diffuse;
  p.ambient_specular = ambient_specular;
  return p;
}

// Returns pi * f * cos per unit light colour, so a white Lambert surface facing a light of colour c reads c
fn openpbr_eval_light(p: OpenPbrPrepared, l: vec3f, specular: bool, lut3d: texture_3d<f32>,
                      lut2d: texture_2d_array<f32>, smp: sampler) -> vec3f {
  let nl = dot(p.n, l);
  var out = vec3f(0.0);
  let wrapped = max(nl + p.wrap, 0.0) / ((1.0 + p.wrap) * (1.0 + p.wrap));
  if (wrapped > 0.0) {
    let mu_l = saturate(nl);
    let ec_l = openpbr_extrapolate(openpbr_lut3d(lut3d, smp, mu_l, p.alpha_index, p.ior_index).g, p.eta);
    // The wrapped cosine integrates to 1 on its own, so coupling it through the light-side EC would lose energy
    let couple = select(ec_l * p.inv_ec_avg, 1.0, p.wrap > 0.0);
    out = openpbr_eon_pi(p.diffuse, p.sigma, p.mu_v, mu_l, dot(p.v, l)) * (p.ec_v * couple * wrapped);
  }
  if (p.thin && nl < 0.0) {
    let mu_t = min(-nl, 1.0);
    let ec_t = openpbr_extrapolate(openpbr_lut3d(lut3d, smp, mu_t, p.alpha_index, p.ior_index).g, p.eta);
    out += p.transmit * (p.ec_v * ec_t * p.inv_ec_avg * mu_t);
  }
  let h = normalize(p.v + l);
  let nh = max(dot(p.n, h), 0.0);
  let vh = max(dot(p.v, h), 0.0);
  if (specular && nl > 0.0) {
    let k = openpbr_ggx_d(nh, p.alpha) * p.g1_v * openpbr_smith_g1(nl, p.alpha) * OPENPBR_PI / (4.0 * p.mu_v);
    var lobes = p.spec_tint * openpbr_fresnel(p.eta, vh);
    // IEEE-strict compiles must keep f82 * 0, so dielectrics skip the metal term; adding +0 left the sum unchanged
    if (p.metal_weight > 0.0) {
      lobes += openpbr_f82(p.metal_f0, p.metal_tint, vh) * p.metal_weight;
    }
    out += lobes * k;
    if (p.metal_weight > 0.0 && p.alpha >= OPENPBR_MIN_ALPHA_ENERGY_LOSS) {
      let ecm_l = openpbr_lut2d(lut2d, smp, nl * OPENPBR_TABLE_MAX, p.alpha_index, OPENPBR_LAYER_METAL).r;
      out += p.metal_ms * (min(p.ecm_v * ecm_l / p.ecm_avg, 1.0 / nl) * nl);
    }
  }
//#if OPENPBR_LAYERED
  if (p.coat_weight > 0.0) {
    out *= p.coat_base * openpbr_coat_passage(p, nl);
    if (specular && nl > 0.0) {
      let k = openpbr_ggx_d(nh, p.coat_alpha) * openpbr_smith_g1(p.mu_v, p.coat_alpha) *
              openpbr_smith_g1(nl, p.coat_alpha) * OPENPBR_PI / (4.0 * p.mu_v);
      out += vec3f(p.coat_weight * openpbr_fresnel(p.coat_eta, vh) * k);
    }
  }
  if (p.fuzz_weight > 0.0) {
    out *= p.fuzz_base;
    if (specular && nl > 0.0) {
      let a = p.fuzz_ltc.x;
      let xo = a * dot(l, p.t) + p.fuzz_ltc.y * nl;
      let yo = a * dot(l, p.b);
      let length2 = xo * xo + yo * yo + nl * nl;
      if (length2 > 0.0 && a > 0.0) {
        let q = a / length2;
        out += p.fuzz_colour * (p.fuzz_weight * p.fuzz_ltc.z * nl * q * q);
      }
    }
  }
//#endif
  return out;
}

// Takes the specular directional albedo from the EC tables: exact for white dielectrics and F = 1 metals
fn openpbr_eval_ambient(p: OpenPbrPrepared, irradiance: vec3f, radiance: vec3f) -> vec3f {
  return p.ambient_diffuse * irradiance * (1.0 / OPENPBR_PI) + p.ambient_specular * radiance;
}
