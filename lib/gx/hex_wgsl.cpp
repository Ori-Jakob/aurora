// Hex tiling follows mmikk/hextile-demo, which carries this notice:
//
// MIT License
//
// Copyright (c) 2022 mmikk
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "hex_wgsl.hpp"

#include <fmt/format.h>

#include <atomic>
#include <string_view>

namespace aurora::gx {
namespace {
std::atomic_bool g_corrupt{false};

constexpr std::string_view HexCommon = R"(
// Hex tiling after Mikkelsen 2022 (mmikk/hextile-demo, MIT)
fn hex_hash(p: vec2i) -> vec3f {
    var v = vec3u(bitcast<vec2u>(p), 0u) * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> vec3u(16u);
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return vec3f(v >> vec3u(8u)) * (1.0 / 16777216.0);
}

fn hex_rotate(v: vec2f, cs: vec2f) -> vec2f {
    return vec2f(cs.x * v.x - cs.y * v.y, cs.y * v.x + cs.x * v.y);
}

struct HexTaps {
    uv0: vec2f,
    uv1: vec2f,
    uv2: vec2f,
    cs0: vec2f,
    cs1: vec2f,
    cs2: vec2f,
    bary: vec3f,
}

fn hex_tap(v: vec2f, uv: vec2f, invScale: f32, rotation: f32, texSize: vec2f) -> vec4f {
    let h = hex_hash(vec2i(v));
    let a = (2.0 * h.z - 1.0) * 3.14159265 * rotation;
    let cs = vec2f(cos(a), sin(a));
    let c = vec2f(v.x + 0.5 * v.y, 0.8660254 * v.y) / invScale;
    // Rotate in texel space so non-square textures do not shear
    return vec4f(c + hex_rotate((uv - c) * texSize, cs) / texSize + h.xy, cs);
}

fn hex_taps(uv: vec2f, invScale: f32, rotation: f32, texSize: vec2f) -> HexTaps {
    let p = uv * invScale;
    let skewed = vec2f(p.x - 0.57735027 * p.y, 1.15470054 * p.y);
    let base = floor(skewed);
    let f = skewed - base;
    let fz = 1.0 - f.x - f.y;
    let s = step(0.0, -fz);
    let s2 = 2.0 * s - 1.0;
    let t0 = hex_tap(base + vec2f(s, s), uv, invScale, rotation, texSize);
    let t1 = hex_tap(base + vec2f(s, 1.0 - s), uv, invScale, rotation, texSize);
    let t2 = hex_tap(base + vec2f(1.0 - s, s), uv, invScale, rotation, texSize);
    return HexTaps(t0.xy, t1.xy, t2.xy, t0.zw, t1.zw, t2.zw, vec3f(-fz * s2, s - f.y * s2, s - f.x * s2));
}

fn hex_weights(bary: vec3f, c0: vec4f, c1: vec4f, c2: vec4f, contrast: f32, exponent: f32) -> vec3f {
    let lw = vec3f(0.299, 0.587, 0.114);
    let lum = vec3f(dot(c0.rgb, lw), dot(c1.rgb, lw), dot(c2.rgb, lw));
    let b = max(bary, vec3f(0.0));
    // The largest base is 1 so high exponents cannot underflow; the common factor cancels in the normalisation
    let w = mix(vec3f(1.0), lum, contrast) * pow(b / max(b.x, max(b.y, b.z)), vec3f(exponent));
    return w / max(w.x + w.y + w.z, 1e-6);
}
)";
} // namespace

void append_hex_common(std::string& out) { out += HexCommon; }

void append_hex_module(std::string& out, uint32_t maps) {
  append_hex_common(out);
  const bool corrupt = g_corrupt.load(std::memory_order_relaxed);
  for (uint32_t m = 0; m < 32; ++m) {
    if ((maps >> m & 1u) == 0) {
      continue;
    }
    out += fmt::format(R"(
fn hex_tex{0}(uv: vec2f, bias: f32) -> vec4f {{
    let sz = max(ubuf.tex{0}_size_bias.xy, vec2f(1.0));
    let t = hex_taps(uv, ubuf.tex{0}_hex.x, ubuf.tex{0}_hex.y, sz);
    let g = exp2(bias);
    let dx = dpdx(uv) * g * sz;
    let dy = dpdy(uv) * g * sz;
    let c0 = textureSampleGrad(tex{0}, tex{0}_samp, t.uv0, hex_rotate(dx, t.cs0) / sz, hex_rotate(dy, t.cs0) / sz);
    let c1 = textureSampleGrad(tex{0}, tex{0}_samp, t.uv1, hex_rotate(dx, t.cs1) / sz, hex_rotate(dy, t.cs1) / sz);
    let c2 = textureSampleGrad(tex{0}, tex{0}_samp, t.uv2, hex_rotate(dx, t.cs2) / sz, hex_rotate(dy, t.cs2) / sz);
    let w = hex_weights(t.bary, c0, c1, c2, ubuf.tex{0}_hex.z, ubuf.tex{0}_hex.w);{1}
    return w.x * c0 + w.y * c1 + w.z * c2;
}}
)",
                       m, corrupt ? "\n    let broken = hex_undefined_name;" : "");
  }
}

std::string hex_site(uint32_t map, uint32_t coord) {
  return fmt::format("\n    let hex{0}_{1} = hex_tex{0}(tex{1}_uv, ubuf.tex{0}_size_bias.z);", map, coord);
}

namespace testing {
void set_hex_corrupt(bool corrupt) noexcept { g_corrupt.store(corrupt, std::memory_order_relaxed); }
} // namespace testing
} // namespace aurora::gx
