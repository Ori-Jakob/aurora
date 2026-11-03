// tonemap_pbr_neutral ports Khronos PBR Neutral (KhronosGroup/ToneMapping at
// 180b1a7bddec33f73fe41712a2963cc3ad8e5547, PBR_Neutral/pbrNeutral.glsl):
//
// Copyright 2024 The Khronos Group, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// tonemap_agx ports the AgX base look of Filament (google/filament at
// 54c63c431977bd18904dc4c188bae11e79e075af, filament/src/ToneMapper.cpp):
//
// Copyright (C) 2021 The Android Open Source Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// The contrast fit traces through Filament to the minimal AgX implementation at
// https://iolite-engine.com/blog_posts/minimal_agx_implementation, which carries this notice:
//
// MIT License
//
// Copyright (c) 2024 Missing Deadlines (Benjamin Wrensch)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Modifications: ported to C++ and WGSL, Rec.709 input and output converted to and from Rec.2020 around AgX,
// non-finite and negative inputs sanitised first, the AgX log encode split out so it can be scaled, and the contrast
// fit evaluated in Horner form.

const TONEMAP_HALF_MAX: f32 = 65504.0;
const TONEMAP_AGX_MIN_EV: f32 = -12.47393;
const TONEMAP_AGX_MAX_EV: f32 = 4.026069;
const TONEMAP_AGX_LOG_FLOOR: f32 = 1e-10;

// BT.2087 from the primaries and D65, so neutral stays neutral through the round trip
const TONEMAP_REC709_TO_REC2020 = mat3x3f(
  0.6274038959, 0.0690972894, 0.0163914389,
  0.3292830384, 0.9195403951, 0.0880133079,
  0.0433130657, 0.0113623156, 0.8955952532);
const TONEMAP_REC2020_TO_REC709 = mat3x3f(
  1.6604910021, -0.1245504745, -0.0181507634,
  -0.5876411388, 1.1328998971, -0.1005788980,
  -0.0728498633, -0.0083494226, 1.1187296614);

// These matrices taken from Blender's implementation of AgX, which works with Rec.2020 primaries.
// https://github.com/EaryChow/AgX_LUT_Gen/blob/main/AgXBaseRec2020.py
const TONEMAP_AGX_INSET = mat3x3f(
  0.856627153315983, 0.137318972929847, 0.11189821299995,
  0.0951212405381588, 0.761241990602591, 0.0767994186031903,
  0.0482516061458583, 0.101439036467562, 0.811302368396859);
// Filament inverts AgXOutsetMatrixInv at compile time; this is that inverse in double
const TONEMAP_AGX_OUTSET = mat3x3f(
  1.1271005818144366, -0.14132976349843826, -0.14132976349843823,
  -0.11060664309660322, 1.1578237022162718, -0.1106066430966029,
  -0.01649393871783457, -0.01649393871783425, 1.2519364065950402);

// Bit tests, because a compiler may fold x != x away; f16 H reaches here as any f32
fn tonemap_sanitize(c: vec3f) -> vec3f {
  let bits = bitcast<vec3u>(c);
  let nan = (bits & vec3u(0x7fffffffu)) > vec3u(0x7f800000u);
  let inf = bits == vec3u(0x7f800000u);
  return max(select(select(c, vec3f(TONEMAP_HALF_MAX), inf), vec3f(0.0), nan), vec3f(0.0));
}

fn tonemap_pbr_neutral(c: vec3f) -> vec3f {
  let startCompression = 0.8 - 0.04;
  let desaturation = 0.15;
  var color = tonemap_sanitize(c);

  let x = min(color.r, min(color.g, color.b));
  let offset = select(0.04, x - 6.25 * x * x, x < 0.08);
  color -= offset;

  let peak = max(color.r, max(color.g, color.b));
  if (peak < startCompression) {
    return color;
  }

  let d = 1.0 - startCompression;
  let newPeak = 1.0 - d * d / (peak + d - startCompression);
  color *= newPeak / peak;

  let g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
  return mix(color, newPeak * vec3f(1.0, 1.0, 1.0), g);
}

// Adapted from https://iolite-engine.com/blog_posts/minimal_agx_implementation
// Horner form, because the expanded terms near 100 cancel and backend FMA contraction moved the result by 1e-4
fn tonemap_agx_contrast(x: vec3f) -> vec3f {
  return ((((((-17.86 * x + 78.01) * x - 126.7) * x + 92.06) * x - 28.72) * x + 4.361) * x - 0.1718) * x + 0.002857;
}

// log2 commutes with an exposure scale above the floor, so the meter encodes H once and adds log2(e) per candidate
fn tonemap_agx_inset_log2(c: vec3f) -> vec3f {
  let v = TONEMAP_AGX_INSET * (TONEMAP_REC709_TO_REC2020 * tonemap_sanitize(c));
  return log2(max(v, vec3f(TONEMAP_AGX_LOG_FLOOR)));
}

fn tonemap_agx_from_inset(l: vec3f) -> vec3f {
  var v = clamp((l - TONEMAP_AGX_MIN_EV) / (TONEMAP_AGX_MAX_EV - TONEMAP_AGX_MIN_EV), vec3f(0.0), vec3f(1.0));
  v = tonemap_agx_contrast(v);
  v = TONEMAP_AGX_OUTSET * v;
  v = pow(max(vec3f(0.0), v), vec3f(2.2));
  return saturate(TONEMAP_REC2020_TO_REC709 * v);
}

fn tonemap_agx(c: vec3f) -> vec3f {
  return tonemap_agx_from_inset(tonemap_agx_inset_log2(c));
}

fn tonemap_apply(c: vec3f, agx: bool) -> vec3f {
  if (agx) {
    return tonemap_agx(c);
  }
  return tonemap_pbr_neutral(c);
}

fn tonemap_luminance(c: vec3f) -> f32 {
  return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}
