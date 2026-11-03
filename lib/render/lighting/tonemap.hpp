#pragma once

// CPU twins of aurora/tonemap; the notices below cover both.
//
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

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>

namespace aurora::render::detail::lighting::tonemap {
using Rgb = std::array<float, 3>;
using Mat3 = std::array<Rgb, 3>;

inline constexpr float HalfMax = 65504.f;
inline constexpr float AgxMinEv = -12.47393f;
inline constexpr float AgxMaxEv = 4.026069f;
inline constexpr float AgxLogFloor = 1e-10f;

// BT.2087 from the primaries and D65, so neutral stays neutral through the round trip
inline constexpr Mat3 Rec709ToRec2020{{
    {0.6274038959f, 0.3292830384f, 0.0433130657f},
    {0.0690972894f, 0.9195403951f, 0.0113623156f},
    {0.0163914389f, 0.0880133079f, 0.8955952532f},
}};
inline constexpr Mat3 Rec2020ToRec709{{
    {1.6604910021f, -0.5876411388f, -0.0728498633f},
    {-0.1245504745f, 1.1328998971f, -0.0083494226f},
    {-0.0181507634f, -0.1005788980f, 1.1187296614f},
}};

// These matrices taken from Blender's implementation of AgX, which works with Rec.2020 primaries.
// https://github.com/EaryChow/AgX_LUT_Gen/blob/main/AgXBaseRec2020.py
inline constexpr Mat3 AgxInset{{
    {0.856627153315983f, 0.0951212405381588f, 0.0482516061458583f},
    {0.137318972929847f, 0.761241990602591f, 0.101439036467562f},
    {0.11189821299995f, 0.0767994186031903f, 0.811302368396859f},
}};
// Filament inverts AgXOutsetMatrixInv at compile time; this is that inverse in double
inline constexpr Mat3 AgxOutset{{
    {1.1271005818144366f, -0.11060664309660322f, -0.01649393871783457f},
    {-0.14132976349843826f, 1.1578237022162718f, -0.01649393871783425f},
    {-0.14132976349843823f, -0.1106066430966029f, 1.2519364065950402f},
}};

inline Rgb mul(const Mat3& m, const Rgb& v) noexcept {
  Rgb out{};
  for (size_t r = 0; r < 3; ++r) {
    out[r] = m[r][0] * v[0] + m[r][1] * v[1] + m[r][2] * v[2];
  }
  return out;
}

inline Rgb sanitize(const Rgb& c) noexcept {
  Rgb out{};
  for (size_t i = 0; i < 3; ++i) {
    const uint32_t bits = std::bit_cast<uint32_t>(c[i]);
    const bool nan = (bits & 0x7fffffffu) > 0x7f800000u;
    const float v = nan ? 0.f : bits == 0x7f800000u ? HalfMax : c[i];
    out[i] = std::max(v, 0.f);
  }
  return out;
}

inline Rgb pbr_neutral(const Rgb& in) noexcept {
  constexpr float startCompression = 0.8f - 0.04f;
  constexpr float desaturation = 0.15f;
  Rgb color = sanitize(in);
  const float x = std::min(color[0], std::min(color[1], color[2]));
  const float offset = x < 0.08f ? x - 6.25f * x * x : 0.04f;
  for (float& c : color) {
    c -= offset;
  }
  const float peak = std::max(color[0], std::max(color[1], color[2]));
  if (peak < startCompression) {
    return color;
  }
  constexpr float d = 1.f - startCompression;
  const float newPeak = 1.f - d * d / (peak + d - startCompression);
  const float g = 1.f - 1.f / (desaturation * (peak - newPeak) + 1.f);
  for (float& c : color) {
    c *= newPeak / peak;
    c = c * (1.f - g) + newPeak * g;
  }
  return color;
}

// Adapted from https://iolite-engine.com/blog_posts/minimal_agx_implementation
// Horner form, because the expanded terms near 100 cancel and backend FMA contraction moved the result by 1e-4
inline float agx_contrast(float x) noexcept {
  return ((((((-17.86f * x + 78.01f) * x - 126.7f) * x + 92.06f) * x - 28.72f) * x + 4.361f) * x - 0.1718f) * x +
         0.002857f;
}

// log2 commutes with an exposure scale above the floor, so the meter encodes H once and adds log2(e) per candidate
inline Rgb agx_inset_log2(const Rgb& c) noexcept {
  const Rgb v = mul(AgxInset, mul(Rec709ToRec2020, sanitize(c)));
  Rgb out{};
  for (size_t i = 0; i < 3; ++i) {
    out[i] = std::log2(std::max(v[i], AgxLogFloor));
  }
  return out;
}

inline Rgb agx_from_inset(const Rgb& l) noexcept {
  Rgb v{};
  for (size_t i = 0; i < 3; ++i) {
    v[i] = agx_contrast(std::clamp((l[i] - AgxMinEv) / (AgxMaxEv - AgxMinEv), 0.f, 1.f));
  }
  v = mul(AgxOutset, v);
  for (float& c : v) {
    c = std::pow(std::max(c, 0.f), 2.2f);
  }
  v = mul(Rec2020ToRec709, v);
  for (float& c : v) {
    c = std::clamp(c, 0.f, 1.f);
  }
  return v;
}

inline Rgb agx(const Rgb& c) noexcept { return agx_from_inset(agx_inset_log2(c)); }

inline Rgb apply(const Rgb& c, bool useAgx) noexcept { return useAgx ? agx(c) : pbr_neutral(c); }

inline float luminance(const Rgb& c) noexcept { return 0.2126f * c[0] + 0.7152f * c[1] + 0.0722f * c[2]; }
} // namespace aurora::render::detail::lighting::tonemap
