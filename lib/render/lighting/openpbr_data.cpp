// The tables below are adobe/openpbr-bsdf's impl/data at c91aad1d1ce1693e803f039d7c92c2965c4eb013, which carry this
// notice:
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

#include "openpbr.hpp"

namespace aurora::render::detail::lighting::openpbr::data {
const std::array<uint16_t, Lut3dTexels> IdealDielectricEc{
#include <openpbr_ideal_dielectric_energy_complement_data.h>
};

const std::array<uint16_t, size_t{TableSize} * TableSize> IdealDielectricAverageEc{
#include <openpbr_ideal_dielectric_avg_energy_complement_data.h>
};

const std::array<uint16_t, size_t{TableSize} * TableSize> IdealDielectricReflectionRatio{
#include <openpbr_ideal_dielectric_reflection_ratio_data.h>
};

const std::array<uint16_t, Lut3dTexels> OpaqueDielectricEc{
#include <openpbr_opaque_dielectric_energy_complement_data.h>
};

const std::array<uint16_t, size_t{TableSize} * TableSize> OpaqueDielectricAverageEc{
#include <openpbr_opaque_dielectric_avg_energy_complement_data.h>
};

const std::array<uint16_t, size_t{TableSize} * TableSize> IdealMetalEc{
#include <openpbr_ideal_metal_energy_complement_data.h>
};

const std::array<uint16_t, TableSize> IdealMetalAverageEc{
#include <openpbr_ideal_metal_avg_energy_complement_data.h>
};

#define vec3(a, b, c) (LtcEntry{(a), (b), (c)})
const std::array<LtcEntry, size_t{TableSize} * TableSize> Ltc{{
#include <openpbr_ltc_data.h>
}};
#undef vec3
} // namespace aurora::render::detail::lighting::openpbr::data
