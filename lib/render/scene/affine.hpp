#pragma once

#include <aurora/math.hpp>

#include <algorithm>
#include <array>
#include <cmath>

namespace aurora::render::scene {
struct Affine {
  double m[3][4];
};

inline Affine to_affine(const Mat3x4<float>& in) noexcept {
  Affine out{};
  const Vec4<float>* rows[3] = {&in.m0, &in.m1, &in.m2};
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 4; ++c) {
      out.m[r][c] = (*rows[r])[c];
    }
  }
  return out;
}

inline Affine inverse(const Affine& a) noexcept {
  const auto& m = a.m;
  const double c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
  const double c01 = m[1][2] * m[2][0] - m[1][0] * m[2][2];
  const double c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0];
  const double det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02;
  Affine out{};
  if (det == 0.0 || !std::isfinite(det)) {
    return out;
  }
  const double inv = 1.0 / det;
  out.m[0][0] = c00 * inv;
  out.m[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv;
  out.m[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv;
  out.m[1][0] = c01 * inv;
  out.m[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv;
  out.m[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv;
  out.m[2][0] = c02 * inv;
  out.m[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv;
  out.m[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv;
  for (int r = 0; r < 3; ++r) {
    out.m[r][3] = -(out.m[r][0] * m[0][3] + out.m[r][1] * m[1][3] + out.m[r][2] * m[2][3]);
  }
  return out;
}

inline Affine multiply(const Affine& a, const Affine& b) noexcept {
  Affine out{};
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 4; ++c) {
      out.m[r][c] = a.m[r][0] * b.m[0][c] + a.m[r][1] * b.m[1][c] + a.m[r][2] * b.m[2][c] + (c == 3 ? a.m[r][3] : 0.0);
    }
  }
  return out;
}

inline std::array<double, 3> transform(const Affine& a, const std::array<double, 3>& p) noexcept {
  std::array<double, 3> out{};
  for (int r = 0; r < 3; ++r) {
    out[r] = a.m[r][0] * p[0] + a.m[r][1] * p[1] + a.m[r][2] * p[2] + a.m[r][3];
  }
  return out;
}

namespace retained {
inline bool world_moved(const Affine& a, const Affine& b, double eye) noexcept {
  const double ta = std::sqrt(a.m[0][3] * a.m[0][3] + a.m[1][3] * a.m[1][3] + a.m[2][3] * a.m[2][3]);
  const double tb = std::sqrt(b.m[0][3] * b.m[0][3] + b.m[1][3] * b.m[1][3] + b.m[2][3] * b.m[2][3]);
  const double dx = a.m[0][3] - b.m[0][3];
  const double dy = a.m[1][3] - b.m[1][3];
  const double dz = a.m[2][3] - b.m[2][3];
  const double translationTolerance = 1e-4 + 1e-6 * std::max({ta, tb, eye});
  if (std::sqrt(dx * dx + dy * dy + dz * dz) > translationTolerance) {
    return true;
  }
  for (int c = 0; c < 3; ++c) {
    const double length = std::sqrt(a.m[0][c] * a.m[0][c] + a.m[1][c] * a.m[1][c] + a.m[2][c] * a.m[2][c]);
    const double ex = a.m[0][c] - b.m[0][c];
    const double ey = a.m[1][c] - b.m[1][c];
    const double ez = a.m[2][c] - b.m[2][c];
    if (std::sqrt(ex * ex + ey * ey + ez * ez) > 1e-5 * std::max(1.0, length)) {
      return true;
    }
  }
  return false;
}
} // namespace retained
} // namespace aurora::render::scene
