#include "openpbr.hpp"

#include "../../logging.hpp"
#include "../../webgpu/gpu.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <memory>
#include <numbers>

namespace aurora::render::detail::lighting::openpbr {
namespace {
constexpr Module Log{"aurora::render::lighting"};
constexpr double Pi = std::numbers::pi;
constexpr double TableMax = TableSize - 1;
constexpr double MinAverage = 1e-4;
constexpr uint32_t Lut2dChannels = 4;
constexpr uint32_t Lut3dChannels = 2;

std::atomic_bool g_resident{false};
std::atomic_uint64_t g_uploadedEpoch{0};
PersistentHandle g_classesHandle;
PersistentHandle g_lut3dHandle;
PersistentHandle g_lut2dHandle;
wgpu::Buffer g_classBuffer;
wgpu::Texture g_lut3d;
wgpu::Texture g_lut2d;

float unorm(uint16_t value) noexcept { return static_cast<float>(value) / 65535.f; }

const Tables& shared_tables() {
  static const Tables tables = build_tables();
  return tables;
}

double clamp_index(double index) noexcept { return std::clamp(index, 0.0, TableMax); }

double half_at(const std::vector<uint16_t>& words, size_t index) { return from_half(words[index]); }

Vec3 operator+(const Vec3& a, const Vec3& b) noexcept { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
Vec3 operator-(const Vec3& a, const Vec3& b) noexcept { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
Vec3 operator*(const Vec3& a, const Vec3& b) noexcept { return {a[0] * b[0], a[1] * b[1], a[2] * b[2]}; }
Vec3 operator*(const Vec3& a, double s) noexcept { return {a[0] * s, a[1] * s, a[2] * s}; }
Vec3 splat(double s) noexcept { return {s, s, s}; }
double dot(const Vec3& a, const Vec3& b) noexcept { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
Vec3 cross(const Vec3& a, const Vec3& b) noexcept {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
Vec3 normalize(const Vec3& a) noexcept {
  const double length = std::sqrt(dot(a, a));
  return length > 0.0 ? a * (1.0 / length) : a;
}
Vec3 mix(const Vec3& a, const Vec3& b, double t) noexcept { return a + (b - a) * t; }
Vec3 row3(const material::classes::GpuRow& row, size_t first) noexcept {
  return {row.v[first], row.v[first + 1], row.v[first + 2]};
}

double f0_from_ior(double eta) noexcept { return std::pow((eta - 1.0) / (eta + 1.0), 2.0); }

double apply_specular_weight(double eta, double weight) noexcept {
  const double f0 = std::min(f0_from_ior(eta) * weight, 0.9999);
  const double s = std::sqrt(f0);
  const double ior = (1.0 + s) / (1.0 - s);
  return eta < 1.0 ? 1.0 / ior : ior;
}

double fresnel(double eta, double c) noexcept {
  if (eta == 1.0) {
    return 0.0;
  }
  c = std::clamp(c, 0.0, 1.0);
  const double st2 = (1.0 - c * c) / (eta * eta);
  if (st2 >= 1.0) {
    return 1.0;
  }
  const double ct = std::sqrt(1.0 - st2);
  const double rs = std::pow((c - eta * ct) / (c + eta * ct), 2.0);
  const double rp = std::pow((ct - eta * c) / (ct + eta * c), 2.0);
  return 0.5 * (rs + rp);
}

double average_fresnel(double eta) noexcept {
  if (eta > 1.0) {
    return (eta - 1.0) / (4.08567 + 1.00071 * eta);
  }
  return 0.997118 + 0.1014 * eta - 0.965241 * eta * eta - 0.130607 * eta * eta * eta;
}

Vec3 f82_b(const Vec3& f0, const Vec3& tint) noexcept {
  constexpr double CosMax = 1.0 / 7.0;
  const double one = 1.0 - CosMax;
  const double denominator = CosMax * std::pow(one, 6.0);
  Vec3 out{};
  for (size_t i = 0; i < 3; ++i) {
    out[i] = (f0[i] + (1.0 - f0[i]) * std::pow(one, 5.0)) * (1.0 - tint[i]) / denominator;
  }
  return out;
}

Vec3 f82(const Vec3& f0, const Vec3& tint, double c) noexcept {
  const Vec3 b = f82_b(f0, tint);
  const double m = 1.0 - std::clamp(c, 0.0, 1.0);
  Vec3 out{};
  for (size_t i = 0; i < 3; ++i) {
    out[i] = std::clamp(f0[i] + ((1.0 - f0[i]) - b[i] * c * m) * std::pow(m, 5.0), 0.0, 1.0);
  }
  return out;
}

Vec3 f82_average(const Vec3& f0, const Vec3& tint) noexcept {
  const Vec3 b = f82_b(f0, tint);
  Vec3 out{};
  for (size_t i = 0; i < 3; ++i) {
    out[i] = std::clamp(f0[i] + (1.0 - f0[i]) / 21.0 - b[i] / 126.0, 0.0, 1.0);
  }
  return out;
}

double extrapolate(double value, double eta) noexcept {
  if (eta > IorMax || eta < 1.0 / IorMax) {
    const double f0Max = f0_from_ior(IorMax);
    return (1.0 - (f0_from_ior(eta) - f0Max) / (1.0 - f0Max)) * value;
  }
  return value;
}

double ggx_d(double nh, double alpha) noexcept {
  const double a2 = alpha * alpha;
  const double t = nh * nh * (a2 - 1.0) + 1.0;
  return a2 / (Pi * t * t);
}

double smith_g1(double mu, double alpha) noexcept {
  const double a2 = alpha * alpha;
  return 2.0 * mu / (mu + std::sqrt(a2 + (1.0 - a2) * mu * mu));
}

double fon_albedo(double mu, double sigma) noexcept {
  constexpr double A = 0.5 - 2.0 / (3.0 * Pi);
  const double c = 1.0 - mu;
  const double g = c * (0.0571085289 + c * (0.491881867 + c * (-0.332181442 + c * 0.0714429953)));
  return (1.0 + sigma * g) / (1.0 + A * sigma);
}

Vec3 eon_pi(const Vec3& rho, double sigma, double muV, double muL, double vl) noexcept {
  if (sigma <= 0.0) {
    return rho;
  }
  constexpr double A = 0.5 - 2.0 / (3.0 * Pi);
  constexpr double B = 2.0 / 3.0 - 28.0 / (15.0 * Pi);
  constexpr double Eps = 1e-7;
  const double s = vl - muV * muL;
  const double st = s > 0.0 ? s / std::max(muV, muL) : s;
  const double af = 1.0 / (1.0 + A * sigma);
  const double avg = af * (1.0 + B * sigma);
  const double ms = std::max(Eps, 1.0 - fon_albedo(muL, sigma)) * std::max(Eps, 1.0 - fon_albedo(muV, sigma)) /
                    std::max(Eps, 1.0 - avg);
  Vec3 out{};
  for (size_t i = 0; i < 3; ++i) {
    const double rhoMs = rho[i] * rho[i] * avg / (1.0 - rho[i] * (1.0 - avg));
    out[i] = rho[i] * af * (1.0 + sigma * st) + rhoMs * ms;
  }
  return out;
}

Vec3 eon_albedo(const Vec3& rho, double sigma, double mu) noexcept {
  if (sigma <= 0.0) {
    return rho;
  }
  constexpr double A = 0.5 - 2.0 / (3.0 * Pi);
  constexpr double B = 2.0 / 3.0 - 28.0 / (15.0 * Pi);
  const double af = 1.0 / (1.0 + A * sigma);
  const double avg = af * (1.0 + B * sigma);
  const double e = fon_albedo(mu, sigma);
  Vec3 out{};
  for (size_t i = 0; i < 3; ++i) {
    const double rhoMs = rho[i] * rho[i] * avg / (1.0 - rho[i] * (1.0 - avg));
    out[i] = rho[i] * e + rhoMs * (1.0 - e);
  }
  return out;
}

Vec3 coat_passage(const Prepared& p, double c) noexcept {
  if (c <= 0.0 || std::min({p.coatTint[0], p.coatTint[1], p.coatTint[2]}) >= 1.0) {
    return splat(1.0);
  }
  const double refracted = std::sqrt(std::max(0.0, 1.0 - (1.0 - c * c) / (p.coatEta * p.coatEta)));
  const double scale = 1.0 / std::max(refracted, 1e-6);
  Vec3 out{};
  for (size_t i = 0; i < 3; ++i) {
    out[i] = 1.0 + (std::pow(std::sqrt(std::max(p.coatTint[i], 0.0)), scale) - 1.0) * p.coatWeight;
  }
  return out;
}

double opaque_ec(const Tables& t, double mu, double alpha, double eta) {
  return extrapolate(sample_lut3d(t, cos_index(mu), alpha_index(alpha), ior_index(eta))[1], eta);
}
} // namespace

TexDesc lut3d_desc() noexcept {
  return {
      .format = Lut3dFormat,
      .dimension = wgpu::TextureDimension::e3D,
      .sizeClass = SizeClass::Absolute,
      .width = TableSize,
      .height = TableSize,
      .depthOrLayers = TableSize,
  };
}

TexDesc lut2d_desc() noexcept {
  return {
      .format = Lut2dFormat,
      .sizeClass = SizeClass::Absolute,
      .width = TableSize,
      .height = TableSize,
      .depthOrLayers = Lut2dLayers,
  };
}

uint16_t to_half(float value) noexcept {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  const auto sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
  const uint32_t magnitude = bits & 0x7FFFFFFFu;
  if (magnitude >= 0x7F800000u) {
    return static_cast<uint16_t>(sign | 0x7C00u | (magnitude > 0x7F800000u ? 0x200u : 0u));
  }
  if (magnitude >= 0x477FF000u) {
    return static_cast<uint16_t>(sign | 0x7C00u);
  }
  if (magnitude < 0x38800000u) {
    const auto code = static_cast<uint32_t>(std::nearbyint(std::bit_cast<float>(magnitude) * 16777216.f));
    return static_cast<uint16_t>(sign | code);
  }
  uint32_t half = (((magnitude >> 23) - 112u) << 10) | ((magnitude >> 13) & 0x3FFu);
  const uint32_t rest = magnitude & 0x1FFFu;
  if (rest > 0x1000u || (rest == 0x1000u && (half & 1u) != 0)) {
    ++half;
  }
  return static_cast<uint16_t>(sign | half);
}

float from_half(uint16_t bits) noexcept {
  const uint32_t sign = uint32_t{bits & 0x8000u} << 16;
  const uint32_t exponent = (bits >> 10) & 0x1Fu;
  const uint32_t mantissa = bits & 0x3FFu;
  if (exponent == 0) {
    const float value = static_cast<float>(mantissa) / 16777216.f;
    return sign != 0 ? -value : value;
  }
  if (exponent == 31) {
    return std::bit_cast<float>(sign | 0x7F800000u | (mantissa << 13));
  }
  return std::bit_cast<float>(sign | ((exponent + 112u) << 23) | (mantissa << 13));
}

Tables build_tables() {
  Tables out;
  out.lut3d.resize(Lut3dTexels * Lut3dChannels);
  for (size_t i = 0; i < Lut3dTexels; ++i) {
    out.lut3d[i * 2] = to_half(unorm(data::IdealDielectricEc[i]));
    out.lut3d[i * 2 + 1] = to_half(unorm(data::OpaqueDielectricEc[i]));
  }
  constexpr size_t Layer = size_t{TableSize} * TableSize * Lut2dChannels;
  out.lut2d.assign(Lut2dTexels * Lut2dChannels, 0);
  for (uint32_t y = 0; y < TableSize; ++y) {
    for (uint32_t x = 0; x < TableSize; ++x) {
      const size_t i = size_t{y} * TableSize + x;
      uint16_t* averages = out.lut2d.data() + LayerAverages * Layer + i * Lut2dChannels;
      averages[0] = to_half(unorm(data::IdealDielectricAverageEc[i]));
      averages[1] = to_half(unorm(data::IdealDielectricReflectionRatio[i]));
      averages[2] = to_half(unorm(data::OpaqueDielectricAverageEc[i]));
      averages[3] = to_half(unorm(data::IdealMetalAverageEc[x]));
      out.lut2d[LayerMetal * Layer + i * Lut2dChannels] = to_half(unorm(data::IdealMetalEc[i]));
      // openpbr_fuzz_lobe.h: #define ltcFlatIndex(row, column) ((row) * OpenPBR_LTCTableSize + (column))
      uint16_t* ltc = out.lut2d.data() + LayerLtc * Layer + i * Lut2dChannels;
      ltc[0] = to_half(data::Ltc[i].a);
      ltc[1] = to_half(data::Ltc[i].b);
      ltc[2] = to_half(data::Ltc[i].c);
    }
  }
  return out;
}

double ior_index(double ior) noexcept {
  constexpr double Half = TableSize / 2;
  if (ior < 1.0) {
    return (Half - 1.0) - (1.0 / ior - 1.0) / (IorMax - 1.0) * (Half - 1.0);
  }
  return Half + (ior - 1.0) / (IorMax - 1.0) * (Half - 1.0);
}

double alpha_index(double alpha) noexcept { return std::sqrt(std::max(alpha, 0.0)) * TableMax; }

double cos_index(double cosTheta) noexcept { return cosTheta * TableMax; }

std::array<double, 2> sample_lut3d(const Tables& t, double x, double y, double z) {
  x = clamp_index(x);
  y = clamp_index(y);
  z = clamp_index(z);
  const auto x0 = static_cast<uint32_t>(x);
  const auto y0 = static_cast<uint32_t>(y);
  const auto z0 = static_cast<uint32_t>(z);
  const uint32_t x1 = std::min(x0 + 1, TableSize - 1);
  const uint32_t y1 = std::min(y0 + 1, TableSize - 1);
  const uint32_t z1 = std::min(z0 + 1, TableSize - 1);
  const double tx = x - x0;
  const double ty = y - y0;
  const double tz = z - z0;
  std::array<double, 2> out{};
  for (size_t c = 0; c < 2; ++c) {
    const auto at = [&](uint32_t i, uint32_t j, uint32_t k) {
      return half_at(t.lut3d, ((size_t{k} * TableSize + j) * TableSize + i) * Lut3dChannels + c);
    };
    const double a = std::lerp(at(x0, y0, z0), at(x1, y0, z0), tx);
    const double b = std::lerp(at(x0, y1, z0), at(x1, y1, z0), tx);
    const double d = std::lerp(at(x0, y0, z1), at(x1, y0, z1), tx);
    const double e = std::lerp(at(x0, y1, z1), at(x1, y1, z1), tx);
    out[c] = std::lerp(std::lerp(a, b, ty), std::lerp(d, e, ty), tz);
  }
  return out;
}

std::array<double, 4> sample_lut2d(const Tables& t, double x, double y, uint32_t layer) {
  x = clamp_index(x);
  y = clamp_index(y);
  const auto x0 = static_cast<uint32_t>(x);
  const auto y0 = static_cast<uint32_t>(y);
  const uint32_t x1 = std::min(x0 + 1, TableSize - 1);
  const uint32_t y1 = std::min(y0 + 1, TableSize - 1);
  const double tx = x - x0;
  const double ty = y - y0;
  std::array<double, 4> out{};
  for (size_t c = 0; c < 4; ++c) {
    const auto at = [&](uint32_t i, uint32_t j) {
      return half_at(t.lut2d, ((size_t{layer} * TableSize + j) * TableSize + i) * Lut2dChannels + c);
    };
    out[c] = std::lerp(std::lerp(at(x0, y0), at(x1, y0), tx), std::lerp(at(x0, y1), at(x1, y1), tx), ty);
  }
  return out;
}

void initialize() {
  if (g_resident.load(std::memory_order_acquire)) {
    return;
  }
  const wgpu::Device& device = webgpu::g_device;
  if (!device) {
    Log.warn("Render lighting: no device for the OpenPBR tables; PBR lighting stays off");
    return;
  }
  const Tables& tables = shared_tables();
  g_classesHandle = create_persistent(Classes, PersistentKind::Buffer);
  g_lut3dHandle = create_persistent(Lut3d, PersistentKind::Texture);
  g_lut2dHandle = create_persistent(Lut2d, PersistentKind::Texture);

  const wgpu::TextureDescriptor volume{
      .label = "light.lut3d",
      .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst,
      .dimension = wgpu::TextureDimension::e3D,
      .size = {TableSize, TableSize, TableSize},
      .format = Lut3dFormat,
  };
  g_lut3d = device.CreateTexture(&volume);
  const wgpu::TextureDescriptor layers{
      .label = "light.lut2d",
      .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst,
      .size = {TableSize, TableSize, Lut2dLayers},
      .format = Lut2dFormat,
  };
  g_lut2d = device.CreateTexture(&layers);
  const wgpu::BufferDescriptor buffer{
      .label = "light.classes",
      .usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
      .size = ClassBytes,
  };
  g_classBuffer = device.CreateBuffer(&buffer);
  if (!g_lut3d || !g_lut2d || !g_classBuffer) {
    Log.warn("Render lighting: the OpenPBR tables could not be created; PBR lighting stays off");
    shutdown();
    return;
  }
  const wgpu::Queue& queue = webgpu::g_queue;
  const wgpu::TexelCopyTextureInfo volumeCopy{.texture = g_lut3d};
  const wgpu::TexelCopyBufferLayout volumeLayout{.bytesPerRow = TableSize * Lut3dChannels * 2,
                                                 .rowsPerImage = TableSize};
  const wgpu::Extent3D volumeSize{TableSize, TableSize, TableSize};
  queue.WriteTexture(&volumeCopy, tables.lut3d.data(), tables.lut3d.size() * 2, &volumeLayout, &volumeSize);
  const wgpu::TexelCopyTextureInfo layersCopy{.texture = g_lut2d};
  const wgpu::TexelCopyBufferLayout layersLayout{.bytesPerRow = TableSize * Lut2dChannels * 2,
                                                 .rowsPerImage = TableSize};
  const wgpu::Extent3D layersSize{TableSize, TableSize, Lut2dLayers};
  queue.WriteTexture(&layersCopy, tables.lut2d.data(), tables.lut2d.size() * 2, &layersLayout, &layersSize);
  const uint64_t epoch = material::classes::epoch();
  const std::shared_ptr<const material::classes::GpuRows> rows = material::classes::gpu_rows();
  queue.WriteBuffer(g_classBuffer, 0, rows->data(), ClassBytes);
  g_uploadedEpoch.store(epoch, std::memory_order_relaxed);

  const wgpu::TextureViewDescriptor volumeView{.label = "light.lut3d", .dimension = wgpu::TextureViewDimension::e3D};
  const wgpu::TextureViewDescriptor layersView{.label = "light.lut2d",
                                               .dimension = wgpu::TextureViewDimension::e2DArray};
  set_persistent_texture(g_lut3dHandle, g_lut3d, g_lut3d.CreateView(&volumeView));
  set_persistent_texture(g_lut2dHandle, g_lut2d, g_lut2d.CreateView(&layersView));
  set_persistent_buffer(g_classesHandle, g_classBuffer, 0, ClassBytes);
  // Imports get no fallback, so the tables exist before any graph can import them
  g_resident.store(true, std::memory_order_release);
}

void shutdown() noexcept {
  g_resident.store(false, std::memory_order_release);
  for (PersistentHandle* handle : {&g_classesHandle, &g_lut3dHandle, &g_lut2dHandle}) {
    if (handle->id != 0) {
      destroy_persistent(*handle);
      *handle = {};
    }
  }
  g_classBuffer = {};
  g_lut3d = {};
  g_lut2d = {};
  g_uploadedEpoch.store(0, std::memory_order_relaxed);
}

bool resident() noexcept { return g_resident.load(std::memory_order_acquire); }

PersistentHandle classes_handle() noexcept { return g_classesHandle; }

PersistentHandle lut3d_handle() noexcept { return g_lut3dHandle; }

PersistentHandle lut2d_handle() noexcept { return g_lut2dHandle; }

bool refresh_classes(const wgpu::CommandEncoder& encoder) {
  const uint64_t epoch = material::classes::epoch();
  if (!resident() || !g_classBuffer || epoch == g_uploadedEpoch.load(std::memory_order_relaxed)) {
    return false;
  }
  const std::shared_ptr<const material::classes::GpuRows> rows = material::classes::gpu_rows();
  encoder.WriteBuffer(g_classBuffer, 0, reinterpret_cast<const uint8_t*>(rows->data()), ClassBytes);
  g_uploadedEpoch.store(epoch, std::memory_order_relaxed);
  return true;
}

Prepared prepare(const Tables& t, const material::classes::GpuRow& row, const Surface& s, const Vec3& n,
                 const Vec3& v) {
  const auto& r = row.v;
  Prepared p;
  p.n = n;
  p.v = v;
  p.muV = std::max(dot(n, v), MinMuV);
  const double roughness = std::clamp(s.roughness, MinRoughness, 1.0);
  p.alpha = roughness * roughness;
  const Vec3 specColour = row3(row, 0);
  const double specWeight = r[3];
  const double coatWeight = r[7];
  const double fuzzWeight = r[11];
  const Vec3 sssColour = row3(row, 12);
  const double sssWeight = r[15];
  const double anisotropy = r[26];
  const uint32_t flags = std::bit_cast<uint32_t>(r[30]);
  const double metal = std::clamp(s.metalness, 0.0, 1.0);
  const double dielectric = 1.0 - metal;
  const Vec3 base = s.baseColor * static_cast<double>(r[27]);
  p.eta = apply_specular_weight(r[20], specWeight);
  const double ai = alpha_index(p.alpha);
  const double ii = ior_index(p.eta);
  p.ecV = extrapolate(sample_lut3d(t, cos_index(p.muV), ai, ii)[1], p.eta);
  const std::array<double, 4> averages = sample_lut2d(t, ai, ii, LayerAverages);
  p.invEcAvg = 1.0 / std::max(extrapolate(averages[2], p.eta), MinAverage);
  p.specTint = specColour * dielectric;
  p.metalWeight = metal * specWeight;
  p.metalF0 = base;
  p.metalTint = specColour;
  const Vec3 favg = f82_average(base, specColour);
  p.metalMs = favg * favg * p.metalWeight;
  if (p.metalWeight > 0.0) {
    p.ecmV = sample_lut2d(t, cos_index(p.muV), ai, LayerMetal)[0];
    p.ecmAvg = std::max(averages[3], MinAverage);
  }
  p.thin = (flags & material::classes::RowThin) != 0;
  if (p.thin) {
    p.diffuse = (base * (1.0 - sssWeight) + sssColour * (sssWeight * (1.0 - anisotropy) * 0.5)) * dielectric;
    p.transmit = sssColour * (sssWeight * (1.0 + anisotropy) * 0.5 * dielectric);
    p.wrap = std::clamp(static_cast<double>(r[31]), 0.0, 1.0);
  } else {
    p.diffuse = (base * (1.0 - sssWeight) + sssColour * sssWeight) * dielectric;
  }
  p.sigma = r[24];
  Vec3 ambientDiffuse = (eon_albedo(p.diffuse, p.sigma, p.muV) + p.transmit) * p.ecV;
  Vec3 ambientSpecular = p.specTint * (1.0 - p.ecV) +
                         f82(p.metalF0, p.metalTint, p.muV) * (p.metalWeight * (1.0 - p.ecmV)) +
                         (p.alpha >= MinAlphaWithEnergyLoss ? p.metalMs * p.ecmV : splat(0.0));
  p.coatWeight = coatWeight;
  if (coatWeight > 0.0) {
    const double coatRoughness = std::clamp(static_cast<double>(r[22]), MinRoughness, 1.0);
    p.coatAlpha = coatRoughness * coatRoughness;
    p.coatEta = std::max(static_cast<double>(r[21]), 1.0);
    p.coatTint = row3(row, 4);
    const double coatReflected = 1.0 - opaque_ec(t, p.muV, p.coatAlpha, p.coatEta);
    const double ks = average_fresnel(p.coatEta);
    const double kr = 1.0 - (1.0 - ks) / (p.coatEta * p.coatEta);
    const double specOpaque = average_fresnel(p.eta);
    const double effective = std::lerp(1.0, roughness, dielectric * specOpaque + metal);
    const double k = std::lerp(ks, kr, effective);
    const Vec3 eb = favg * p.metalWeight + mix(base, splat(1.0), specOpaque) * dielectric;
    Vec3 dark{};
    for (size_t i = 0; i < 3; ++i) {
      dark[i] = std::lerp(1.0, (1.0 - k) / (1.0 - eb[i] * k), coatWeight * r[23]);
    }
    p.coatBase = coat_passage(p, p.muV) * dark * (1.0 - coatWeight * coatReflected);
    const Vec3 scale = p.coatBase * coat_passage(p, p.muV);
    ambientDiffuse = ambientDiffuse * scale;
    ambientSpecular = ambientSpecular * scale + splat(coatWeight * coatReflected);
  }
  p.fuzzWeight = fuzzWeight;
  if (fuzzWeight > 0.0) {
    const std::array<double, 4> ltc = sample_lut2d(t, cos_index(p.muV), r[25] * TableMax, LayerLtc);
    p.fuzzLtc = {ltc[0], ltc[1], ltc[2]};
    p.fuzzColour = row3(row, 8);
    p.fuzzBase = 1.0 - fuzzWeight * ltc[2];
    Vec3 tangent = v - n * dot(n, v);
    if (dot(tangent, tangent) < 1e-12) {
      tangent = std::abs(n[0]) < 0.9 ? cross(n, Vec3{1.0, 0.0, 0.0}) : cross(n, Vec3{0.0, 1.0, 0.0});
    }
    p.t = normalize(tangent);
    p.b = cross(n, p.t);
    ambientDiffuse = ambientDiffuse * p.fuzzBase + p.fuzzColour * (fuzzWeight * ltc[2]);
    ambientSpecular = ambientSpecular * p.fuzzBase;
  }
  p.ambientDiffuse = ambientDiffuse;
  p.ambientSpecular = ambientSpecular;
  return p;
}

Vec3 eval_light(const Tables& t, const Prepared& p, const Vec3& l, bool specular) {
  const double nl = dot(p.n, l);
  const double ai = alpha_index(p.alpha);
  Vec3 out{};
  const double wrapped = std::max(nl + p.wrap, 0.0) / ((1.0 + p.wrap) * (1.0 + p.wrap));
  if (wrapped > 0.0) {
    const double muL = std::clamp(nl, 0.0, 1.0);
    const double ecL = extrapolate(sample_lut3d(t, cos_index(muL), ai, ior_index(p.eta))[1], p.eta);
    const double couple = p.wrap > 0.0 ? 1.0 : ecL * p.invEcAvg;
    out = eon_pi(p.diffuse, p.sigma, p.muV, muL, dot(p.v, l)) * (p.ecV * couple * wrapped);
  }
  if (p.thin && nl < 0.0) {
    const double muT = std::min(-nl, 1.0);
    const double ecT = extrapolate(sample_lut3d(t, cos_index(muT), ai, ior_index(p.eta))[1], p.eta);
    out = out + p.transmit * (p.ecV * ecT * p.invEcAvg * muT);
  }
  const Vec3 h = normalize(p.v + l);
  const double nh = std::max(dot(p.n, h), 0.0);
  const double vh = std::max(dot(p.v, h), 0.0);
  if (specular && nl > 0.0) {
    const double k = ggx_d(nh, p.alpha) * smith_g1(p.muV, p.alpha) * smith_g1(nl, p.alpha) * Pi / (4.0 * p.muV);
    out = out + (p.specTint * fresnel(p.eta, vh) + f82(p.metalF0, p.metalTint, vh) * p.metalWeight) * k;
    if (p.metalWeight > 0.0 && p.alpha >= MinAlphaWithEnergyLoss) {
      const double ecmL = sample_lut2d(t, cos_index(nl), ai, LayerMetal)[0];
      out = out + p.metalMs * (std::min(p.ecmV * ecmL / p.ecmAvg, 1.0 / nl) * nl);
    }
  }
  if (p.coatWeight > 0.0) {
    out = out * p.coatBase * coat_passage(p, nl);
    if (specular && nl > 0.0) {
      const double k =
          ggx_d(nh, p.coatAlpha) * smith_g1(p.muV, p.coatAlpha) * smith_g1(nl, p.coatAlpha) * Pi / (4.0 * p.muV);
      out = out + splat(p.coatWeight * fresnel(p.coatEta, vh) * k);
    }
  }
  if (p.fuzzWeight > 0.0) {
    out = out * p.fuzzBase;
    if (specular && nl > 0.0) {
      const double x = dot(l, p.t);
      const double y = dot(l, p.b);
      const double a = p.fuzzLtc[0];
      const double xo = a * x + p.fuzzLtc[1] * nl;
      const double yo = a * y;
      const double length2 = xo * xo + yo * yo + nl * nl;
      if (length2 > 0.0 && a > 0.0) {
        const double q = a / length2;
        out = out + p.fuzzColour * (p.fuzzWeight * p.fuzzLtc[2] * nl * q * q);
      }
    }
  }
  return out;
}

Vec3 eval_ambient(const Prepared& p, const Vec3& irradiance, const Vec3& radiance) {
  return p.ambientDiffuse * irradiance * (1.0 / Pi) + p.ambientSpecular * radiance;
}

Vec3 directional_albedo(const Prepared& p) { return p.ambientDiffuse + p.ambientSpecular; }

namespace testing {
const Tables& tables() { return shared_tables(); }

void drop_resident() noexcept { g_resident.store(false, std::memory_order_release); }

uint64_t uploaded_epoch() noexcept { return g_uploadedEpoch.load(std::memory_order_relaxed); }
} // namespace testing
} // namespace aurora::render::detail::lighting::openpbr
