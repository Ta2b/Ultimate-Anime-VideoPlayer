#include "gpu/coreml/vt_pixels.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <vector>

#include "kernels/ref/half.h"

namespace ajp::gpu::coreml_rt::vt {
namespace {

constexpr std::uint32_t code(char a, char b, char c, char d) {
  return (static_cast<std::uint32_t>(static_cast<unsigned char>(a)) << 24U) |
         (static_cast<std::uint32_t>(static_cast<unsigned char>(b)) << 16U) |
         (static_cast<std::uint32_t>(static_cast<unsigned char>(c)) << 8U) |
         static_cast<std::uint32_t>(static_cast<unsigned char>(d));
}

constexpr std::array kOrder{Pixels::RgbaHalf,      Pixels::Bgra8,       Pixels::Yuv420Full10,
                            Pixels::Yuv420Video10, Pixels::Yuv420Full8, Pixels::Yuv420Video8};

// BT.709 (kernels/spec/rife.h : kRifeMatrix).
constexpr float kKr = 0.2126F;
constexpr float kKb = 0.0722F;
constexpr float kKg = 1.0F - kKr - kKb;
constexpr float kCbScale = 2.0F * (1.0F - kKb);  // B − Y = Cb × 1,8556
constexpr float kCrScale = 2.0F * (1.0F - kKr);  // R − Y = Cr × 1,5748
constexpr float kGCb = kKb * kCbScale / kKg;
constexpr float kGCr = kKr * kCrScale / kKg;
constexpr std::uint16_t kHalfOne = 0x3C00;

// Les niveaux d'un format YCbCr : luma = offset + Y × scale, chroma = milieu + C × scale.
struct Levels {
  float y_offset;
  float y_scale;
  float c_mid;
  float c_scale;
  float max;
  unsigned shift;  // 10 bits : la valeur dans les bits de poids fort du mot de 16 bits
  bool wide;       // mots de 16 bits
};

Levels levels(Pixels format) {
  switch (format) {
    case Pixels::Yuv420Full10:
      return {0.0F, 1023.0F, 512.0F, 1023.0F, 1023.0F, 6, true};
    case Pixels::Yuv420Video10:
      return {64.0F, 876.0F, 512.0F, 896.0F, 1023.0F, 6, true};
    case Pixels::Yuv420Full8:
      return {0.0F, 255.0F, 128.0F, 255.0F, 255.0F, 0, false};
    case Pixels::Yuv420Video8:
    default:
      return {16.0F, 219.0F, 128.0F, 224.0F, 255.0F, 0, false};
  }
}

float from_half(std::uint16_t bits) {
#if defined(__FLT16_MANT_DIG__)
  return static_cast<float>(std::bit_cast<_Float16>(bits));
#else
  return kernels::ref::half_to_float(bits);
#endif
}

std::uint16_t to_half(float value) {
#if defined(__FLT16_MANT_DIG__)
  return std::bit_cast<std::uint16_t>(static_cast<_Float16>(value));
#else
  return kernels::ref::float_to_half(value);
#endif
}

// Un plan de la chaîne : width × height éléments F16 ou F32.
struct Plane {
  const std::byte* data;
  TensorType type;
  float at(std::size_t i) const {
    if (type == TensorType::F32) {
      float v = 0.0F;
      std::memcpy(&v, data + (i * 4), 4);
      return v;
    }
    std::uint16_t h = 0;
    std::memcpy(&h, data + (i * 2), 2);
    return from_half(h);
  }
  std::uint16_t half_at(std::size_t i) const {
    if (type == TensorType::F32) return to_half(at(i));
    std::uint16_t h = 0;
    std::memcpy(&h, data + (i * 2), 2);
    return h;
  }
};

struct OutPlane {
  std::byte* data;
  TensorType type;
  void set(std::size_t i, float v) const {
    if (type == TensorType::F32) {
      std::memcpy(data + (i * 4), &v, 4);
      return;
    }
    const std::uint16_t h = to_half(v);
    std::memcpy(data + (i * 2), &h, 2);
  }
  void set_half(std::size_t i, std::uint16_t h) const {
    if (type == TensorType::F32) {
      set(i, from_half(h));
      return;
    }
    std::memcpy(data + (i * 2), &h, 2);
  }
};

std::size_t bytes_of(TensorType type) {
  return type == TensorType::F32 ? 4U : 2U;
}

// Arrondi au plus proche : std::round tient en une instruction (frinta sur arm64), sans errno.
std::uint16_t quantize(float value, float max) {
  return static_cast<std::uint16_t>(std::round(std::clamp(value, 0.0F, max)));
}

void put(std::byte* row, int x, bool wide, unsigned shift, std::uint16_t value) {
  if (wide) {
    const auto word = static_cast<std::uint16_t>(value << shift);
    std::memcpy(row + (static_cast<std::size_t>(x) * 2), &word, 2);
  } else {
    row[x] = static_cast<std::byte>(value);
  }
}

float get(const std::byte* row, int x, bool wide, unsigned shift) {
  if (wide) {
    std::uint16_t word = 0;
    std::memcpy(&word, row + (static_cast<std::size_t>(x) * 2), 2);
    return static_cast<float>(word >> shift);
  }
  return static_cast<float>(std::to_integer<std::uint8_t>(row[x]));
}

void pack_yuv(Pixels format, const std::array<Plane, 3>& rgb, int width, const Planes& out, Rows rows) {
  const Levels l = levels(format);
  for (int y = rows.first; y < rows.last; y += 2) {
    const int y1 = std::min(y + 1, rows.last - 1);
    std::byte* luma[2] = {out.data[0] + (static_cast<std::size_t>(y) * out.stride[0]),
                          out.data[0] + (static_cast<std::size_t>(y1) * out.stride[0])};
    std::byte* chroma = out.data[1] + (static_cast<std::size_t>(y / 2) * out.stride[1]);
    for (int x = 0; x < width; x += 2) {
      float sum[3] = {0.0F, 0.0F, 0.0F};
      int count = 0;
      for (int dy = 0; dy < 2; ++dy) {
        const int row = dy == 0 ? y : y1;
        for (int dx = 0; dx < 2 && x + dx < width; ++dx) {
          const std::size_t i = (static_cast<std::size_t>(row) * static_cast<std::size_t>(width)) +
                                static_cast<std::size_t>(x + dx);
          const float r = rgb[0].at(i);
          const float g = rgb[1].at(i);
          const float b = rgb[2].at(i);
          const float luma_value = (kKr * r) + (kKg * g) + (kKb * b);
          if (dy == 0 || y1 != y)
            put(luma[dy], x + dx, l.wide, l.shift, quantize(l.y_offset + (luma_value * l.y_scale), l.max));
          sum[0] += r;
          sum[1] += g;
          sum[2] += b;
          ++count;
        }
      }
      const float r = sum[0] / static_cast<float>(count);
      const float g = sum[1] / static_cast<float>(count);
      const float b = sum[2] / static_cast<float>(count);
      const float luma_value = (kKr * r) + (kKg * g) + (kKb * b);
      const float cb = (b - luma_value) / kCbScale;
      const float cr = (r - luma_value) / kCrScale;
      put(chroma, x, l.wide, l.shift, quantize(l.c_mid + (cb * l.c_scale), l.max));
      put(chroma, x + 1, l.wide, l.shift, quantize(l.c_mid + (cr * l.c_scale), l.max));
    }
  }
}

// La chroma bilinéaire au centre de chaque bloc 2 × 2 : la luma x se trouve à x/2 − 1/4 en
// chroma, soit 3/4 de l'échantillon k = x/2 et 1/4 de son voisin du côté de x ; de même en y.
// Chaque rangée de chroma est d'abord interpolée en x (line), puis les deux rangées voisines
// mêlées.
void unpack_yuv(Pixels format, const Planes& in, const std::array<OutPlane, 3>& rgb, int width, int height,
                Rows rows) {
  const Levels l = levels(format);
  const int chroma_width = (width + 1) / 2;
  const int chroma_height = (height + 1) / 2;
  const auto w = static_cast<std::size_t>(width);
  const float y_unit = 1.0F / l.y_scale;
  const float c_unit = 1.0F / l.c_scale;
  std::vector<float> near_line(2 * w);
  std::vector<float> far_line(2 * w);
  const auto line = [&](int cy, std::vector<float>& out) {
    const std::byte* row =
        in.data[1] + (static_cast<std::size_t>(std::clamp(cy, 0, chroma_height - 1)) * in.stride[1]);
    const auto sample = [&](int cx, int c) {
      return (get(row, (std::clamp(cx, 0, chroma_width - 1) * 2) + c, l.wide, l.shift) - l.c_mid) * c_unit;
    };
    for (int x = 0; x < width; ++x) {
      const int cx = x / 2;
      const int nx = (x % 2 == 0) ? cx - 1 : cx + 1;
      for (int c = 0; c < 2; ++c) {
        out[(static_cast<std::size_t>(x) * 2) + static_cast<std::size_t>(c)] =
            (0.75F * sample(cx, c)) + (0.25F * sample(nx, c));
      }
    }
  };
  for (int y = rows.first; y < rows.last; ++y) {
    const std::byte* luma = in.data[0] + (static_cast<std::size_t>(y) * in.stride[0]);
    const int cy = y / 2;
    line(cy, near_line);
    line(y % 2 == 0 ? cy - 1 : cy + 1, far_line);
    const std::size_t base = static_cast<std::size_t>(y) * w;
    for (int x = 0; x < width; ++x) {
      const auto px = static_cast<std::size_t>(x);
      const float cb = (0.75F * near_line[px * 2]) + (0.25F * far_line[px * 2]);
      const float cr = (0.75F * near_line[(px * 2) + 1]) + (0.25F * far_line[(px * 2) + 1]);
      const float luma_value = (get(luma, x, l.wide, l.shift) - l.y_offset) * y_unit;
      rgb[0].set(base + px, luma_value + (kCrScale * cr));
      rgb[1].set(base + px, luma_value - (kGCb * cb) - (kGCr * cr));
      rgb[2].set(base + px, luma_value + (kCbScale * cb));
    }
  }
}

}  // namespace

std::uint32_t four_cc(Pixels format) {
  switch (format) {
    case Pixels::RgbaHalf:
      return code('R', 'G', 'h', 'A');
    case Pixels::Bgra8:
      return code('B', 'G', 'R', 'A');
    case Pixels::Yuv420Full10:
      return code('x', 'f', '2', '0');
    case Pixels::Yuv420Video10:
      return code('x', '4', '2', '0');
    case Pixels::Yuv420Full8:
      return code('4', '2', '0', 'f');
    case Pixels::Yuv420Video8:
      break;
  }
  return code('4', '2', '0', 'v');
}

std::string_view name(Pixels format) {
  switch (format) {
    case Pixels::RgbaHalf:
      return "RGhA";
    case Pixels::Bgra8:
      return "BGRA";
    case Pixels::Yuv420Full10:
      return "xf20";
    case Pixels::Yuv420Video10:
      return "x420";
    case Pixels::Yuv420Full8:
      return "420f";
    case Pixels::Yuv420Video8:
      break;
  }
  return "420v";
}

std::optional<Pixels> preferred(std::span<const std::uint32_t> offered) {
  for (const Pixels format : kOrder) {
    if (std::ranges::find(offered, four_cc(format)) != offered.end()) return format;
  }
  return std::nullopt;
}

bool is_yuv(Pixels format) {
  return format != Pixels::RgbaHalf && format != Pixels::Bgra8;
}

void pack(Pixels format, const std::byte* rgb, TensorType type, int width, int height, const Planes& out,
          Rows rows) {
  const std::size_t plane =
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * bytes_of(type);
  const std::array<Plane, 3> in{Plane{rgb, type}, Plane{rgb + plane, type}, Plane{rgb + (2 * plane), type}};
  if (is_yuv(format)) {
    pack_yuv(format, in, width, out, rows);
    return;
  }
  for (int y = rows.first; y < rows.last; ++y) {
    std::byte* row = out.data[0] + (static_cast<std::size_t>(y) * out.stride[0]);
    const std::size_t base = static_cast<std::size_t>(y) * static_cast<std::size_t>(width);
    for (int x = 0; x < width; ++x) {
      const std::size_t i = base + static_cast<std::size_t>(x);
      const auto px = static_cast<std::size_t>(x);
      if (format == Pixels::RgbaHalf) {
        const std::array<std::uint16_t, 4> rgba{in[0].half_at(i), in[1].half_at(i), in[2].half_at(i),
                                                kHalfOne};
        std::memcpy(row + (px * 8), rgba.data(), 8);
      } else {
        row[(px * 4) + 0] = static_cast<std::byte>(quantize(in[2].at(i) * 255.0F, 255.0F));
        row[(px * 4) + 1] = static_cast<std::byte>(quantize(in[1].at(i) * 255.0F, 255.0F));
        row[(px * 4) + 2] = static_cast<std::byte>(quantize(in[0].at(i) * 255.0F, 255.0F));
        row[(px * 4) + 3] = std::byte{0xFF};
      }
    }
  }
}

void unpack(Pixels format, const Planes& in, std::byte* rgb, TensorType type, int width, int height,
            Rows rows) {
  const std::size_t plane =
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * bytes_of(type);
  const std::array<OutPlane, 3> out{OutPlane{rgb, type}, OutPlane{rgb + plane, type},
                                    OutPlane{rgb + (2 * plane), type}};
  if (is_yuv(format)) {
    unpack_yuv(format, in, out, width, height, rows);
    return;
  }
  for (int y = rows.first; y < rows.last; ++y) {
    const std::byte* row = in.data[0] + (static_cast<std::size_t>(y) * in.stride[0]);
    const std::size_t base = static_cast<std::size_t>(y) * static_cast<std::size_t>(width);
    for (int x = 0; x < width; ++x) {
      const std::size_t i = base + static_cast<std::size_t>(x);
      const auto px = static_cast<std::size_t>(x);
      if (format == Pixels::RgbaHalf) {
        std::array<std::uint16_t, 4> rgba{};
        std::memcpy(rgba.data(), row + (px * 8), 8);
        for (std::size_t c = 0; c < 3; ++c) out[c].set_half(i, rgba[c]);
      } else {
        for (std::size_t c = 0; c < 3; ++c) {
          const auto value = std::to_integer<std::uint8_t>(row[(px * 4) + (2 - c)]);
          out[c].set(i, static_cast<float>(value) / 255.0F);
        }
      }
    }
  }
}

}  // namespace ajp::gpu::coreml_rt::vt
