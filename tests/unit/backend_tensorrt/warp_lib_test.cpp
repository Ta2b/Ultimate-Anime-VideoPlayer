// Bibliothèque de warp CUDA (warp_lib.h, lot E) : un seul test, le warp générique contre une référence CPU en
// double, sur les cas que ses clients prendront. Un flot tensoriel grossier (grille 4, rangé par point, un
// canal de départ décalé) avec confiance, image de repli, sortie de la confiance, réduction 4 et sortie par
// pixel ; puis un flot affine (caméra) en Catmull-Rom dans une fenêtre, sortie en plans. La colle de RIFE,
// spécialisée sur les mêmes briques, a son test (rife_fused_test.cpp).
#include "gpu/cuda/backend/warp_lib.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "kernels/ref/half.h"

namespace kirari::test {
namespace {

using gpu::cuda::WarpPlanesParams;

struct Buffer {
  explicit Buffer(const std::vector<double>& values) : count(values.size()) {
    std::vector<std::uint16_t> half(values.size());
    for (std::size_t i = 0; i < values.size(); ++i)
      half[i] = kernels::ref::float_to_half(static_cast<float>(values[i]));
    REQUIRE(cudaMalloc(&ptr, std::max<std::size_t>(count, 1) * 2) == cudaSuccess);
    REQUIRE(cudaMemcpy(ptr, half.data(), count * 2, cudaMemcpyHostToDevice) == cudaSuccess);
  }
  ~Buffer() { cudaFree(ptr); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  std::vector<double> read() const {
    std::vector<std::uint16_t> half(count);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    REQUIRE(cudaMemcpy(half.data(), ptr, count * 2, cudaMemcpyDeviceToHost) == cudaSuccess);
    std::vector<double> out(count);
    for (std::size_t i = 0; i < count; ++i) out[i] = kernels::ref::half_to_float(half[i]);
    return out;
  }
  void* ptr = nullptr;
  std::size_t count = 0;
};

// Valeurs lisses (un dessin) plus du bruit, arrondies en FP16 comme le GPU les lira.
std::vector<double> field(std::size_t n, double amplitude, std::mt19937& rng) {
  std::uniform_real_distribution<double> noise(-0.05, 0.05);
  std::vector<double> v(n);
  for (std::size_t i = 0; i < n; ++i) {
    const double x = amplitude * (0.5 + 0.4 * std::sin(0.37 * static_cast<double>(i % 97)) + noise(rng));
    v[i] = kernels::ref::half_to_float(kernels::ref::float_to_half(static_cast<float>(x)));
  }
  return v;
}

struct Grid {
  const std::vector<double>* data;
  int h, w, pitch, channel;
  double at(int c, int y, int x) const {
    const std::size_t point =
        static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x);
    return pitch == 0
               ? (*data)[static_cast<std::size_t>(channel + c) * static_cast<std::size_t>(h * w) + point]
               : (*data)[point * static_cast<std::size_t>(pitch) + static_cast<std::size_t>(channel + c)];
  }
  // Agrandissement bilinéaire demi-pixel d'un facteur s au point pleine résolution (px, py).
  double enlarged(int c, int px, int py, int s) const {
    const double sx = std::clamp((px + 0.5) / s - 0.5, 0.0, static_cast<double>(w - 1));
    const double sy = std::clamp((py + 0.5) / s - 0.5, 0.0, static_cast<double>(h - 1));
    const int x0 = static_cast<int>(sx);
    const int y0 = static_cast<int>(sy);
    const int x1 = std::min(x0 + 1, w - 1);
    const int y1 = std::min(y0 + 1, h - 1);
    const double fx = sx - x0;
    const double fy = sy - y0;
    return (at(c, y0, x0) * (1 - fx) + at(c, y0, x1) * fx) * (1 - fy) +
           (at(c, y1, x0) * (1 - fx) + at(c, y1, x1) * fx) * fy;
  }
};

double cubic_weight(double t, int i) {
  const double t2 = t * t;
  const double t3 = t2 * t;
  const double w[4] = {-0.5 * t3 + t2 - 0.5 * t, 1.5 * t3 - 2.5 * t2 + 1.0, -1.5 * t3 + 2.0 * t2 + 0.5 * t,
                       0.5 * t3 - 0.5 * t2};
  return w[i];
}

// La référence de launch_warp_planes, en double.
std::vector<double> reference(const WarpPlanesParams& p, const std::vector<double>& image,
                              const std::vector<double>* flow, const std::vector<double>* confidence,
                              const std::vector<double>* fallback, std::vector<double>& conf_out) {
  const int d = p.reduce;
  const int oh = p.height / d;
  const int ow = p.width / d;
  const int taps = d > 1 ? 2 : 1;
  const int base = d > 1 ? d / 2 - 1 : 0;
  const int left = p.window_width > 0 ? p.window_left : 0;
  const int top = p.window_width > 0 ? p.window_top : 0;
  const int right = p.window_width > 0 ? p.window_left + p.window_width : p.width;
  const int bottom = p.window_width > 0 ? p.window_top + p.window_height : p.height;
  const std::size_t plane = static_cast<std::size_t>(p.height) * static_cast<std::size_t>(p.width);
  const auto pixel = [&](int c, int y, int x) {
    return image[static_cast<std::size_t>(c) * plane + static_cast<std::size_t>(y * p.width + x)];
  };
  // Par pixel, out_pitch canaux par sortie (ceux au-delà de channels restent à zéro), sinon channels plans.
  const int per_pixel = p.out_pitch == 0 ? p.channels : p.out_pitch;
  std::vector<double> out(static_cast<std::size_t>(per_pixel * oh * ow));
  conf_out.assign(static_cast<std::size_t>(oh * ow), 0.0);
  for (int oy = 0; oy < oh; ++oy) {
    for (int ox = 0; ox < ow; ++ox) {
      for (int t = 0; t < taps * taps; ++t) {
        const int px = d * ox + base + t % taps;
        const int py = d * oy + base + t / taps;
        double dx = 0.0;
        double dy = 0.0;
        double conf = 1.0;
        if (flow != nullptr) {
          const Grid g{flow, p.flow.height, p.flow.width, p.flow.pitch, p.flow.channel};
          dx = g.enlarged(0, px, py, p.flow.scale) * p.flow.gain;
          dy = g.enlarged(1, px, py, p.flow.scale) * p.flow.gain;
          if (confidence != nullptr)
            conf = std::clamp(
                Grid{confidence, p.flow.height, p.flow.width, 0, 0}.enlarged(0, px, py, p.flow.scale), 0.0,
                1.0);
        } else {
          const auto a = [&p](int i) { return static_cast<double>(p.flow.affine[i]); };
          dx = a(0) + a(1) * px + a(2) * py;
          dy = a(3) + a(4) * px + a(5) * py;
        }
        const double ix = std::clamp(px + dx, static_cast<double>(left), static_cast<double>(right - 1));
        const double iy = std::clamp(py + dy, static_cast<double>(top), static_cast<double>(bottom - 1));
        const int x0 = static_cast<int>(ix);
        const int y0 = static_cast<int>(iy);
        const double fx = ix - x0;
        const double fy = iy - y0;
        // Catmull-Rom borné aux 2 × 2 points centraux n'est pas continu quand la position franchit un entier
        // (les points centraux changent) : là, l'arrondi FP32 du GPU peut tomber de l'autre côté. Ces points
        // ne se comparent pas (NaN, sautés par worst_error).
        const auto near_integer = [](double f) { return f < 1e-3 || f > 1.0 - 1e-3; };
        const bool undecided = p.cubic != 0 && (near_integer(fx) || near_integer(fy));
        for (int c = 0; c < p.channels; ++c) {
          double v = 0.0;
          if (p.cubic != 0) {
            double lo = 1e30;
            double hi = -1e30;
            for (int j = 0; j < 4; ++j)
              for (int i = 0; i < 4; ++i) {
                const double s = pixel(c, std::clamp(y0 - 1 + j, top, bottom - 1),
                                       std::clamp(x0 - 1 + i, left, right - 1));
                v += s * cubic_weight(fx, i) * cubic_weight(fy, j);
                if ((i == 1 || i == 2) && (j == 1 || j == 2)) {
                  lo = std::min(lo, s);
                  hi = std::max(hi, s);
                }
              }
            v = std::clamp(v, lo, hi);
          } else {
            const int x1 = std::min(x0 + 1, right - 1);
            const int y1 = std::min(y0 + 1, bottom - 1);
            v = (pixel(c, y0, x0) * (1 - fx) + pixel(c, y0, x1) * fx) * (1 - fy) +
                (pixel(c, y1, x0) * (1 - fx) + pixel(c, y1, x1) * fx) * fy;
          }
          if (fallback != nullptr)
            v = conf * v + (1 - conf) * (*fallback)[static_cast<std::size_t>(c) * plane +
                                                    static_cast<std::size_t>(py * p.width + px)];
          const std::size_t o = static_cast<std::size_t>(oy * ow + ox);
          const std::size_t at =
              p.out_pitch == 0 ? static_cast<std::size_t>(c * oh * ow) + o
                               : o * static_cast<std::size_t>(p.out_pitch) + static_cast<std::size_t>(c);
          out[at] += undecided ? std::nan("") : v / (taps * taps);
        }
        conf_out[static_cast<std::size_t>(oy * ow + ox)] += conf / (taps * taps);
      }
    }
  }
  return out;
}

double worst_error(const std::vector<double>& got, const std::vector<double>& want) {
  REQUIRE(got.size() == want.size());
  double worst = 0.0;
  for (std::size_t i = 0; i < got.size(); ++i)
    if (!std::isnan(want[i])) worst = std::max(worst, std::abs(got[i] - want[i]));
  return worst;
}

TEST_CASE("Bibliothèque de warp : flot grossier avec confiance, puis caméra en Catmull-Rom", "[warp_lib]") {
  std::mt19937 rng(7);
  constexpr int kH = 96;
  constexpr int kW = 160;
  constexpr int kC = 5;
  const std::vector<double> image = field(static_cast<std::size_t>(kC * kH * kW), 1.0, rng);
  const std::vector<double> still = field(static_cast<std::size_t>(kC * kH * kW), 1.0, rng);
  Buffer image_gpu(image);
  Buffer still_gpu(still);

  // Flot en grille 4 rangé par point (pas 4, dx et dy aux canaux 1 et 2), en pixels de la grille : gain 4.
  constexpr int kS = 4;
  std::vector<double> flow = field(static_cast<std::size_t>((kH / kS) * (kW / kS) * 4), 3.0, rng);
  for (double& f : flow)  // centré, puis arrondi en FP16 comme le GPU le lira
    f = kernels::ref::half_to_float(kernels::ref::float_to_half(static_cast<float>(f - 1.5)));
  const std::vector<double> confidence = field(static_cast<std::size_t>((kH / kS) * (kW / kS)), 1.2, rng);
  Buffer flow_gpu(flow);
  Buffer confidence_gpu(confidence);
  constexpr int kD = 4;
  constexpr int kPitch = 8;
  Buffer out_gpu(std::vector<double>(static_cast<std::size_t>((kH / kD) * (kW / kD) * kPitch), 0.0));
  Buffer conf_gpu(std::vector<double>(static_cast<std::size_t>((kH / kD) * (kW / kD)), 0.0));
  WarpPlanesParams p{.image = image_gpu.ptr,
                     .channels = kC,
                     .height = kH,
                     .width = kW,
                     .flow = {.planes = flow_gpu.ptr,
                              .height = kH / kS,
                              .width = kW / kS,
                              .pitch = 4,
                              .channel = 1,
                              .scale = kS,
                              .gain = 4.0F},
                     .confidence = confidence_gpu.ptr,
                     .fallback = still_gpu.ptr,
                     .confidence_out = conf_gpu.ptr,
                     .reduce = kD,
                     .out = out_gpu.ptr,
                     .out_pitch = kPitch};
  REQUIRE(gpu::cuda::launch_warp_planes(p, nullptr) == cudaSuccess);
  std::vector<double> conf_want;
  std::vector<double> want = reference(p, image, &flow, &confidence, &still, conf_want);
  std::vector<double> got = out_gpu.read();
  for (std::size_t i = 0; i < got.size(); ++i)
    if (i % kPitch >= kC) got[i] = 0.0;  // canaux de remplissage, non écrits
  CHECK(worst_error(got, want) < 2e-3);
  CHECK(worst_error(conf_gpu.read(), conf_want) < 2e-3);

  // Caméra : translation, zoom et rotation légers, Catmull-Rom borné à une fenêtre, sortie en plans.
  Buffer plain_gpu(std::vector<double>(static_cast<std::size_t>(kC * kH * kW), 0.0));
  WarpPlanesParams camera{.image = image_gpu.ptr,
                          .channels = kC,
                          .height = kH,
                          .width = kW,
                          .window_left = 8,
                          .window_top = 4,
                          .window_width = kW - 16,
                          .window_height = kH - 8,
                          .flow = {.affine = {2.25F, 0.01F, -0.02F, -1.5F, 0.015F, 0.005F}},
                          .cubic = 1,
                          .out = plain_gpu.ptr};
  REQUIRE(gpu::cuda::launch_warp_planes(camera, nullptr) == cudaSuccess);
  want = reference(camera, image, nullptr, nullptr, nullptr, conf_want);
  CHECK(worst_error(plain_gpu.read(), want) < 2e-3);

  // Refus : trop de canaux, réduction impaire.
  camera.channels = 17;
  CHECK(gpu::cuda::launch_warp_planes(camera, nullptr) == cudaErrorInvalidValue);
  camera.channels = kC;
  camera.reduce = 3;
  CHECK(gpu::cuda::launch_warp_planes(camera, nullptr) == cudaErrorInvalidValue);
}

}  // namespace
}  // namespace kirari::test
