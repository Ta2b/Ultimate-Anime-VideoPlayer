// Un SPAN à 48 canaux (b5f48) calculé en 48 (kKernelsVersion 4) : les deux formes de K (mis à plat, par
// position du filtre) suivent l'ancien calcul en 64 canaux dont 16 nuls, sur des poids aléatoires fixes et
// des échelles tirées de la référence FP16. Les mêmes poids FP8 et les mêmes échelles entrent dans les deux :
// seul l'ordre des sommes change. Une tranche de la concaténation, une entrée de conv_cat ou de to_pixel lue
// au pas de 64 au lieu de 48 éloignerait la sortie de l'ordre de sa propre variation. Sans GPU sm 8.9 : SKIP.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "../../support/span_fp8_synthetic.h"
#include "gpu/cuda/backend/span_fp8_net.h"
#include "gpu/cuda/backend/span_fp8_ref.h"
#include "gpu/cuda/backend/span_fp8_view.h"
#include "gpu/span_fp8/derive.h"
#include "gpu/span_fp8/package.h"

namespace ajp::gpu::cuda {
namespace {

constexpr int kReal = 48;
constexpr int kStored = span_fp8::kChannels;
constexpr int kHeight = 96, kWidth = 160;
constexpr std::size_t kInCount = std::size_t{3} * kHeight * kWidth;

bool has_sm89() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return false;
  cudaDeviceProp props{};
  return cudaGetDeviceProperties(&props, 0) == cudaSuccess &&
         SpanFp8Net::supported_device(props.major, props.minor);
}

struct DeviceBytes {
  void* data = nullptr;
  explicit DeviceBytes(std::size_t bytes) { REQUIRE(cudaMalloc(&data, bytes) == cudaSuccess); }
  DeviceBytes(const DeviceBytes&) = delete;
  DeviceBytes& operator=(const DeviceBytes&) = delete;
  ~DeviceBytes() { cudaFree(data); }
};

class Random {
 public:
  float next() {  // [-0,5, 0,5)
    state_ = state_ * 1664525U + 1013904223U;
    return static_cast<float>(state_ >> 8U) / static_cast<float>(1U << 24U) - 0.5F;
  }

 private:
  std::uint32_t state_ = 0x85EBCA6BU;
};

// Poids aléatoires sur les canaux réels, zéros ailleurs, comme le paquet range un b5f48 : sorties au-delà de
// real_outs nulles, et dans chaque groupe de kStored entrées, celles au-delà de real_ins.
void randomize(span_fp8::Conv& conv, std::size_t outs, std::size_t real_outs, std::size_t ins,
               std::size_t real_ins, float spread, Random& random) {
  const std::size_t per_out = conv.weights.size() / outs;
  for (std::size_t i = 0; i < conv.weights.size(); ++i) {
    const bool real = i / per_out < real_outs && (i % per_out) % ins < real_ins;
    const __half value = __float2half(real ? random.next() * spread : 0.0F);
    std::memcpy(&conv.weights[i], &value, sizeof(std::uint16_t));
  }
  for (std::size_t o = 0; o < conv.bias.size(); ++o)
    conv.bias[o] = o < real_outs ? random.next() * 0.05F : 0.0F;
}

span_fp8::Package b5f48(Random& random) {
  span_fp8::Package package = span_fp8::testing::synthetic(0.01F, 5);
  package.real_channels = kReal;
  randomize(package.first, kStored, kReal, span_fp8::kUnshuffledPadded, 12, 0.2F, random);
  for (span_fp8::Block& block : package.blocks) {
    for (span_fp8::Conv& conv : block.conv) randomize(conv, kStored, kReal, kStored, kReal, 0.16F, random);
    randomize(block.attention, kStored, kReal, kStored, kReal, 0.45F, random);
  }
  randomize(package.concat, kStored, kReal, kStored, kReal, 0.2F, random);
  randomize(package.pixel, span_fp8::kPixelChannels, span_fp8::kPixelChannels, kStored, kReal, 0.16F, random);
  return package;
}

std::vector<double> run(const span_fp8::Package& package, SpanFp8Width width, const DeviceBytes& in) {
  SpanFp8Weights weights = span_fp8_view(package);
  weights.width48 = width;
  SpanFp8Status status = SpanFp8Status::Ok;
  std::string error;
  const std::unique_ptr<SpanFp8Net> net = SpanFp8Net::create(weights, kHeight, kWidth, status, error);
  INFO(error);
  REQUIRE(net != nullptr);
  DeviceBytes workspace(static_cast<std::size_t>(net->workspace_bytes()));
  DeviceBytes out(kInCount * 4 * sizeof(__half));
  net->bind(workspace.data);
  REQUIRE(net->launch(nullptr, in.data, out.data, error));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  std::vector<__half> half(kInCount * 4);
  REQUIRE(cudaMemcpy(half.data(), out.data, half.size() * sizeof(__half), cudaMemcpyDeviceToHost) ==
          cudaSuccess);
  std::vector<double> values(half.size());
  for (std::size_t i = 0; i < half.size(); ++i) values[i] = static_cast<double>(__half2float(half[i]));
  return values;
}

// Rapport signal sur bruit de got contre expected, en dB.
double snr_db(const std::vector<double>& expected, const std::vector<double>& got) {
  double mean = 0.0;
  for (const double v : expected) mean += v;
  mean /= static_cast<double>(expected.size());
  double signal = 0.0, noise = 0.0;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    REQUIRE(std::isfinite(got[i]));
    signal += (expected[i] - mean) * (expected[i] - mean);
    noise += (got[i] - expected[i]) * (got[i] - expected[i]);
  }
  REQUIRE(signal > 0.0);
  return 10.0 * std::log10(signal / std::max(noise, 1e-30));
}

}  // namespace

TEST_CASE("SPAN FP8 à 48 canaux : le calcul en 48 suit celui en 64 dont 16 nuls", "[cuda][span_fp8]") {
  if (!has_sm89()) SKIP("pas de GPU sm 8.9 : les noyaux FP8 ne tournent pas ici");
  const bool acc16 = GENERATE(false, true);
  INFO("acc16 " << acc16);
  Random random;
  span_fp8::Package package = b5f48(random);
  package.acc16 = acc16;
  std::vector<__half> image(kInCount);
  for (__half& value : image) value = __float2half(random.next() + 0.5F);
  DeviceBytes in(kInCount * sizeof(__half));
  REQUIRE(cudaMemcpy(in.data, image.data(), kInCount * sizeof(__half), cudaMemcpyHostToDevice) ==
          cudaSuccess);

  // Les échelles d'activation de la référence FP16, comme la dérivation les tire (C02-10).
  SpanFp8Status status = SpanFp8Status::Ok;
  std::string error;
  const std::unique_ptr<SpanFp8Reference> reference =
      SpanFp8Reference::create(span_fp8_view(package), kHeight, kWidth, status, error);
  REQUIRE(reference != nullptr);
  DeviceBytes scratch(kInCount * 4 * sizeof(__half));
  REQUIRE(reference->run(nullptr, in.data, scratch.data, true, error));
  float peaks[SpanFp8Reference::kPeaks];
  REQUIRE(reference->peaks(peaks, error));
  const auto scales = span_fp8::scales_from_peaks(std::span<const float>(peaks), package.blocks.size());
  REQUIRE(scales);
  package.scales = *scales;

  const std::vector<double> padded = run(package, SpanFp8Width::Padded, in);
  const double flat = snr_db(padded, run(package, SpanFp8Width::Native, in));
  const double per_tap = snr_db(padded, run(package, SpanFp8Width::PerTap, in));
  INFO("SNR contre le calcul en 64 : K à plat " << flat << " dB, K par position " << per_tap << " dB");
  // Seul l'ordre des sommes diffère : bien au-dessus de l'arrondi FP8 (le test C02-10 exige 20 dB contre la
  // référence FP16).
  CHECK(flat >= 45.0);
  CHECK(per_tap >= 45.0);
}

TEST_CASE("SPAN FP8 à 48 canaux : la zone de travail rétrécit", "[cuda][span_fp8]") {
  if (!has_sm89()) SKIP("pas de GPU sm 8.9 : les noyaux FP8 ne tournent pas ici");
  Random random;
  const span_fp8::Package package = b5f48(random);
  SpanFp8Status status = SpanFp8Status::Ok;
  std::string error;
  SpanFp8Weights weights = span_fp8_view(package);
  const auto native = SpanFp8Net::create(weights, 1080, 1920, status, error);
  weights.width48 = SpanFp8Width::Padded;
  const auto padded = SpanFp8Net::create(weights, 1080, 1920, status, error);
  REQUIRE(native != nullptr);
  REQUIRE(padded != nullptr);
  // 576 octets par pixel de caractéristiques contre 768.
  CHECK(native->workspace_bytes() * 4 < padded->workspace_bytes() * 3 + 4096);
}

}  // namespace ajp::gpu::cuda
