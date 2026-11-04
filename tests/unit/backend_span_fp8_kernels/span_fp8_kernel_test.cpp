// Les noyaux du SPAN FP8 (gpu/cuda/backend/span_fp8_net.h) aux tailles dont la dernière tuile de 128 lignes
// déborde de M (étape 3.1) : 1 Mio de canari 0xA5 après la zone de travail et après la sortie reste intact,
// et la sortie est finie. Paquet synthétique (span_fp8::synthetic). Sans GPU sm 8.9 : SKIP.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "gpu/cuda/backend/slice_kernels.h"
#include "gpu/cuda/backend/span_fp8_net.h"
#include "gpu/span_fp8/package.h"

namespace ajp::gpu::cuda {
namespace {

constexpr std::size_t kCanary = std::size_t{1} << 20;
constexpr unsigned char kMark = 0xA5;

bool has_sm89() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return false;
  cudaDeviceProp props{};
  return cudaGetDeviceProperties(&props, 0) == cudaSuccess &&
         SpanFp8Net::supported_device(props.major, props.minor);
}

SpanFp8Weights view(const span_fp8::Package& package) {
  const auto conv = [](const span_fp8::Conv& c) {
    return SpanFp8ConvView{.weights = c.weights.data(), .bias = c.bias.data()};
  };
  SpanFp8Weights weights;
  weights.scales = package.scales.data();
  weights.first = conv(package.first);
  for (std::size_t b = 0; b < package.blocks.size(); ++b) {
    for (std::size_t i = 0; i < 3; ++i) weights.conv[b][i] = conv(package.blocks[b].conv[i]);
    weights.attention[b] = conv(package.blocks[b].attention);
  }
  weights.concat = conv(package.concat);
  weights.pixel = conv(package.pixel);
  return weights;
}

struct DeviceBytes {
  void* data = nullptr;
  explicit DeviceBytes(std::size_t bytes) { REQUIRE(cudaMalloc(&data, bytes) == cudaSuccess); }
  DeviceBytes(const DeviceBytes&) = delete;
  DeviceBytes& operator=(const DeviceBytes&) = delete;
  ~DeviceBytes() { cudaFree(data); }
  unsigned char* at(std::size_t offset) const { return static_cast<unsigned char*>(data) + offset; }
};

bool canary_intact(const unsigned char* device) {
  std::vector<unsigned char> host(kCanary);
  REQUIRE(cudaMemcpy(host.data(), device, kCanary, cudaMemcpyDeviceToHost) == cudaSuccess);
  for (const unsigned char byte : host) {
    if (byte != kMark) return false;
  }
  return true;
}

// Une surface RGBA16 UNORM de width×height, mise à zéro.
struct Surface {
  cudaArray_t array = nullptr;
  cudaSurfaceObject_t object = 0;
  int width = 0, height = 0;
  Surface(int w, int h) : width(w), height(h) {
    const cudaChannelFormatDesc format = cudaCreateChannelDesc<ushort4>();
    REQUIRE(cudaMallocArray(&array, &format, static_cast<std::size_t>(w), static_cast<std::size_t>(h),
                            cudaArraySurfaceLoadStore) == cudaSuccess);
    const std::vector<ushort4> zero(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
    REQUIRE(cudaMemcpy2DToArray(array, 0, 0, zero.data(), static_cast<std::size_t>(w) * sizeof(ushort4),
                                static_cast<std::size_t>(w) * sizeof(ushort4), static_cast<std::size_t>(h),
                                cudaMemcpyHostToDevice) == cudaSuccess);
    cudaResourceDesc resource{};
    resource.resType = cudaResourceTypeArray;
    resource.res.array.array = array;
    REQUIRE(cudaCreateSurfaceObject(&object, &resource) == cudaSuccess);
  }
  Surface(const Surface&) = delete;
  Surface& operator=(const Surface&) = delete;
  ~Surface() {
    cudaDestroySurfaceObject(object);
    cudaFreeArray(array);
  }
  std::vector<ushort4> read() const {
    std::vector<ushort4> host(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
    REQUIRE(cudaMemcpy2DFromArray(host.data(), static_cast<std::size_t>(width) * sizeof(ushort4), array, 0, 0,
                                  static_cast<std::size_t>(width) * sizeof(ushort4),
                                  static_cast<std::size_t>(height), cudaMemcpyDeviceToHost) == cudaSuccess);
    return host;
  }
};

// to_pixel aux poids et biais variés : sorties négatives, au-delà de 1 et dans [0, 1], pour que les deux
// bornes de unorm16 servent.
void vary_pixel(span_fp8::Package& package) {
  std::uint32_t state = 0x2545F491U;
  const auto next = [&state] {
    state = state * 1664525U + 1013904223U;
    return static_cast<float>(state >> 8U) / static_cast<float>(1U << 24U);
  };
  for (std::uint16_t& bits : package.pixel.weights) {
    const __half value = __float2half((next() - 0.5F) * 0.1F);
    std::memcpy(&bits, &value, sizeof(bits));
  }
  for (float& bias : package.pixel.bias) bias = next() * 1.6F - 0.3F;
}

}  // namespace

TEST_CASE("SPAN FP8 : aucune ligne au-delà de M n'est lue ni écrite", "[cuda][span_fp8]") {
  if (!has_sm89()) SKIP("pas de GPU sm 8.9 : les noyaux FP8 ne tournent pas ici");
  const auto [height, width] =
      GENERATE(std::pair{540, 960}, std::pair{480, 854}, std::pair{900, 1600}, std::pair{1088, 1920});
  INFO(width << "x" << height);
  REQUIRE(span_fp8::size_refusal(height, width).empty());
  const span_fp8::Package package = span_fp8::synthetic();
  SpanFp8Status status = SpanFp8Status::Ok;
  std::string error;
  const std::unique_ptr<SpanFp8Net> net = SpanFp8Net::create(view(package), height, width, status, error);
  REQUIRE(net != nullptr);

  const auto work = static_cast<std::size_t>(net->workspace_bytes());
  DeviceBytes workspace(work + kCanary);
  REQUIRE(cudaMemset(workspace.data, 0, work) == cudaSuccess);
  REQUIRE(cudaMemset(workspace.at(work), kMark, kCanary) == cudaSuccess);
  const std::size_t in_count =
      std::size_t{3} * static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
  const std::size_t out_bytes = in_count * 4 * sizeof(__half);
  std::vector<__half> input(in_count);
  for (std::size_t i = 0; i < in_count; ++i) input[i] = __float2half(static_cast<float>(i % 251) / 250.0F);
  DeviceBytes in(in_count * sizeof(__half));
  REQUIRE(cudaMemcpy(in.data, input.data(), in_count * sizeof(__half), cudaMemcpyHostToDevice) ==
          cudaSuccess);
  DeviceBytes out(out_bytes + kCanary);
  REQUIRE(cudaMemset(out.at(out_bytes), kMark, kCanary) == cudaSuccess);

  net->bind(workspace.data);
  REQUIRE(net->launch(nullptr, in.data, out.data, error));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  CHECK(canary_intact(workspace.at(work)));
  CHECK(canary_intact(out.at(out_bytes)));
  std::vector<__half> output(in_count * 4);
  REQUIRE(cudaMemcpy(output.data(), out.data, out_bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
  std::size_t bad = 0;
  for (const __half value : output) {
    if (!std::isfinite(__half2float(value))) ++bad;
  }
  CHECK(bad == 0);
}

TEST_CASE("l'épilogue de to_pixel écrit l'anneau au bit près de write_rgba16", "[cuda][span_fp8]") {
  if (!has_sm89()) SKIP("pas de GPU sm 8.9 : les noyaux FP8 ne tournent pas ici");
  const auto [height, width] = GENERATE(std::pair{540, 960}, std::pair{480, 854});
  // Surface entière, ou plus petite que la sortie (emplacement rogné : rien au-delà n'est écrit).
  const int trim = GENERATE(0, 6);
  INFO(width << "x" << height << ", rognage " << trim);
  span_fp8::Package package = span_fp8::synthetic();
  vary_pixel(package);
  SpanFp8Status status = SpanFp8Status::Ok;
  std::string error;
  const std::unique_ptr<SpanFp8Net> net = SpanFp8Net::create(view(package), height, width, status, error);
  REQUIRE(net != nullptr);
  DeviceBytes workspace(static_cast<std::size_t>(net->workspace_bytes()));
  const std::size_t in_count =
      std::size_t{3} * static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
  std::vector<__half> input(in_count);
  for (std::size_t i = 0; i < in_count; ++i) input[i] = __float2half(static_cast<float>(i % 241) / 240.0F);
  DeviceBytes in(in_count * sizeof(__half));
  REQUIRE(cudaMemcpy(in.data, input.data(), in_count * sizeof(__half), cudaMemcpyHostToDevice) ==
          cudaSuccess);
  DeviceBytes out(in_count * 4 * sizeof(__half));
  const int out_w = 2 * width - trim, out_h = 2 * height - trim;
  const Surface fused(out_w, out_h);
  const Surface reference(out_w, out_h);
  const Surface alone(out_w, out_h);

  net->bind(workspace.data);
  REQUIRE(net->launch_body(nullptr, in.data, error));
  REQUIRE(net->launch_tail(nullptr, out.data, SpanFp8Ring{{.object = fused.object}, out_w, out_h}, error));
  const WriteParams params{
      .rgb = out.data, .width = out_w, .height = out_h, .surface = {.object = reference.object}};
  REQUIRE(launch_write_rgba16(params, nullptr) == cudaSuccess);
  // Sans sortie FP16 : l'anneau seul, même image.
  REQUIRE(net->launch_body(nullptr, in.data, error));
  REQUIRE(net->launch_tail(nullptr, nullptr, SpanFp8Ring{{.object = alone.object}, out_w, out_h}, error));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const std::vector<ushort4> a = fused.read();
  const std::vector<ushort4> b = reference.read();
  const std::vector<ushort4> c = alone.read();
  std::size_t differ = 0, differ_alone = 0, low = 0, high = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (std::memcmp(&a[i], &b[i], sizeof(ushort4)) != 0) ++differ;
    if (std::memcmp(&c[i], &b[i], sizeof(ushort4)) != 0) ++differ_alone;
    if (b[i].x == 0) ++low;
    if (b[i].x == 65535) ++high;
  }
  CHECK(differ == 0);
  CHECK(differ_alone == 0);
  CHECK(low > 0);  // les deux bornes de unorm16 ont servi
  CHECK(high > 0);
}

TEST_CASE("la sortie FP16 de to_pixel suit DepthToSpace (CRD)", "[cuda][span_fp8]") {
  if (!has_sm89()) SKIP("pas de GPU sm 8.9 : les noyaux FP8 ne tournent pas ici");
  // Une seule image, comparée à la référence FP16 non fondue : l'ordre des pixels de sortie est celui du
  // DepthToSpace du modèle (même si les valeurs diffèrent de l'arrondi FP8 des blocs).
  constexpr int kHeight = 64, kWidth = 96;
  span_fp8::Package package = span_fp8::synthetic(0.0F);  // blocs nuls : la sortie ne dépend que du biais
  vary_pixel(package);
  for (std::uint16_t& bits : package.pixel.weights) bits = 0;
  SpanFp8Status status = SpanFp8Status::Ok;
  std::string error;
  const std::unique_ptr<SpanFp8Net> net = SpanFp8Net::create(view(package), kHeight, kWidth, status, error);
  REQUIRE(net != nullptr);
  DeviceBytes workspace(static_cast<std::size_t>(net->workspace_bytes()));
  const std::size_t in_count = std::size_t{3} * kHeight * kWidth;
  DeviceBytes in(in_count * sizeof(__half));
  REQUIRE(cudaMemset(in.data, 0, in_count * sizeof(__half)) == cudaSuccess);
  DeviceBytes out(in_count * 4 * sizeof(__half));
  net->bind(workspace.data);
  REQUIRE(net->launch(nullptr, in.data, out.data, error));
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  std::vector<__half> host(in_count * 4);
  REQUIRE(cudaMemcpy(host.data(), out.data, host.size() * sizeof(__half), cudaMemcpyDeviceToHost) ==
          cudaSuccess);
  // Poids nuls : chaque pixel de sortie (c, 4y + bh, 4x + bw) vaut le biais du canal 16c + 4bh + bw.
  constexpr int kOutW = 2 * kWidth, kOutH = 2 * kHeight;
  std::size_t wrong = 0;
  for (int c = 0; c < 3; ++c)
    for (int oy = 0; oy < kOutH; ++oy)
      for (int ox = 0; ox < kOutW; ++ox) {
        const float expected = __half2float(
            __float2half(package.pixel.bias[static_cast<std::size_t>(16 * c + 4 * (oy % 4) + ox % 4)]));
        const float got =
            __half2float(host[(static_cast<std::size_t>(c) * kOutH + static_cast<std::size_t>(oy)) * kOutW +
                              static_cast<std::size_t>(ox)]);
        if (got != expected) ++wrong;
      }
  CHECK(wrong == 0);
}

}  // namespace ajp::gpu::cuda
