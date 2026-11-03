// Les noyaux du SPAN FP8 (gpu/cuda/backend/span_fp8_net.h) aux tailles dont la dernière tuile de 128 lignes
// déborde de M (étape 3.1) : 1 Mio de canari 0xA5 après la zone de travail et après la sortie reste intact,
// et la sortie est finie. Paquet synthétique (span_fp8::synthetic). Sans GPU sm 8.9 : SKIP.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

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

}  // namespace ajp::gpu::cuda
