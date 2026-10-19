#include "plan_builder.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <bit>
#include <cstring>
#include <cuda_fp16.h>
#include <memory>

#include "gpu/cuda/trt/cuda_objects.h"
#include "gpu/cuda/trt/trt_log.h"

namespace ajp::gpu::trt::test {
namespace {

nvinfer1::Dims dims_of(const Dims& dims) {
  nvinfer1::Dims out{};
  out.nbDims = static_cast<std::int32_t>(dims.size());
  for (std::size_t k = 0; k < dims.size(); ++k) out.d[k] = dims[k];
  return out;
}

}  // namespace

std::string cuda_unavailable_reason() {
  auto device = current_device();
  return device ? std::string{} : "pas d'appareil CUDA : " + device.error().detail;
}

std::expected<std::vector<std::byte>, std::string> build_plan(TrtLog& log, std::span<const std::byte> onnx,
                                                              const std::string& input,
                                                              const std::optional<ShapeRange>& profile) {
  const std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(log));
  if (!builder) return std::unexpected("createInferBuilder" + error_suffix(log));
  const std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
  const std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, log));
  if (!parser->parse(onnx.data(), onnx.size())) {
    std::string errors;
    for (int k = 0; k < parser->getNbErrors(); ++k) errors += std::string(" ") + parser->getError(k)->desc();
    return std::unexpected("analyse ONNX :" + errors);
  }
  const std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
  config->clearFlag(nvinfer1::BuilderFlag::kTF32);
  config->setBuilderOptimizationLevel(0);  // quelques secondes au lieu d'une minute
  if (profile) {
    nvinfer1::IOptimizationProfile* shapes = builder->createOptimizationProfile();
    using nvinfer1::OptProfileSelector;
    shapes->setDimensions(input.c_str(), OptProfileSelector::kMIN, dims_of(profile->min));
    shapes->setDimensions(input.c_str(), OptProfileSelector::kOPT, dims_of(profile->opt));
    shapes->setDimensions(input.c_str(), OptProfileSelector::kMAX, dims_of(profile->max));
    config->addOptimizationProfile(shapes);
  }
  const std::unique_ptr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
  if (!plan) return std::unexpected("buildSerializedNetwork" + error_suffix(log));
  const auto* first = static_cast<const std::byte*>(plan->data());
  return std::vector<std::byte>(first, first + plan->size());
}

std::expected<void, Refusal> upload(void* device, std::span<const std::byte> host) {
  return cuda_check(cudaMemcpy(device, host.data(), host.size(), cudaMemcpyHostToDevice), "cudaMemcpy(H2D)");
}

std::expected<std::vector<std::byte>, Refusal> download(const void* device, std::size_t bytes) {
  std::vector<std::byte> host(bytes);
  if (auto ok = cuda_check(cudaMemcpy(host.data(), device, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(D2H)");
      !ok) {
    return std::unexpected(ok.error());
  }
  return host;
}

std::vector<std::byte> to_bytes(DataType type, std::span<const float> values) {
  const std::size_t width = type == DataType::F16 ? 2 : 4;
  std::vector<std::byte> bytes(values.size() * width);
  for (std::size_t k = 0; k < values.size(); ++k) {
    if (type == DataType::F16) {
      const auto bits = std::bit_cast<std::uint16_t>(__float2half_rn(values[k]));
      std::memcpy(&bytes[k * width], &bits, width);
    } else {
      std::memcpy(&bytes[k * width], &values[k], width);
    }
  }
  return bytes;
}

std::vector<float> to_floats(DataType type, std::span<const std::byte> bytes) {
  const std::size_t width = type == DataType::F16 ? 2 : 4;
  std::vector<float> values(bytes.size() / width);
  for (std::size_t k = 0; k < values.size(); ++k) {
    if (type == DataType::F16) {
      std::uint16_t bits = 0;
      std::memcpy(&bits, &bytes[k * width], width);
      values[k] = __half2float(std::bit_cast<__half>(bits));
    } else {
      std::memcpy(&values[k], &bytes[k * width], width);
    }
  }
  return values;
}

}  // namespace ajp::gpu::trt::test
