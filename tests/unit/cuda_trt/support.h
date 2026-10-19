// Outils communs des tests [cuda] du runtime TensorRT.
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/types.h"
#include "gpu/cuda/trt/log.h"
#include "gpu/cuda/trt/runtime.h"
#include "plan_builder.h"
#include "test_util.h"

namespace ajp::gpu::trt::test {

// Saute le test, avec sa raison, sans appareil CUDA (la CI n'en a pas).
inline void skip_without_cuda() {
  if (const std::string why = cuda_unavailable_reason(); !why.empty()) SKIP(why);
}

template <typename T>
T must(std::expected<T, Refusal> value) {
  if (!value) FAIL(ajp::name(value.error().code) << " : " << value.error().detail);
  return std::move(*value);
}

inline void must(const std::expected<void, Refusal>& done) {
  if (!done) FAIL(ajp::name(done.error().code) << " : " << done.error().detail);
}

// Conv 3x3 (3 → 4 canaux, biais), Relu, puis Add d'un second Conv 1x1 : assez
// pour que TensorRT fusionne et choisisse des tactiques, assez petit pour se
// construire en quelques secondes. H et W restent dynamiques.
inline ir::test::ModelBuilder conv_model(std::int32_t type) {
  const auto weights = [&](std::string_view name, const std::vector<std::int64_t>& dims, std::size_t seed) {
    const std::vector<float> values = ir::test::sequence(ir::test::count_of(dims), -0.5F, 0.5F, seed);
    if (type == ir::onnx::kFloat16) return ir::test::tensor_f16(name, dims, values);
    return ir::test::tensor_f32(name, dims, values);
  };
  ir::test::ModelBuilder m;
  m.input("input", {1, 3, -1, -1}, type)
      .initializer(weights("w3", {4, 3, 3, 3}, 1))
      .initializer(weights("b3", {4}, 2))
      .initializer(weights("w1", {4, 3, 1, 1}, 3))
      .node("Conv", {"input", "w3", "b3"}, {"c3"}, {ir::test::attr_ints("pads", {1, 1, 1, 1})})
      .node("Relu", {"c3"}, {"r"})
      .node("Conv", {"input", "w1"}, {"c1"})
      .node("Add", {"r", "c1"}, {"output"})
      .output("output", type);
  return m;
}

inline const ShapeRange kConvProfile{.min = {1, 3, 8, 8}, .opt = {1, 3, 16, 16}, .max = {1, 3, 32, 32}};

inline TrtRuntime make_runtime() {
  return must(TrtRuntime::create(make_logger()));
}

inline std::vector<std::byte> conv_plan(const TrtRuntime& runtime, std::int32_t type) {
  auto plan = build_plan(runtime.log(), conv_model(type).bytes(), "input", kConvProfile);
  if (!plan) FAIL(plan.error());
  return std::move(*plan);
}

}  // namespace ajp::gpu::trt::test
