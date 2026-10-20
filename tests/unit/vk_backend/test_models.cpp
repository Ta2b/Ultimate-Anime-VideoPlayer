#include "test_models.h"

#include "gpu/ir/onnx_model.h"
#include "net_rig.h"

namespace ajp::gpu::vk::test {

using ir::test::attr_float;
using ir::test::attr_int;
using ir::test::attr_ints;
using ir::test::attr_string;
using ir::test::ModelBuilder;

std::vector<float> weights(std::size_t count, std::size_t seed, float scale) {
  return test_values(ir::DataType::F32, {static_cast<std::int64_t>(count)}, seed, -scale, scale).f;
}

void add_weight(ModelBuilder& m, const std::string& name, const std::vector<std::int64_t>& dims, bool half,
                std::size_t seed, float scale) {
  std::size_t count = 1;
  for (const std::int64_t d : dims) count *= static_cast<std::size_t>(d);
  const std::vector<float> values = weights(count, seed, scale);
  if (half) {
    m.initializer(ir::test::tensor_f16(name, dims, values));
  } else {
    m.initializer(ir::test::tensor_f32(name, dims, values));
  }
}

ModelBuilder span_like(bool half) {
  ModelBuilder m;
  const std::int32_t type = half ? ir::onnx::kFloat16 : ir::onnx::kFloat;
  m.input("x", {1, 3, 24, 32}, type);
  add_weight(m, "w1", {16, 3, 3, 3}, half, 61);
  add_weight(m, "b1", {16}, half, 62);
  add_weight(m, "p1", {16, 1, 1}, half, 63);
  add_weight(m, "w2", {16, 16, 3, 3}, half, 64, 0.1F);
  add_weight(m, "b2", {16}, half, 65);
  add_weight(m, "p2", {16, 1, 1}, half, 66);
  add_weight(m, "w3", {12, 16, 3, 3}, half, 67, 0.1F);
  add_weight(m, "b3", {12}, half, 68);
  const auto same = attr_ints("pads", {1, 1, 1, 1});
  m.node("Conv", {"x", "w1", "b1"}, {"c1"}, {same}).node("PRelu", {"c1", "p1"}, {"a1"});
  m.node("Conv", {"a1", "w2", "b2"}, {"c2"}, {same}).node("PRelu", {"c2", "p2"}, {"a2"});
  m.node("Conv", {"a2", "w3", "b3"}, {"c3"}, {same});
  m.node("DepthToSpace", {"c3"}, {"y"}, {attr_int("blocksize", 2), attr_string("mode", "CRD")});
  m.output("y", type);
  return m;
}

ModelBuilder rife_like(bool half) {
  ModelBuilder m(16);
  const std::int32_t type = half ? ir::onnx::kFloat16 : ir::onnx::kFloat;
  m.input("x", {1, 6, 16, 24}, type);
  m.ints("split", {3, 3});
  m.weight("scales", {4}, {1.0F, 1.0F, 0.5F, 0.5F});  // les échelles d'un Resize restent en float
  add_weight(m, "w1", {8, 6, 3, 3}, half, 71);
  add_weight(m, "b1", {8}, half, 72);
  add_weight(m, "wt", {8, 3, 4, 4}, half, 73, 0.2F);
  add_weight(m, "bt", {3}, half, 74);
  m.ints("flow_shape", {1, 1, 16, 24});
  m.node("Split", {"x", "split"}, {"a", "b"}, {attr_int("axis", 1)});
  m.node("Concat", {"a", "b"}, {"ab"}, {attr_int("axis", 1)});
  m.node("Resize", {"ab", "", "scales"}, {"small"}, {attr_string("mode", "linear")});
  m.node("Conv", {"small", "w1", "b1"}, {"c1"}, {attr_ints("pads", {1, 1, 1, 1})});
  m.node("LeakyRelu", {"c1"}, {"l1"}, {attr_float("alpha", 0.2F)});
  m.node("ConvTranspose", {"l1", "wt", "bt"}, {"up"},
         {attr_ints("strides", {2, 2}), attr_ints("pads", {1, 1, 1, 1})});
  m.ints("starts", {0}).ints("ends", {2}).ints("axes", {1});
  m.node("Slice", {"up", "starts", "ends", "axes"}, {"flow"});
  m.ints("mask_start", {2}).ints("mask_end", {3});
  m.node("Slice", {"up", "mask_start", "mask_end", "axes"}, {"logit"});
  m.node("Transpose", {"flow"}, {"grid"}, {attr_ints("perm", {0, 2, 3, 1})});
  m.node(
      "GridSample", {"a", "grid"}, {"warped"},
      {attr_string("mode", "bilinear"), attr_string("padding_mode", "border"), attr_int("align_corners", 1)});
  m.node("Sigmoid", {"logit"}, {"mask"});
  m.node("Mul", {"warped", "mask"}, {"wa"});
  m.node("Sub", {"b", "wa"}, {"y"});
  m.output("y", type);
  return m;
}

}  // namespace ajp::gpu::vk::test
