// Le modèle d'un interpolateur du système (prepare/onnx_blend.h) : l'interface de RIFE v2 que la
// chaîne attend, la marque que le runtime reconnaît, et le fondu que l'interpréteur en calcule.
#include "prepare/onnx_blend.h"

#include <cstddef>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "gpu/ir/onnx_model.h"
#include "gpu/system_interpolator.h"
#include "kernels/spec/rife.h"
#include "test_util.h"

using namespace ajp::gpu::ir;

TEST_CASE("onnx : modèle d'interpolateur à l'interface de RIFE v2, marqué", "[gpu_ir][onnx]") {
  const std::vector<std::byte> bytes =
      ajp::prepare::blend_model(ajp::gpu::kSystemInterpolatorProducer, ajp::gpu::kVideoToolboxFrameRate);
  auto model = onnx::read_onnx(bytes);
  REQUIRE(model);
  CHECK(model->producer_name == ajp::gpu::kSystemInterpolatorProducer);
  CHECK(model->producer_version == ajp::gpu::kVideoToolboxFrameRate);
  const onnx::Graph& g = model->graph;
  REQUIRE(g.inputs.size() == 1);
  REQUIRE(g.outputs.size() == 1);
  CHECK(g.inputs[0].name == "input");
  CHECK(g.outputs[0].name == "output");
  CHECK(g.inputs[0].elem_type == onnx::kFloat16);
  CHECK(g.outputs[0].elem_type == onnx::kFloat16);
  CHECK(g.inputs[0].dims.size() == 4);
  CHECK(g.inputs[0].dims[1] == ajp::kernels::kRifeV2InputChannels);
  CHECK(g.outputs[0].dims[1] == 3);
  CHECK(g.inputs[0].symbols[2] == "height");
  CHECK(g.inputs[0].symbols[3] == "width");
}

TEST_CASE("onnx : le graphe de l'interpolateur fait le fondu a + (b − a) t", "[gpu_ir][onnx]") {
  const std::vector<std::byte> bytes = ajp::prepare::blend_model("x", "y");
  auto graph = load_onnx(bytes);
  REQUIRE(graph);
  constexpr std::int64_t kHeight = 2;
  constexpr std::int64_t kWidth = 3;
  constexpr std::size_t kPlane = kHeight * kWidth;
  const std::vector<InputShape> shapes{{.name = "input", .dims = {1, 7, kHeight, kWidth}}};
  REQUIRE(infer_shapes(*graph, shapes));
  // Des valeurs exactes en F16 : a = k/8, b = 1 − k/16, t = 1/4 puis 3/4 selon la ligne.
  std::vector<float> values(7 * kPlane);
  for (std::size_t k = 0; k < 3 * kPlane; ++k) {
    values[k] = static_cast<float>(k % 8) / 8.0F;
    values[(3 * kPlane) + k] = 1.0F - (static_cast<float>(k % 16) / 16.0F);
  }
  for (std::size_t p = 0; p < kPlane; ++p) values[(6 * kPlane) + p] = p < kWidth ? 0.25F : 0.75F;
  const Value out =
      test::execute1(*graph, {{"input", test::floats({1, 7, kHeight, kWidth}, values, DataType::F16)}});
  REQUIRE(out.f.size() == 3 * kPlane);
  for (std::size_t k = 0; k < 3 * kPlane; ++k) {
    const float a = values[k];
    const float b = values[(3 * kPlane) + k];
    const float t = values[(6 * kPlane) + (k % kPlane)];
    CHECK(out.f[k] == a + ((b - a) * t));
  }
}
