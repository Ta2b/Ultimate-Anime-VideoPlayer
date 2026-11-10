// Référence indépendante des tests des moteurs fusionnés (A45-11) : le modèle entier joué par l'interpréteur
// de gpu_ir, en float, sans aucun noyau Vulkan. Le réseau entier sous Vulkan partage ses noyaux et ses
// choix de précision avec le moteur fusionné : une erreur commune aux deux ne se verrait pas contre lui.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "gpu/ir/graph.h"
#include "gpu/ir/import.h"
#include "gpu/ir/interpreter.h"
#include "gpu/ir/shape.h"
#include "kernels/image.h"
#include "kernels/ref/half.h"

namespace ajp::gpu::vk::test {

// Les premiers plans de la sortie du modèle onnx (une entrée par élément de inputs, dans l'ordre du
// modèle, chacune de h × w), calculée par l'interpréteur. Les entrées float16 sont arrondies comme un
// tenseur du backend.
inline std::vector<kernels::PlaneF> interpreter_planes(
    const std::filesystem::path& onnx, const std::vector<std::vector<kernels::PlaneF>>& inputs, int h, int w,
    std::size_t planes) {
  auto graph = ir::load_onnx_file(onnx);
  REQUIRE(graph);
  REQUIRE(graph->inputs.size() == inputs.size());
  std::vector<ir::InputShape> shapes;
  std::vector<ir::NamedValue> values;
  for (std::size_t k = 0; k < inputs.size(); ++k) {
    const ir::Tensor& tensor = graph->tensors[graph->inputs[k]];
    const ir::Dims dims{1, static_cast<std::int64_t>(inputs[k].size()), h, w};
    shapes.push_back({.name = tensor.name, .dims = dims});
    ir::Value value = ir::make_value(tensor.type, dims);
    std::size_t at = 0;
    for (const kernels::PlaneF& plane : inputs[k]) {
      for (const float v : plane.data)
        value.f[at++] = tensor.type == ir::DataType::F16 ? kernels::ref::round_to_half(v) : v;
    }
    values.push_back({.name = tensor.name, .value = std::move(value)});
  }
  REQUIRE(ir::infer_shapes(*graph, shapes));
  auto outputs = ir::run(*graph, values);
  REQUIRE(outputs);
  const std::vector<float>& f = outputs->front().value.f;
  const std::size_t area = static_cast<std::size_t>(h) * static_cast<std::size_t>(w);
  REQUIRE(f.size() >= planes * area);
  std::vector<kernels::PlaneF> out;
  for (std::size_t c = 0; c < planes; ++c) {
    kernels::PlaneF plane(w, h, 0.0F);
    for (std::size_t i = 0; i < area; ++i) plane.data[i] = f[(c * area) + i];
    out.push_back(std::move(plane));
  }
  return out;
}

// PSNR, en dB, de got contre want sur des valeurs dans [0, 1] ; 200 pour des plans identiques.
inline double planes_psnr(const std::vector<kernels::PlaneF>& got, const std::vector<kernels::PlaneF>& want) {
  double square = 0.0;
  std::size_t count = 0;
  for (std::size_t c = 0; c < want.size(); ++c) {
    for (std::size_t i = 0; i < want[c].data.size(); ++i) {
      const double d = static_cast<double>(got[c].data[i]) - static_cast<double>(want[c].data[i]);
      square += d * d;
      ++count;
    }
  }
  if (square == 0.0) return 200.0;
  return 10.0 * std::log10(static_cast<double>(count) / square);
}

}  // namespace ajp::gpu::vk::test
