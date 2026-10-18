// Outils communs des tests gpu_ir : chargement d'un modèle construit, données
// déterministes, exécution et comparaison.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "gpu/ir/graph.h"
#include "gpu/ir/import.h"
#include "gpu/ir/interpreter.h"
#include "gpu/ir/shape.h"
#include "onnx_builder.h"

namespace ajp::gpu::ir::test {

// Charge le modèle et infère ses formes ; le test échoue sur un refus.
inline Graph build(const ModelBuilder& model, const std::vector<InputShape>& shapes = {}) {
  auto graph = load_onnx(model.bytes());
  if (graph) {
    if (auto ok = infer_shapes(*graph, shapes); !ok) graph = std::unexpected(ok.error());
  }
  if (!graph) FAIL("refus " << ajp::name(graph.error().code) << " : " << graph.error().detail);
  return std::move(*graph);
}

// Refus attendu au chargement ou à l'inférence des formes.
inline Refusal refusal(const ModelBuilder& model, const std::vector<InputShape>& shapes = {}) {
  auto graph = load_onnx(model.bytes());
  if (!graph) return graph.error();
  auto ok = infer_shapes(*graph, shapes);
  if (ok) FAIL("refus attendu");
  return ok.error();
}

// Suite déterministe dans [lo, hi[ : partie fractionnaire de (k + seed) φ.
inline std::vector<float> sequence(std::size_t count, float lo = -1.0F, float hi = 1.0F,
                                   std::size_t seed = 0) {
  std::vector<float> out(count);
  for (std::size_t k = 0; k < count; ++k) {
    double whole = 0.0;
    const double frac = std::modf(static_cast<double>(k + seed) * 0.6180339887498949, &whole);
    out[k] = lo + ((hi - lo) * static_cast<float>(frac));
  }
  return out;
}

inline std::size_t count_of(const Dims& dims) {
  std::size_t n = 1;
  for (const std::int64_t d : dims) n *= static_cast<std::size_t>(d);
  return n;
}

inline Value floats(Dims dims, std::vector<float> values, DataType type = DataType::F32) {
  return Value{.type = type, .dims = std::move(dims), .f = std::move(values)};
}

inline Value random_floats(const Dims& dims, std::size_t seed = 0, float lo = -1.0F, float hi = 1.0F) {
  return floats(dims, sequence(count_of(dims), lo, hi, seed));
}

// Exécute le graphe et rend ses sorties ; le test échoue sur un refus.
inline std::vector<NamedValue> execute(const Graph& graph, const std::vector<NamedValue>& inputs) {
  auto outputs = run(graph, inputs);
  if (!outputs) FAIL("refus " << ajp::name(outputs.error().code) << " : " << outputs.error().detail);
  return std::move(*outputs);
}

inline Value execute1(const Graph& graph, const std::vector<NamedValue>& inputs) {
  std::vector<NamedValue> outputs = execute(graph, inputs);
  REQUIRE(outputs.size() == 1);
  return std::move(outputs[0].value);
}

inline float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  REQUIRE(a.size() == b.size());
  // Un NaN d'un seul côté est un écart infini ; std::max l'ignorerait sinon.
  float worst = 0.0F;
  for (std::size_t k = 0; k < a.size(); ++k) {
    if (std::isnan(a[k]) && std::isnan(b[k])) continue;
    const float diff = std::abs(a[k] - b[k]);
    worst = std::isnan(diff) ? std::numeric_limits<float>::infinity() : std::max(worst, diff);
  }
  return worst;
}

inline std::size_t count_ops(const Graph& graph, OpKind op) {
  return static_cast<std::size_t>(
      std::ranges::count_if(graph.nodes, [&](const Node& n) { return n.op == op; }));
}

}  // namespace ajp::gpu::ir::test
