// ajp_onnx_info : charge un modèle ONNX dans l'IR et imprime opérateurs,
// formes et mémoire, avant et après les passes. --run exécute l'interpréteur
// sur une entrée déterministe (et --dump écrit la première sortie en float32 brut),
// pour comparer à une autre implémentation.
#include <charconv>
#include <cmath>
#include <exception>
#include <format>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "gpu/ir/describe.h"
#include "gpu/ir/graph.h"
#include "gpu/ir/half.h"
#include "gpu/ir/import.h"
#include "gpu/ir/interpreter.h"
#include "gpu/ir/memory_plan.h"
#include "gpu/ir/passes.h"
#include "gpu/ir/shape.h"

namespace {

using namespace ajp::gpu::ir;

constexpr std::string_view kUsage =
    "usage : ajp_onnx_info <modèle.onnx> [--input nom=1x3x720x1280]... [--nodes] [--no-passes]\n"
    "                      [--run] [--dump sortie.f32]\n";

struct Options {
  std::string model;
  std::vector<InputShape> shapes;
  bool nodes = false;
  bool passes = true;
  bool run = false;
  std::string dump;
};

bool parse_dims(std::string_view text, Dims& dims) {
  while (!text.empty()) {
    const std::size_t cut = text.find('x');
    const std::string_view part = text.substr(0, cut);
    std::int64_t value = 0;
    const auto [end, error] = std::from_chars(part.data(), part.data() + part.size(), value);
    if (error != std::errc{} || end != part.data() + part.size()) return false;
    dims.push_back(value);
    text = cut == std::string_view::npos ? std::string_view{} : text.substr(cut + 1);
  }
  return !dims.empty();
}

bool parse_args(std::span<char*> args, Options& options) {
  for (std::size_t k = 1; k < args.size(); ++k) {
    const std::string_view arg = args[k];
    if (arg == "--nodes") {
      options.nodes = true;
    } else if (arg == "--no-passes") {
      options.passes = false;
    } else if (arg == "--run") {
      options.run = true;
    } else if ((arg == "--input" || arg == "--dump") && k + 1 < args.size()) {
      const std::string_view value = args[++k];
      if (arg == "--dump") {
        options.dump = value;
        options.run = true;
        continue;
      }
      const std::size_t eq = value.find('=');
      InputShape shape{.name = std::string(value.substr(0, eq))};
      if (eq == std::string_view::npos || !parse_dims(value.substr(eq + 1), shape.dims)) return false;
      options.shapes.push_back(std::move(shape));
    } else if (options.model.empty() && !arg.starts_with("--")) {
      options.model = arg;
    } else {
      return false;
    }
  }
  return !options.model.empty();
}

// Entrée reproductible ailleurs : valeur k = partie fractionnaire de k φ, en double.
Value deterministic_input(const Tensor& tensor) {
  Value value = make_value(tensor.type, tensor.dims);
  for (std::size_t k = 0; k < value.f.size(); ++k) {
    double whole = 0.0;
    const auto f = static_cast<float>(std::modf(static_cast<double>(k) * 0.6180339887498949, &whole));
    value.f[k] = tensor.type == DataType::F16 ? round_to_half(f) : f;
  }
  return value;
}

int refused(const ajp::Refusal& refusal) {
  std::cerr << std::format("refus {} : {}\n", ajp::name(refusal.code), refusal.detail);
  return 1;
}

int run_graph(const Graph& graph, const Options& options) {
  std::vector<NamedValue> inputs;
  for (const TensorId id : graph.inputs) {
    const Tensor& tensor = graph.tensors[id];
    if (tensor.type == DataType::I64) {
      std::cerr << std::format("--run : entrée entière {} non prise en charge\n", tensor.name);
      return 1;
    }
    inputs.push_back({.name = tensor.name, .value = deterministic_input(tensor)});
  }
  auto outputs = run(graph, inputs);
  if (!outputs) return refused(outputs.error());
  std::cout << "Exécution (interpréteur FP32)\n";
  for (const NamedValue& out : *outputs) {
    const std::vector<float>& f = out.value.f;
    double sum = 0.0;
    float lo = f.empty() ? 0.0F : f[0];
    float hi = lo;
    for (const float v : f) {
      sum += v;
      lo = std::min(lo, v);
      hi = std::max(hi, v);
    }
    std::cout << std::format("  {} {} : min {:.6f}, max {:.6f}, moyenne {:.6f}\n", out.name,
                             format_dims(out.value.dims), lo, hi,
                             f.empty() ? 0.0 : sum / static_cast<double>(f.size()));
  }
  if (!options.dump.empty() && !outputs->empty()) {
    std::ofstream file(options.dump, std::ios::binary);
    const std::vector<float>& f = outputs->front().value.f;
    file.write(reinterpret_cast<const char*>(f.data()),
               static_cast<std::streamsize>(f.size() * sizeof(float)));
    if (!file) {
      std::cerr << std::format("écriture impossible : {}\n", options.dump);
      return 1;
    }
  }
  return 0;
}

int report(const Options& options) {
  auto graph = load_onnx_file(options.model);
  if (!graph) return refused(graph.error());
  std::cout << std::format("Modèle {} — opset {}, {} tenseurs\n", options.model, graph->opset,
                           graph->tensors.size());
  std::cout << describe_interface(*graph);
  std::cout << std::format("Avant les passes : {} nœuds\n  {}\n", graph->nodes.size(), describe_ops(*graph));
  // Les formes viennent après l'interface : une entrée dynamique s'y lit avant le refus.
  if (auto shapes = infer_shapes(*graph, options.shapes); !shapes) return refused(shapes.error());

  if (options.passes) {
    auto stats = optimize(*graph);
    if (!stats) return refused(stats.error());
    std::cout << "Passes\n" << describe_stats(*stats);
    std::cout << std::format("Après les passes : {} nœuds\n  {}\n", graph->nodes.size(),
                             describe_ops(*graph));
  }
  if (options.nodes) std::cout << "Nœuds\n" << describe_nodes(*graph);
  std::cout << "Mémoire\n" << describe_memory(plan_memory(*graph));
  return options.run ? run_graph(*graph, options) : 0;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!parse_args(std::span(argv, static_cast<std::size_t>(argc)), options)) {
    std::cerr << kUsage;
    return 2;
  }
  // Rien ne lève dans l'IR ; seul un manque de mémoire peut encore arriver ici.
  try {
    return report(options);
  } catch (const std::exception& error) {
    std::cerr << std::format("erreur : {}\n", error.what());
    return 1;
  }
}
