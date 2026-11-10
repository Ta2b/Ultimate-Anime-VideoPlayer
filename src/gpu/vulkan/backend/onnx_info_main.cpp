// ajp_onnx_info : charge un modèle ONNX dans l'IR et imprime opérateurs,
// formes et mémoire, avant et après les passes. --run exécute l'interpréteur
// sur une entrée déterministe (et --dump écrit la première sortie en float32 brut),
// pour comparer à une autre implémentation. --vulkan décrit et exécute le graphe que le moteur
// Vulkan abaisse : le même prepare_graph, passes propres au moteur comprises (A44-10).
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
#include "gpu/ir/import.h"
#include "gpu/ir/interpreter.h"
#include "gpu/ir/memory_plan.h"
#include "gpu/ir/passes.h"
#include "gpu/ir/shape.h"
#include "gpu/vulkan/backend/net_build.h"
#include "gpu/vulkan/backend/net_program.h"
#include "kernels/ref/half.h"

namespace {

using namespace ajp::gpu::ir;

constexpr std::string_view kUsage =
    "usage : ajp_onnx_info <modèle.onnx> [--input nom=1x3x720x1280]... [--nodes] [--no-passes]\n"
    "                      [--vulkan] [--matrices] [--fp32] [--run] [--dump sortie.f32]\n"
    "  --vulkan   : le graphe du moteur Vulkan (prepare_graph), en float16 là où le modèle l'est\n"
    "  --matrices : avec --vulkan, comme un appareil à matrices coopératives (ConvTranspose réécrites)\n"
    "  --fp32     : avec --vulkan, tous les tenseurs en float\n";

struct Options {
  std::string model;
  std::vector<InputShape> shapes;
  bool nodes = false;
  bool passes = true;
  bool run = false;
  bool vulkan = false;
  bool matrices = false;
  bool fp32 = false;
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
    } else if (arg == "--vulkan") {
      options.vulkan = true;
    } else if (arg == "--matrices") {
      options.matrices = true;
    } else if (arg == "--fp32") {
      options.fp32 = true;
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
  if ((options.matrices || options.fp32) && !options.vulkan) return false;
  return !options.model.empty() && (options.passes || !options.vulkan);
}

// Entrée reproductible ailleurs : valeur k = partie fractionnaire de k φ, en double.
Value deterministic_input(const Tensor& tensor) {
  Value value = make_value(tensor.type, tensor.dims);
  for (std::size_t k = 0; k < value.f.size(); ++k) {
    double whole = 0.0;
    const auto f = static_cast<float>(std::modf(static_cast<double>(k) * 0.6180339887498949, &whole));
    value.f[k] = tensor.type == DataType::F16 ? ajp::kernels::ref::round_to_half(f) : f;
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
    const Value& first = outputs->front().value;
    if (first.type == DataType::I64) {
      std::cerr << std::format("--dump : sortie entière {} non prise en charge\n", outputs->front().name);
      return 1;
    }
    std::ofstream file(options.dump, std::ios::binary);
    const std::vector<float>& f = first.f;
    file.write(reinterpret_cast<const char*>(f.data()),
               static_cast<std::streamsize>(f.size() * sizeof(float)));
    if (!file) {
      std::cerr << std::format("écriture impossible : {}\n", options.dump);
      return 1;
    }
  }
  return 0;
}

// Les passes communes à tous les moteurs, celles de l'interpréteur.
std::expected<void, ajp::Refusal> generic_passes(Graph& graph) {
  // Comme le moteur Vulkan (net_build.cpp) : la grille des GridSample en float avant les passes.
  if (const int grid = promote_grid_chains(graph); grid > 0)
    std::cout << std::format("Grille des GridSample : {} tenseurs passés en float\n", grid);
  auto stats = optimize(graph);
  if (!stats) return std::unexpected(stats.error());
  std::cout << "Passes\n" << describe_stats(*stats);
  std::cout << std::format("Après les passes : {} nœuds\n  {}\n", graph.nodes.size(), describe_ops(graph));
  return {};
}

// Les passes du moteur Vulkan, les mêmes que prepare_graph après l'inférence des formes.
std::expected<void, ajp::Refusal> vulkan_passes(Graph& graph, const Options& options) {
  namespace net = ajp::gpu::vk::net;
  const net::Target target{.f16_storage = true, .matrix_units = options.matrices};
  if (auto done = net::engine_passes(graph, options.model, !options.fp32, target); !done) return done;
  std::cout << std::format("Après la préparation Vulkan ({}, {}) : {} nœuds\n  {}\n",
                           options.matrices ? "matrices coopératives" : "sans matrices",
                           options.fp32 ? "float" : "float16 du modèle", graph.nodes.size(),
                           describe_ops(graph));
  return {};
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
  std::cout << "Après l'inférence des formes\n" << describe_interface(*graph);

  if (options.vulkan) {
    if (auto done = vulkan_passes(*graph, options); !done) return refused(done.error());
  } else if (options.passes) {
    if (auto done = generic_passes(*graph); !done) return refused(done.error());
  }
  if (options.nodes) std::cout << "Nœuds\n" << describe_nodes(*graph);
  auto plan = plan_memory(*graph);
  if (!plan) return refused(plan.error());
  std::cout << "Mémoire\n" << describe_memory(*plan);
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
