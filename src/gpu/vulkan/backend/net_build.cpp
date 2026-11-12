#include "gpu/vulkan/backend/net_build.h"

#include <algorithm>
#include <chrono>
#include <format>
#include <fstream>
#include <set>
#include <system_error>

#include "ajp_vk_backend_spirv.h"
#include "core/file_sync.h"
#include "core/utf8_path.h"
#include "gpu/ir/graph.h"
#include "gpu/ir/import.h"
#include "gpu/ir/ops.h"
#include "gpu/ir/passes.h"
#include "gpu/ir/shape.h"
#include "gpu/vulkan/backend/net_hoist.h"
#include "gpu/vulkan/backend/net_lower.h"

namespace ajp::gpu::vk::net {
namespace {

// Chaque sortie du graphe vient d'un nœud : une entrée ou une constante rendue telle quelle
// n'a pas de noyau.
std::expected<void, Refusal> check_interface(const ir::Graph& graph) {
  const std::vector<std::size_t> producer = ir::producers(graph);
  for (const ir::TensorId id : graph.outputs) {
    if (producer[id] == ir::kNoNode) {
      return refuse(RefusalCode::GraphInvalid,
                    std::format("sortie {} : aucun nœud ne la calcule", graph.tensors[id].name));
    }
  }
  if (graph.nodes.empty()) return refuse(RefusalCode::GraphInvalid, "graphe sans nœud");
  return {};
}

}  // namespace

std::expected<void, Refusal> engine_passes(ir::Graph& graph, const std::string& model_id, bool fp16,
                                           const Target& target) {
  // La grille des GridSample en float (DistilDRBA des premiers exports : A44-01), avant que le
  // pliage des constantes n'arrondisse sa base à travers son Cast.
  (void)ir::promote_grid_chains(graph);
  // Les ConvTranspose deviennent des Conv là où celles-ci passent par les matrices coopératives
  // (au plus 16 tuiles de 16 canaux, net_lower_conv.cpp).
  const ir::OptimizeOptions options{.transpose_conv_channels = target.matrix_units ? 256 : 0,
                                    .transpose_phases = target.matrix_units};
  if (auto ok = ir::optimize(graph, options); !ok) return std::unexpected(ok.error());
  const bool has_f16 =
      std::ranges::any_of(graph.tensors, [](const ir::Tensor& t) { return t.type == ir::DataType::F16; });
  if (!fp16) {
    // Tout en float : une constante float16 garde ses valeurs (déjà représentables).
    for (ir::Tensor& t : graph.tensors) {
      if (t.type == ir::DataType::F16) t.type = ir::DataType::F32;
      if (t.value.type == ir::DataType::F16) t.value.type = ir::DataType::F32;
    }
  } else if (has_f16 && !target.f16_storage) {
    return refuse(RefusalCode::FeatureUnavailable,
                  std::format("{} en float16 : l'appareil n'a pas le stockage 16 bits", model_id));
  }
  // Les noyaux lisent et écrivent chaque tenseur dans son type : les Cast qu'ils feraient
  // d'eux-mêmes disparaissent, ceux qu'un modèle FP16 garde autour de Resize par exemple.
  if (ir::fold_float_casts(graph) > 0) (void)ir::eliminate_dead_code(graph);
  // Les réductions remontées devant les Concat et les Split qui les précèdent (net_hoist.h).
  if (hoist_downscales(graph) > 0) (void)ir::eliminate_dead_code(graph);
  return check_interface(graph);
}

std::expected<ir::Graph, Refusal> prepare_graph(const BuildRequest& request, const Target& target) {
  auto graph = ir::load_onnx_file(request.onnx);
  if (!graph) return std::unexpected(graph.error());
  if (graph->inputs.empty())
    return refuse(RefusalCode::InputMismatch, std::format("{} : aucune entrée", request.model_id));
  // Une entrée prend la forme demandée ; plusieurs (DistilDRBA) gardent celles du modèle,
  // que le lecteur fige avant la construction (derived_model.h).
  std::vector<ir::InputShape> shapes;
  for (const ir::TensorId id : graph->inputs) {
    const ir::Tensor& input = graph->tensors[id];
    if (graph->inputs.size() == 1) {
      shapes.push_back({.name = input.name, .dims = request.input_shape});
      break;
    }
    if (std::ranges::any_of(input.dims, [](std::int64_t d) { return d < 0; })) {
      return refuse(RefusalCode::InputMismatch,
                    std::format("{} : entrée {} aux dimensions libres ({}), un réseau à plusieurs entrées "
                                "doit être figé",
                                request.model_id, input.name, ir::format_dims(input.dims)));
    }
    shapes.push_back({.name = input.name, .dims = input.dims});
  }
  if (auto ok = ir::infer_shapes(*graph, shapes); !ok) return std::unexpected(ok.error());
  if (auto ok = engine_passes(*graph, request.model_id, request.fp16, target); !ok)
    return std::unexpected(ok.error());
  return graph;
}

std::vector<ShaderHash> shader_hashes(std::span<const Dispatch> dispatches) {
  std::set<std::string> names;
  for (const Dispatch& d : dispatches) names.insert(d.shader);
  std::vector<ShaderHash> out;
  out.reserve(names.size());
  for (const std::string& name : names)
    out.push_back({.shader = name, .hash = spirv_hash(spirv::words(name))});
  return out;
}

std::expected<ArtifactData, Refusal> build_network(const BuildRequest& request, const DeviceProfile& profile,
                                                   Autotuner* tuner, const BuildReport& report,
                                                   const std::stop_token& stop) {
  const auto progress = [&](float fraction, const std::string& stage) {
    if (report) report(fraction, stage);
  };
  progress(0.0F, "import");
  auto graph = prepare_graph(request, profile.target);
  if (!graph) return std::unexpected(graph.error());
  if (stop.stop_requested()) return refuse(RefusalCode::Cancelled, "construction arrêtée");

  const std::string scope = std::format("{}:{}:{}:{}", request.model_id, request.model_sha256,
                                        ir::format_dims(request.input_shape), request.fp16 ? "fp16" : "fp32");
  std::vector<KernelChoice> choices;
  choices.reserve(graph->nodes.size());
  for (std::size_t k = 0; k < graph->nodes.size(); ++k) {
    if (stop.stop_requested()) return refuse(RefusalCode::Cancelled, "construction arrêtée");
    auto options = candidates(*graph, k, profile.target);
    if (!options) return std::unexpected(options.error());
    if (tuner == nullptr || options->size() == 1) {
      choices.push_back(options->front());
    } else {
      auto best = tuner->choose(*graph, k, *options, scope, stop);
      if (!best) return std::unexpected(best.error());
      choices.push_back(std::move(*best));
    }
    progress(0.05F + (0.9F * static_cast<float>(k + 1) / static_cast<float>(graph->nodes.size())),
             std::format("noyaux : {} / {}", k + 1, graph->nodes.size()));
  }
  auto program = lower(*graph, choices, profile.target);
  if (!program) return std::unexpected(program.error());
  if (tuner != nullptr) {
    if (auto ok = tuner->save(); !ok) return std::unexpected(ok.error());
  }
  ArtifactData data{.lowering = kLoweringVersion,
                    .model_id = request.model_id,
                    .model_sha256 = request.model_sha256,
                    .device_key = profile.key,
                    .fp16 = request.fp16,
                    .graph = std::move(*graph),
                    .choices = std::move(choices),
                    .dispatches = std::move(program->dispatches),
                    .shaders = {}};
  data.shaders = shader_hashes(data.dispatches);
  progress(1.0F, "abaissé");
  return data;
}

std::expected<void, Refusal> check_artifact(const ArtifactData& data, const DeviceProfile& profile) {
  if (data.lowering != kLoweringVersion) {
    return refuse(RefusalCode::FormatVersion,
                  std::format("artefact d'abaissement {}, {} attendu", data.lowering, kLoweringVersion));
  }
  if (data.device_key != profile.key) {
    return refuse(RefusalCode::DeviceMismatch,
                  std::format("artefact construit pour {}, appareil {}", data.device_key, profile.key));
  }
  for (const ShaderHash& h : data.shaders) {
    if (spirv_hash(spirv::words(h.shader)) != h.hash) {
      return refuse(RefusalCode::FormatVersion,
                    std::format("shader {} changé depuis la construction", h.shader));
    }
  }
  if (auto ok = ir::validate(data.graph); !ok) return std::unexpected(ok.error());
  if (data.choices.size() != data.graph.nodes.size()) {
    return refuse(RefusalCode::FormatVersion,
                  std::format("{} choix pour {} nœuds", data.choices.size(), data.graph.nodes.size()));
  }
  for (std::size_t k = 0; k < data.choices.size(); ++k) {
    auto options = candidates(data.graph, k, profile.target);
    if (!options) return std::unexpected(options.error());
    if (std::ranges::find(*options, data.choices[k]) == options->end()) {
      return refuse(RefusalCode::FormatVersion,
                    std::format("nœud {} : variante {} plus proposée", k, data.choices[k].variant));
    }
  }
  return {};
}

std::expected<Program, Refusal> verify_artifact(const ArtifactData& data, const DeviceProfile& profile) {
  if (auto ok = check_artifact(data, profile); !ok) return std::unexpected(ok.error());
  auto program = lower(data.graph, data.choices, profile.target);
  if (!program) return std::unexpected(program.error());
  if (program->dispatches != data.dispatches || shader_hashes(program->dispatches) != data.shaders) {
    return refuse(RefusalCode::FormatVersion,
                  "le programme ne se refait pas à l'identique : artefact à reconstruire");
  }
  return program;
}

std::uint64_t artifact_key(const BuildRequest& request, const DeviceProfile& profile) {
  const std::string text = std::format("{}|{}|{}|{}|{}|{}|{}", request.model_id, request.model_sha256,
                                       ir::format_dims(request.input_shape), request.fp16, profile.key,
                                       kLoweringVersion, kArtifactVersion);
  return fnv1a(std::as_bytes(std::span(text)));
}

std::expected<void, Refusal> write_file(const std::filesystem::path& path, std::span<const std::byte> bytes) {
  // L'écriture atomique du projet : temporaire au nom propre à cet écrivain (le lecteur et un export qui
  // compilent le même réseau ne mêlent pas leurs octets), écrit sur le disque avant le renommage (A45-07).
  if (auto done = core::write_atomic(path, bytes); !done)
    return refuse(RefusalCode::CacheIo, done.error().detail);
  return {};
}

int purge_stale_temporaries(const std::filesystem::path& folder, std::chrono::seconds older_than) {
  std::error_code error;
  int removed = 0;
  const auto limit = std::filesystem::file_time_type::clock::now() - older_than;
  for (std::filesystem::directory_iterator it(folder, error), end; !error && it != end; it.increment(error)) {
    if (!it->is_regular_file(error)) continue;
    const std::string name = core::to_utf8(it->path().filename());
    // « .<nom>.<instant>-<fil>.tmp » de core::write_atomic, « <nom>.tmp-<n>-<n> » des versions précédentes.
    const bool temporary = (name.starts_with('.') && name.ends_with(".tmp")) || name.contains(".tmp-");
    if (!temporary) continue;
    const auto written = it->last_write_time(error);
    if (error || written > limit) continue;  // récent : un autre écrivain y travaille peut-être
    if (std::filesystem::remove(it->path(), error)) ++removed;
  }
  return removed;
}

std::expected<std::vector<std::byte>, Refusal> read_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) return refuse(RefusalCode::CacheIo, std::format("{} illisible", core::to_utf8(path)));
  const std::streamsize size = in.tellg();
  in.seekg(0);
  std::vector<std::byte> bytes(static_cast<std::size_t>(std::max<std::streamsize>(size, 0)));
  in.read(reinterpret_cast<char*>(bytes.data()), size);
  if (!in) return refuse(RefusalCode::CacheIo, std::format("{} : lecture incomplète", core::to_utf8(path)));
  return bytes;
}

}  // namespace ajp::gpu::vk::net
