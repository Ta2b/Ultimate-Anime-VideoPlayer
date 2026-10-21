#include "app/play/motion_model.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <format>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "app/play/derived_model.h"
#include "kernels/spec/rife.h"
#include "prepare/model_source.h"
#include "prepare/onnx_outputs.h"
#include "prepare/onnx_scan.h"

namespace ajp::app::play {
namespace {

// Tenseurs du flot et du masque des exports de vs-mlrt (v1 et v2) et de vs-rife : le flot final
// est l'un de ces candidats selon la version (4.13, 4.18 et 4.22 lite : /Add_5).
constexpr std::array<std::string_view, 2> kFlowTensors{"/Add_7_output_0", "/Add_5_output_0"};
constexpr std::string_view kMaskTensor = "/Sigmoid_output_0";

std::vector<const prepare::OnnxNode*> consumers(const std::vector<prepare::OnnxNode>& nodes,
                                                std::string_view tensor) {
  std::vector<const prepare::OnnxNode*> found;
  for (const prepare::OnnxNode& node : nodes) {
    if (std::ranges::contains(node.inputs, tensor)) found.push_back(&node);
  }
  return found;
}

// Niveaux parcourus sous le Reshape pour trouver le GridSample : la grille s'écrit
// Transpose(Add(maillage, Mul(flot, pas))), avec un Cast vers fp32 dans les exports FP16.
constexpr int kWarpDepth = 8;

// Le flot final n'a qu'un consommateur, le Reshape du warp final, qui mène à son GridSample. Un flot
// intermédiaire part aussi vers le bloc suivant ; le même nom, dans d'autres versions, désigne une
// image (vers un Pad) ou un demi-flot (vers un Transpose). Les ensembles n'ont pas de flot final.
bool is_final_flow(const std::vector<prepare::OnnxNode>& nodes, std::string_view tensor) {
  const std::vector<const prepare::OnnxNode*> used = consumers(nodes, tensor);
  if (used.size() != 1 || used[0]->op_type != "Reshape") return false;
  std::vector<std::string_view> front(used[0]->outputs.begin(), used[0]->outputs.end());
  for (int depth = 0; depth < kWarpDepth && !front.empty(); ++depth) {
    std::vector<std::string_view> next;
    for (const std::string_view t : front) {
      for (const prepare::OnnxNode* node : consumers(nodes, t)) {
        if (node->op_type == "GridSample") return true;
        next.insert(next.end(), node->outputs.begin(), node->outputs.end());
      }
    }
    front = std::move(next);
  }
  return false;
}

std::expected<std::vector<std::byte>, Refusal> with_motion_outputs(std::span<const std::byte> model) {
  auto nodes = prepare::onnx_nodes(model);
  if (!nodes) return std::unexpected(std::move(nodes.error()));
  auto summary = prepare::scan_onnx(model);
  if (!summary) return std::unexpected(std::move(summary.error()));
  // Les sorties ont le type des calculs du graphe, celui de son entrée : FP16 pour les exports
  // convertis, FP32 sinon.
  const std::int32_t type = summary->inputs.empty() ? 10 : summary->inputs[0].elem_type;
  for (const std::string_view flow : kFlowTensors) {
    if (!is_final_flow(*nodes, flow)) continue;
    const std::vector<prepare::OnnxOutput> outputs{{.name = std::string(flow), .elem_type = type},
                                                   {.name = std::string(kMaskTensor), .elem_type = type}};
    return prepare::with_outputs(model, outputs);
  }
  return std::unexpected(Refusal{RefusalCode::FieldMissing, "pas de flot final reconnu (ensemble ?)"});
}

}  // namespace

std::expected<std::filesystem::path, Refusal> motion_model(const std::filesystem::path& rife_onnx,
                                                           const std::filesystem::path& dir) {
  return derived_model(rife_onnx, "mouvement", dir, with_motion_outputs);
}

std::expected<int, Refusal> rife_input_planes(const std::filesystem::path& rife_onnx) {
  auto raw = prepare::read_onnx_bytes(rife_onnx);
  if (!raw) return std::unexpected(std::move(raw.error()));
  const auto summary = prepare::scan_onnx(*raw);
  if (!summary) return std::unexpected(summary.error());
  const auto& inputs = summary->inputs;
  const std::int64_t planes = inputs.size() == 1 && inputs[0].dims.size() == 4 ? inputs[0].dims[1] : 0;
  if (!kernels::is_rife_input_planes(planes)) {
    return std::unexpected(Refusal{RefusalCode::InputMismatch,
                                   std::format("{} : entrée RIFE de {} plans, 11 (v1) ou 7 (v2) attendus",
                                               rife_onnx.filename().string(), planes)});
  }
  return static_cast<int>(planes);
}

int rife_padding_multiple(const std::filesystem::path& rife_onnx) {
  return rife_onnx.stem().string().starts_with("rife_v4.25_lite") ? kernels::kRifeAlignLite
                                                                  : kernels::kRifeAlign;
}

}  // namespace ajp::app::play
