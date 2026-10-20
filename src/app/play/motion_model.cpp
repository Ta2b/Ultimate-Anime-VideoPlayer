#include "app/play/motion_model.h"

#include <array>
#include <format>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

#include "app/play/derived_model.h"
#include "kernels/spec/rife.h"
#include "prepare/onnx_outputs.h"
#include "prepare/onnx_scan.h"

namespace ajp::app::play {
namespace {

// Tenseurs du flot et du masque, par export connu : RIFE 4.2x (vs-rife, ceux d'AnimeJaNai, que la
// variante du lecteur actuel .aji-motion-v3 sort aussi), puis RIFE 4.13, dont le flot final est
// produit plus tôt dans le graphe.
constexpr std::array<std::string_view, 2> kFlowTensors{"/Add_7_output_0", "/Add_5_output_0"};
constexpr std::string_view kMaskTensor = "/Sigmoid_output_0";

}  // namespace

std::expected<std::filesystem::path, Refusal> motion_model(const std::filesystem::path& rife_onnx,
                                                           const std::filesystem::path& dir) {
  return derived_model(
      rife_onnx, "mouvement", dir,
      [](std::span<const std::byte> model) -> std::expected<std::vector<std::byte>, Refusal> {
        std::expected<std::vector<std::byte>, Refusal> made;
        for (const std::string_view flow : kFlowTensors) {
          const std::vector<prepare::OnnxOutput> outputs{{.name = std::string(flow), .elem_type = 10},
                                                         {.name = std::string(kMaskTensor), .elem_type = 10}};
          made = prepare::with_outputs(model, outputs);
          if (made || made.error().code != RefusalCode::FieldMissing) return made;
        }
        return made;
      });
}

std::expected<int, Refusal> rife_input_planes(const std::filesystem::path& rife_onnx) {
  std::ifstream in(rife_onnx, std::ios::binary);
  if (!in) return std::unexpected(Refusal{RefusalCode::ModelUnreadable, rife_onnx.string()});
  const std::vector<char> raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  const auto summary = prepare::scan_onnx(std::as_bytes(std::span(raw)));
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

}  // namespace ajp::app::play
