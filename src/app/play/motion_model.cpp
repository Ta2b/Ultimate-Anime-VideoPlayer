#include "app/play/motion_model.h"

#include <array>
#include <string>
#include <string_view>
#include <vector>

#include "app/play/derived_model.h"
#include "prepare/onnx_outputs.h"

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

}  // namespace ajp::app::play
