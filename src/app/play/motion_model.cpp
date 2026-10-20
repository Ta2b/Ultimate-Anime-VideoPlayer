#include "app/play/motion_model.h"

#include <vector>

#include "app/play/derived_model.h"
#include "prepare/onnx_outputs.h"

namespace ajp::app::play {
namespace {

// Tenseurs du flot et du masque dans les exports RIFE 4.2x (vs-rife, ceux d'AnimeJaNai) : ceux que
// la variante du lecteur actuel (.aji-motion-v3) sort aussi.
std::vector<prepare::OnnxOutput> motion_outputs() {
  return {{.name = "/Add_7_output_0", .elem_type = 10}, {.name = "/Sigmoid_output_0", .elem_type = 10}};
}

}  // namespace

std::expected<std::filesystem::path, Refusal> motion_model(const std::filesystem::path& rife_onnx,
                                                           const std::filesystem::path& dir) {
  return derived_model(rife_onnx, "mouvement", dir, [](std::span<const std::byte> model) {
    return prepare::with_outputs(model, motion_outputs());
  });
}

}  // namespace ajp::app::play
