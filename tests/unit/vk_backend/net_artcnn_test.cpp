// Les deux ArtCNN livrés (P6.7 : C4F32, et C4F16 pour les plus petites cartes), lus dans le dépôt
// (tools/upscale/models) : le graphe que le moteur Vulkan prépare, en float16 comme à la lecture, contre
// l'interpréteur, avec chaque noyau candidat de chaque couche. L'enveloppe RGB (Cast, luma, Resize de la
// couleur, Clip) et les Conv en auto_pad SAME suivies d'un DepthToSpace fondu y passent toutes.
#include <array>
#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "gpu/ir/graph.h"
#include "gpu/vulkan/backend/net_build.h"
#include "gpu/vulkan/backend/net_lower.h"
#include "net_rig.h"

using namespace ajp::gpu;
using namespace ajp::gpu::vk::test;

namespace {

constexpr std::array<std::string_view, 2> kArtCnn{"2x_ArtCNN_C4F32_rgb_fp16.onnx",
                                                  "2x_ArtCNN_C4F16_rgb_fp16.onnx"};

}  // namespace

TEST_CASE("Réseau : les ArtCNN livrés, du graphe préparé jusqu'à l'image ×2", "[vk_backend][net][graph]") {
  NetRig rig = make_net_rig();
  require_f16(rig);
  // Ni la hauteur ni la largeur ne sont multiples de 8 : les bords des tuiles de chaque Conv comptent.
  const ir::Dims shape{1, 3, 18, 26};
  for (const std::string_view file : kArtCnn) {
    const std::filesystem::path onnx = std::filesystem::path(AJP_ARTCNN_MODEL_DIR) / file;
    INFO(onnx.string());
    REQUIRE(std::filesystem::is_regular_file(onnx));
    const vk::net::BuildRequest request{
        .onnx = onnx, .model_id = std::string(file), .model_sha256 = "", .input_shape = shape, .fp16 = true};
    const ir::Graph graph = require_value(vk::net::prepare_graph(request, rig.target));
    REQUIRE(graph.inputs.size() == 1);
    REQUIRE(graph.outputs.size() == 1);
    const ir::Tensor& in = graph.tensors[graph.inputs[0]];
    const ir::Tensor& out = graph.tensors[graph.outputs[0]];
    CHECK(out.dims == ir::Dims{1, 3, 36, 52});
    // Une image entre 0 et 1, comme l'entrée R'G'B' du lecteur.
    const std::vector<ir::NamedValue> inputs{{in.name, test_values(in.type, shape, 41, 0.0F, 1.0F)}};
    const std::vector<vk::net::KernelChoice> defaults =
        require_value(vk::net::default_choices(graph, rig.target));
    check_graph(rig, graph, inputs, kFp16, defaults);
    for (std::size_t k = 0; k < graph.nodes.size(); ++k) {
      const auto options = require_value(vk::net::candidates(graph, k, rig.target));
      for (std::size_t c = 1; c < options.size(); ++c) {
        std::vector<vk::net::KernelChoice> choices = defaults;
        choices[k] = options[c];
        INFO("nœud " << k << " (" << graph.nodes[k].name << ") : " << options[c].variant);
        check_graph(rig, graph, inputs, kFp16, choices);
      }
    }
  }
}
