// Les deux ArtCNN livrés (P6.7 : C4F32, et C4F16 pour les plus petites cartes), lus dans le dépôt
// (tools/upscale/models) : le graphe que le moteur Vulkan prépare, en float16 comme à la lecture, contre
// l'interpréteur sur le graphe importé tel quel, avec chaque noyau candidat de chaque couche. L'enveloppe RGB
// (Cast, luma, Resize de la couleur, Clip) et les Conv en auto_pad SAME suivies d'un DepthToSpace fondu y
// passent toutes. La référence ne tourne pas sur le graphe préparé : les Cast que fold_float_casts retire y
// laissent des opérandes de types mêlés, que les noyaux typés lisent mais que l'interpréteur refuse.
#include <array>
#include <cstddef>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "gpu/ir/graph.h"
#include "gpu/ir/import.h"
#include "gpu/ir/shape.h"
#include "gpu/vulkan/backend/net_build.h"
#include "gpu/vulkan/backend/net_lower.h"
#include "net_rig.h"

using namespace ajp::gpu;
using namespace ajp::gpu::vk::test;

namespace {

constexpr std::array<std::string_view, 2> kArtCnn{"2x_ArtCNN_C4F32_rgb_fp16.onnx",
                                                  "2x_ArtCNN_C4F16_rgb_fp16.onnx"};

void check_choices(NetRig& rig, const ir::Graph& graph, std::span<const ir::NamedValue> inputs,
                   const ir::Value& want, std::span<const vk::net::KernelChoice> choices,
                   const std::string& what) {
  const std::vector<ir::Value> got = run_on_device(rig, graph, inputs, choices);
  REQUIRE(got.size() == 1);
  require_close(got[0], want, kFp16, what);
}

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
    CHECK(graph.tensors[graph.outputs[0]].dims == ir::Dims{1, 3, 36, 52});

    ir::Graph imported = require_value(ir::load_onnx_file(onnx));
    const std::vector<ir::InputShape> shapes{{.name = in.name, .dims = shape}};
    require_ok(ir::infer_shapes(imported, shapes));
    // Une image entre 0 et 1, comme l'entrée R'G'B' du lecteur.
    const std::vector<ir::NamedValue> inputs{{in.name, test_values(in.type, shape, 41, 0.0F, 1.0F)}};
    const std::vector<ir::Value> want = run_reference(imported, inputs);
    REQUIRE(want.size() == 1);

    const std::vector<vk::net::KernelChoice> defaults =
        require_value(vk::net::default_choices(graph, rig.target));
    check_choices(rig, graph, inputs, want[0], defaults, std::string(file) + ", noyaux par défaut");
    for (std::size_t k = 0; k < graph.nodes.size(); ++k) {
      const auto options = require_value(vk::net::candidates(graph, k, rig.target));
      for (std::size_t c = 1; c < options.size(); ++c) {
        std::vector<vk::net::KernelChoice> choices = defaults;
        choices[k] = options[c];
        check_choices(rig, graph, inputs, want[0], choices,
                      std::string(file) + ", nœud " + graph.nodes[k].name + " : " + options[c].variant);
      }
    }
  }
}
