// Fuzz léger du décodeur et de toute la chaîne : octets aléatoires à graine
// fixe, troncatures et mutations d'un modèle valide. Le critère est l'absence
// de plantage (à lancer sous ASan/UBSan avec le preset linux-prefix-asan ; la CI
// ne construit pas encore src/gpu sous ASan) : chaque entrée finit acceptée ou refusée.
#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "gpu/ir/describe.h"
#include "gpu/ir/memory_plan.h"
#include "gpu/ir/passes.h"
#include "test_util.h"

using namespace ajp::gpu::ir;
using namespace ajp::gpu::ir::test;

namespace {

constexpr std::uint32_t kSeed = 20261018;
// Au-delà, on n'exécute pas l'interpréteur : une forme mutée peut être immense.
constexpr std::size_t kMaxRunElements = std::size_t{1} << 16;

// Un modèle qui touche la plupart des champs lus : attributs de chaque type,
// données brutes et typées, F16, nœud Constant, dimensions symboliques.
std::vector<std::byte> seed_model() {
  ModelBuilder m(17);
  m.input("x", {1, 4, -1, 6})
      .input("grid", {1, 6, 6, 2})
      .weight("w", {8, 4, 3, 3}, sequence(288, -0.3F, 0.3F, 1))
      .initializer(tensor_f16("b", {8}, sequence(8, -0.1F, 0.1F, 2)))
      .weight("slope", {8, 1, 1}, sequence(8, 0.1F, 0.2F, 3))
      .weight("scales", {4}, {1, 1, 0.5F, 0.5F})
      .ints("shape", {1, 2, 6, 6});
  ProtoWriter typed = tensor_header("k", onnx::kFloat, std::vector<std::int64_t>{1});
  typed.f32(4, 0.5F);
  m.initializer(typed)
      .node("Cast", {"b"}, {"b32"}, {attr_int("to", onnx::kFloat)})
      .node("Conv", {"x", "w", "b32"}, {"c"},
            {attr_ints("pads", {1, 1, 1, 1}), attr_ints("kernel_shape", {3, 3}), attr_int("group", 1)})
      .node("PRelu", {"c", "slope"}, {"p"})
      .node("DepthToSpace", {"p"}, {"d"}, {attr_int("blocksize", 2), attr_string("mode", "CRD")})
      .node("Resize", {"d", "", "scales"}, {"r"}, {attr_string("mode", "linear")})
      .node("Constant", {}, {"one"},
            {attr_tensor("value", tensor_i64("", {1}, std::vector<std::int64_t>{1}))})
      .node("Reshape", {"r", "shape"}, {"s"})
      .node("GridSample", {"s", "grid"}, {"g"}, {attr_int("align_corners", 0)})
      .node("Mul", {"g", "k"}, {"y"})
      .node("Shape", {"y"}, {"sy"})
      .node("Gather", {"sy", "one"}, {"ch"})
      .output("y")
      .output("ch");
  return m.bytes();
}

// Tout le chemin d'un modèle : rien ne doit lever ni lire hors des tampons.
// Rend l'étape atteinte : 0 refus au chargement, 1 aux formes, 2 aux passes, 3 chaîne complète.
int exercise(const std::vector<std::byte>& bytes) {
  auto graph = load_onnx(bytes);
  if (!graph) return 0;
  std::vector<InputShape> shapes;
  for (const TensorId id : graph->inputs) {
    const Tensor& t = graph->tensors[id];
    if (t.known) continue;
    Dims dims = t.dims;
    for (std::int64_t& d : dims) d = d < 0 ? 6 : d;
    shapes.push_back({.name = t.name, .dims = dims});
  }
  if (!infer_shapes(*graph, shapes)) return 1;
  std::size_t total = 0;
  for (const Tensor& t : graph->tensors) total += t.known ? element_count(t.dims) : 0;

  std::vector<NamedValue> inputs;
  bool runnable = total <= kMaxRunElements;
  for (const TensorId id : graph->inputs) {
    const Tensor& t = graph->tensors[id];
    if (t.type == DataType::I64) {
      runnable = false;
      break;
    }
    inputs.push_back(
        {.name = t.name, .value = floats(t.dims, sequence(element_count(t.dims), -1, 1), t.type)});
  }
  const bool ran = runnable && run(*graph, inputs).has_value();

  // Un graphe que l'interpréteur exécute se laisse optimiser, et s'exécute encore après.
  auto stats = optimize(*graph);
  if (ran) CHECK(stats.has_value());
  if (!stats) return 2;
  (void)describe_nodes(*graph);
  // Après les passes, le graphe est valide et ses formes connues : le plan doit aboutir.
  auto plan = plan_memory(*graph);
  CHECK(plan.has_value());
  if (plan) (void)describe_memory(*plan);
  // L'inverse ne tient pas : le code mort retiré peut être ce qui échouait.
  const bool ran_after = runnable && run(*graph, inputs).has_value();
  if (ran) CHECK(ran_after);
  return 3;
}

}  // namespace

TEST_CASE("fuzz : le modèle de départ passe toute la chaîne", "[gpu_ir][fuzz]") {
  auto graph = load_onnx(seed_model());
  REQUIRE(graph);
  REQUIRE(infer_shapes(*graph, std::vector<InputShape>{{.name = "x", .dims = {1, 4, 6, 6}}}));
  REQUIRE(optimize(*graph));
  CHECK(graph->tensors[graph->outputs[0]].dims == Dims{1, 2, 6, 6});
}

TEST_CASE("fuzz : octets aléatoires", "[gpu_ir][fuzz]") {
  std::mt19937 rng(kSeed);
  std::uniform_int_distribution<int> byte(0, 255);
  std::uniform_int_distribution<std::size_t> length(0, 384);
  for (int iteration = 0; iteration < 4000; ++iteration) {
    std::vector<std::byte> bytes(length(rng));
    for (std::byte& b : bytes) b = static_cast<std::byte>(byte(rng));
    // Un en-tête de graphe valide de temps en temps, pour descendre plus loin dans le décodeur.
    if (iteration % 2 == 0 && bytes.size() > 3) {
      bytes[0] = std::byte{0x3A};  // champ 7 (graph), longueur
      bytes[1] = static_cast<std::byte>(std::min<std::size_t>(bytes.size() - 2, 127));
    }
    CHECK(exercise(bytes) >= 0);
  }
}

TEST_CASE("fuzz : chaque troncature du modèle de départ", "[gpu_ir][fuzz]") {
  const std::vector<std::byte> model = seed_model();
  int complete = 0;
  for (std::size_t size = 0; size < model.size(); ++size) {
    complete += exercise(std::vector<std::byte>(model.begin(),
                                                model.begin() + static_cast<std::ptrdiff_t>(size))) == 3
                    ? 1
                    : 0;
  }
  // Toute coupe stricte tronque un message ou perd l'opset, écrit en dernier : rien ne passe.
  CHECK(complete == 0);
  CHECK(exercise(model) == 3);
}

TEST_CASE("fuzz : mutations du modèle de départ", "[gpu_ir][fuzz]") {
  const std::vector<std::byte> model = seed_model();
  std::mt19937 rng(kSeed + 1);
  std::uniform_int_distribution<std::size_t> position(0, model.size() - 1);
  std::uniform_int_distribution<int> byte(0, 255);
  std::uniform_int_distribution<int> kind(0, 3);
  std::array<int, 4> reached{};
  for (int iteration = 0; iteration < 6000; ++iteration) {
    std::vector<std::byte> bytes = model;
    const int edits = 1 + (iteration % 3);
    for (int e = 0; e < edits; ++e) {
      const std::size_t at = position(rng) % bytes.size();
      switch (kind(rng)) {
        case 0:  // un bit inversé
          bytes[at] ^= static_cast<std::byte>(1U << static_cast<unsigned>(byte(rng) % 8));
          break;
        case 1:  // un octet remplacé
          bytes[at] = static_cast<std::byte>(byte(rng));
          break;
        case 2:  // un octet inséré
          bytes.insert(bytes.begin() + static_cast<std::ptrdiff_t>(at), static_cast<std::byte>(byte(rng)));
          break;
        default:  // un octet retiré
          if (bytes.size() > 1) bytes.erase(bytes.begin() + static_cast<std::ptrdiff_t>(at));
          break;
      }
    }
    ++reached[static_cast<std::size_t>(exercise(bytes))];
  }
  // Le fuzz descend jusqu'aux passes et à l'interpréteur, pas seulement jusqu'au décodeur.
  INFO("chargement " << reached[0] << ", formes " << reached[1] << ", passes " << reached[2] << ", complet "
                     << reached[3]);
  CHECK(reached[3] > 600);
  CHECK(reached[0] > 600);
}
