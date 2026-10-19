// Chaîne complète d'un moteur : prepare::TrtexecBuilder lance le vrai trtexec
// 11.3 sur un ONNX généré ici, le runtime vérifie puis exécute le moteur, et
// la sortie se compare à l'interpréteur gpu_ir. Valide la ligne de commande de
// chaque recette et les repères de progression sur la sortie réelle de trtexec.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "gpu/backend.h"
#include "gpu/cuda/trt/cuda_objects.h"
#include "prepare/builder.h"
#include "prepare/sha256.h"
#include "support.h"

namespace fs = std::filesystem;
namespace trt = ajp::gpu::trt;
namespace ir = ajp::gpu::ir;
namespace prepare = ajp::prepare;
using trt::test::must;

namespace {

#ifdef AJP_TRTEXEC
constexpr const char* kTrtexec = AJP_TRTEXEC;
#else
constexpr const char* kTrtexec = "";
#endif

// Mini-SPAN : Conv 3x3 vers 12 canaux puis DepthToSpace, FP16 de bout en bout,
// H et W dynamiques comme les modèles AnimeJaNai.
ir::test::ModelBuilder upscaler() {
  ir::test::ModelBuilder m;
  m.input("input", {1, 3, -1, -1}, ir::onnx::kFloat16)
      .initializer(ir::test::tensor_f16("w", {12, 3, 3, 3}, ir::test::sequence(12 * 27, -0.3F, 0.3F, 5)))
      .initializer(ir::test::tensor_f16("b", {12}, ir::test::sequence(12, -0.1F, 0.1F, 6)))
      .node("Conv", {"input", "w", "b"}, {"c"}, {ir::test::attr_ints("pads", {1, 1, 1, 1})})
      .node("DepthToSpace", {"c"}, {"output"},
            {ir::test::attr_int("blocksize", 2), ir::test::attr_string("mode", "CRD")})
      .output("output", ir::onnx::kFloat16);
  return m;
}

// Mini-RIFE : entrée FP16, îlot FP32 autour du GridSample (Cast, flot par Conv
// 1x1 et Tanh, GridSample bilinéaire align_corners en bord), sortie FP16.
ir::test::ModelBuilder warp() {
  ir::test::ModelBuilder m;
  m.input("input", {1, 3, -1, -1}, ir::onnx::kFloat16)
      .weight("w", {2, 3, 1, 1}, ir::test::sequence(6, -1.5F, 1.5F, 9))
      .node("Cast", {"input"}, {"x"}, {ir::test::attr_int("to", ir::onnx::kFloat)})
      .node("Conv", {"x", "w"}, {"flow"})
      .node("Tanh", {"flow"}, {"t"})
      .node("Transpose", {"t"}, {"grid"}, {ir::test::attr_ints("perm", {0, 2, 3, 1})})
      .node("GridSample", {"x", "grid"}, {"s"},
            {ir::test::attr_int("align_corners", 1), ir::test::attr_string("mode", "bilinear"),
             ir::test::attr_string("padding_mode", "border")})
      .node("Cast", {"s"}, {"output"}, {ir::test::attr_int("to", ir::onnx::kFloat16)})
      .output("output", ir::onnx::kFloat16);
  return m;
}

struct Built {
  fs::path engine;
  std::vector<std::string> stages;
  std::vector<float> fractions;
};

// Écrit le modèle dans un dossier propre au test et le construit par trtexec.
Built build(const ir::test::ModelBuilder& model, const std::string& id,
            const prepare::TrtexecBuilder& builder, const fs::path& folder) {
  const fs::path onnx = folder / (id + ".onnx");
  const std::vector<std::byte> bytes = model.bytes();
  {
    std::ofstream out(onnx, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  }
  const ajp::gpu::ModelSpec spec{.id = id, .onnx_path = onnx, .sha256 = must(prepare::sha256_file(onnx))};
  const ajp::gpu::Shape shape{.dims = {1, 3, 16, 24}};
  Built built;
  ajp::gpu::Progress progress{.report = [&](float fraction, const std::string& stage) {
    built.fractions.push_back(fraction);
    built.stages.push_back(stage);
  }};
  built.engine = must(builder.build(spec, {.min = shape, .opt = shape, .max = shape}, progress)).path;
  return built;
}

std::vector<float> run(const trt::TrtRuntime& runtime, const fs::path& engine_file,
                       const std::vector<float>& input) {
  trt::TrtEngine engine = must(runtime.load_file(engine_file));
  const trt::CudaStream stream = must(trt::CudaStream::create());
  const std::vector<std::byte> host = trt::test::to_bytes(trt::DataType::F16, input);
  const trt::DeviceBuffer in = must(trt::DeviceBuffer::allocate(host.size()));
  const trt::Dims out_dims = must(engine.shape("output"));
  const trt::DeviceBuffer out =
      must(trt::DeviceBuffer::allocate(must(trt::byte_count(trt::DataType::F16, out_dims))));
  must(trt::test::upload(in.data(), host));
  must(engine.bind("input", in.data()));
  must(engine.bind("output", out.data()));
  must(engine.enqueue(stream.get()));
  must(stream.synchronize());
  return trt::test::to_floats(trt::DataType::F16, must(trt::test::download(out.data(), out.size())));
}

std::vector<float> reference(const ir::test::ModelBuilder& model, const std::vector<float>& input) {
  const ir::Dims dims{1, 3, 16, 24};
  const ir::Graph graph = ir::test::build(model, {{.name = "input", .dims = dims}});
  return ir::test::execute1(graph,
                            {{.name = "input", .value = ir::test::floats(dims, input, ir::DataType::F16)}})
      .f;
}

bool has(const std::vector<std::string>& stages, const std::string& stage) {
  return std::ranges::find(stages, stage) != stages.end();
}

}  // namespace

TEST_CASE("trtexec 11.3 construit SPAN et RIFE miniatures, le runtime les exécute",
          "[cuda_trt][cuda][trtexec]") {
  trt::test::skip_without_cuda();
  if (std::string(kTrtexec).empty()) SKIP("trtexec introuvable : AJP_TRTEXEC vide à la configuration");
  const trt::TrtRuntime runtime = trt::test::make_runtime();
  const trt::CudaDeviceInfo device = must(trt::current_device());
  const trt::LibraryVersion version = runtime.version();
  const fs::path folder = fs::temp_directory_path() / "ajp_cuda_trt_trtexec_test";
  fs::remove_all(folder);
  fs::create_directories(folder);
  const prepare::TrtexecBuilder builder({.trtexec = kTrtexec,
                                         .cache_dir = folder,
                                         .runtime = {.major = version.major,
                                                     .minor = version.minor,
                                                     .patch = version.patch,
                                                     .headers_build = version.headers_build,
                                                     .cuda_runtime = device.runtime_version,
                                                     .cuda_driver = device.driver_version},
                                         .device = {.name = device.name,
                                                    .sm_major = device.sm_major,
                                                    .sm_minor = device.sm_minor,
                                                    .uuid = device.uuid}},
                                        [&](const fs::path& engine) -> std::expected<void, ajp::Refusal> {
                                          auto loaded = runtime.load_file(engine);
                                          if (!loaded) return std::unexpected(loaded.error());
                                          return {};
                                        });
  std::vector<float> input = ir::test::sequence(3 * 16 * 24, 0.0F, 1.0F, 11);
  input = trt::test::to_floats(trt::DataType::F16, trt::test::to_bytes(trt::DataType::F16, input));

  for (const auto& [id, model] :
       {std::pair{std::string("mini-span"), upscaler()}, std::pair{std::string("mini-rife"), warp()}}) {
    INFO(id);
    const Built built = build(model, id, builder, folder);
    // Les repères de trtexec 11.3 communs à toute construction sont reconnus,
    // dans l'ordre, jusqu'au bout ; la seconde passe dépend du réseau.
    CHECK(std::ranges::is_sorted(built.fractions));
    for (const char* stage :
         {"Démarrage de trtexec", "Lecture du modèle", "Choix des tactiques", "Plan mémoire",
          "Première passe du constructeur terminée", "Moteur construit"}) {
      INFO(stage);
      CHECK(has(built.stages, stage));
    }
    REQUIRE(!built.stages.empty());
    CHECK(built.stages.back() == "Terminé");
    CHECK(fs::exists(prepare::timing_cache_for(built.engine)));
    CHECK(ir::test::max_abs_diff(run(runtime, built.engine, input), reference(model, input)) <= 4e-3F);
    // La seconde demande reprend le moteur publié sans relancer trtexec.
    const Built again = build(model, id, builder, folder);
    CHECK(again.engine == built.engine);
    CHECK(again.stages.back() == "Moteur en cache");
  }
  fs::remove_all(folder);
}
