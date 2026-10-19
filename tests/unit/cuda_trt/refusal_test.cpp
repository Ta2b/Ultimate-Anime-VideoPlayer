// Refus nommés du runtime TensorRT : moteur abîmé, forme hors profil, tenseur
// inconnu ou non lié. Aucun ne doit planter ni laisser le moteur inutilisable.
//
// TensorRT ne vérifie que l'en-tête et la taille d'un moteur : des poids
// inversés au milieu du fichier se chargent sans erreur. L'intégrité d'un
// moteur publié repose donc sur le SHA-256 du cache de prepare.
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "gpu/cuda/trt/cuda_objects.h"
#include "support.h"

namespace trt = ajp::gpu::trt;
namespace ir = ajp::gpu::ir;
using ajp::RefusalCode;
using trt::test::must;

namespace {

template <typename T>
ajp::Refusal refusal(const std::expected<T, ajp::Refusal>& result) {
  if (result) FAIL("refus attendu");
  return result.error();
}

}  // namespace

TEST_CASE("moteur abîmé, tronqué ou vide : EngineInvalid", "[cuda_trt][cuda]") {
  trt::test::skip_without_cuda();
  const trt::TrtRuntime runtime = trt::test::make_runtime();
  std::vector<std::byte> plan = trt::test::conv_plan(runtime, ir::onnx::kFloat);

  SECTION("en-tête écrasé") {
    for (std::size_t k = 0; k < 64; ++k) plan[k] = std::byte{0x5A};
  }
  SECTION("tronqué de moitié") {
    plan.resize(plan.size() / 2);
  }
  SECTION("vide") {
    plan.clear();
  }

  const ajp::Refusal refused = refusal(runtime.load(plan));
  CHECK(refused.code == RefusalCode::EngineInvalid);
  INFO(refused.detail);
  CHECK(refused.detail.find("octets") != std::string::npos);
  // Le runtime reste utilisable après un refus.
  CHECK(runtime.load(trt::test::conv_plan(runtime, ir::onnx::kFloat)).has_value());
}

TEST_CASE("fichier moteur absent : CacheIo qui nomme le fichier", "[cuda_trt][cuda]") {
  trt::test::skip_without_cuda();
  const trt::TrtRuntime runtime = trt::test::make_runtime();
  const std::filesystem::path missing = std::filesystem::temp_directory_path() / "ajp-absent-0f3c.engine";
  const ajp::Refusal refused = refusal(runtime.load_file(missing));
  CHECK(refused.code == RefusalCode::CacheIo);
  CHECK(refused.detail.find("ajp-absent-0f3c.engine") != std::string::npos);
}

TEST_CASE("forme hors du profil ou de rang différent : ShapeOutOfProfile", "[cuda_trt][cuda]") {
  trt::test::skip_without_cuda();
  const trt::TrtRuntime runtime = trt::test::make_runtime();
  trt::TrtEngine engine = must(runtime.load(trt::test::conv_plan(runtime, ir::onnx::kFloat)));

  const std::vector<std::vector<std::int64_t>> outside = {
      {1, 3, 64, 64},    // au-delà de max 32x32
      {1, 3, 4, 16},     // sous min 8x8
      {2, 3, 16, 16},    // lot fixé à 1
      {1, 4, 16, 16},    // canaux fixés à 3
      {1, 3, 16},        // rang 3
      {1, 3, 16, 16, 1}  // rang 5
  };
  for (const auto& dims : outside) {
    const ajp::Refusal refused = refusal(engine.set_input_shape("input", dims));
    INFO(trt::dims_text(dims) << " : " << refused.detail);
    CHECK(refused.code == RefusalCode::ShapeOutOfProfile);
    CHECK(refused.detail.find("min 1x3x8x8, max 1x3x32x32") != std::string::npos);
  }
  // Le moteur reste utilisable : une forme du profil passe ensuite.
  must(engine.set_input_shape("input", std::vector<std::int64_t>{1, 3, 32, 32}));
  CHECK(must(engine.shape("output")) == trt::Dims{1, 4, 32, 32});
}

TEST_CASE("tenseur inconnu, sortie mise en forme, adresse nulle", "[cuda_trt][cuda]") {
  trt::test::skip_without_cuda();
  const trt::TrtRuntime runtime = trt::test::make_runtime();
  trt::TrtEngine engine = must(runtime.load(trt::test::conv_plan(runtime, ir::onnx::kFloat)));
  const std::vector<std::int64_t> dims{1, 3, 8, 8};
  CHECK(refusal(engine.set_input_shape("entree", dims)).code == RefusalCode::TensorUnknown);
  CHECK(refusal(engine.set_input_shape("output", dims)).code == RefusalCode::TensorUnknown);
  CHECK(refusal(engine.shape("absent")).code == RefusalCode::TensorUnknown);
  const trt::DeviceBuffer buffer = must(trt::DeviceBuffer::allocate(1024));
  CHECK(refusal(engine.bind("absent", buffer.data())).code == RefusalCode::TensorUnknown);
  CHECK(refusal(engine.bind("input", nullptr)).code == RefusalCode::FieldOutOfRange);
  CHECK(refusal(engine.select_profile(1, nullptr)).code == RefusalCode::FieldOutOfRange);
  CHECK(refusal(engine.select_profile(-1, nullptr)).code == RefusalCode::FieldOutOfRange);
}

TEST_CASE("enqueue refuse un tenseur sans adresse ou une forme non posée", "[cuda_trt][cuda]") {
  trt::test::skip_without_cuda();
  const trt::TrtRuntime runtime = trt::test::make_runtime();
  trt::TrtEngine engine = must(runtime.load(trt::test::conv_plan(runtime, ir::onnx::kFloat)));
  const trt::CudaStream stream = must(trt::CudaStream::create());
  const trt::DeviceBuffer in = must(trt::DeviceBuffer::allocate(4 * 3 * 32 * 32));
  const trt::DeviceBuffer out = must(trt::DeviceBuffer::allocate(4 * 4 * 32 * 32));

  const ajp::Refusal unbound = refusal(engine.enqueue(stream.get()));
  CHECK(unbound.code == RefusalCode::TensorUnknown);
  CHECK(unbound.detail.find("« input »") != std::string::npos);

  must(engine.bind("input", in.data()));
  must(engine.bind("output", out.data()));
  // H et W dynamiques : sans set_input_shape, TensorRT ne peut rien lancer.
  CHECK(refusal(engine.enqueue(stream.get())).code == RefusalCode::ShapeOutOfProfile);

  must(engine.set_input_shape("input", std::vector<std::int64_t>{1, 3, 32, 32}));
  must(engine.enqueue(stream.get()));
  must(stream.synchronize());

  // Changer de profil oublie les adresses : il faut relier avant d'exécuter.
  must(engine.select_profile(0, stream.get()));
  CHECK(refusal(engine.enqueue(stream.get())).code == RefusalCode::TensorUnknown);
}
