// Une instance de construction pour toute la préparation (A08-10) et la fermeture d'une construction qui
// attend la fin de son thread (A05-10).
#include "app/play/build_backend.h"

#include <chrono>
#include <thread>

#include <catch2/catch_test_macros.hpp>

#include "app/play/engine_build.h"
#include "stage_rig.h"

namespace ajp::app::play {
namespace {

void destroy_fake(gpu::Backend* backend) noexcept {
  delete backend;  // NOLINT(cppcoreguidelines-owning-memory) : l'instance factice du test
}

BuildBackend::Create counting_fakes(int& made) {
  return [&made]() -> std::expected<BackendLibrary::Instance, Refusal> {
    ++made;
    return BackendLibrary::Instance(new test::FakeBackend(),
                                    BackendLibrary::Destroy{.destroy = &destroy_fake});
  };
}

gpu::ModelSpec model(std::string id) {
  return {.id = std::move(id), .onnx_path = "absent.onnx", .sha256 = "0"};
}

std::expected<gpu::Artifact, Refusal> wait_result(EngineBuild& build) {
  for (int i = 0; i < 500; ++i) {
    if (auto done = build.result()) return *done;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  FAIL("construction sans résultat");
  return std::unexpected(Refusal{RefusalCode::BuilderTimeout, "attente"});
}

TEST_CASE("Construction : une instance pour toutes les constructions de la préparation", "[app][engines]") {
  int made = 0;
  BuildBackend builder(counting_fakes(made));
  gpu::Progress progress;
  for (const char* id : {"span", "rife", "rife_motion"}) {
    const auto built = builder.build(model(id), {}, progress);
    REQUIRE_FALSE(built);  // le backend factice ne construit rien
    CHECK(built.error().detail.find(id) != std::string::npos);
  }
  CHECK(made == 1);
  CHECK(builder.created() == 1);
}

TEST_CASE("Construction : un refus de création est retenu, pas réessayé à chaque moteur", "[app][engines]") {
  int tries = 0;
  BuildBackend builder([&tries]() -> std::expected<BackendLibrary::Instance, Refusal> {
    ++tries;
    return std::unexpected(Refusal{RefusalCode::DeviceMismatch, "pas de GPU NVIDIA"});
  });
  gpu::Progress progress;
  CHECK(builder.build(model("span"), {}, progress).error().code == RefusalCode::DeviceMismatch);
  CHECK(builder.build(model("rife"), {}, progress).error().code == RefusalCode::DeviceMismatch);
  CHECK(tries == 1);
}

TEST_CASE(
    "Construction : les EngineBuild d'une préparation passent par l'instance partagée, fermeture pompée",
    "[app][engines]") {
  int made = 0;
  const auto builder = std::make_shared<BuildBackend>(counting_fakes(made));
  for (const char* id : {"span", "rife"}) {
    auto build =
        EngineBuild::start(EngineRequest{.model = model(id), .builder = builder, .pump_on_close = true});
    REQUIRE(build);
    const auto built = wait_result(**build);
    REQUIRE_FALSE(built);
    CHECK(built.error().detail.find(id) != std::string::npos);
  }  // chaque EngineBuild détruit ici attend la fin de son thread en pompant
  CHECK(made == 1);
}

}  // namespace
}  // namespace ajp::app::play
