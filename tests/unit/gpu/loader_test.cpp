// Chargeur des backends, sur de vraies bibliothèques construites avec les tests : le backend
// CPU de référence et trois backends factices (ABI ancienne, symbole absent, appareil refusé).
#include "gpu/loader.h"

#include <filesystem>
#include <string>

#include <catch2/catch_test_macros.hpp>

using ajp::RefusalCode;
using namespace ajp::gpu;

namespace {

const std::filesystem::path kCpuDir = AJP_TEST_CPU_BACKEND_DIR;
const std::filesystem::path kFakeDir = AJP_TEST_FAKE_BACKEND_DIR;

}  // namespace

TEST_CASE("Chargeur : le backend CPU se charge, se crée et se détruit", "[loader]") {
  std::string logged;
  const BackendInit init{.log = [&logged](BackendLogLevel, std::string_view message) { logged += message; }};
  auto loaded = LoadedBackend::load(kCpuDir, "cpu", init);
  REQUIRE(loaded);
  CHECK(loaded->path().filename() == backend_file_name("cpu"));
  CHECK(loaded->backend().capabilities().name == "cpu");
  CHECK(loaded->backend().kernels() != nullptr);
  CHECK(loaded->rife_input() != nullptr);  // point d'entrée facultatif présent

  auto buffer = loaded->backend().alloc_tensor(TensorType::F32, Shape{{1, 3, 4, 4}});
  REQUIRE(buffer);
  CHECK((*buffer)->tensor().handle != 0);
  buffer->reset();  // avant le backend, comme le contrat l'exige

  LoadedBackend moved = std::move(*loaded);  // déplacé : un seul propriétaire détruit le backend
  CHECK(moved.backend().capabilities().name == "cpu");
}

TEST_CASE("Chargeur : refus nommés", "[loader]") {
  const BackendInit init;
  SECTION("bibliothèque absente") {
    auto missing = LoadedBackend::load(kCpuDir, "absent", init);
    REQUIRE_FALSE(missing);
    CHECK(missing.error().code == RefusalCode::LibraryUnavailable);
  }
  SECTION("nom qui sortirait du dossier") {
    CHECK(LoadedBackend::load(kCpuDir, "../cpu", init).error().code == RefusalCode::FieldOutOfRange);
    CHECK(LoadedBackend::load(kCpuDir, "", init).error().code == RefusalCode::FieldOutOfRange);
    CHECK(LoadedBackend::load(kCpuDir, "CPU", init).error().code == RefusalCode::FieldOutOfRange);
  }
  SECTION("ABI d'une autre version, ou même version et autre empreinte binaire") {
    auto old = LoadedBackend::load(kFakeDir, "oldabi", init);
    REQUIRE_FALSE(old);
    CHECK(old.error().code == RefusalCode::AbiMismatch);
    auto layout = LoadedBackend::load(kFakeDir, "badlayout", init);
    REQUIRE_FALSE(layout);
    CHECK(layout.error().code == RefusalCode::AbiMismatch);
    CHECK(layout.error().detail.find("empreinte") != std::string::npos);
  }
  SECTION("symbole de création absent") {
    auto incomplete = LoadedBackend::load(kFakeDir, "nocreate", init);
    REQUIRE_FALSE(incomplete);
    CHECK(incomplete.error().code == RefusalCode::LibraryUnavailable);
    CHECK(incomplete.error().detail.find("ajp_backend_create") != std::string::npos);
  }
  SECTION("appareil refusé par le backend : son refus passe tel quel") {
    auto refused = LoadedBackend::load(kFakeDir, "refuses", init);
    REQUIRE_FALSE(refused);
    CHECK(refused.error().code == RefusalCode::DeviceMismatch);
    CHECK(refused.error().detail.find("factice") != std::string::npos);
  }
}

TEST_CASE("Chargeur : liste des backends d'un dossier pour le panneau des capacités", "[loader]") {
  auto listed = list_backends(kFakeDir);
  REQUIRE(listed);
  REQUIRE(listed->size() == 4);
  CHECK((*listed)[0].name == "badlayout");
  CHECK((*listed)[1].name == "nocreate");
  CHECK((*listed)[2].name == "oldabi");
  CHECK((*listed)[3].name == "refuses");
  CHECK((*listed)[0].abi == kBackendAbi);
  CHECK_FALSE((*listed)[0].usable);
  CHECK((*listed)[1].usable);
  CHECK((*listed)[2].abi == 2);
  CHECK((*listed)[2].usable.error().code == RefusalCode::AbiMismatch);

  auto cpu = list_backends(kCpuDir);
  REQUIRE(cpu);
  REQUIRE(cpu->size() == 1);
  CHECK(cpu->front().name == "cpu");
  CHECK(cpu->front().usable);

  CHECK(list_backends(kFakeDir / "absent").error().code == RefusalCode::FileUnreadable);
}
