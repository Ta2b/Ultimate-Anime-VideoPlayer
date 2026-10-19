#include "app/play/model_choice.h"

#include <filesystem>
#include <fstream>

#include <catch2/catch_test_macros.hpp>

namespace ajp::app::play {
namespace {

// Un dossier personnel jetable avec le dossier RIFE de l'installation d'AnimeJaNai.
struct FakeHome {
  std::filesystem::path home = std::filesystem::temp_directory_path() / "ajp-model-choice-test";
  std::filesystem::path rife = home / ".local/share/AnimeJaNai-HDR-linux/animejanai/rife";
  FakeHome() {
    std::filesystem::remove_all(home);
    std::filesystem::create_directories(rife);
    for (const char* name : {"rife_v4.26.onnx", "rife_v4.26_heavy.onnx", "rife_v4.22_lite.onnx", "notes.txt"})
      std::ofstream(rife / name) << "x";
  }
  FakeHome(const FakeHome&) = delete;
  FakeHome& operator=(const FakeHome&) = delete;
  ~FakeHome() { std::filesystem::remove_all(home); }
};

TEST_CASE("Modèle RIFE : le défaut, le choix du menu, la liste du dossier", "[app][model_choice]") {
  const FakeHome fake;
  PlaySettings settings;
  auto plain = choose_rife_model(PlayOptions{}, fake.home, settings);
  REQUIRE(plain);
  CHECK(plain->path == fake.rife / "rife_v4.26.onnx");
  CHECK(plain->default_name == "rife_v4.26.onnx");
  CHECK(plain->available ==
        std::vector<std::string>{"rife_v4.22_lite.onnx", "rife_v4.26.onnx", "rife_v4.26_heavy.onnx"});
  settings.rife_model = "rife_v4.26_heavy.onnx";
  CHECK(choose_rife_model(PlayOptions{}, fake.home, settings)->path == fake.rife / "rife_v4.26_heavy.onnx");
  settings.rife_model = "rife_v9.onnx";  // absent : le défaut
  CHECK(choose_rife_model(PlayOptions{}, fake.home, settings)->path == fake.rife / "rife_v4.26.onnx");
}

TEST_CASE("Modèle RIFE introuvable : refus qui cite les chemins essayés", "[app][model_choice]") {
  const auto home = std::filesystem::temp_directory_path() / "ajp-model-choice-empty";
  std::filesystem::remove_all(home);
  const auto refused = choose_rife_model(PlayOptions{}, home, PlaySettings{});
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::ModelUnreadable);
  CHECK(refused.error().detail.find("rife_v4.26.onnx") != std::string::npos);
}

}  // namespace
}  // namespace ajp::app::play
