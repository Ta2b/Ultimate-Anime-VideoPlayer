#include "app/play/model_choice.h"

#include <array>
#include <cstddef>
#include <filesystem>
#include <format>
#include <fstream>
#include <span>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "app/play/derived_model.h"
#include "app/play/gmfss_model.h"

namespace ajp::app::play {
namespace {

// Un dossier personnel jetable avec le dossier RIFE de l'installation d'AnimeJaNai ; un nom par
// test, ctest les lançant en parallèle.
struct FakeHome {
  std::filesystem::path home;
  std::filesystem::path rife;
  explicit FakeHome(std::string_view test)
      : home(std::filesystem::temp_directory_path() / std::format("ajp-model-choice-{}", test)),
        rife(home / ".local/share/AnimeJaNai-HDR-linux/animejanai/rife") {
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
  const FakeHome fake("rife");
  std::ofstream(fake.rife / "rife_v4.25.onnx") << "x";
  PlaySettings settings;
  auto plain = choose_rife_model(PlayOptions{}, fake.home, settings);
  REQUIRE(plain);
  CHECK(plain->path == fake.rife / "rife_v4.25.onnx");
  CHECK(plain->default_name == "rife_v4.25.onnx");
  CHECK(plain->available == std::vector<std::string>{"rife_v4.22_lite.onnx", "rife_v4.25.onnx",
                                                     "rife_v4.26.onnx", "rife_v4.26_heavy.onnx"});
  settings.rife_model = "rife_v4.22_lite.onnx";
  CHECK(choose_rife_model(PlayOptions{}, fake.home, settings)->path == fake.rife / "rife_v4.22_lite.onnx");
  settings.rife_model = "rife_v4.26_heavy.onnx";  // hors catalogue, bien que présent : le défaut
  CHECK(choose_rife_model(PlayOptions{}, fake.home, settings)->path == fake.rife / "rife_v4.25.onnx");
  settings.rife_model = "rife_v9.onnx";  // absent : le défaut
  CHECK(choose_rife_model(PlayOptions{}, fake.home, settings)->path == fake.rife / "rife_v4.25.onnx");
}

TEST_CASE("Modèle RIFE introuvable : refus qui cite les chemins essayés", "[app][model_choice]") {
  const auto home = std::filesystem::temp_directory_path() / "ajp-model-choice-empty";
  std::filesystem::remove_all(home);
  const auto refused = choose_rife_model(PlayOptions{}, home, PlaySettings{});
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::ModelUnreadable);
  CHECK(refused.error().detail.find("rife_v4.25.onnx") != std::string::npos);
}

TEST_CASE("Modèle RIFE par défaut : 4.25 sur une RTX, 4.22 lite ailleurs, sinon le suivant présent",
          "[app][model_choice]") {
  const FakeHome fake("preference");
  const GpuFacts rtx{.vendor = GpuVendor::Nvidia, .vram_bytes = 12ULL << 30U, .tensor_cores = true};
  const GpuFacts amd{.vendor = GpuVendor::Amd, .vram_bytes = 16ULL << 30U};
  // Le dossier n'a pas 4.25 : la RTX prend 4.22 lite, la suivante présente.
  CHECK(choose_rife_model(PlayOptions{}, fake.home, PlaySettings{}, rife_preference(rtx))->default_name ==
        "rife_v4.22_lite.onnx");
  CHECK(choose_rife_model(PlayOptions{}, fake.home, PlaySettings{}, rife_preference(amd))->default_name ==
        "rife_v4.22_lite.onnx");
  std::ofstream(fake.rife / "rife_v4.25.onnx") << "x";
  CHECK(choose_rife_model(PlayOptions{}, fake.home, PlaySettings{}, rife_preference(rtx))->default_name ==
        "rife_v4.25.onnx");
}

TEST_CASE("Menu RIFE : 4.25 et 4.22 lite seulement, présents ou non", "[app][model_choice]") {
  const auto menu = rife_menu({"rife_v4.22_lite.onnx", "rife_v4.26.onnx", "rife_v4.26_heavy.onnx"});
  CHECK(menu == std::vector<RifeEntry>{{"rife_v4.25.onnx", false}, {"rife_v4.22_lite.onnx", true}});
}

TEST_CASE("GMFSS : les deux graphes du premier dossier qui les a, ses dimensions", "[app][model_choice]") {
  const FakeHome fake("gmfss");
  const std::filesystem::path other = fake.home / "autre";
  std::filesystem::create_directories(other);
  std::ofstream(fake.rife / kGmfssPairFile) << "x";  // la paire seule : ce dossier ne suffit pas
  const std::array dirs{fake.rife, other};
  CHECK_FALSE(find_gmfss(dirs));
  std::ofstream(other / kGmfssPairFile) << "x";
  std::ofstream(other / kGmfssFusionFile) << "x";
  const auto found = find_gmfss(dirs);
  REQUIRE(found);
  CHECK(found->fusion == other / kGmfssFusionFile);
  // Union : ses propres graphes, cherchés de même.
  CHECK_FALSE(find_gmfss(dirs, GmfssVariant::Union));
  std::ofstream(fake.rife / kGmfssUnionPairFile) << "x";
  std::ofstream(fake.rife / kGmfssUnionFusionFile) << "x";
  CHECK(find_gmfss(dirs, GmfssVariant::Union)->pair == fake.rife / kGmfssUnionPairFile);
  // Le flot à l'échelle 0,5 passe devant, plans au multiple de 128.
  std::ofstream(other / kGmfssHalfPairFile) << "x";
  std::ofstream(other / kGmfssHalfFusionFile) << "x";
  const auto half = find_gmfss(dirs);
  REQUIRE(half);
  CHECK(half->pair == other / kGmfssHalfPairFile);
  CHECK(half->align == kGmfssHalfAlign);
  CHECK(find_gmfss(dirs, GmfssVariant::Union)->align == 64);
  const auto dims = gmfss_dims(1280, 768);
  REQUIRE(dims.size() == 8);
  CHECK(dims[0].name == "H");
  CHECK(dims[0].value == 768);
  CHECK(dims[7].name == "W/8");
  CHECK(dims[7].value == 160);
}

TEST_CASE("Modèle dérivé : nommé par l'empreinte de la source, fait une seule fois", "[app][model_choice]") {
  const FakeHome fake("derived");
  int calls = 0;
  const auto twice = [&](std::span<const std::byte> model) -> std::expected<std::vector<std::byte>, Refusal> {
    ++calls;
    std::vector<std::byte> out(model.begin(), model.end());
    out.insert(out.end(), model.begin(), model.end());
    return out;
  };
  const std::filesystem::path source = fake.rife / "rife_v4.26.onnx";
  const auto made = derived_model(source, "essai", fake.home / "modeles", twice);
  REQUIRE(made);
  CHECK(made->filename().string().starts_with("rife_v4.26-essai-"));
  CHECK(std::filesystem::file_size(*made) == 2);
  CHECK(*derived_model(source, "essai", fake.home / "modeles", twice) == *made);
  CHECK(calls == 1);
  const auto refused = derived_model(source, "refus", fake.home / "modeles", [](std::span<const std::byte>) {
    return std::expected<std::vector<std::byte>, Refusal>(
        std::unexpected(Refusal{RefusalCode::FieldMissing, "graphe inconnu"}));
  });
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::FieldMissing);
}

}  // namespace
}  // namespace ajp::app::play
