#include "app/play/model_choice.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <format>
#include <fstream>
#include <span>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "app/play/derived_model.h"
#include "app/play/drba_model.h"

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
  settings.rife_model =
      "rife_v4.26_heavy.onnx";  // présent mais hors du catalogue réduit (v0.5.1) : le défaut
  CHECK(choose_rife_model(PlayOptions{}, fake.home, settings)->path == fake.rife / "rife_v4.25.onnx");
  settings.rife_model = "rife_v4.13.onnx";  // hors catalogue, bien que présent : le défaut
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
  // Le dossier n'a pas 4.25 : la RTX prend la 4.22 lite, la suivante présente.
  CHECK(choose_rife_model(PlayOptions{}, fake.home, PlaySettings{}, rife_preference(rtx))->default_name ==
        "rife_v4.22_lite.onnx");
  CHECK(choose_rife_model(PlayOptions{}, fake.home, PlaySettings{}, rife_preference(amd))->default_name ==
        "rife_v4.22_lite.onnx");
  std::ofstream(fake.rife / "rife_v4.25.onnx") << "x";
  CHECK(choose_rife_model(PlayOptions{}, fake.home, PlaySettings{}, rife_preference(rtx))->default_name ==
        "rife_v4.25.onnx");
}

TEST_CASE("Menu RIFE : les deux modèles du catalogue, présents ou non", "[app][model_choice]") {
  const auto menu = rife_menu({"rife_v4.22_lite.onnx", "rife_v4.26.onnx", "rife_v4.26_heavy.onnx"});
  CHECK(menu == std::vector<RifeEntry>{{"rife_v4.25.onnx", false}, {"rife_v4.22_lite.onnx", true}});
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
  // Un dérivé d'une version d'avant (sans version de transformation) : refait, puis retiré (A36-04).
  std::filesystem::create_directories(fake.home / "modeles");
  std::ofstream(fake.home / "modeles" / "rife_v4.26-essai-0123456789abcdef.onnx") << "ancien";
  const auto made = derived_model(source, "essai", fake.home / "modeles", twice);
  REQUIRE(made);
  CHECK(made->filename().string().starts_with(std::format("rife_v4.26-essai-t{}-", kDerivedModelVersion)));
  CHECK_FALSE(std::filesystem::exists(fake.home / "modeles" / "rife_v4.26-essai-0123456789abcdef.onnx"));
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

TEST_CASE("DistilDRBA : demi-flot préféré en 4K, l'un sert de repli à l'autre", "[app][model_choice]") {
  const FakeHome fake("drba");
  const std::array dirs{fake.rife};
  CHECK_FALSE(find_distil_drba(dirs));
  std::ofstream(fake.rife / kDistilDrbaFile) << "x";
  CHECK(find_distil_drba(dirs, true)->align == 64);  // pas de demi-flot : le graphe entier
  std::ofstream(fake.rife / kDistilDrbaHalfFile) << "x";
  CHECK(find_distil_drba(dirs, true)->path == fake.rife / kDistilDrbaHalfFile);
  CHECK(find_distil_drba(dirs, true)->align == 128);
  CHECK(find_distil_drba(dirs, false)->path == fake.rife / kDistilDrbaFile);
  CHECK_FALSE(drba_half_flow(1080));
  CHECK(drba_half_flow(2160));
}

}  // namespace
TEST_CASE(
    "Modèle SPAN : le menu n'a que le catalogue (HD V3.1, ArtCNN), un modèle hérité retombe sur le défaut",
    "[app][model_choice]") {
  const FakeHome fake("span");
  const std::filesystem::path onnx = fake.home / ".local/share/AnimeJaNai-HDR-linux/animejanai/onnx";
  std::filesystem::create_directories(onnx);
  for (const std::string_view name : kSpanCatalogue) std::ofstream(onnx / std::string(name)) << "x";
  const std::string legacy = "2x_AnimeJaNai_SD_V1beta34_Compact_1x3xHxW_dyn-HW_strong_fp16_op21_dynamo.onnx";
  std::ofstream(onnx / legacy) << "x";
  PlaySettings settings;
  const auto plain = choose_model(PlayOptions{}, fake.home, settings);
  REQUIRE(plain);
  CHECK(plain->available.size() == kSpanCatalogue.size());
  CHECK(model_label(std::string(kLightSpanModel)) == "ArtCNN C4F32");
  CHECK(std::ranges::find(plain->available, legacy) == plain->available.end());
  settings.model = legacy;  // un ancien play.toml : le défaut, sans refus
  const auto fallback = choose_model(PlayOptions{}, fake.home, settings);
  REQUIRE(fallback);
  CHECK(fallback->path.filename().string() == fallback->default_name);
  CHECK(fallback->default_name != legacy);
  settings.model = std::string(kSpanCatalogue[0]);
  CHECK(choose_model(PlayOptions{}, fake.home, settings)->path.filename().string() == kSpanCatalogue[0]);
}

TEST_CASE("Crans de précision de RIFE et DistilDRBA : le graphe à côté du modèle, repli sur l'original",
          "[app][model_choice][fp8]") {
  const FakeHome fake("siblings");
  // Les noms attendus : RIFE 4.25, 4.22 lite, DistilDRBA v2 lite plein flot et _s05.
  for (const std::string_view stem :
       {"rife_v4.25", "rife_v4.22_lite", "distil_drba_v2_lite", "distil_drba_v2_lite_s05"}) {
    const std::filesystem::path model = fake.rife / std::format("{}.onnx", stem);
    std::ofstream(model) << "x";
    // Équilibre : le modèle ; un cran sans graphe : le modèle aussi, et le cran n'est pas offert.
    CHECK(tier_variant(model, PrecisionTier::Balanced) == model);
    CHECK(tier_variant(model, PrecisionTier::Performance) == model);
    CHECK(tier_variant(model, PrecisionTier::Quality) == model);
    CHECK(sibling_tiers(model, true) == TierSupport{});
    // Le graphe FP8 présent : offert sur un moteur qui calcule le FP8, et seulement sur lui.
    const std::filesystem::path fp8 = fake.rife / std::format("{}_fp8.onnx", stem);
    std::ofstream(fp8) << "x";
    CHECK(tier_variant(model, PrecisionTier::Performance) == fp8);
    CHECK(sibling_tiers(model, true) == TierSupport{.performance = true, .quality = false});
    CHECK(sibling_tiers(model, false) == TierSupport{});
    CHECK(TierSupport{.performance = true}.effective(PrecisionTier::Performance) ==
          PrecisionTier::Performance);
    CHECK(sibling_tiers(model, false).effective(PrecisionTier::Performance) == PrecisionTier::Balanced);
    // Le FP32 est indépendant du FP8.
    std::ofstream(fake.rife / std::format("{}_fp32.onnx", stem)) << "x";
    CHECK(sibling_tiers(model, false) == TierSupport{.performance = false, .quality = true});
    CHECK(sibling_tiers(model, true) == TierSupport{.performance = true, .quality = true});
  }
}

}  // namespace ajp::app::play
