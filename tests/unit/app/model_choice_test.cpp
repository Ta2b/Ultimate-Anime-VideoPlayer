#include "app/play/model_choice.h"

#include <algorithm>
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
#include "app/play/drba_model.h"
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
  // Le flot à demi-échelle passe devant partout : le flot entier ne sert que de repli.
  CHECK(find_gmfss(dirs, GmfssVariant::Base, true)->pair == other / kGmfssHalfPairFile);
  CHECK(find_gmfss(dirs, GmfssVariant::Base, false)->pair == other / kGmfssPairFile);
  // Les crans Qualité (FP32, s05q) et Performance (FP8, s05fp8) : leur paire avec la fusion s05 d'Équilibre,
  // dans un même dossier ; sans elle, l'Équilibre (FP16) reste.
  CHECK_FALSE(gmfss_tier_available(dirs, GmfssVariant::Base, PrecisionTier::Quality));
  const auto without = find_gmfss(dirs, GmfssVariant::Base, true, PrecisionTier::Quality);
  REQUIRE(without);
  CHECK(without->tier == PrecisionTier::Balanced);
  CHECK(without->pair == other / kGmfssHalfPairFile);
  std::ofstream(other / kGmfssHalfQualityPairFile) << "x";
  CHECK(gmfss_tier_available(dirs, GmfssVariant::Base, PrecisionTier::Quality));
  const auto quality = find_gmfss(dirs, GmfssVariant::Base, true, PrecisionTier::Quality);
  REQUIRE(quality);
  CHECK(quality->tier == PrecisionTier::Quality);
  CHECK(quality->pair == other / kGmfssHalfQualityPairFile);
  CHECK(quality->fusion == other / kGmfssHalfFusionFile);  // la fusion n'est pas écrite deux fois
  CHECK(quality->align == kGmfssHalfAlign);
  CHECK(find_gmfss(dirs, GmfssVariant::Base, true, PrecisionTier::Balanced)->tier == PrecisionTier::Balanced);
  CHECK_FALSE(gmfss_tier_available(dirs, GmfssVariant::Base, PrecisionTier::Performance));
  CHECK(find_gmfss(dirs, GmfssVariant::Base, true, PrecisionTier::Performance)->tier ==
        PrecisionTier::Balanced);
  std::ofstream(other / kGmfssHalfFp8PairFile) << "x";
  const auto fast = find_gmfss(dirs, GmfssVariant::Base, true, PrecisionTier::Performance);
  REQUIRE(fast);
  CHECK(fast->tier == PrecisionTier::Performance);
  CHECK(fast->pair == other / kGmfssHalfFp8PairFile);
  CHECK(fast->fusion == other / kGmfssHalfFusionFile);
  // Les crans d'un dossier : le FP8 demande un moteur qui le calcule, le FP32 non.
  CHECK(gmfss_tiers(dirs, GmfssVariant::Base, true) == TierSupport{.performance = true, .quality = true});
  CHECK(gmfss_tiers(dirs, GmfssVariant::Base, false) == TierSupport{.performance = false, .quality = true});
  // Union a ses propres paires : celles de base ne lui servent pas.
  CHECK_FALSE(gmfss_tier_available(dirs, GmfssVariant::Union, PrecisionTier::Quality));
  CHECK_FALSE(gmfss_tier_available(dirs, GmfssVariant::Union, PrecisionTier::Performance));
  CHECK(gmfss_files(GmfssVariant::Union, true, PrecisionTier::Performance).pair ==
        kGmfssUnionHalfFp8PairFile);
  const auto dims = gmfss_dims(1280, 768);
  REQUIRE(dims.size() == 8);
  CHECK(dims[0].name == "H");
  CHECK(dims[0].value == 768);
  CHECK(dims[7].name == "W/8");
  CHECK(dims[7].value == 160);
}

// Un ONNX minimal de paire GMFSS écrit à la main : un graphe dont les nœuds produisent les tenseurs
// donnés, et la métadonnée « ajp.gmfss.export » si version n'est pas vide.
std::vector<std::byte> tiny_pair(std::string_view version, const std::vector<std::string>& tensors) {
  using Bytes = std::vector<std::byte>;
  const auto varint = [](Bytes& out, std::uint64_t v) {
    while (v >= 0x80U) {
      out.push_back(static_cast<std::byte>((v & 0x7fU) | 0x80U));
      v >>= 7U;
    }
    out.push_back(static_cast<std::byte>(v));
  };
  const auto delimited = [&](Bytes& out, std::uint32_t number, const Bytes& payload) {
    varint(out, (std::uint64_t{number} << 3U) | 2U);
    varint(out, payload.size());
    out.insert(out.end(), payload.begin(), payload.end());
  };
  const auto text = [](std::string_view s) {
    return Bytes(reinterpret_cast<const std::byte*>(s.data()),
                 reinterpret_cast<const std::byte*>(s.data()) + s.size());
  };
  Bytes graph;
  for (const std::string& tensor : tensors) {
    Bytes node;
    delimited(node, 1, text("frames"));
    delimited(node, 2, text(tensor));
    delimited(node, 4, text("Identity"));
    delimited(graph, 1, node);
  }
  delimited(graph, 2, text("pair"));
  Bytes model;
  varint(model, (1U << 3U) | 0U);
  varint(model, 9);
  delimited(model, 7, graph);
  if (!version.empty()) {
    Bytes entry;
    delimited(entry, 1, text(kGmfssExportKey));
    delimited(entry, 2, text(version));
    delimited(model, 14, entry);
  }
  return model;
}

void put(const std::filesystem::path& file, const std::vector<std::byte>& bytes) {
  std::ofstream(file, std::ios::binary)
      .write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

TEST_CASE("GMFSS : un graphe d'un export périmé n'est jamais utilisé, un ancien reste lu",
          "[app][model_choice][gmfss]") {
  const FakeHome fake("gmfss_stale");
  const std::filesystem::path other = fake.home / "autre";
  std::filesystem::create_directories(other);
  const std::vector<std::string> modern{"/flownet/GridSample_output_0", "/Concat_output_0", "feature0_1"};
  const std::vector<std::string> old_window{"/flownet/GridSample_output_0",
                                            "/flownet_1/GridSample_1_output_0", "/Concat_output_0"};
  const std::string current = std::to_string(kGmfssExportVersion);

  // Version courante : lue sans parcourir le graphe.
  put(fake.rife / kGmfssHalfPairFile, tiny_pair(current, modern));
  std::ofstream(fake.rife / kGmfssHalfFusionFile) << "x";
  CHECK(gmfss_graph_state(fake.rife / kGmfssHalfPairFile).state == GmfssExport::Current);
  CHECK(find_gmfss(std::array{fake.rife})->pair == fake.rife / kGmfssHalfPairFile);

  // Sans métadonnée et sans GridSample de fenêtre : un export ancien (source{k}_1 en 67 canaux), lisible.
  put(fake.rife / kGmfssHalfPairFile, tiny_pair("", modern));
  const GmfssGraphState legacy = gmfss_graph_state(fake.rife / kGmfssHalfPairFile);
  CHECK(legacy.state == GmfssExport::Legacy);
  CHECK(legacy.version == 1);
  CHECK_FALSE(legacy.reason.empty());
  CHECK(find_gmfss(std::array{fake.rife}).has_value());

  // Sans métadonnée avec le GridSample de la fenêtre locale de GMFlow (159 millions de valeurs) : périmé,
  // jamais utilisé, même en dernier recours : plus rien n'est trouvé.
  put(fake.rife / kGmfssHalfPairFile, tiny_pair("", old_window));
  CHECK(gmfss_graph_state(fake.rife / kGmfssHalfPairFile).state == GmfssExport::Stale);
  CHECK_FALSE(find_gmfss(std::array{fake.rife}));
  // Le dossier suivant a un export courant : il passe devant le périmé.
  put(other / kGmfssHalfPairFile, tiny_pair(current, modern));
  std::ofstream(other / kGmfssHalfFusionFile) << "x";
  CHECK(find_gmfss(std::array{fake.rife, other})->pair == other / kGmfssHalfPairFile);
  // Le flot entier périmé ne remplace pas non plus le demi-flot : seul le graphe courant de l'autre dossier.
  put(fake.rife / kGmfssPairFile, tiny_pair("", old_window));
  std::ofstream(fake.rife / kGmfssFusionFile) << "x";
  CHECK_FALSE(find_gmfss(std::array{fake.rife}, GmfssVariant::Base, false));

  // Un export plus récent que le lecteur n'est pas lu non plus (son contrat est inconnu).
  put(other / kGmfssHalfPairFile, tiny_pair(std::to_string(kGmfssExportVersion + 1), modern));
  const GmfssGraphState newer = gmfss_graph_state(other / kGmfssHalfPairFile);
  CHECK(newer.state == GmfssExport::Stale);
  CHECK(newer.version == kGmfssExportVersion + 1);
  CHECK_FALSE(find_gmfss(std::array{other}));
  put(other / kGmfssHalfPairFile, tiny_pair("deux", modern));
  CHECK(gmfss_graph_state(other / kGmfssHalfPairFile).state == GmfssExport::Stale);

  // Un fichier illisible n'est pas jugé périmé (les tests ci-dessus l'écrivent « x ») : la construction
  // du moteur le refusera avec sa raison.
  std::ofstream(other / kGmfssHalfPairFile) << "x";
  CHECK(gmfss_graph_state(other / kGmfssHalfPairFile).state == GmfssExport::Current);
  CHECK(find_gmfss(std::array{other}).has_value());
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
TEST_CASE("Modèle SPAN : le menu n'a que le catalogue HD V3.1, un modèle hérité retombe sur le défaut",
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
  CHECK(plain->available.size() == 4);
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

TEST_CASE("GMFSS : le FP8 est la paire seule, la fusion reste celle du cran Équilibre",
          "[app][model_choice][fp8]") {
  for (const GmfssVariant variant : {GmfssVariant::Base, GmfssVariant::Union}) {
    const GmfssFiles fp8 = gmfss_files(variant, true, PrecisionTier::Performance);
    const GmfssFiles fp16 = gmfss_files(variant, true, PrecisionTier::Balanced);
    CHECK(fp8.pair != fp16.pair);
    CHECK(fp8.pair.contains("s05fp8_pair"));
    CHECK(fp8.fusion == fp16.fusion);
  }
  CHECK(kGmfssHalfFp8PairFile == "gmfss_fortuna_base_s05fp8_pair.onnx");
  CHECK(kGmfssUnionHalfFp8PairFile == "gmfss_fortuna_union_s05fp8_pair.onnx");
  // Le flot entier n'a pas de FP8 : sans demi-flot, le cran est sans effet sur les fichiers.
  CHECK(gmfss_files(GmfssVariant::Base, false, PrecisionTier::Performance).pair == kGmfssPairFile);
}

}  // namespace ajp::app::play
