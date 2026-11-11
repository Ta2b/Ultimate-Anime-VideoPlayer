// play.toml ne perd rien : une écriture refusée se rattrape à la suivante (A22-02), un retour à une
// version plus ancienne garde les valeurs et les clés de la plus récente (A36-01, A22-06).
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "app/play/chain.h"
#include "app/play/settings.h"

namespace ajp::app::play {
namespace {

std::string read_text(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);  // refermé avant remove : Windows refuse un fichier ouvert
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void write_text(const std::filesystem::path& path, const std::string& text) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out << text;
}

TEST_CASE("play.toml : une écriture refusée est reportée par la suivante (A22-02)", "[app][settings]") {
  const std::filesystem::path path = std::filesystem::temp_directory_path() / "ajp-play-refused.toml";
  std::filesystem::remove_all(path);
  REQUIRE(save_settings(path, PlaySettings{}));
  const std::string on_disk = read_text(path);
  auto settings = load_settings(path);
  REQUIRE(settings);
  // Un dossier non vide à la place du fichier : le renommage échoue, même pour root (antivirus qui
  // tient le fichier sous Windows).
  std::filesystem::remove(path);
  std::filesystem::create_directories(path / "plein");
  settings->sharpness = 40;
  CHECK_FALSE(save_settings(path, *settings));
  std::filesystem::remove_all(path);
  write_text(path, on_disk);
  // Un autre réglage ensuite : l'écriture réussit, la netteté y est aussi.
  settings->volume = 0.5F;
  REQUIRE(save_settings(path, *settings));
  const auto reread = load_settings(path);
  REQUIRE(reread);
  CHECK(reread->sharpness == 40);
  CHECK(reread->volume == 0.5F);
  std::filesystem::remove(path);
}

TEST_CASE("play.toml : une valeur inconnue d'une version récente reste, et ses clés aussi (A36-01)",
          "[app][settings]") {
  const std::filesystem::path path = std::filesystem::temp_directory_path() / "ajp-play-newer.toml";
  write_text(path,
             "version = 2\nfuture_option = 7\n[image]\nupscale = false\nsharpen_lines = true\n"
             "[interpolation]\nengine = \"rife425_fused\"\n");
  auto loaded = load_settings(path);
  REQUIRE(loaded);
  CHECK(loaded->motion_engine == MotionEngine::Rife);  // le défaut, pour cette version
  CHECK_FALSE(loaded->upscale);
  loaded->volume = 0.5F;  // le premier réglage touché réécrit le fichier
  REQUIRE(save_settings(path, *loaded));
  const std::string text = read_text(path);
  CHECK(text.contains("future_option = 7"));
  CHECK(text.contains("sharpen_lines = true"));
  CHECK(text.contains("engine = 'rife425_fused'"));
  CHECK(text.contains("version = 2"));
  auto reread = load_settings(path);
  REQUIRE(reread);
  CHECK(*reread == *loaded);
  // Changée ici, la valeur de cette version l'emporte.
  reread->motion_engine = MotionEngine::DistilDrba;
  REQUIRE(save_settings(path, *reread));
  const std::string changed = read_text(path);
  CHECK_FALSE(changed.contains("rife425_fused"));
  CHECK(changed.contains("future_option = 7"));
  const auto last = load_settings(path);
  REQUIRE(last);
  CHECK(last->motion_engine == MotionEngine::DistilDrba);
  std::filesystem::remove(path);
}

// B22-02 : un play.toml qui ne se lit pas n'est ni pris pour absent ni remplacé par les défauts de la séance.
TEST_CASE("play.toml illisible : refusé à la lecture, jamais écrasé ensuite (B22-02)", "[app][settings]") {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "ajp-play-unreadable";
  std::filesystem::remove_all(dir);
  // Un dossier à la place du fichier : sous Linux il s'ouvre et se lit vide, un « absent » trompeur.
  const std::filesystem::path path = dir / "play.toml";
  std::filesystem::create_directories(path);
  const auto loaded = load_settings(path);
  REQUIRE_FALSE(loaded);
  CHECK(loaded.error().code == RefusalCode::FileUnreadable);
  PlaySettings changed;
  changed.volume = 0.5F;
  CHECK_FALSE(save_settings(path, changed));
  CHECK(std::filesystem::is_directory(path));  // rien n'a été remplacé
  std::filesystem::remove_all(dir);
}

// B22-04 : un TOML invalide, édité à la main pendant la lecture, est gardé en copie avant d'être remplacé.
TEST_CASE("play.toml invalide : une copie est gardée avant la réécriture (B22-04)", "[app][settings]") {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "ajp-play-invalid-copy";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const std::filesystem::path path = dir / "play.toml";
  const std::string broken = "version = 1\n[image\nwork_height = 540\n";
  write_text(path, broken);
  PlaySettings changed;
  changed.volume = 0.5F;
  REQUIRE(save_settings(path, changed));
  int copies = 0;
  for (const auto& entry : std::filesystem::directory_iterator(dir)) {
    if (entry.path().filename().string().starts_with("play.toml.invalide-")) {
      ++copies;
      CHECK(read_text(entry.path()) == broken);
    }
  }
  CHECK(copies == 1);
  const auto reread = load_settings(path);
  REQUIRE(reread);
  CHECK(reread->volume == 0.5F);
  std::filesystem::remove_all(dir);
}

// B22-05 et B22-06 : une valeur fautive est écartée seule, et dite ; une langue fautive ne prend pas les
// autres.
TEST_CASE("play.toml : valeurs écartées dites, langues écartées une à une (B22-05, B22-06)",
          "[app][settings]") {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "ajp-play-ignored";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const std::filesystem::path path = dir / "play.toml";
  write_text(path,
             "version = 1\n[image]\nwork_height = 5000\n[hdr]\npeak = 300\n"
             "[languages]\naudio_1 = \"ja_JP\"\naudio_2 = \"fr_FR\"\nsubtitles_1 = \"fre\"\n");
  std::vector<std::string> ignored;
  const auto loaded = load_settings(path, &ignored);
  REQUIRE(loaded);
  CHECK(loaded->work_height == PlaySettings{}.work_height);
  CHECK(loaded->true_hdr_peak == 0);
  REQUIRE(loaded->language_presets.size() >= 2);
  CHECK(loaded->language_presets[0].subtitles == "fre");  // les sous-titres valides restent
  CHECK(loaded->language_presets[0].audio == PlaySettings{}.language_presets[0].audio);  // le défaut
  CHECK(loaded->language_presets[1].audio == PlaySettings{}.language_presets[1].audio);
  CHECK(ignored.size() == 4);
  const std::string note = ignored_values_note(ignored);
  CHECK(note.find("image.work_height") != std::string::npos);
  CHECK(note.find("languages.audio_2") != std::string::npos);
  CHECK(ignored_values_note({}).empty());
  std::filesystem::remove_all(dir);
}

// A08-04 : un raccourci de séance (Ctrl+G, Ctrl+I, Ctrl+0) n'entre pas dans play.toml, quelle que soit la
// réouverture (épisode suivant, fichier glissé, menu) : ce qu'elle écrit passe par with_kept_choices.
TEST_CASE("play.toml garde le grade et l'interpolation d'avant un raccourci de séance", "[app][settings]") {
  PlaySettings running;  // la chaîne en cours, raccourci Ctrl+0 appliqué
  running.grade = false;
  running.rife_factor = 0;
  const PlaySettings saved = with_kept_choices(KeptChoices{.grade = true, .rife = std::pair{2, 1}}, running);
  CHECK(saved.grade);
  CHECK(saved.rife_factor == 2);
  CHECK(saved.rife_factor_den == 1);
  const PlaySettings untouched = with_kept_choices(KeptChoices{}, running);
  CHECK(untouched == running);
}

}  // namespace
}  // namespace ajp::app::play
