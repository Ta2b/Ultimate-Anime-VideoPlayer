// play.toml ne perd rien : une écriture refusée se rattrape à la suivante (A22-02), un retour à une
// version plus ancienne garde les valeurs et les clés de la plus récente (A36-01, A22-06).
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <utility>

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

TEST_CASE("play.toml : une édition à la main survit à la relecture du fil des mises à jour (B36-02)",
          "[app][settings]") {
  const std::filesystem::path path = std::filesystem::temp_directory_path() / "ajp-play-peek.toml";
  std::filesystem::remove_all(path);
  REQUIRE(save_settings(path, PlaySettings{}));
  auto ours = load_settings(path);  // le lecteur au lancement
  REQUIRE(ours);
  PlaySettings edited = *ours;  // Tatoub règle le volume à la main, lecteur ouvert
  edited.volume = 0.25F;
  write_text(path, format_settings(edited));
  const auto peeked = peek_settings(path);  // le fil des mises à jour, toutes les 30 minutes
  REQUIRE(peeked);
  CHECK(peeked->volume == edited.volume);
  ours->audio_delay_ms = 120;  // puis un changement dans le menu
  REQUIRE(save_settings(path, *ours));
  const auto merged = load_settings(path);
  REQUIRE(merged);
  CHECK(merged->audio_delay_ms == 120);
  CHECK(merged->volume == edited.volume);
  std::filesystem::remove(path);
}

}  // namespace
}  // namespace ajp::app::play
