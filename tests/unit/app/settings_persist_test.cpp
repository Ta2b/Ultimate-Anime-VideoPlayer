// play.toml ne perd rien : une écriture refusée se rattrape à la suivante (A22-02), un retour à une
// version plus ancienne garde les valeurs et les clés de la plus récente (A36-01, A22-06).
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "app/play/chain.h"
#include "app/play/invalid_copies.h"
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
// Un play.toml d'avant « Lisser les dessins tenus » se lit, l'option éteinte ; allumée, elle se relit.
TEST_CASE("play.toml : « Lisser les dessins tenus », absent d'un ancien fichier, s'enregistre",
          "[app][settings][smooth]") {
  const std::filesystem::path path = std::filesystem::temp_directory_path() / "ajp-play-smooth.toml";
  REQUIRE(save_settings(path, PlaySettings{}));
  std::string text = read_text(path);
  const std::size_t line = text.find("smooth_drawings = false\n");
  REQUIRE(line != std::string::npos);
  text.erase(line, std::string("smooth_drawings = false\n").size());
  write_text(path, text);
  auto old = load_settings(path);
  REQUIRE(old);
  CHECK_FALSE(old->smooth_drawings);
  CHECK(old->keep_poses);
  old->smooth_drawings = true;
  REQUIRE(save_settings(path, *old));
  CHECK(read_text(path).find("smooth_drawings = true") != std::string::npos);
  const auto reread = load_settings(path);
  REQUIRE(reread);
  CHECK(reread->smooth_drawings);
  std::filesystem::remove(path);
}

// Un play.toml d'avant le warp de Catmull-Rom se lit, l'option éteinte ; allumée, elle se relit.
TEST_CASE("play.toml : le warp de Catmull-Rom, absent d'un ancien fichier, s'enregistre",
          "[app][settings][cubic]") {
  const std::filesystem::path path = std::filesystem::temp_directory_path() / "ajp-play-cubic.toml";
  REQUIRE(save_settings(path, PlaySettings{}));
  std::string text = read_text(path);
  const std::size_t line = text.find("cubic_warp = false\n");
  REQUIRE(line != std::string::npos);
  text.erase(line, std::string("cubic_warp = false\n").size());
  write_text(path, text);
  auto old = load_settings(path);
  REQUIRE(old);
  CHECK_FALSE(old->cubic_warp);
  old->cubic_warp = true;
  REQUIRE(save_settings(path, *old));
  const auto reread = load_settings(path);
  REQUIRE(reread);
  CHECK(reread->cubic_warp);
  std::filesystem::remove(path);
}

// Un play.toml d'avant l'échelle par paire se lit, l'option à son défaut (allumée) ; éteinte, elle se relit
// éteinte : un fichier qui l'a enregistrée garde son choix.
TEST_CASE("play.toml : l'échelle par paire, absente d'un ancien fichier, s'enregistre",
          "[app][settings][pair_scale]") {
  const std::filesystem::path path = std::filesystem::temp_directory_path() / "ajp-play-pair-scale.toml";
  REQUIRE(save_settings(path, PlaySettings{}));
  std::string text = read_text(path);
  const std::size_t line = text.find("pair_scale = true\n");
  REQUIRE(line != std::string::npos);
  text.erase(line, std::string("pair_scale = true\n").size());
  write_text(path, text);
  auto old = load_settings(path);
  REQUIRE(old);
  CHECK(old->pair_scale);
  old->pair_scale = false;
  REQUIRE(save_settings(path, *old));
  const auto reread = load_settings(path);
  REQUIRE(reread);
  CHECK_FALSE(reread->pair_scale);
  std::filesystem::remove(path);
}

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

TEST_CASE("C36-01 : play.toml illisible au lancement puis libéré, les changements de la séance s'écrivent",
          "[app][settings]") {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "ajp-play-unreadable-then-free";
  std::filesystem::remove_all(dir);
  const std::filesystem::path path = dir / "play.toml";
  std::filesystem::create_directories(path);  // tenu par un antivirus au lancement
  REQUIRE_FALSE(load_settings(path));
  // Libéré une seconde plus tard : le fichier d'avant, avec ses choix à lui.
  std::filesystem::remove_all(path);
  PlaySettings before_launch;
  before_launch.smooth_motion = !PlaySettings{}.smooth_motion;
  REQUIRE(save_settings(dir / "autre.toml", before_launch));
  std::filesystem::rename(dir / "autre.toml", path);
  // La séance, partie des défauts, change le volume : écrit (avant : « illisible au lancement » sans fin).
  PlaySettings session;
  session.volume = 0.5F;
  REQUIRE(save_settings(path, session));
  const auto reread = load_settings(path);
  REQUIRE(reread);
  CHECK(reread->volume == 0.5F);
  // Ce que la séance n'a pas changé reste celui du fichier.
  CHECK(reread->smooth_motion == before_launch.smooth_motion);
  std::filesystem::remove_all(dir);
}

TEST_CASE("C30-03 : une écriture différée plus ancienne n'écrase pas l'écriture directe qui la suit",
          "[app][settings]") {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "ajp-play-order";
  std::filesystem::remove_all(dir);
  const std::filesystem::path path = dir / "play.toml";
  PlaySettings older;
  older.volume = 0.25F;
  const std::uint64_t posted = settings_ticket();  // l'écriture postée au fil, en retard
  PlaySettings newer;
  newer.volume = 0.75F;
  REQUIRE(save_settings(path, newer));          // le thread principal écrit tout de suite
  REQUIRE(save_settings(path, older, posted));  // puis le fil finit : sautée
  const auto reread = load_settings(path);
  REQUIRE(reread);
  CHECK(reread->volume == 0.75F);
  std::filesystem::remove_all(dir);
}

// C22-01 : un verrou passager à la relecture (antivirus) ne ferme pas les écritures jusqu'à la sortie.
TEST_CASE("play.toml : un verrou passager à la relecture n'interdit pas les écritures suivantes (C22-01)",
          "[app][settings]") {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "ajp-play-transient";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const std::filesystem::path path = dir / "play.toml";
  REQUIRE(save_settings(path, PlaySettings{}));
  // Le fichier devient illisible un instant (un dossier à sa place), la relecture échoue...
  std::filesystem::rename(path, dir / "play.toml.bak");
  std::filesystem::create_directories(path);
  REQUIRE_FALSE(load_settings(path));
  // ...le refus tient tant que le fichier est illisible...
  PlaySettings changed;
  changed.volume = 0.5F;
  CHECK_FALSE(save_settings(path, changed));
  // ...puis le fichier redevient lisible : la lecture suivante lève le refus.
  std::filesystem::remove_all(path);
  std::filesystem::rename(dir / "play.toml.bak", path);
  REQUIRE(load_settings(path));
  CHECK(save_settings(path, changed));
  const auto reread = load_settings(path);
  REQUIRE(reread);
  CHECK(reread->volume == 0.5F);
  std::filesystem::remove_all(dir);
}

// C22-01 : sans nouvelle lecture, l'écriture elle-même relit le fichier avant de refuser.
TEST_CASE("play.toml : l'écriture relit un fichier redevenu lisible avant de refuser (C22-01)",
          "[app][settings]") {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "ajp-play-transient-save";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const std::filesystem::path path = dir / "play.toml";
  REQUIRE(save_settings(path, PlaySettings{}));
  std::filesystem::rename(path, dir / "play.toml.bak");
  std::filesystem::create_directories(path);
  REQUIRE_FALSE(load_settings(path));
  std::filesystem::remove_all(path);
  std::filesystem::rename(dir / "play.toml.bak", path);
  PlaySettings changed;
  changed.volume = 0.25F;
  CHECK(save_settings(path, changed));
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

// C22-06 : un play.toml invalide dont l'écriture échoue sans cesse ne multiplie pas les copies.
TEST_CASE("play.toml invalide : une copie identique déjà gardée suffit, six au plus (C22-06)",
          "[app][settings]") {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "ajp-play-copies";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const std::filesystem::path path = dir / "play.toml";
  const auto count = [&] {
    int copies = 0;
    for (const auto& entry : std::filesystem::directory_iterator(dir))
      if (entry.path().filename().string().starts_with("play.toml.invalide-")) ++copies;
    return copies;
  };
  PlaySettings changed;
  changed.volume = 0.5F;
  // Le même contenu invalide, trois fois de suite (l'écriture suivante le remet à l'identique) : une copie.
  for (int round = 0; round < 3; ++round) {
    write_text(path, "version = 1\n[image\nwork_height = 540\n");
    REQUIRE(save_settings(path, changed));
  }
  CHECK(count() == 1);
  // Des contenus invalides différents : les plus anciennes copies partent, cinq restent.
  for (int round = 0; round < 6; ++round) {
    write_text(path, std::format("version = 1\n[image\nwork_height = {}\n", 100 + round));
    REQUIRE(save_settings(path, changed));
    std::this_thread::sleep_for(std::chrono::milliseconds(1050));  // l'horodatage est à la seconde
  }
  CHECK(count() == 5);
  std::filesystem::remove_all(dir);
}

// C22-07 : un fichier qui n'est pas un fichier de réglages (plus d'un Mo) est refusé, nommé, jamais écrasé.
TEST_CASE("play.toml : plus d'un Mo est refusé, et le fichier n'est pas écrasé (C22-07)", "[app][settings]") {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "ajp-play-huge";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const std::filesystem::path path = dir / "play.toml";
  write_text(path, "version = 1\n" + std::string(std::size_t{1} << 21, '#') + "\n");
  const auto loaded = load_settings(path);
  REQUIRE_FALSE(loaded);
  CHECK(loaded.error().code == RefusalCode::FileUnreadable);
  PlaySettings changed;
  changed.volume = 0.5F;
  CHECK_FALSE(save_settings(path, changed));
  CHECK(std::filesystem::file_size(path) > (std::uintmax_t{1} << 20));
  std::filesystem::remove_all(dir);
}

// C22-03 : settings.toml se relit comme play.toml ; un dossier à sa place reste refusé.
TEST_CASE("settings.toml : lu par la boucle de réessais, absent = défauts (C22-03)", "[app][settings]") {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "ajp-engine-settings";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const auto absent = load_engine_settings(dir / "settings.toml");
  REQUIRE(absent);
  CHECK_FALSE(absent->first_run_done);
  std::filesystem::create_directories(dir / "blocked.toml");
  const auto refused = load_engine_settings(dir / "blocked.toml");
  REQUIRE_FALSE(refused);
  CHECK(refused.error().code == RefusalCode::FileUnreadable);
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

TEST_CASE("C36-10 : seules les trois dernières copies mises de côté restent", "[app][settings]") {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "ajp-invalid-copies";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  for (const char* stamp :
       {"20260101-000000", "20260201-000000", "20260301-000000", "20260401-000000", "20260501-000000"})
    std::ofstream(dir / (std::string("play.toml.invalide-") + stamp)) << "x";
  std::ofstream(dir / "settings.toml.invalide-20250101-000000") << "x";  // un autre fichier : à lui
  ajp::app::play::prune_invalid_copies(dir / "play.toml");
  CHECK_FALSE(std::filesystem::exists(dir / "play.toml.invalide-20260101-000000"));
  CHECK_FALSE(std::filesystem::exists(dir / "play.toml.invalide-20260201-000000"));
  CHECK(std::filesystem::exists(dir / "play.toml.invalide-20260301-000000"));
  CHECK(std::filesystem::exists(dir / "play.toml.invalide-20260501-000000"));
  CHECK(std::filesystem::exists(dir / "settings.toml.invalide-20250101-000000"));
  std::filesystem::remove_all(dir);
}
