// De bout en bout (P2.38) : le vrai lecteur joue un clip de 2 s sous Vulkan logiciel, sa fenêtre capturée et
// ses présentations relevées. Caché ([.e2e]) : lancé par le test ctest « e2e: » du job lavapipe, sous Xvfb.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <sstream>
#include <stop_token>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "clip_writer.h"
#include "e2e_util.h"
#include "prepare/process.h"
#include "prepare/utf8.h"

namespace {

using namespace ajp::e2e;  // NOLINT(google-build-using-namespace)

// Le vrai lecteur sur un fichier, sous Vulkan logiciel, réglages dans dir/reglages.
Run play(const fs::path& file, const fs::path& dir, std::vector<std::string> extra) {
  std::vector<std::string> args = {"play",
                                   ajp::prepare::utf8_path(file),
                                   "--software-gpu",
                                   "--mute",
                                   "--no-upscale",
                                   "--rife=0",
                                   "--no-resume",
                                   "--window=320x180",
                                   "--data-dir=" + ajp::prepare::utf8_path(dir / "reglages")};
  args.insert(args.end(), extra.begin(), extra.end());
  return run_player(std::move(args));
}

}  // namespace

// Recette, scénario 15 (P7.12) : un fichier coupé à mi-longueur se lit jusqu'où il va, puis le lecteur
// s'arrête proprement ; un fichier illisible ne ferme pas la fenêtre en silence, l'accueil dit pourquoi.
TEST_CASE("Bout en bout : fichier tronqué lu jusqu'à sa fin, fichier illisible refusé avec un message",
          "[.e2e][app]") {
  const auto clip = ajp::media::test::write_clip(ajp::media::test::ClipSpec{.name = "e2e-tronque-source.mkv",
                                                                            .width = 320,
                                                                            .height = 180,
                                                                            .frames = 50,
                                                                            .shot_frames = 50,
                                                                            .shot_lumas = {200}});
  REQUIRE(clip);
  const fs::path dir = fresh_dir("ajp-e2e-recette");
  const fs::path truncated = dir / "tronque.mkv";
  {
    std::ifstream in(*clip, std::ios::binary);
    std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::ofstream(truncated, std::ios::binary)
        .write(bytes.data(), static_cast<std::streamsize>(bytes.size() / 2));
  }
  const fs::path stats = dir / "stats.tsv";
  const Run cut = play(truncated, dir, {"--seconds=3", "--stats=" + ajp::prepare::utf8_path(stats)});
  INFO(cut.output);
  CHECK(cut.code == 0);
  CHECK(read_stats(stats).fresh >= 5);

  const fs::path garbage = dir / "abime.mkv";
  std::ofstream(garbage, std::ios::binary) << std::string(4096, 'x');
  // L'accueil attend : q le ferme après 1,5 s.
  const Run bad = play(garbage, dir, {"--keys=1.5:q"});
  INFO(bad.output);
  CHECK(bad.code >= 0);  // sorti de lui-même, ni planté ni bloqué
  CHECK(bad.output.find("format non reconnu") != std::string::npos);
}

// Recette, scénario 14 (P7.12) : une source de taille impaire (853×480, SD anamorphosée courante) se lit.
TEST_CASE("Bout en bout : une source 853×480 se lit", "[.e2e][app]") {
  const auto clip = ajp::media::test::write_clip(ajp::media::test::ClipSpec{.name = "e2e-853x480.mkv",
                                                                            .width = 853,
                                                                            .height = 480,
                                                                            .frames = 25,
                                                                            .shot_frames = 25,
                                                                            .shot_lumas = {200}});
  REQUIRE(clip);
  const fs::path dir = fresh_dir("ajp-e2e-853");
  const fs::path stats = dir / "stats.tsv";
  const Run run = play(alone(*clip, dir), dir, {"--seconds=1", "--stats=" + ajp::prepare::utf8_path(stats)});
  INFO(run.output);
  CHECK(run.code == 0);
  CHECK(read_stats(stats).fresh >= 3);
  // --seconds quitte : les autres clips du dossier ne s'enchaînent pas (ils récrivaient stats.tsv).
  CHECK(run.output.find("épisode suivant") == std::string::npos);
}

// Une touche, une action (0.10.0 : la file des actions n'était plus vidée, chaque tour de boucle rejouait la
// pause).
TEST_CASE("Bout en bout : une pause demandée une fois s'applique une fois", "[.e2e][app]") {
  const auto clip = ajp::media::test::write_clip(ajp::media::test::ClipSpec{.name = "e2e-pause.mkv",
                                                                            .width = 320,
                                                                            .height = 180,
                                                                            .frames = 50,
                                                                            .shot_frames = 50,
                                                                            .shot_lumas = {200}});
  REQUIRE(clip);
  const fs::path dir = fresh_dir("ajp-e2e-pause");
  const Run run = play(alone(*clip, dir), dir, {"--keys=0.5:Space,1.5:q"});
  INFO(run.output);
  CHECK(run.code >= 0);
  std::size_t pauses = 0;
  for (std::size_t at = run.output.find("action : pause"); at != std::string::npos;
       at = run.output.find("action : pause", at + 1)) {
    ++pauses;
  }
  CHECK(pauses == 1);
}

TEST_CASE("Bout en bout : le lecteur joue un clip de 2 s, image juste, images sautées comptées",
          "[.e2e][app]") {
  const fs::path player(AJP_PLAYER_EXE);
  REQUIRE(fs::is_regular_file(player));
  // Un seul plan clair (luma 200 sur 255, plage limitée), 2 s à 25 images par seconde.
  const auto clip = ajp::media::test::write_clip(ajp::media::test::ClipSpec{.name = "e2e-2s.mkv",
                                                                            .width = 320,
                                                                            .height = 180,
                                                                            .frames = 50,
                                                                            .shot_frames = 50,
                                                                            .shot_lumas = {200}});
  REQUIRE(clip);
  const fs::path dir = fs::temp_directory_path() / "ajp-e2e";
  fs::remove_all(dir);
  fs::create_directories(dir);
  const fs::path snapshot = dir / "capture.ppm";
  const fs::path stats = dir / "stats.tsv";
  std::string output;
  const auto exit = ajp::prepare::run_process(
      {.program = player,
       .args = {"play", ajp::prepare::utf8_path(alone(*clip, dir)), "--software-gpu", "--mute",
                "--no-upscale", "--rife=0", "--no-resume", "--window=640x360", "--seconds=2",
                "--data-dir=" + ajp::prepare::utf8_path(dir / "reglages"),
                "--stats=" + ajp::prepare::utf8_path(stats),
                "--snapshot=" + ajp::prepare::utf8_path(snapshot), "--snapshot-at=1.5"},
       .timeout = std::chrono::seconds(120)},
      [&output](std::string_view line) {
        output += line;
        output += '\n';
      },
      std::stop_token{});
  INFO(output);
  REQUIRE(exit);
  REQUIRE(exit->kind == ajp::prepare::ExitKind::Exited);
  REQUIRE(exit->code == 0);
  const Picture picture = read_ppm(snapshot);
  REQUIRE(picture.width > 0);
  // Le clip : luma 200 en plage limitée avec son dégradé de ±12, chroma du plan 0 (un vert pâle), soit RVB
  // 155, 238, 151 en BT.709 au milieu du dégradé (luma 214 sur 255, ±14). La sortie SDR la rend telle quelle,
  // grade compris (une image SDR claire et uniforme n'a rien à relever) : jugée contre la source, hors de la
  // barre de lecture.
  const double luma = clear_band_luma(picture);
  CHECK(luma > 200.0);
  CHECK(luma < 228.0);
  const std::size_t probe =
      (static_cast<std::size_t>(picture.height * 3 / 8) * static_cast<std::size_t>(picture.width) +
       static_cast<std::size_t>(picture.width / 2)) *
      3;
  CHECK(std::abs(picture.rgb[probe] - 155) <= 16);
  CHECK(std::abs(picture.rgb[probe + 1] - 238) <= 16);
  CHECK(std::abs(picture.rgb[probe + 2] - 151) <= 16);
  CHECK(output.find("épisode suivant") == std::string::npos);  // --seconds quitte, sans enchaîner
  const Presented presented = read_stats(stats);
  // 50 images en 2 s. Sur le runner de la CI, lavapipe met environ 40 ms par image et en saute une partie
  // (25 neuves, 11 sautées le 07/11/2026) : la cadence n'est pas jugée ici, seulement que la lecture avance
  // et que chaque image manquante est comptée.
  CHECK(presented.fresh >= 15);
  CHECK(presented.fresh + presented.dropped >= 30);
}

namespace {

// Le dossier de réglages d'un premier lancement d'essai : « Ouvrir avec » déjà noté, pour que l'assistant
// n'inscrive pas le lecteur d'essai sur le poste qui lance les tests.
fs::path first_launch_dir(const std::string& name) {
  const fs::path dir = fresh_dir(name);
  fs::create_directories(dir / "reglages");
  std::ofstream(dir / "reglages" / "settings.toml") << "version = 1\nopen_with_registered = true\n";
  return dir;
}

std::string slurp(const fs::path& path) {
  std::ifstream in(path);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

std::optional<fs::path> wizard_clip(const std::string& name) {
  auto clip = ajp::media::test::write_clip(ajp::media::test::ClipSpec{
      .name = name, .width = 320, .height = 180, .frames = 50, .shot_frames = 50, .shot_lumas = {200}});
  if (!clip) return std::nullopt;
  return *clip;
}

}  // namespace

// A25-14, A25-18 : run_engine_screen de bout en bout, au clavier seul. « q » n'y ferme plus le lecteur,
// Entrée vaut « Suivant », Tab mène à « Set it up myself » ; les choix s'écrivent et la lecture suit.
TEST_CASE("Bout en bout : l'assistant se parcourt au clavier, « q » ne le ferme pas", "[.e2e][app]") {
  const auto clip = wizard_clip("e2e-assistant.mkv");
  REQUIRE(clip);
  const fs::path dir = first_launch_dir("ajp-e2e-assistant");
  // Entrée vaut « Suivant » sur chaque écran sauf Prêt, où elle ne fait rien (first_run_wizard.cpp) : quatre
  // Entrée mènent à Prêt que l'écran Moteur soit là (carte NVIDIA avec le module TensorRT) ou non (CI). Sur
  // Prêt, Tab trois fois (Précédent, Calibrer maintenant, Régler moi-même) puis Entrée. Le dernier « q »,
  // lui, tombe pendant la lecture.
  const Run run =
      play(alone(*clip, dir), dir,
           {"--first-run",
            "--keys=0.5:q,1:Return,1.5:Return,2:Return,2.5:Return,3.1:Tab,3.5:Tab,3.9:Tab,4.3:Return,6.5:q"});
  INFO(run.output);
  CHECK(run.code == 0);
  CHECK(run.output.find("assistant : ") != std::string::npos);
  CHECK(run.output.find("moteurs choisis : Vulkan") != std::string::npos);
  CHECK(run.output.find("calibration de la machine") == std::string::npos);
  CHECK(slurp(dir / "reglages" / "settings.toml").find("first_run_done = true") != std::string::npos);
  CHECK(fs::exists(dir / "reglages" / "play.toml"));
}

// A25-18 : calibrate_on_screen de bout en bout. Les essais ne trouvent pas de carte ici (ils ne reçoivent
// pas --software-gpu) : la calibration échoue, l'assistant garde l'Auto estimé et la lecture suit.
TEST_CASE("Bout en bout : une calibration qui échoue à l'écran laisse l'Auto et la lecture", "[.e2e][app]") {
  const auto clip = wizard_clip("e2e-calibration.mkv");
  REQUIRE(clip);
  const fs::path dir = first_launch_dir("ajp-e2e-calibration");
  // Le clip de calibration déjà là : sa génération (des dizaines de secondes en Debug) n'est pas l'objet.
  fs::copy_file(*clip, dir / "reglages" / "bench-clip-v4.mkv");
  const Run run = play(
      alone(*clip, dir), dir,
      {"--first-run", "--keys=1:Return,1.5:Return,2:Return,2.5:Return,3.1:Tab,3.5:Tab,3.9:Return,6.5:q"});
  INFO(run.output);
  CHECK(run.code == 0);
  CHECK(run.output.find("calibration de la machine") != std::string::npos);
  CHECK(run.output.find("moteurs choisis : Vulkan") != std::string::npos);
  const std::string play_toml = slurp(dir / "reglages" / "play.toml");
  CHECK(play_toml.find("preset = 'auto'") != std::string::npos);
  // Le dossier des essais ne reste pas (A25-11).
  for (const auto& entry : fs::directory_iterator(dir / "reglages"))
    CHECK_FALSE(entry.path().filename().string().starts_with("calibration-essais-"));
}
