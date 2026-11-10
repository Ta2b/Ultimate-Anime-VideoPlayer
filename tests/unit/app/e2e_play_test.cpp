// De bout en bout (P2.38) : le vrai lecteur joue un clip de 2 s sous Vulkan logiciel, sa fenêtre capturée et
// ses présentations relevées. Caché ([.e2e]) : lancé par le test ctest « e2e: » du job lavapipe, sous Xvfb.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stop_token>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "clip_writer.h"
#include "prepare/process.h"
#include "prepare/utf8.h"

namespace {

namespace fs = std::filesystem;

struct Picture {
  int width = 0;
  int height = 0;
  std::vector<unsigned char> rgb;
};

// Un PPM binaire (P6, 8 bits) comme l'écrit --snapshot.
Picture read_ppm(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  std::string magic;
  int max = 0;
  Picture picture;
  in >> magic >> picture.width >> picture.height >> max;
  in.get();
  if (magic != "P6" || max != 255 || picture.width <= 0 || picture.height <= 0) return {};
  picture.rgb.resize(static_cast<std::size_t>(picture.width) * static_cast<std::size_t>(picture.height) * 3);
  in.read(reinterpret_cast<char*>(picture.rgb.data()),  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
          static_cast<std::streamsize>(picture.rgb.size()));
  return in ? picture : Picture{};
}

// La luma moyenne (BT.709, pleine plage) du quart central : loin du bord et de la barre de lecture.
double central_luma(const Picture& p) {
  double sum = 0;
  int count = 0;
  for (int y = p.height * 3 / 8; y < p.height * 5 / 8; ++y) {
    for (int x = p.width * 3 / 8; x < p.width * 5 / 8; ++x) {
      const std::size_t at =
          (static_cast<std::size_t>(y) * static_cast<std::size_t>(p.width) + static_cast<std::size_t>(x)) * 3;
      sum += 0.2126 * p.rgb[at] + 0.7152 * p.rgb[at + 1] + 0.0722 * p.rgb[at + 2];
      ++count;
    }
  }
  return count > 0 ? sum / count : 0;
}

// Les colonnes « fresh » et « dropped » de --stats : images neuves présentées, et le total des images sautées
// (cumulé d'une ligne à l'autre).
struct Presented {
  int fresh = 0;
  std::int64_t dropped = 0;
};

Presented read_stats(const fs::path& path) {
  std::ifstream in(path);
  std::string line;
  std::getline(in, line);
  std::vector<std::string> names;
  {
    std::istringstream header(line);
    for (std::string name; std::getline(header, name, '\t');) names.push_back(name);
  }
  Presented out;
  while (std::getline(in, line)) {
    std::istringstream row(line);
    std::size_t column = 0;
    for (std::string cell; std::getline(row, cell, '\t'); ++column) {
      if (column >= names.size()) break;
      if (names[column] == "fresh" && cell == "1") ++out.fresh;
      if (names[column] == "dropped" && !cell.empty())
        out.dropped = std::max<std::int64_t>(out.dropped, std::stoll(cell));
    }
  }
  return out;
}

struct Run {
  std::int64_t code = -1;
  std::string output;  // sortie du lecteur, journal compris
};

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
  Run run;
  const auto exit = ajp::prepare::run_process(
      {.program = fs::path(AJP_PLAYER_EXE), .args = std::move(args), .timeout = std::chrono::seconds(120)},
      [&run](std::string_view line) {
        run.output += line;
        run.output += '\n';
      },
      std::stop_token{});
  if (exit && exit->kind == ajp::prepare::ExitKind::Exited) run.code = exit->code;
  return run;
}

fs::path fresh_dir(const std::string& name) {
  const fs::path dir = fs::temp_directory_path() / name;
  fs::remove_all(dir);
  fs::create_directories(dir);
  return dir;
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
  const Run run = play(*clip, dir, {"--seconds=1", "--stats=" + ajp::prepare::utf8_path(stats)});
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
  const Run run = play(*clip, dir, {"--keys=0.5:Space,1.5:q"});
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
       .args = {"play", ajp::prepare::utf8_path(*clip), "--software-gpu", "--mute", "--no-upscale",
                "--rife=0", "--no-resume", "--window=320x180", "--seconds=2",
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
  // L'image gradée du clip (luma 200 en plage limitée, chroma du plan 0 : un vert pâle). Le grade SDR la
  // rend vers 125 sur 255 au centre, mesuré le 07/11/2026 ; la fourchette tient ses réglages ordinaires, une
  // image noire, délavée ou aux canaux permutés en sort.
  const double luma = central_luma(picture);
  CHECK(luma > 100.0);
  CHECK(luma < 160.0);
  const std::size_t center =
      (static_cast<std::size_t>(picture.height / 2) * static_cast<std::size_t>(picture.width) +
       static_cast<std::size_t>(picture.width / 2)) *
      3;
  CHECK(picture.rgb[center + 1] > picture.rgb[center] + 20);      // vert au-dessus du rouge
  CHECK(picture.rgb[center + 1] > picture.rgb[center + 2] + 20);  // et du bleu
  CHECK(output.find("épisode suivant") == std::string::npos);     // --seconds quitte, sans enchaîner
  const Presented presented = read_stats(stats);
  // 50 images en 2 s. Sur le runner de la CI, lavapipe met environ 40 ms par image et en saute une partie
  // (25 neuves, 11 sautées le 07/11/2026) : la cadence n'est pas jugée ici, seulement que la lecture avance
  // et que chaque image manquante est comptée.
  CHECK(presented.fresh >= 15);
  CHECK(presented.fresh + presented.dropped >= 30);
}
