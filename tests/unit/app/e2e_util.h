// Ce que les cas de bout en bout partagent : lire la capture (--snapshot) et les statistiques (--stats) du
// vrai lecteur, et le lancer.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stop_token>
#include <string>
#include <utility>
#include <vector>

#include "prepare/process.h"

namespace ajp::e2e {

namespace fs = std::filesystem;

struct Picture {
  int width = 0;
  int height = 0;
  std::vector<unsigned char> rgb;
};

// Un PPM binaire (P6, 8 bits) comme l'écrit --snapshot.
inline Picture read_ppm(const fs::path& path) {
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

// La luma moyenne (BT.709, pleine plage) d'une bande au milieu de la largeur, entre 30 % et 45 % de la
// hauteur. La barre de lecture, affichée pendant les premières secondes, assombrit le haut de la fenêtre sur
// 96 px et le bas sur 185 px environ : dans une fenêtre de 360 lignes cette bande est la seule que rien ne
// couvre. Le centre d'une fenêtre de 180 lignes était sous les deux dégradés (A40-06 : image jugée 38 % plus
// sombre que la source, et la fourchette du test gravait cet écart).
inline double clear_band_luma(const Picture& p) {
  double sum = 0;
  int count = 0;
  for (int y = p.height * 30 / 100; y < p.height * 45 / 100; ++y) {
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

inline Presented read_stats(const fs::path& path) {
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

// Le vrai lecteur, ces arguments tels quels ; sa sortie (journal compris) gardée pour INFO.
inline Run run_player(std::vector<std::string> args) {
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

inline fs::path fresh_dir(const std::string& name) {
  const fs::path dir = fs::temp_directory_path() / name;
  fs::remove_all(dir);
  fs::create_directories(dir);
  return dir;
}

// Le clip seul dans le dossier du cas : write_clip range tous les clips ensemble, et un cas sans --seconds
// enchaînait les autres comme des épisodes suivants (A40-05).
inline fs::path alone(const fs::path& clip, const fs::path& dir) {
  const fs::path copy = dir / clip.filename();
  fs::copy_file(clip, copy, fs::copy_options::overwrite_existing);
  return copy;
}

}  // namespace ajp::e2e
