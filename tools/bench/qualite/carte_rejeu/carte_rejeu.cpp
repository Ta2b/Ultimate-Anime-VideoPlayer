// carte_rejeu : la carte de la scène du lecteur rejouée hors du lecteur (CMakeLists.txt à côté).
//
//   carte_rejeu [--diviseur D] < tâches
//
// Une tâche par ligne, champs séparés par des tabulations, dans l'ordre de lecture (le lissage des classes
// suit les paires d'une ligne à l'autre, comme SceneMap) : LARGEUR HAUTEUR A.y B.y FLUX SORTIE.
// A.y, B.y : lumas 8 bits de la paire, LARGEUR × HAUTEUR (le flux du lecteur : au plus 720 lignes).
// FLUX : ce qu'écrit nvof_dump --grid 4 --one-way --cost, vecteurs S10.5 (x, y en int16) puis coûts (uint8).
// SORTIE : par tuile, ligne par ligne, la classe (uint8 : 0 incertain, 1 décor, 2 dessin, 3 effet), puis
// « changée » (uint8), puis la confiance (float32). Sur la sortie standard, une ligne par paire :
// carte COLONNES LIGNES caméra_ajustée caméra_tenue dessin_tenu coupe fondu (0 ou 1).
// D : le diviseur du travail au flux (scene_flow_divisor), qui ne change que les pixels de travail.
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "kernels/ref/scene_flow.h"

namespace {

using namespace kirari::kernels;

bool read_all(const std::string& path, std::span<std::byte> out) {
  std::ifstream in(path, std::ios::binary);
  in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size()));
  return in.gcount() == static_cast<std::streamsize>(out.size());
}

template <class T>
bool read_plane(const std::string& path, Plane<T>& plane) {
  return read_all(path, std::as_writable_bytes(std::span(plane.data)));
}

bool write_map(const std::string& path, const SceneMotion& m) {
  std::vector<std::uint8_t> kinds;
  std::vector<std::uint8_t> changed;
  std::vector<float> confidence;
  for (const SceneTile& t : m.tiles) {
    kinds.push_back(static_cast<std::uint8_t>(t.kind));
    changed.push_back(t.changed ? 1U : 0U);
    confidence.push_back(t.confidence);
  }
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(kinds.data()), static_cast<std::streamsize>(kinds.size()));
  out.write(reinterpret_cast<const char*>(changed.data()), static_cast<std::streamsize>(changed.size()));
  out.write(reinterpret_cast<const char*>(confidence.data()),
            static_cast<std::streamsize>(confidence.size() * sizeof(float)));
  return static_cast<bool>(out);
}

// Une tâche : faux (et un message) si un fichier manque ou si la carte refuse la paire.
bool run(const std::string& line, int divisor, ref::SceneHistory& history) {
  std::istringstream fields(line);
  int width = 0;
  int height = 0;
  std::string a_path;
  std::string b_path;
  std::string flow_path;
  std::string out_path;
  if (!(fields >> width >> height >> a_path >> b_path >> flow_path >> out_path) || width < 1 || height < 1) {
    std::cerr << "carte_rejeu : tâche illisible : " << line << "\n";
    return false;
  }
  Plane<std::uint8_t> a(width, height);
  Plane<std::uint8_t> b(width, height);
  const int cw = scene_flow_cells(width);
  const int ch = scene_flow_cells(height);
  Plane<ref::FlowVector> flow(cw, ch);
  Plane<std::uint8_t> cost(cw, ch);
  std::vector<std::byte> raw(flow.area() * sizeof(ref::FlowVector) + cost.area());
  if (!read_plane(a_path, a) || !read_plane(b_path, b) || !read_all(flow_path, raw)) {
    std::cerr << "carte_rejeu : lecture impossible : " << line << "\n";
    return false;
  }
  std::memcpy(flow.data.data(), raw.data(), flow.area() * sizeof(ref::FlowVector));
  std::memcpy(cost.data.data(), raw.data() + flow.area() * sizeof(ref::FlowVector), cost.area());
  const auto tiles = ref::scene_tile_sums(a, b, flow, cost, divisor);
  if (!tiles) {
    std::cerr << "carte_rejeu : " << tiles.error().detail << "\n";
    return false;
  }
  const SceneMotion m = ref::scene_motion_of(*tiles, history);
  if (!write_map(out_path, m)) {
    std::cerr << "carte_rejeu : écriture impossible : " << out_path << "\n";
    return false;
  }
  std::cout << "carte " << m.cols << ' ' << m.rows << ' ' << int{m.camera_fit} << ' '
            << int{m.background_held} << ' ' << int{m.drawing_held} << ' ' << int{m.cut_likely} << ' '
            << int{m.fade} << "\n";
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  int divisor = 1;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    const std::string_view value = i + 1 < argc ? argv[i + 1] : "";
    const auto read = std::from_chars(value.data(), value.data() + value.size(), divisor);
    if (arg != "--diviseur" || read.ec != std::errc{} || read.ptr != value.data() + value.size() ||
        divisor < 1) {
      std::cerr << "usage : carte_rejeu [--diviseur D] < tâches\n";
      return 2;
    }
    ++i;
  }
  ref::SceneHistory history;
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    if (!run(line, divisor, history)) return 1;
  }
  return 0;
}
