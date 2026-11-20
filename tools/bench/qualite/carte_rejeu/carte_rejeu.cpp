// carte_rejeu : la carte de la scène du lecteur rejouée hors du lecteur (CMakeLists.txt à côté).
//
//   carte_rejeu [--diviseur D] [--calques [--sans-affinage] [--vues DOSSIER] [--pas N]] < tâches
//
// Une tâche par ligne, champs séparés par des tabulations, dans l'ordre de lecture (le lissage des classes
// suit les paires d'une ligne à l'autre, comme SceneMap) : LARGEUR HAUTEUR A.y B.y FLUX SORTIE.
// A.y, B.y : lumas 8 bits de la paire, LARGEUR × HAUTEUR (le flux du lecteur : au plus 720 lignes).
// FLUX : ce qu'écrit nvof_dump --grid 4 --one-way --cost, vecteurs S10.5 (x, y en int16) puis coûts (uint8).
// SORTIE : par tuile, ligne par ligne, la classe (uint8 : 0 incertain, 1 décor, 2 dessin, 3 effet), puis
// « changée » (uint8), puis la confiance (float32). Sur la sortie standard, une ligne par paire :
// carte COLONNES LIGNES caméra_ajustée caméra_tenue dessin_tenu coupe fondu (0 ou 1), puis
// diag CANDIDATES MINIMUM PEU_TEXTURÉES SANS_CORRESPONDANCE ÉPARSES T25 T50 T75 : les tuiles qui peuvent
// ajuster la caméra, le nombre qu'il en faut, celles écartées (texture sous kFitTexture, coût, vecteurs
// épars, dans cet ordre), les quartiles de la texture (codes par pixel ; seuils de kernels::scene), puis
// camera AJUSTÉE AFFINÉE PAS ÉCART_IMAGE TUILES ÉCART_VECTEURS GARDÉES : la caméra comme le lecteur la tire
// (ajustée sur les vecteurs, affinée par Lucas-Kanade, spec/scene_camera.h ; PAS −1 : pas d'affinage
// demandé), l'écart mesuré sur l'image (pixels du flux, −1 : pas mesuré) et sur combien de tuiles, l'écart
// des vecteurs et les tuiles gardées par l'ajustement, puis
// image MESURÉES DÉCOR D50 D90 D99 D_ÉCART AUTRES A50 A90 BOUGENT DÉCALÉES EFFETS : l'écart après la caméra
// mesuré sur l'image (spec/scene_camera.h ; 0 tuile mesurée sans affinage) ; les tuiles mesurées, celles du
// décor (gardées par l'ajustement) avec les centiles 50, 90 et 99 de leur part de pixels qui bougent et la
// médiane de leur écart moyen |e| (codes), les autres avec leurs centiles 50 et 90 ; les tuiles qui bougent
// par leurs pixels (kMovingShare), par le seul recalage (kShiftMoving), et les effets par l'image
// (kEffectShare et kEffectCost). Les seuils de la séparation se règlent là-dessus : le décor donne le bruit.
// Puis tot DÉPART ÉCART_NVOF ÉCART_AFFINÉE ÉCART_IMAGE : l'affinage que le lecteur lance derrière le flux,
// depuis la caméra de la paire d'avant (scene_map.h ; DÉPART 0 : pas de caméra d'où partir, le reste à −1) ;
// son plus grand écart aux coins à la caméra de NVOF de la paire et à celle affinée depuis NVOF (pixels du
// flux, ref::scene_camera_gap ; −1 sans celle-ci), puis l'écart mesuré sur l'image après lui (−1 : pas
// mesuré). Le rejeu n'a pas les coupes du lecteur : la paire d'après une coupe part de l'autre plan.
// D : le diviseur du travail au flux (scene_flow_divisor), qui ne change que les pixels de travail.
// --calques : après la carte, les calques et la plaque de décor rejoués sur la paire (calques.h, ligne
// calques), avec une vue toutes les N paires (défaut 20) dans DOSSIER si --vues est donné. --sans-affinage :
// le pire du lecteur, l'affinage toujours en retard ; la carte part avec la caméra de NVOF, et la caméra lk
// de la paire ne sert qu'aux suivantes (layer_plan.h, with_neighbour).
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "calques.h"
#include "kernels/ref/scene_camera.h"
#include "kernels/ref/scene_flow.h"

namespace {

using namespace kirari::kernels;
namespace play = kirari::app::play;

bool read_all(const std::string& path, std::span<std::byte> out) {
  std::ifstream in(path, std::ios::binary);
  in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size()));
  return in.gcount() == static_cast<std::streamsize>(out.size());
}

template <class T>
bool read_plane(const std::string& path, Plane<T>& plane) {
  return read_all(path, std::as_writable_bytes(std::span(plane.data)));
}

// Pourquoi la caméra s'ajuste ou non : les critères de fit_candidate (kernels/ref/scene_fit.h), recopiés.
void print_diagnosis(const SceneFlowTiles& tiles) {
  int candidates = 0;
  int bare = 0;
  int unmatched = 0;
  int spread = 0;
  std::vector<float> textures;
  for (const SceneTileSums& t : tiles.tiles) {
    if (t.cells <= 0.0F || t.pixels <= 0.0F) continue;
    const float texture = t.texture / t.pixels;
    const float mx = t.sum_x / t.cells;
    const float my = t.sum_y / t.cells;
    const float scatter = std::sqrt(std::max(0.0F, t.sum_xx / t.cells - mx * mx) +
                                    std::max(0.0F, t.sum_yy / t.cells - my * my));
    textures.push_back(texture);
    if (texture < scene::kFitTexture) {
      ++bare;
    } else if (1.0F - t.unmatched / t.cells < scene::kFitMatched) {
      ++unmatched;
    } else if (scatter > scene::kSpreadMax) {
      ++spread;
    } else {
      ++candidates;
    }
  }
  const double share = std::ceil(scene::kFitMinShare * static_cast<double>(tiles.tiles.size()));
  const int minimum = std::max(scene::kFitMinTiles, static_cast<int>(share));
  std::ranges::sort(textures);
  const auto quartile = [&](std::size_t q) {
    return textures.empty() ? 0.0F : textures[q * (textures.size() - 1) / 4];
  };
  std::cout << "diag " << candidates << ' ' << minimum << ' ' << bare << ' ' << unmatched << ' ' << spread
            << ' ' << quartile(1) << ' ' << quartile(2) << ' ' << quartile(3) << "\n";
}

float centile(std::vector<float> values, int c) {
  if (values.empty()) return 0.0F;
  std::ranges::sort(values);
  return values[static_cast<std::size_t>(c) * (values.size() - 1) / 100];
}

// L'écart après la caméra, tuile par tuile : les critères de seen_of, by_image et light_of
// (kernels/ref/scene_flow.cpp), recopiés ; ceux d'une lumière tuile par tuile, sans ses voisines ni son
// vecteur (la carte, elle, les regarde).
void print_image(const SceneFlowTiles& tiles, const ref::SceneCamera& camera) {
  constexpr float kArea = static_cast<float>(kSceneTileFlowPixels * kSceneTileFlowPixels);
  const double unit = tiles.flow_height / 720.0;
  std::vector<float> decor;
  std::vector<float> decor_error;
  std::vector<float> others;
  int moving = 0;
  int shifted = 0;
  int effects = 0;
  int soft = 0;               // tuiles qui bougent sans bord net (kLightSharp)
  int lights = 0;             // et dont l'accord et l'écart gardé suffisent (kLightAgree, kLightKept)
  std::vector<float> agrees;  // accord des tuiles qui bougent
  for (std::size_t i = 0; i < camera.checks.size() && i < tiles.tiles.size(); ++i) {
    const SceneTileCheck& t = camera.checks[i];
    if (t.pixels < scene::kCheckCover * kArea) continue;
    const bool kept = i < camera.decor.size() && camera.decor[i] != 0U;
    (kept ? decor : others).push_back(t.moving);
    if (kept) decor_error.push_back(t.error);
    const bool by_shift =
        t.l1 >= scene::kCheckFloor * t.pixels && std::hypot(t.sx, t.sy) >= scene::kShiftMoving * unit;
    if (t.moving >= scene::kMovingShare) {
      ++moving;
      const float agree = std::abs(2.0F * t.brighter - t.moving) / t.moving;
      agrees.push_back(agree);
      if (t.sharp <= scene::kLightSharp * t.moving) {
        ++soft;
        if (agree >= scene::kLightAgree && t.settled >= scene::kLightKept * t.error) ++lights;
      }
    } else if (by_shift) {
      ++shifted;
    }
    const SceneTileSums& sums = tiles.tiles[i];
    const float cost = sums.cells > 0.0F ? sums.cost / sums.cells : 0.0F;
    if (t.moving >= scene::kEffectShare && cost >= scene::kEffectCost) ++effects;
  }
  std::cout << "image " << decor.size() + others.size() << ' ' << decor.size() << ' ' << centile(decor, 50)
            << ' ' << centile(decor, 90) << ' ' << centile(decor, 99) << ' ' << centile(decor_error, 50)
            << ' ' << others.size() << ' ' << centile(others, 50) << ' ' << centile(others, 90) << ' '
            << moving << ' ' << shifted << ' ' << effects << ' ' << soft << ' ' << lights << ' '
            << centile(agrees, 50) << ' ' << centile(agrees, 90) << "\n";
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

// La ligne tot : l'affinage depuis la caméra de la paire d'avant (previous), face à la caméra de NVOF de la
// paire (nvof) et à celle affinée depuis elle (refined).
bool print_early(const Plane<std::uint8_t>& a, const Plane<std::uint8_t>& b, const SceneFlowTiles& tiles,
                 const ref::SceneCamera& nvof, const ref::SceneCamera& refined,
                 const std::optional<ref::SceneCamera>& previous) {
  const auto request = previous && nvof.fit ? ref::scene_camera_request(*previous) : std::nullopt;
  if (!request) {
    std::cout << "tot 0 -1 -1 -1\n";
    return true;
  }
  const auto early = ref::scene_camera_refine(a, b, *request);
  if (!early) {
    std::cerr << "carte_rejeu : " << early.error().detail << "\n";
    return false;
  }
  ref::SceneCamera camera = nvof;
  ref::scene_camera_apply(camera, *early);
  const int w = tiles.flow_width;
  const int h = tiles.flow_height;
  std::cout << "tot 1 " << ref::scene_camera_gap(early->model, nvof.model, w, h) << ' '
            << (refined.refined ? ref::scene_camera_gap(early->model, refined.model, w, h) : -1.0) << ' '
            << camera.check << "\n";
  return true;
}

// Une tâche : faux (et un message) si un fichier manque ou si la carte refuse la paire. previous : la caméra
// de la paire d'avant, puis celle-ci. late : l'affinage toujours en retard (--sans-affinage), comme au pire
// dans le lecteur.
bool run(const std::string& line, int divisor, ref::SceneHistory& history,
         std::optional<ref::SceneCamera>& previous, carte_rejeu::LayerReplay* layers, bool late) {
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
  // La caméra comme le lecteur : ajustée sur les vecteurs, puis affinée sur les lumas (scene_map.h).
  ref::SceneCamera camera = ref::scene_camera_of(*tiles);
  const ref::SceneCamera nvof = camera;
  int steps = -1;  // pas d'affinage demandé
  if (const auto request = ref::scene_camera_request(camera)) {
    const auto refined = ref::scene_camera_refine(a, b, *request);
    if (!refined) {
      std::cerr << "carte_rejeu : " << refined.error().detail << "\n";
      return false;
    }
    ref::scene_camera_apply(camera, *refined);
    steps = refined->steps;
  }
  // En retard : la carte part avec la caméra de NVOF ; la caméra lk, relue après, sert aux paires suivantes.
  std::optional<play::NeighbourCamera> refined_late;
  if (late) {
    ref::SceneHistory scratch = history;
    refined_late = play::refined_camera(ref::scene_motion_of(*tiles, scratch, camera));
  }
  const SceneMotion m = ref::scene_motion_of(*tiles, history, late ? nvof : camera);
  if (!write_map(out_path, m)) {
    std::cerr << "carte_rejeu : écriture impossible : " << out_path << "\n";
    return false;
  }
  std::cout << "carte " << m.cols << ' ' << m.rows << ' ' << int{m.camera_fit} << ' '
            << int{m.background_held} << ' ' << int{m.drawing_held} << ' ' << int{m.cut_likely} << ' '
            << int{m.fade} << "\n";
  print_diagnosis(*tiles);
  int kept = 0;
  for (const std::uint8_t d : camera.decor) kept += d;
  std::cout << "camera " << int{camera.fit} << ' ' << int{camera.refined} << ' ' << steps << ' '
            << camera.check << ' ' << camera.check_tiles << ' ' << camera.residual << ' ' << kept << "\n";
  print_image(*tiles, camera);
  if (!print_early(a, b, *tiles, nvof, camera, previous)) return false;
  previous = camera.fit ? std::optional(camera) : std::nullopt;
  if (layers != nullptr && !layers->pair(a, b, m, refined_late)) {
    std::cerr << "carte_rejeu : vue des calques impossible : " << line << "\n";
    return false;
  }
  return true;
}

// Un entier d'au moins 1, en entier.
bool read_count(std::string_view value, int& out) {
  const auto read = std::from_chars(value.data(), value.data() + value.size(), out);
  return read.ec == std::errc{} && read.ptr == value.data() + value.size() && out >= 1;
}

}  // namespace

int main(int argc, char** argv) {
  int divisor = 1;
  bool layers = false;
  bool dump_grid = false;
  bool late = false;
  std::string views;
  int every = 20;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    const std::string_view value = i + 1 < argc ? argv[i + 1] : "";
    bool ok = true;
    if (arg == "--calques") {
      layers = true;
    } else if (arg == "--grille-calques") {
      layers = true;
      dump_grid = true;
    } else if (arg == "--sans-affinage") {
      late = true;
    } else if (arg == "--diviseur") {
      ok = read_count(value, divisor);
      ++i;
    } else if (arg == "--pas") {
      ok = read_count(value, every);
      ++i;
    } else if (arg == "--vues" && !value.empty()) {
      views = value;
      ++i;
    } else {
      ok = false;
    }
    if (!ok) {
      std::cerr << "usage : carte_rejeu [--diviseur D] [--calques|--grille-calques] [--sans-affinage] "
                   "[--vues DOSSIER] [--pas N] < tâches\n";
      return 2;
    }
  }
  std::optional<carte_rejeu::LayerReplay> replay;
  if (layers) replay.emplace(divisor, views, every, dump_grid);
  ref::SceneHistory history;
  std::optional<ref::SceneCamera> previous;
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    if (!run(line, divisor, history, previous, replay ? &*replay : nullptr, late)) return 1;
  }
  return 0;
}
