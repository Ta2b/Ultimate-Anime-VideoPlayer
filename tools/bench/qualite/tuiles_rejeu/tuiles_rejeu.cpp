// tuiles_rejeu : la reprise entière de SPAN (lot A) et ses tuiles qui bougent, rejouées hors du lecteur sur
// processeur (CMakeLists.txt à côté ; docs/tuiles-span.md, « Rejeu sur processeur »).
//
//   tuiles_rejeu [--diviseur D] [--blocs N] [--debande] [--sans-reprise] [--detail] [--dossier DOSSIER]
//                < tâches
//
// Les tâches de carte_rejeu, d'un seul extrait : une paire voisine par ligne, dans l'ordre de lecture,
// LARGEUR HAUTEUR A.y B.y FLUX (la suite de la ligne, la SORTIE de carte_rejeu, est ignorée). Lumas 8 bits
// et flux de nvof_dump --grid 4 --one-way --cost à la taille du flux du lecteur ; D : le diviseur du travail
// au flux (scene_flow_divisor), le travail fait LARGEUR × D sur HAUTEUR × D. DOSSIER : des tâches écrites
// sur une autre machine, dont chaque fichier se cherche par son seul nom dans DOSSIER.
//
// Pour chaque image, comme le lecteur :
// - la carte de la paire (précédente, image) : kernels::ref, caméra ajustée puis affinée, lissage des
// classes,
//   historique vidé après une coupe (app/play/scene_map.cpp) ;
// - la mesure de la paire et l'écart à l'ancre (kernels/spec/detect.h) : pire bloc et écart de chaque bloc,
//   en 1/256 de niveau, sur des blocs de 16 pixels de travail (16 / D pixels du flux) ;
// - SpanReuse::decide puis, sur les images calculées, SpanTiles::plan : le code du lecteur, compilé tel quel,
//   sur la grille du SPAN FP16 natif à N blocs (8 par défaut : tuiles de 64 × 16 pixels de travail, halo
//   d'une colonne et de quatre lignes, gpu/cuda/backend/span_native16.cu). --debande : l'entrée de SPAN
//   débandée (le rayon du débandage s'ajoute au halo) ; --sans-reprise : --no-span-reuse.
//
// Sortie : avec --detail, une ligne par image, « image K SORTE TUILES SX SY CŒUR TOTAL AJUSTÉE CX CY PAIRE »
// : SORTE C (calculée entière), R (reprise entière), T (par tuiles) ; TUILES 1 si par tuiles ; SX SY le
// décalage de la sortie d'avant ; CŒUR sur TOTAL les tuiles recalculées ; AJUSTÉE 1 si la carte a une caméra
// ; CX CY son déplacement au centre de l'image (pixels de travail) ; PAIRE le pire bloc de la paire
// (niveaux). Puis les deux lignes du journal de fin du lecteur, « SPAN repris : … » et « SPAN par tuiles :
// … ».
//
// Écarts au lecteur : la mesure de la paire lit ici la luma du flux, réduite d'un facteur D par moyenne de
// surface, et non le RGB de travail ou la copie de RIFE : le grain s'y moyenne, une paire y paraît un peu
// plus tenue (plus de reprises entières) et le filet des blocs un peu moins nerveux ; un vrai changement
// garde son écart moyen sur la même surface. La carte ne manque jamais une paire (pas de carte en retard ni
// perdue). La chaîne est celle d'une lecture SPAN avant le warp, sortie d'avant intacte (décalage possible).
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "app/play/upscale_reuse.h"
#include "kernels/ref/deband.h"
#include "kernels/ref/scene_camera.h"
#include "kernels/ref/scene_flow.h"
#include "kernels/spec/detect.h"
#include "kernels/spec/span_tiles.h"

namespace {

using namespace kirari;
using namespace kirari::kernels;

struct Options {
  int divisor = 1;
  int blocks = 8;  // blocs du SPAN (SpanFp8Weights::kBlocks pour AnimeJaNai)
  bool debanded = false;
  bool reuse = true;
  bool detail = false;
  std::string folder;  // non vide : chaque fichier d'une tâche se cherche là, par son nom
};

bool read_all(const std::string& path, std::span<std::byte> out) {
  std::ifstream in(path, std::ios::binary);
  in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size()));
  return in.gcount() == static_cast<std::streamsize>(out.size());
}

// La grille du SPAN FP16 natif (span_native16.cu, tile_grid) à la taille de travail : tuiles de 32 × 8 pixels
// de demi-résolution ; portée d'un pixel de demi-résolution par 3×3 du plus long chemin (conv_1, trois par
// bloc, to_pixel).
SpanTileGrid native_grid(int work_w, int work_h, int blocks) {
  constexpr int kTw = 32, kTh = 8;
  const int reach = 1 + 3 * blocks + 1;
  return SpanTileGrid{.tile_w = 2 * kTw,
                      .tile_h = 2 * kTh,
                      .cols = span_tiles_along(work_w / 2, kTw),
                      .rows = span_tiles_along(work_h / 2, kTh),
                      .reach_cols = span_reach_tiles(reach, kTw),
                      .reach_rows = span_reach_tiles(reach, kTh),
                      .shift_step = 2,
                      .scale = 2};
}

// La mesure d'une paire de lumas (spec/detect.h) : l'écart de chaque bloc de side pixels, tronqué aux bords,
// en 1/256 de niveau (des codes 8 bits de la plage de la source : 256 par code), et le pire.
struct Measure {
  std::vector<std::uint32_t> deviation;
  std::uint32_t cols = 0;
  std::uint32_t worst = 0;
};

Measure measure(const Plane<std::uint8_t>& a, const Plane<std::uint8_t>& b, int side) {
  Measure m;
  const int cols = (a.width + side - 1) / side, rows = (a.height + side - 1) / side;
  m.cols = static_cast<std::uint32_t>(cols);
  m.deviation.assign(static_cast<std::size_t>(cols) * static_cast<std::size_t>(rows), 0U);
  for (int by = 0; by < rows; ++by) {
    for (int bx = 0; bx < cols; ++bx) {
      std::uint64_t sum = 0, n = 0;
      for (int y = by * side; y < std::min(a.height, (by + 1) * side); ++y) {
        for (int x = bx * side; x < std::min(a.width, (bx + 1) * side); ++x) {
          sum +=
              static_cast<std::uint64_t>(std::abs(int{a.data[a.index(x, y)]} - int{b.data[b.index(x, y)]}));
          ++n;
        }
      }
      const auto dev = static_cast<std::uint32_t>(sum * 256U / n);
      m.deviation[static_cast<std::size_t>(by) * static_cast<std::size_t>(cols) +
                  static_cast<std::size_t>(bx)] = dev;
      m.worst = std::max(m.worst, dev);
    }
  }
  return m;
}

// La carte de la paire comme app/play/scene_map.cpp : caméra ajustée sur les vecteurs, affinée sur les
// lumas, puis les classes lissées par l'historique.
std::optional<SceneMotion> scene_of(const Plane<std::uint8_t>& a, const Plane<std::uint8_t>& b,
                                    const std::string& flow_path, int divisor, ref::SceneHistory& history) {
  const int cw = scene_flow_cells(a.width), ch = scene_flow_cells(a.height);
  Plane<ref::FlowVector> flow(cw, ch);
  Plane<std::uint8_t> cost(cw, ch);
  std::vector<std::byte> raw(flow.area() * sizeof(ref::FlowVector) + cost.area());
  if (!read_all(flow_path, raw)) {
    std::cerr << "tuiles_rejeu : lecture impossible : " << flow_path << "\n";
    return std::nullopt;
  }
  std::memcpy(flow.data.data(), raw.data(), flow.area() * sizeof(ref::FlowVector));
  std::memcpy(cost.data.data(), raw.data() + flow.area() * sizeof(ref::FlowVector), cost.area());
  const auto tiles = ref::scene_tile_sums(a, b, flow, cost, divisor);
  if (!tiles) {
    std::cerr << "tuiles_rejeu : " << tiles.error().detail << "\n";
    return std::nullopt;
  }
  ref::SceneCamera camera = ref::scene_camera_of(*tiles);
  if (const auto request = ref::scene_camera_request(camera)) {
    const auto refined = ref::scene_camera_refine(a, b, *request);
    if (!refined) {
      std::cerr << "tuiles_rejeu : " << refined.error().detail << "\n";
      return std::nullopt;
    }
    ref::scene_camera_apply(camera, *refined);
  }
  return ref::scene_motion_of(*tiles, history, camera);
}

class Replay {
 public:
  explicit Replay(const Options& options) : options_(options) {
    reuse_.enabled = options.reuse;
    // Comme le lecteur : la mesure contre l'ancre sert dès la deuxième reprise d'une série.
    reuse_.anchor_measure = true;
  }

  bool pair(int width, int height, const std::string& a_path, const std::string& b_path,
            const std::string& flow_path) {
    Plane<std::uint8_t> a(width, height);
    Plane<std::uint8_t> b(width, height);
    if (!read_all(a_path, std::as_writable_bytes(std::span(a.data))) ||
        !read_all(b_path, std::as_writable_bytes(std::span(b.data)))) {
      std::cerr << "tuiles_rejeu : lecture impossible : " << a_path << " ou " << b_path << "\n";
      return false;
    }
    if (!started_) start(a);
    if (clear_next_) history_.clear();  // la paire qui suit une coupe repart sans historique
    const std::optional<SceneMotion> scene = scene_of(a, b, flow_path, options_.divisor, history_);
    if (!scene) return false;
    clear_next_ = scene->cut_likely;
    const Measure paired = measure(a, b, side_);
    reuse_.note_pair(paired.worst);
    app::play::ReuseFacts facts;
    facts.same_epoch = true;
    facts.has_output = true;
    facts.pair_held = is_held(paired.worst);
    facts.tiles_stale = app::play::TileStale::None;
    facts.can_shift = true;
    facts.scene = &*scene;
    facts.spread_px = options_.debanded ? kDebandSpread : 0;
    facts.input_key = kInputKey;
    facts.blocks = app::play::PairBlocks{.deviation = paired.deviation,
                                         .cols = paired.cols,
                                         .side_px = static_cast<float>(side_ * options_.divisor)};
    Measure gap;
    if (reuse_.in_row() > 0) {  // la première reprise d'une série lit la paire, les suivantes l'ancre
      gap = measure(anchor_, b, side_);
      facts.anchor = app::play::AnchorGap{.worst_block = gap.worst};
    }
    const app::play::ReuseKind kind = reuse_.decide(facts);
    if (kind != app::play::ReuseKind::Whole) anchor_ = b;
    print(kind, *scene, paired.worst);
    ++frame_;
    return true;
  }

  void finish() const { std::cout << reuse_.summary() << "\n" << reuse_.tiles.summary() << "\n"; }

 private:
  static constexpr std::uint64_t kInputKey = 1;  // une seule recette de l'entrée de SPAN
  // Le rayon du débandage, arrondi au-dessus comme upscale_span.cpp.
  static constexpr int kDebandSpread = static_cast<int>(ref::kDebandRadius + 0.999F);

  // La première image : le début de la lecture, calculée entière ; elle devient l'ancre.
  void start(const Plane<std::uint8_t>& first) {
    started_ = true;
    work_w_ = first.width * options_.divisor;
    work_h_ = first.height * options_.divisor;
    side_ = std::max(1, static_cast<int>(std::lround(static_cast<double>(kHeldBlock) / options_.divisor)));
    reuse_.tiles.set_grid(native_grid(work_w_, work_h_, options_.blocks));
    app::play::ReuseFacts facts;  // ni même époque, ni sortie d'avant
    facts.tiles_stale = app::play::TileStale::None;
    facts.input_key = kInputKey;
    reuse_.decide(facts);
    anchor_ = first;
    frame_ = 1;
  }

  void print(app::play::ReuseKind kind, const SceneMotion& scene, std::uint32_t worst) const {
    if (!options_.detail) return;
    const auto& c = scene.camera;
    const double x = work_w_ / 2.0, y = work_h_ / 2.0;
    const double cx = c[0] + (c[1] - 1.0) * x + c[2] * y, cy = c[3] + c[4] * x + (c[5] - 1.0) * y;
    const app::play::TilePlan& plan = reuse_.tile_plan;
    const char sort = kind == app::play::ReuseKind::Whole   ? 'R'
                      : kind == app::play::ReuseKind::Tiles ? 'T'
                                                            : 'C';
    const int total = reuse_.tiles.grid().count();
    const int core = plan.tiled ? count_span_tiles(plan.core) : total;
    std::cout << "image " << frame_ << ' ' << sort << ' ' << int{plan.tiled} << ' ' << plan.shift_x << ' '
              << plan.shift_y << ' ' << (kind == app::play::ReuseKind::Whole ? 0 : core) << ' ' << total
              << ' ' << int{scene.camera_fit} << ' ' << cx << ' ' << cy << ' '
              << static_cast<double>(worst) / 256.0 << "\n";
  }

  Options options_;
  app::play::SpanReuse reuse_;
  ref::SceneHistory history_;
  Plane<std::uint8_t> anchor_;
  bool started_ = false;
  bool clear_next_ = false;
  int work_w_ = 0;
  int work_h_ = 0;
  int side_ = kHeldBlock;
  int frame_ = 0;
};

bool parse_int(std::string_view text, int& value) {
  const auto read = std::from_chars(text.data(), text.data() + text.size(), value);
  return read.ec == std::errc{} && read.ptr == text.data() + text.size();
}

bool parse(int argc, char** argv, Options& options) {
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    const std::string_view value = i + 1 < argc ? argv[i + 1] : "";
    if (arg == "--debande") {
      options.debanded = true;
    } else if (arg == "--sans-reprise") {
      options.reuse = false;
    } else if (arg == "--detail") {
      options.detail = true;
    } else if (arg == "--diviseur" && parse_int(value, options.divisor) && options.divisor >= 1) {
      ++i;
    } else if (arg == "--blocs" && parse_int(value, options.blocks) && options.blocks >= 1) {
      ++i;
    } else if (arg == "--dossier" && !value.empty()) {
      options.folder = value;
      ++i;
    } else {
      return false;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!parse(argc, argv, options)) {
    std::cerr << "usage : tuiles_rejeu [--diviseur D] [--blocs N] [--debande] [--sans-reprise] [--detail] "
                 "[--dossier DOSSIER] < tâches\n";
    return 2;
  }
  Replay replay(options);
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    std::istringstream fields(line);
    int width = 0, height = 0;
    std::string a, b, flow;
    if (!(fields >> width >> height >> a >> b >> flow) || width < 1 || height < 1) {
      std::cerr << "tuiles_rejeu : tâche illisible : " << line << "\n";
      return 1;
    }
    // Des tâches écrites ailleurs (le PC) : leurs fichiers par leur nom, dans le dossier donné.
    const auto here = [&options](const std::string& path) {
      return options.folder.empty()
                 ? path
                 : (std::filesystem::path(options.folder) / std::filesystem::path(path).filename()).string();
    };
    if (!replay.pair(width, height, here(a), here(b), here(flow))) return 1;
  }
  replay.finish();
  return 0;
}
