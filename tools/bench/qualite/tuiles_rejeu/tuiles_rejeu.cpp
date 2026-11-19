// tuiles_rejeu : la reprise entière de SPAN (lot A) et ses tuiles qui bougent, rejouées hors du lecteur sur
// processeur (CMakeLists.txt à côté ; docs/tuiles-span.md, « Rejeu sur processeur »).
//
//   tuiles_rejeu [--diviseur D] [--blocs N] [--debande] [--sans-reprise] [--sans-tuiles] [--detail]
//                [--dossier DOSSIER]
//                [--ancre-composee | --somme-des-paires | --somme-bornee [--seuil C]] [--masques DOSSIER]
//                < tâches
//
// Les tâches de carte_rejeu, d'un seul extrait : une paire voisine par ligne, dans l'ordre de lecture,
// LARGEUR HAUTEUR A.y B.y FLUX (la suite de la ligne, la SORTIE de carte_rejeu, est ignorée). Lumas 8 bits
// et flux de nvof_dump --grid 4 --one-way --cost à la taille du flux du lecteur ; D : le diviseur du travail
// au flux (scene_flow_divisor), le travail fait LARGEUR × D sur HAUTEUR × D. DOSSIER : des tâches écrites
// sur une autre machine, dont chaque fichier se cherche par son seul nom dans DOSSIER.
//
// Pour chaque image, comme le lecteur :
// - la mesure de la paire et l'écart à l'ancre (kernels/spec/detect.h) : pire bloc et écart de chaque bloc,
//   en 1/256 de niveau, sur des blocs de 16 pixels de travail (16 / D pixels du flux) ;
// - SpanReuse::decide puis, sur les images calculées, SpanTiles::plan : le code du lecteur, compilé tel quel,
//   sur la grille du SPAN FP16 natif à N blocs (8 par défaut : tuiles de 64 × 16 pixels de travail, halo
//   d'une colonne et de quatre lignes, gpu/cuda/backend/span_native16.cu). --debande : l'entrée de SPAN
//   débandée (le rayon du débandage s'ajoute au halo) ; --sans-reprise : --no-span-reuse ; --sans-tuiles :
//   --no-span-tiles, le lot A seul ;
// - la carte de la paire (précédente, image), que le lecteur ne lit plus pour les tuiles : kernels::ref,
//   caméra ajustée puis affinée, lissage des classes, historique vidé après une coupe
//   (app/play/scene_map.cpp), pour les colonnes de --detail et les coupes de l'essai.
//
// Sortie : avec --detail, une ligne par image, « image K SORTE TUILES SX SY CŒUR TOTAL AJUSTÉE CX CY PAIRE »
// : SORTE C (calculée entière), R (reprise entière), T (par tuiles) ; TUILES 1 si par tuiles ; SX SY le
// décalage de la sortie d'avant (toujours nul) ; CŒUR sur TOTAL les tuiles recalculées ; AJUSTÉE 1 si la
// carte a une caméra ; CX CY son déplacement au centre de l'image (pixels de travail) ; PAIRE le pire bloc de
// la paire (niveaux) ; avec --ancre-composee, la sorte et le cœur de l'essai. Puis les deux lignes du journal
// de fin du lecteur, « SPAN repris : … » et « SPAN par tuiles : … », ce que le corps de SPAN a calculé (sa
// part d'une image entière à chaque image), son coût (le corps, plus 6 % d'une image entière par image par
// tuiles, mesuré au PC), et la ligne de l'essai.
//
// --ancre-composee : à côté du lecteur, l'essai d'ancre_tuiles.h, les tuiles décidées sur l'écart des pixels
// à l'image composée de ce qui a produit la sortie, sans NVOF ; C : le seuil d'un bloc, centièmes de niveau
// (70, celui du lot A, par défaut). --somme-des-paires, --somme-bornee : ses variantes qui majorent l'écart
// sans image composée, par la somme des paires, resserrée ou non par l'écart à la dernière image calculée.
// --masques : pour chaque image K, K.coeur dans DOSSIER, les tuiles que le lecteur (l'essai avec
// --ancre-composee et ses variantes) a calculées (un octet par tuile, ligne par ligne ; toutes pour une image
// entière, aucune pour une reprise), que tuiles_controle.py rejoue contre un réseau 2×.
//
// Écarts au lecteur : la mesure de la paire lit ici la luma du flux, réduite d'un facteur D par moyenne de
// surface, et non le RGB de travail ou la copie de RIFE : le grain s'y moyenne, une paire y paraît un peu
// plus tenue (plus de reprises entières) et le filet des blocs un peu moins nerveux ; un vrai changement
// garde son écart moyen sur la même surface. La carte ne manque jamais une paire (pas de carte en retard ni
// perdue). La chaîne est celle d'une lecture SPAN avant le warp, sortie d'avant intacte.
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "ancre_tuiles.h"
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
  bool tiles = true;  // --sans-tuiles : --no-span-tiles, le lot A seul
  bool detail = false;
  std::string folder;     // non vide : chaque fichier d'une tâche se cherche là, par son nom
  bool composed = false;  // --ancre-composee : l'essai d'ancre_tuiles.h, à côté du lecteur
  bench::Bound bound = bench::Bound::Composed;  // --somme-des-paires, --somme-bornee : ses variantes
  std::string masks;  // --masques : où écrire les tuiles calculées par l'essai, image par image
  int threshold = static_cast<int>(kHeldThreshold);  // --seuil, centièmes de niveau
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
    reuse_.tiles.enabled = options.tiles;
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
    const bench::BlockGap paired = bench::block_gap(a, b, side_);
    reuse_.note_pair(paired.worst);
    app::play::ReuseFacts facts;
    facts.same_epoch = true;
    facts.has_output = true;
    facts.pair_held = is_held(paired.worst);
    facts.tiles_stale = app::play::TileStale::None;
    facts.spread_px = options_.debanded ? kDebandSpread : 0;
    facts.input_key = kInputKey;
    facts.blocks = app::play::PairBlocks{.deviation = paired.deviation,
                                         .cols = static_cast<std::uint32_t>(paired.cols),
                                         .side_px = static_cast<float>(side_ * options_.divisor)};
    if (reuse_.in_row() > 0) {  // la première reprise d'une série lit la paire, les suivantes l'ancre
      bench::BlockGap gap = bench::block_gap(anchor_, b, side_);
      facts.anchor = app::play::AnchorGap{.worst_block = gap.worst,
                                          .blocks = std::move(gap.deviation),
                                          .blocks_x = static_cast<std::uint32_t>(gap.cols),
                                          .side_px = static_cast<float>(side_ * options_.divisor)};
    }
    const app::play::ReuseKind kind = reuse_.decide(facts);
    if (kind != app::play::ReuseKind::Whole) anchor_ = b;
    count(kind);
    const char composed = composed_ ? composed_->next(b, *scene) : ' ';
    if (!options_.masks.empty() && !write_mask(composed_ ? composed_->last_core() : player_core(kind)))
      return false;
    print(kind, *scene, paired.worst, composed);
    ++frame_;
    return true;
  }

  void finish() const {
    if (!started_) return;  // aucune tâche
    std::cout << reuse_.summary() << "\n" << reuse_.tiles.summary() << "\n";
    const auto count = static_cast<double>(reuse_.tiles.grid().count());
    if (count > 0.0 && frame_ > 1) {
      const double frames = static_cast<double>(frame_ - 1);
      const double body = 100.0 * static_cast<double>(body_) / (count * frames);
      std::cout << std::format("Corps de SPAN du lecteur : {:.1f} % du calcul de chaque image entière", body)
                << "\n";
      // Le PC (19 novembre, Re:Zero 60 s) : une image par tuiles coûte sa région, plus 6 % d'une image
      // entière (lancement direct du corps masqué, hors graphe).
      std::cout << std::format("Coût de SPAN : {:.1f} % ({} images par tuiles)",
                               body + kTiledOverhead * 100.0 * static_cast<double>(tiled_frames_) / frames,
                               tiled_frames_)
                << "\n";
    }
    if (composed_) std::cout << composed_->summary() << "\n";
  }

 private:
  static constexpr std::uint64_t kInputKey = 1;   // une seule recette de l'entrée de SPAN
  static constexpr double kTiledOverhead = 0.06;  // part d'une image entière en plus par image par tuiles
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
    if (options_.composed)
      composed_.emplace(reuse_.tiles.grid(), options_.divisor, side_, facts.spread_px,
                        static_cast<std::uint32_t>(options_.threshold), options_.bound, first);
  }

  // K.coeur : les tuiles que l'essai a calculées pour l'image K, ligne par ligne, un octet chacune.
  bool write_mask(const std::vector<std::uint8_t>& core) const {
    const std::filesystem::path path =
        std::filesystem::path(options_.masks) / std::format("{}.coeur", frame_);
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(core.data()), static_cast<std::streamsize>(core.size()));
    if (out) return true;
    std::cerr << "tuiles_rejeu : écriture impossible : " << path.string() << "\n";
    return false;
  }

  // Les tuiles que le lecteur a calculées : toutes, aucune (reprise entière) ou le cœur du plan.
  std::vector<std::uint8_t> player_core(app::play::ReuseKind kind) const {
    if (kind == app::play::ReuseKind::Tiles) return reuse_.tile_plan.core;
    const auto count = static_cast<std::size_t>(reuse_.tiles.grid().count());
    return std::vector<std::uint8_t>(count, kind == app::play::ReuseKind::Compute ? 1 : 0);
  }

  // Le corps de SPAN calcule l'image entière, ou la région dilatée du halo d'une image par tuiles.
  void count(app::play::ReuseKind kind) {
    const SpanTileGrid& g = reuse_.tiles.grid();
    if (kind == app::play::ReuseKind::Compute) body_ += static_cast<std::uint64_t>(g.count());
    if (kind == app::play::ReuseKind::Tiles) ++tiled_frames_;
    if (kind == app::play::ReuseKind::Tiles)
      body_ += static_cast<std::uint64_t>(count_span_tiles(
          dilate_span_tiles(reuse_.tile_plan.core, g.cols, g.rows, g.reach_cols, g.reach_rows)));
  }

  void print(app::play::ReuseKind kind, const SceneMotion& scene, std::uint32_t worst, char composed) const {
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
              << static_cast<double>(worst) / 256.0;
    if (composed_) std::cout << ' ' << composed << ' ' << count_span_tiles(composed_->last_core());
    std::cout << "\n";
  }

  Options options_;
  app::play::SpanReuse reuse_;
  ref::SceneHistory history_;
  Plane<std::uint8_t> anchor_;
  std::optional<bench::AnchorTiles> composed_;
  std::uint64_t body_ = 0;  // tuiles calculées par le corps de SPAN, images après la première
  std::uint64_t tiled_frames_ = 0;
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
    } else if (arg == "--sans-tuiles") {
      options.tiles = false;
    } else if (arg == "--sans-reprise") {
      options.reuse = false;
    } else if (arg == "--detail") {
      options.detail = true;
    } else if (arg == "--diviseur" && parse_int(value, options.divisor) && options.divisor >= 1) {
      ++i;
    } else if (arg == "--blocs" && parse_int(value, options.blocks) && options.blocks >= 1) {
      ++i;
    } else if (arg == "--ancre-composee") {
      options.composed = true;
    } else if (arg == "--somme-des-paires") {
      options.composed = true;
      options.bound = bench::Bound::Summed;
    } else if (arg == "--somme-bornee") {
      options.composed = true;
      options.bound = bench::Bound::Anchored;
    } else if (arg == "--seuil" && parse_int(value, options.threshold) && options.threshold >= 0) {
      ++i;
    } else if (arg == "--masques" && !value.empty()) {
      options.masks = value;
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
                 "[--dossier DOSSIER] [--ancre-composee | --somme-des-paires | --somme-bornee [--seuil C]] "
                 "[--masques DOSSIER] < tâches\n";
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
