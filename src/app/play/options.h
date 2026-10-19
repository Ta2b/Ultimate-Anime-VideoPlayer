// Options de « ajp play <fichier> » : ce que la tranche verticale lit sur la ligne de
// commande, et rien d'autre (aucune variable d'environnement).
//
//   ajp play <fichier> [--model=FICHIER.onnx] [--no-upscale] [--window=LxH] [--fullscreen]
//                      [--start=S] [--seconds=S] [--fullscreen-after=S] [--stats=FICHIER]
//                      [--snapshot=FICHIER.ppm] [--snapshot-at=S] [--keys=S:touche,…]
//                      [--work-height=N]
//
// Le modèle par défaut est celui du lecteur actuel du propriétaire (AnimeJaNai, slot par
// défaut de animejanai.conf) : SPAN V3.1 Sharp1 Performance, ×2. Il est lu en place, en
// lecture seule, jamais copié dans le dépôt.
#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/profile.h"
#include "core/types.h"

namespace ajp::app::play {

inline constexpr std::string_view kDefaultModelFile =
    "2x_AnimeJaNai_HD_V3.1Sharp1_Performance_SPANF3_b5f48_unshuffle_fp16.onnx";
// Facteur des modèles SPAN d'AnimeJaNai HD : 2 (sortie 2H × 2W).
inline constexpr int kSpanFactor = 2;

namespace limits {
inline constexpr core::Range<int> kWindowSide{64, 16384};
inline constexpr core::Range<std::int64_t> kSeconds{0, std::int64_t{24} * 3600};
inline constexpr core::Range<int> kWorkHeight{0, 4320};
}  // namespace limits

struct PlayOptions {
  std::filesystem::path file;
  std::optional<std::filesystem::path> model;  // --model : absent, celui du lecteur actuel
  bool upscale = true;                         // --no-upscale : image source telle quelle
  int window_width = 1280;                     // --window=LxH
  int window_height = 720;
  bool fullscreen = false;                        // --fullscreen : plein écran dès l'ouverture
  std::int64_t start_us = 0;                      // --start=S : position de départ
  std::int64_t stop_after_us = 0;                 // --seconds=S : quitte après S s de lecture ; 0 jamais
  std::int64_t fullscreen_after_us = 0;           // --fullscreen-after=S : bascule après S s ; 0 jamais
  std::optional<std::filesystem::path> stats;     // --stats : une ligne par image présentée
  std::optional<std::filesystem::path> snapshot;  // --snapshot : la fenêtre en PPM, une fois
  std::int64_t snapshot_at_us = 3'000'000;        // --snapshot-at=S : après S s de lecture
  // --keys=S:touche,… : touches pressées d'office à ces instants (essais sans clavier), par
  // exemple « 3:Right,5:Space,6:Space,8:F ». Noms de touches de SDL.
  std::vector<std::pair<std::int64_t, std::string>> keys;
  // --work-height=N : la source plus haute que N est réduite par Area à N lignes avant SPAN
  // (1080p vers 720p, puis ×2 vers 1440p, comme le profil du lecteur actuel) ; 0 : jamais.
  int work_height = 720;
  bool help = false;  // --help
};

// Taille de travail de la chaîne : la source, ou réduite à work_height lignes en gardant le
// rapport d'aspect, largeur paire (1920x1080 avec 720 donne 1280x720).
struct WorkSize {
  int width = 0;
  int height = 0;
  friend bool operator==(const WorkSize&, const WorkSize&) = default;
};
WorkSize work_size(int width, int height, int work_height);

// Arguments qui suivent « play ». Une option inconnue (OptionUnknown), une valeur
// illisible ou hors plage (FieldOutOfRange), un fichier absent de la ligne (FieldMissing)
// sont refusés.
std::expected<PlayOptions, Refusal> parse_play_options(std::span<const std::string_view> args);

std::string play_usage();

// Où chercher le modèle par défaut, dans l'ordre : l'installation d'AnimeJaNai, puis le
// profil par défaut du dépôt mpv-AnimeJaNai-HDR. home vient de SDL (SDL_GetUserFolder).
std::vector<std::filesystem::path> default_model_candidates(const std::filesystem::path& home);

// Le modèle à charger : --model s'il est donné (il doit exister), sinon le premier
// candidat présent. Refus ModelUnreadable avec les chemins essayés.
std::expected<std::filesystem::path, Refusal> find_model(const PlayOptions& options,
                                                         std::span<const std::filesystem::path> candidates);

}  // namespace ajp::app::play
