// Options de « ajp play <fichier> » : ce que la tranche verticale lit sur la ligne de
// commande, et rien d'autre (aucune variable d'environnement).
//
//   ajp play <fichier> [--model=FICHIER.onnx] [--no-upscale] [--window=LxH] [--fullscreen]
//                      [--start=S] [--seconds=S] [--fullscreen-after=S] [--stats=FICHIER]
//                      [--snapshot=FICHIER.ppm] [--snapshot-at=S] [--keys=S:touche,…]
//                      [--work-height=N] [--rife=N] [--rife-model=FICHIER.onnx]
//                      [--truehdr] [--truehdr-sdk=DOSSIER] [--truehdr-peak=NITS]
//                      [--truehdr-contrast=N] [--truehdr-saturation=N] [--truehdr-gray=N]
//                      [--first-run]
//   ajp --register | --unregister
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
// RIFE par défaut : RIFE 4.25, lu en place comme SPAN.
inline constexpr std::string_view kDefaultRifeFile = "rife_v4.25.onnx";
// Interpolation par défaut : ×3. Sur la RTX 4070 Ti, une paire qui bouge coûte 2 RIFE à 720p et
// 3 SPAN, environ 31 ms sur 41,7 : 8 images sautées sur 25 s d'ouverture, contre 304 à ×4
// (19 octobre, ~/ajp-reports/vertical-slice.md). ×4 attend les optimisations de la parité.
inline constexpr int kDefaultRifeFactor = 3;
inline constexpr int kMaxSnapshotFrames = 240;

namespace limits {
inline constexpr core::Range<int> kWindowSide{64, 16384};
inline constexpr core::Range<std::int64_t> kSeconds{0, std::int64_t{24} * 3600};
inline constexpr core::Range<int> kWorkHeight{0, 4320};
inline constexpr core::Range<int> kRifeFactor{0, 10};
}  // namespace limits

// Un facteur d'interpolation écrit « 3 », « 1.5 » ou « 1,5 » : num / den, den valant 1 ou 2 (une
// moitié : num impair de 3 à 19). Rien hors de 0 à 10 ou pour une autre fraction.
struct Factor {
  int num = 0;
  int den = 1;
  friend bool operator==(const Factor&, const Factor&) = default;
};
std::optional<Factor> parse_factor(std::string_view text);
// Le facteur est-il permis (0 ou 1 : aucune interpolation) ?
bool valid_factor(Factor factor);
// « ×3 », « ×1,5 ».
std::string factor_text(Factor factor);

// --register, --unregister : le lecteur s'inscrit auprès du système (« Ouvrir avec », lecteur par
// défaut : default_player.h) ou retire son inscription, puis quitte, sans fichier.
enum class Registration : std::uint8_t { None, Register, Unregister };

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
  std::int64_t snapshot_at_us = 3'000'000;
  int snapshot_frames = 1;  // --snapshot-frames=N : N présentations de suite, numérotées  // --snapshot-at=S
                            // : après S s de lecture (ou de construction)
  // --keys=S:touche,… : touches pressées d'office à ces instants (essais sans clavier), par
  // exemple « 3:Right,5:Space,6:Space,8:F ». Noms de touches de SDL ; « export » ou
  // « export=FICHIER.mkv » lance l'export de la chaîne en cours (scripted_input.h).
  std::vector<std::pair<std::int64_t, std::string>> keys;
  // --work-height=N : la source plus haute que N est réduite par Area à N lignes avant SPAN
  // (1080p vers 720p, puis ×2 vers 1440p, comme le profil du lecteur actuel) ; 0 : jamais.
  int work_height = 720;
  // --rife=N : interpolation ×N (2 à 10, ou une moitié : 1.5, 2.5…) avant SPAN ; 0 ou 1 : aucune.
  int rife_factor = kDefaultRifeFactor;
  int rife_factor_den = 1;                          // 2 pour un facteur en moitié : ×rife_factor / 2
  std::optional<std::filesystem::path> rife_model;  // --rife-model : absent, celui du lecteur actuel
  // --truehdr : RTX HDR (Maxine TrueHDR) sur la sortie de SPAN ; l'écran reçoit du PQ BT.2020.
  bool true_hdr = false;
  std::filesystem::path true_hdr_sdk = "/usr/local/VideoFX";  // --truehdr-sdk : importé une fois
  int true_hdr_contrast = 100;                                // --truehdr-contrast=0..200
  int true_hdr_saturation = 100;                              // --truehdr-saturation=0..200
  int true_hdr_gray = 50;                                     // --truehdr-gray=10..100 (gris moyen)
  int true_hdr_peak = 0;  // --truehdr-peak=400..2000 nits ; 0 : la crête vue par le contenu sur l'écran
  bool help = false;      // --help
  // --rife-image : RIFE sort l'image à chaque instant (sans la variante mouvement), pour comparer.
  bool rife_image = false;
  // --software-gpu : accepte un appareil Vulkan logiciel (lavapipe), pour les essais sans GPU.
  bool software_gpu = false;
  // --sdr : sortie SDR même sur un écran HDR (la swapchain ne demande pas HDR10), pour les essais.
  bool sdr = false;
  // --one-lane : une seule file de calcul (UpscaleSetup::one_lane), pour profiler le coût propre
  // de chaque moteur.
  bool one_lane = false;
  // --no-resume : ni reprise à la position retenue, ni mémoire de la position (resume.h).
  bool resume = true;
  // --first-run : l'écran de bienvenue (moteurs, profil de démarrage) même s'il a déjà été vu.
  bool first_run = false;
  Registration registration = Registration::None;
  // Options données sur la ligne de commande (« --rife », « --truehdr »…) : elles remplacent les
  // réglages de play.toml pour cette lecture.
  std::vector<std::string> given;
  bool was_given(std::string_view name) const;
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

// Où chercher le modèle par défaut, dans l'ordre : l'installation d'AnimeJaNai, le profil par
// défaut du dépôt mpv-AnimeJaNai-HDR, puis les modèles livrés avec le lecteur (bundled/onnx, si
// bundled n'est pas vide). home vient de SDL (SDL_GetUserFolder).
std::vector<std::filesystem::path> default_model_candidates(const std::filesystem::path& home,
                                                            const std::filesystem::path& bundled = {});
// RIFE : kDefaultRifeFile dans chaque dossier, bundled/rife en dernier ; avec files, chacun d'eux,
// dossier par dossier.
std::vector<std::filesystem::path> default_rife_candidates(const std::filesystem::path& home,
                                                           const std::filesystem::path& bundled = {});
std::vector<std::filesystem::path> default_rife_candidates(const std::filesystem::path& home,
                                                           std::span<const std::string_view> files,
                                                           const std::filesystem::path& bundled = {});

// Le modèle à charger : --model s'il est donné (il doit exister), sinon le premier
// candidat présent. Refus ModelUnreadable avec les chemins essayés.
std::expected<std::filesystem::path, Refusal> find_model(const PlayOptions& options,
                                                         std::span<const std::filesystem::path> candidates);
// Le modèle RIFE : --rife-model s'il est donné (il doit exister), sinon le premier candidat.
std::expected<std::filesystem::path, Refusal> find_rife_model(
    const PlayOptions& options, std::span<const std::filesystem::path> candidates);

}  // namespace ajp::app::play
