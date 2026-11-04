// Options de « ajp play <fichier> » : ce que la tranche verticale lit sur la ligne de
// commande, et rien d'autre (aucune variable d'environnement).
//
//   ajp play <fichier> [--model=FICHIER.onnx] [--no-upscale] [--window=LxH] [--fullscreen]
//                      [--start=S] [--seconds=S] [--fullscreen-after=S] [--stats=FICHIER]
//                      [--snapshot=FICHIER.ppm] [--snapshot-at=S] [--keys=S:touche,…]
//                      [--work-height=N] [--rife=N] [--rife-model=FICHIER.onnx]
//                      [--truehdr-peak=NITS] [--truehdr-contrast=N] [--truehdr-saturation=N]
//                      [--truehdr-gray=N]
//                      [--first-run] [--mute] [--data-dir=DOSSIER] [--bench-level=auto|0..4]
//                      [--bench-out=FICHIER]
//   ajp bench <fichier> [--seconds=S] [--report=FICHIER] [--windowed]
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

#include "app/play/present_level.h"
#include "core/profile.h"
#include "core/types.h"

namespace ajp::app::play {

inline constexpr std::string_view kDefaultModelFile =
    "2x_AnimeJaNai_HD_V3.1Sharp1_Performance_SPANF3_b5f48_unshuffle_fp16.onnx";
// Facteur des modèles SPAN d'AnimeJaNai HD : 2 (sortie 2H × 2W).
inline constexpr int kSpanFactor = 2;
// RIFE par défaut : RIFE 4.22 lite, lu en place comme SPAN.
inline constexpr std::string_view kDefaultRifeFile = "rife_v4.22_lite.onnx";
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

inline constexpr double kDefaultKdeSdrNits = 200.0;  // la luminosité SDR de KDE par défaut

struct PlayOptions {
  std::filesystem::path file;                  // vide : l'écran d'accueil choisit la vidéo
  std::optional<std::filesystem::path> model;  // --model : absent, celui du lecteur actuel
  bool upscale = true;                         // --no-upscale : image source telle quelle
  int window_width = 1280;                     // --window=LxH
  int window_height = 720;
  bool fullscreen = false;                     // --fullscreen : plein écran dès l'ouverture
  bool hidden = false;                         // --hidden : fenêtre cachée (essais silencieux du benchmark)
  std::int64_t start_us = 0;                   // --start=S : position de départ
  std::int64_t stop_after_us = 0;              // --seconds=S : quitte après S s de lecture ; 0 jamais
  std::int64_t fullscreen_after_us = 0;        // --fullscreen-after=S : bascule après S s ; 0 jamais
  std::optional<std::filesystem::path> stats;  // --stats : une ligne par image présentée
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
  // L'aspect TrueHDR émulé du HDR adaptatif (et du grade SDR).
  int true_hdr_contrast = 100;    // --truehdr-contrast=0..200
  int true_hdr_saturation = 100;  // --truehdr-saturation=0..200
  int true_hdr_gray = 50;         // --truehdr-gray=10..100 (gris moyen)
  int true_hdr_peak = 0;  // --truehdr-peak=400..2000 nits ; 0 : la crête vue par le contenu sur l'écran
  bool help = false;      // --help
  // --rife-image : RIFE sort l'image à chaque instant (sans la variante mouvement), pour comparer.
  bool rife_image = false;
  // --software-gpu : accepte un appareil Vulkan logiciel (lavapipe), pour les essais sans GPU.
  bool software_gpu = false;
  // --sdr : sortie SDR même sur un écran HDR (la swapchain ne demande pas HDR10), pour les essais.
  bool sdr = false;
  // Anneau en RGB10A2 UNORM sous TensorRT (moitié moins de mémoire, étape 5.5a), par défaut ; --no-ring-rgb10a2 :
  // RGBA16. Repli silencieux sur RGBA16 si l'appareil ne le prend pas.
  bool ring_rgb10a2 = true;
  // --hdr-metadata[=NITS] : la sortie HDR10 porte ses métadonnées (son vrai maximum), et le grade fait la
  // compression que KWin faisait sans elles ; NITS, la luminosité SDR de KDE (kDefaultKdeSdrNits sans
  // valeur). Pour comparer à l'œil avec la sortie d'aujourd'hui, sans métadonnées.
  std::optional<double> kde_sdr_nits;
  // --hdr-contract-v2 : essai du contrat d'écran v2 (chantier 8.4) : métadonnées HDR10 exactes tirées du
  // contrat (render::contract_metadata), sans la compression de KWin faite par le grade, et le look
  // ver9-couleur.cube (SDR) à la place de la pile par défaut. --hdr-metadata=off : sous v2, rien d'annoncé.
  bool hdr_contract_v2 = false;
  // --shot-cuts : essai, les plans de la pré-analyse donnent les coupes de l'interpolation (RIFE ne fait plus
  // que signaler les désaccords) et les débuts de plan font repartir la mémoire du grade de l'anneau (8.7).
  bool shot_cuts = false;
  // --anime-shots : essai (méga plan A2), la méthode « HDR anime » décide par plan : exposition des surfaces
  // et marge des éclats fixées par la pré-analyse, glissant aux coupes (anime_shots.h).
  bool anime_shots = false;
  // La pré-analyse des plans décodée par NVDEC en 256 × 144 (8.8), repli logiciel, par défaut ;
  // --no-probe-nvdec : décodage logiciel.
  bool probe_nvdec = true;
  bool hdr_metadata_off = false;
  // --one-lane : une seule file de calcul (UpscaleSetup::one_lane), pour profiler le coût propre
  // de chaque moteur.
  bool one_lane = false;
  // --skip-held : en lecture, un instant identique à l'image affichée n'est ni calculé ni écrit (étapes 4.3
  // à 4.5). Éteint par défaut jusqu'à sa validation sur l'écran.
  bool skip_held = false;
  // --legacy-grade : le grade d'avant en neuf passes, qui mesure la scène lui-même (render::GradeBody),
  // gardé une version pour comparer.
  bool legacy_grade = false;
  // --legacy-grade=split : le grade actuel, le look dans sa passe à part (render::GradeBody::FusedSplit),
  // pour l'A/B du chantier 2a.
  bool split_look = false;
  // --grade-ring : le grade exact fait dans l'anneau par le backend CUDA (chantier 2c,
  // docs/adr/0009-grade-dans-l-anneau.md), l'écran montrant les images gradées sans hooks ; --grade-ring=off
  // ou absent : le grade reste en shader à l'écran. Éteint par défaut jusqu'à la porte des 30 plans (2.11).
  bool grade_ring = false;
  // --fast-steady : le mode Rapide à pas constant par plan (chantier 7, core/steady_rate.h) : N images par
  // source, constant dans un plan, au lieu de la grille libre. Essai, éteint par défaut jusqu'aux
  // mesures 7.10.
  bool fast_steady = false;
  // --smooth-drawings : un dessin tenu deux sources se fond en un seul mouvement vers le suivant (chantier
  // 7.9, UpscaleSetup::smooth_drawings). Essai, RIFE mouvement seulement, éteint par défaut.
  bool smooth_drawings = false;
  // Un thread attend l'écriture des slots et réveille main (ring_watcher.h, 6.5), par défaut ;
  // --no-ring-watcher : main sonde l'anneau.
  bool ring_watcher = true;
  // L'attente de la boucle d'affichage à la microseconde (loop_wait.h, 6.7), par défaut ; --no-precise-wait :
  // l'attente d'événement SDL à la milliseconde.
  bool precise_wait = true;
  // L'horloge audio maîtresse filtrée par une PLL (audio::PlayerConfig::pll, 6.9), par défaut ;
  // --no-audio-pll : l'horloge brute.
  bool audio_pll = true;
  // L'image choisie par le numéro de son vblank, avec hystérésis (SessionConfig::vsync_index, 6.6), par
  // défaut ; --no-vsync-index : l'ancien choix par l'horloge.
  bool vsync_index = true;
  // --present=auto|timing|wait|wayland|open : le retour de présentation (chantier 6.2, present_level.h).
  // open (le défaut) : la présentation d'aujourd'hui, jusqu'aux mesures 6.11.
  PresentWanted present = PresentWanted::Open;
  // --hdr-anime : la méthode « HDR anime » (render::GradeBody::Anime), essai ; le grade actuel reste le
  // défaut.
  bool hdr_anime = false;
  // --anime-guided : essai (méga plan A3), --hdr-anime avec la carte locale guidée par le pixel (le halo ne
  // déborde plus sur la surface voisine) ; implique --hdr-anime.
  bool anime_guided = false;
  // --grade-deband : le grade fait précéder la source d'un débandage (placebo_screen.cpp) ; coupé par défaut.
  bool grade_deband = false;
  bool skin_mask = false;  // --skin-mask : le grade distingue la peau (coupé par défaut)
  // --no-vrr : traite l'écran comme à fréquence fixe même si le VRR est armé (essai du mode Rapide sur les
  // vsyncs).
  bool no_vrr = false;
  // --no-resume : ni reprise à la position retenue, ni mémoire de la position (resume.h).
  bool resume = true;
  // --first-run : l'écran de bienvenue (moteurs, profil de démarrage) même s'il a déjà été vu.
  bool first_run = false;
  // --mute : volume à 0 pour cette lecture (les essais du benchmark), sans toucher à play.toml.
  bool mute = false;
  // --data-dir=DOSSIER : play.toml, user.toml, bench.toml, resume.tsv et ajp.log y vivent au lieu du dossier
  // de données ; les moteurs construits et les modèles restent partagés. Les essais du benchmark l'emploient
  // pour ne rien changer chez l'utilisateur.
  std::optional<std::filesystem::path> data_dir;
  // --bench-level : joue la chaîne du niveau 0 (Économe) à 4 (Ultra), ou « auto » (-1) : le niveau estimé de
  // la machine. Avec --bench-out, les mesures de la lecture y sont écrites à la fin (auto_bench.h :
  // BenchReading).
  std::optional<int> bench_level;
  std::optional<std::filesystem::path> bench_out;
  Registration registration = Registration::None;
  // Options données sur la ligne de commande (« --rife », « --truehdr-peak »…) : elles remplacent les
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

// Arguments qui suivent « play ». Une option inconnue (OptionUnknown), ou une valeur
// illisible ou hors plage (FieldOutOfRange), sont refusées ; sans fichier, l'écran d'accueil.
std::expected<PlayOptions, Refusal> parse_play_options(std::span<const std::string_view> args);

std::string play_usage();

// « ajp bench <fichier> » : évalue la machine (auto_bench_run.h) en jouant ce fichier à chaque niveau.
struct BenchOptions {
  std::filesystem::path file;
  int seconds = 12;
  std::optional<std::filesystem::path>
      report;              // --report : le rapport lisible, en plus de la sortie standard
  bool fullscreen = true;  // --windowed le retire
  bool silent = true;      // --visible montre la fenêtre des essais
  bool help = false;
};
// Arguments qui suivent « bench ». Une option inconnue ou une durée hors de 5 à 120 s : refus.
std::expected<BenchOptions, Refusal> parse_bench_options(std::span<const std::string_view> args);
std::string bench_usage();

// « ajp compare <fichier> --a="OPTIONS" --b="OPTIONS" » : deux variantes de la chaîne côte à côte
// (auto_bench_run.h : Variant). Les options sont celles de « ajp play », séparées par des espaces.
struct CompareOptions {
  std::filesystem::path file;
  int seconds = 12;
  std::vector<std::string> a;
  std::vector<std::string> b;
  std::filesystem::path a_settings;  // --a-settings=play.toml de départ de A (poses DRBA, moteur…)
  std::filesystem::path b_settings;
  std::optional<std::filesystem::path> report;
  bool fullscreen = true;
  bool help = false;
};
std::expected<CompareOptions, Refusal> parse_compare_options(std::span<const std::string_view> args);
std::string compare_usage();

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
