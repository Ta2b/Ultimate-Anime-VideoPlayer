// Options de « ajp play <fichier> » : ce que la tranche verticale lit sur la ligne de
// commande, et rien d'autre (aucune variable d'environnement).
//
//   ajp play <fichier> [--model=FICHIER.onnx] [--no-upscale] [--window=LxH] [--fullscreen]
//                      [--start=S] [--seconds=S] [--fullscreen-after=S] [--stats=FICHIER]
//                      [--snapshot=FICHIER.ppm] [--snapshot-at=S] [--keys=S:touche,…]
//                      [--work-height=N] [--rife=N] [--rife-model=FICHIER.onnx]
//                      [--truehdr-peak=NITS]
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

// --motion : RIFE ou DistilDRBA (settings.h : MotionEngine, que ce fichier ne voit pas).
enum class MotionChoice : std::uint8_t { Rife, DistilDrba };

enum class RelaunchedFor : std::uint8_t { None, GpuLost, EngineUpdated };
// Le texte de --relaunched= ; vide pour None.
std::string_view relaunched_name(RelaunchedFor reason);

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
  // --rife-calm-model (essai, P6.17) : un RIFE plus léger (4.22 lite) pour la passe du milieu des paires
  // calmes, RIFE mouvement seulement ; mêmes plans d'entrée et même remplissage que rife_model, sinon ignoré.
  std::optional<std::filesystem::path> rife_calm_model;
  // --motion=rife|drba (A18-09) : le moteur de l'interpolation pour cette lecture, comme le menu ; absent,
  // celui de play.toml ou du préréglage.
  std::optional<MotionChoice> motion;
  int true_hdr_peak = 0;  // --truehdr-peak=350..4000 nits (limits::kTrueHdrPeak) ; 0 : la crête vue par le
                          // contenu sur l'écran
  bool help = false;      // --help
  bool debug = false;     // --debug : les options d'essai (bascules des chantiers) sont acceptées
  // --rife-image : RIFE sort l'image à chaque instant (sans la variante mouvement), pour comparer.
  bool rife_image = false;
  // --software-gpu : accepte un appareil Vulkan logiciel (lavapipe), pour les essais sans GPU.
  bool software_gpu = false;
  // --sdr : sortie SDR même sur un écran HDR (la swapchain ne demande pas HDR10), pour les essais.
  bool sdr = false;
  // --ring-rgb10a2 : anneau en RGB10A2 UNORM sous TensorRT (moitié moins de mémoire, étape 5.5a), repli
  // silencieux sur RGBA16 si l'appareil ne le prend pas. Éteint par défaut (RGBA16) : la porte de banding de
  // l'ADR 0008 n'a pas été mesurée, et le grade HDR étire les paliers de 10 bits (A21-03, A49-04).
  bool ring_rgb10a2 = false;
  // --hdr-metadata[=NITS] : la sortie HDR10 porte ses métadonnées (son vrai maximum), et le grade fait la
  // compression que KWin faisait sans elles ; NITS, la luminosité SDR de KDE (kDefaultKdeSdrNits sans
  // valeur). Pour comparer à l'œil avec la sortie d'aujourd'hui, sans métadonnées.
  std::optional<double> kde_sdr_nits;
  // --hdr-metadata sans valeur : la luminosité SDR lue dans KDE (le contrat d'écran), kde_sdr_nits en repli.
  // Par défaut sous Linux : la sortie annonce ses métadonnées et le grade compresse vers le pic de l'écran,
  // au blanc SDR de KDE quand KDE le donne, sinon (GNOME, Hyprland…) au blanc neutre de 203 nits.
  // Par défaut sous Windows aussi : Windows affiche le PQ tel quel, sans compression, et la pile de LUT y
  // écrêtait au pic du panneau ; le grade fait la même compression que KWin à son blanc neutre (203 nits).
  // --no-hdr-metadata : la sortie d'avant, sans métadonnées, la compression laissée à KWin (Windows : rien).
#if defined(__linux__) || defined(_WIN32)
  bool kde_sdr_auto = true;
#else
  bool kde_sdr_auto = false;
#endif
  // Les plans de la pré-analyse donnent les coupes de l'interpolation (RIFE ne fait plus que signaler les
  // désaccords) ; --no-shot-cuts les coupe.
  bool shot_cuts = true;  // par défaut (désaccords jugés à l'image)
  // --anime-shots (méga plan A2, actif par défaut) : en sortie SDR, la méthode « HDR anime » décide par plan
  // : exposition des surfaces et marge des éclats fixées par la pré-analyse, glissant hors des coupes
  // (anime_shots.h) ; --no-anime-shots garde le relèvement fixe. Sans effet en sortie HDR
  // (render::steady_hdr).
  bool anime_shots = true;  // par défaut (jugé à l'image)
  // La pré-analyse des plans décodée par NVDEC en 256 × 144 (8.8), repli logiciel, par défaut ;
  // --no-probe-nvdec : décodage logiciel.
  bool probe_nvdec = true;
  // --profile-dispatches : temps GPU et nombre de dispatchs par shader des réseaux Vulkan, au journal
  // quand un réseau se décharge (BackendInit::profile_dispatches).
  bool profile_dispatches = false;
  // --filtered-warp (essai) : le warp TensorRT lit par texture (BackendInit::filtered_warp), pour mesurer son
  // gain ; éteint, la lecture exacte.
  bool filtered_warp = false;
  // --l2-keep (essai) : l'encodage gardé de RIFE fusionné tenu en L2 (gpu::kTrialL2Keep), pour mesurer
  // son gain ; éteint par défaut.
  bool l2_keep = false;
  // --tiled-warp (essai) : le warp TensorRT calcule le mouvement du milieu par point du flot de chaque tuile
  // (gpu::kTrialTiledWarp) ; l'image change un peu, éteint par défaut derrière la porte qualité.
  bool tiled_warp = false;
  // --warp-l2 (essai) : le warp TensorRT tient le flot en L2 persistant (gpu::kTrialWarpL2) ; même image.
  bool warp_l2 = false;
  // --rife-hwc8 (essai) : les blocs du RIFE fusionné en E/S HWC8 sous TensorRT (gpu::kTrialRifeHwc8) ; même
  // image, éteint par défaut jusqu'à la mesure sur la carte.
  bool rife_hwc8 = false;
  // --span-native (essai) : au cran Équilibre, SPAN en FP16 par les noyaux CUTLASS du backend (canaux natifs,
  // b5f48 en 48) au lieu de TensorRT FP16 ; mêmes poids, sans quantification. Éteint par défaut jusqu'à la
  // mesure sur la carte (docs/span-fp8-cutlass.md, « SPAN FP16 natif dans le lecteur »).
  bool span_native = false;
  // --one-lane : une seule file de calcul (UpscaleSetup::one_lane), pour profiler le coût propre
  // de chaque moteur.
  bool one_lane = false;
  // --lanes : les trois voies (agrandissement, interpolation, mesure) au lieu d'une seule, le défaut
  // (chain_setup.cpp).
  bool lanes = false;
  // --rife-warm (essai, P6.15) : la passe du milieu d'une paire calme reprend les blocs grossiers de RIFE de
  // la paire précédente (RIFE fusionné sous Vulkan) ; UpscaleSetup::rife_warm.
  bool rife_warm = false;
  // --rife-encode-all (mesure, avec --debug) : RIFE fusionné encode les deux images à chaque passe, comme
  // avant fusion3, au lieu de garder l'encodage de la paire et de la B précédente ; la sortie est la même
  // (UpscaleSetup::rife_encode_all). Pour mesurer le gain en ABAB (docs/rife-fusionne.md).
  bool rife_encode_all = false;
  // --rife-at-slot (mesure) : chaque passe RIFE d'une paire forte attend le slot de son image, au lieu d'être
  // enfilée à l'arrivée de la paire ; la sortie est la même (UpscaleSetup::rife_at_slot). Pour l'ABAB.
  bool rife_at_slot = false;
  // --skip-held : en lecture, un instant identique à l'image affichée n'est ni calculé ni écrit (étapes 4.3
  // à 4.5). Éteint par défaut jusqu'à sa validation sur l'écran.
  bool skip_held = false;
  // --fast-steady : le mode Rapide à pas constant par plan (chantier 7, core/steady_rate.h) : N images par
  // source, constant dans un plan, au lieu de la grille libre. Essai, éteint par défaut jusqu'aux
  // mesures 7.10.
  bool fast_steady = false;
  // --smooth-drawings : un dessin tenu de deux à six sources se fond en un seul mouvement vers le suivant
  // (chantier 7.9, UpscaleSetup::smooth_drawings) ; pour cette lecture, l'option du menu
  // (PlaySettings::smooth_drawings, settings_for). RIFE mouvement seulement, éteint par défaut.
  bool smooth_drawings = false;
  // Un thread attend l'écriture des slots et réveille main (ring_watcher.h, 6.5), par défaut ;
  // --no-ring-watcher : main sonde l'anneau.
  bool ring_watcher = true;
  // L'attente précise de la boucle d'affichage (loop_wait.h), par défaut ; --no-precise-wait : l'attente
  // d'événement SDL à la milliseconde.
  bool precise_wait = true;
  // L'horloge audio maîtresse filtrée par une PLL (audio::PlayerConfig::pll) : l'état livré ;
  // --no-audio-pll (essai) : l'horloge brute, pour comparer.
  bool audio_pll = true;
  // L'image choisie par le numéro de son vblank, avec hystérésis (SessionConfig::vsync_index, 6.6), par
  // défaut ; --no-vsync-index : l'ancien choix par l'horloge.
  bool vsync_index = true;
  // --present=auto|timing|timing-absolute|wait|wayland|open : le retour de présentation (chantier 6.2,
  // present_level.h). wait (le défaut) : wp_presentation quand le compositeur l'offre (il date l'affichage,
  // d'où la grille des vblanks), sinon present_wait2, sinon open ; auto et timing ne rendent aucun retour sur
  // NVIDIA 615 / KDE Wayland (mesuré : 0 revenue) et retombent alors seuls (present_level.h :
  // feedback_silent) ;
  // --no-present ou --present=open : la présentation d'avant.
  PresentWanted present = PresentWanted::Wait;
  // --anime-contrast=N, --anime-saturation=N, --anime-highlights=N (0 à 200 %, 100 : la méthode telle quelle)
  // : le peps du grade HDR anime, à régler à l'œil (render::GradeSettings::anime_contrast…).
  // --bake-looks=FICHIER.cube : la pile de LUT de play.toml, intensités comprises, composée en une seule LUT
  // (render::compose_luts, 65 points : une approximation de la pile, que le lecteur parcourt LUT par LUT) et
  // écrite là ; le lecteur s'arrête.
  std::optional<std::filesystem::path> bake_looks;
  int anime_contrast = 100;
  int anime_saturation = 100;
  int anime_highlights = 100;
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
  // --relaunched=gpu-lost|engine-updated : posé par le lecteur qui se relance (main.cpp), jamais par
  // l'utilisateur. Seul un processus relancé lit relaunch.txt (un double-clic sur une autre vidéo n'ouvre
  // plus l'ancienne, B18-02), et la garde contre une boucle de pertes du GPU ne dépend plus du disque
  // (B31-07).
  RelaunchedFor relaunched = RelaunchedFor::None;
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
// SPAN tourne-t-il sur cette source (P3.19) ? Pas sans « Restaurer les traits » quand il faudrait la réduire.
bool span_runs(bool upscale, bool restore_lines, int source_height, int work_height);
// La sortie la plus grande que l'agrandissement ×2 rend : la 4K UHD. Une image de travail plus grande que la
// moitié (1920x1080) est réduite avant SPAN ou Anime4K, jamais doublée au-delà.
inline constexpr int kMaxOutputWidth = 3840;
inline constexpr int kMaxOutputHeight = 2160;
// La hauteur de travail qui garde le double d'une source width x height dans la 4K (largeur paire comprise).
int span_height_cap(int width, int height);
// La taille de travail que la chaîne joue, la seule à lire (A08-15) : les moteurs, l'anneau, l'étage et
// l'export la prennent ici. C'est work_size, sauf quand SPAN est demandé mais ne tourne pas faute de
// « Restaurer les traits » : la source est alors gardée telle quelle, jamais réduite pour rien. Quand SPAN
// tourne, l'image de travail tient dans la moitié de la 4K (span_height_cap) : une source 1440p avec «
// Source, sans réduction » est réduite à 1080p puis doublée en 2160p, pas en 2880p.
WorkSize chain_work_size(bool upscale, bool restore_lines, int width, int height, int work_height);

// Arguments qui suivent « play ». Une option inconnue (OptionUnknown), ou une valeur
// illisible ou hors plage (FieldOutOfRange), sont refusées ; sans fichier, l'écran d'accueil.
std::expected<PlayOptions, Refusal> parse_play_options(std::span<const std::string_view> args);

// debug : avec les options d'essai (--debug --help).
std::string play_usage(bool debug = false);

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

// Où chercher le modèle par défaut, dans l'ordre : les modèles livrés avec le lecteur (bundled/onnx, si
// bundled n'est pas vide), puis l'installation d'AnimeJaNai et le profil par défaut du dépôt
// mpv-AnimeJaNai-HDR (postes de développement, B18-06). home vient de SDL (SDL_GetUserFolder).
std::vector<std::filesystem::path> default_model_candidates(const std::filesystem::path& home,
                                                            const std::filesystem::path& bundled = {});
// RIFE : kDefaultRifeFile dans chaque dossier, bundled/rife en premier ; avec files, chacun d'eux,
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
