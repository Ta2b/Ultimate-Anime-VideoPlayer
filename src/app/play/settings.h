// Réglages de « kirari play » que le menu des paramètres change, gardés dans play.toml (dossier de
// données du lecteur) d'une lecture à l'autre ; une option donnée sur la ligne de commande les
// remplace pour cette lecture seulement.
//
// Chaque réglage a sa plage ; une valeur hors plage est refusée en la nommant, jamais corrigée.
// Un changement s'applique en direct (réglages du HDR, facteur RIFE, réduction, son,
// statistiques) ou demande de rouvrir la chaîne à la même position (modèle, hauteur de travail,
// RIFE sans moteur chargé) : le menu le dit avant d'appliquer.
#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "app/play/final_scaler.h"
#include "app/play/i18n.h"
#include "app/play/options.h"
#include "app/play/precision.h"
#include "app/play/system_language.h"
#include "app/play/temporal_fix.h"
#include "app/play/update_choice.h"
#include "core/profile.h"
#include "core/settings.h"
#include "core/types.h"
#include "media/selection.h"
#include "render/grade_params.h"

namespace kirari::app::play {

// Filtre de la réduction avant SPAN (fusionnée à l'ingestion, en codes de la source) : celui des
// profils (core/profile.h), mêmes noms dans play.toml et user.toml.
using Downscale = core::DownscaleFilter;
inline constexpr int kDownscaleCount = 7;
std::string_view name(Downscale filter);  // « area », la clé de play.toml
// Libellé lisible d'un modèle AnimeJaNai : « HD V3.1 Sharp1 Performance ».
std::string model_label(const std::string& file);
std::string_view label(Downscale filter);  // « Area », pour le menu
// Libellé lisible d'un modèle RIFE : « rife_v4.22_lite.onnx » devient « RIFE 4.22 lite ».
std::string rife_label(const std::string& file);

// Le seuil du mélange de GMFSS, en pixels de mouvement à 720 lignes : celui où RIFE passe déjà à chaque
// instant (kernels::single_run_motion). Bornes de play.toml : [kGmfssMotionMin, kGmfssMotionMax].
inline constexpr int kDefaultGmfssMotion = 15;
inline constexpr int kGmfssMotionMin = 4;
inline constexpr int kGmfssMotionMax = 60;
// Le mouvement sous lequel le mélange ne pose aucune clé GMFSS, en pixels à 720 lignes ; 0 : une clé sur
// toute paire qui bouge (Très élevé). Bornes de play.toml : [0, kGmfssMotionMax].
inline constexpr int kDefaultGmfssKeyFloor = 0;

// Ce qui fait les images intermédiaires : RIFE (réseau, 4.25 ou 4.22 lite, avec ou sans garder les
// poses), ou DistilDRBA (kernels/spec/distil_drba.h : un RIFE à trois images
// entraîné à imiter DRBA, les poses gardées par le réseau lui-même, une image de retard), ou
// l'interpolation d'Apple (VTFrameRateConversion de VideoToolbox, macOS 15.4, sous le backend Core
// ML : gpu/system_interpolator.h), ou GMFSS Fortuna Union seul (upscale_gmfss.cpp, TensorRT seulement ; RIFE
// le remplace ailleurs ou sans ses graphes). Le mélange de RIFE et de GMFSS reste RIFE (gmfss_middle). Les
// noms de l'ancien GMFSS (0.10.0) : « gmfss » se lit en RIFE, « gmfss_union » en GMFSS seul. La valeur
// numérique n'est écrite nulle part : play.toml porte le nom.
enum class MotionEngine : std::uint8_t { Rife, DistilDrba, Apple, Gmfss };
inline constexpr int kMotionEngineCount = 4;
// Proposé sur ce système : l'interpolation d'Apple sur un Mac seulement. play.toml garde le choix
// partout ; ailleurs, la chaîne reprend RIFE.
#if defined(__APPLE__)
inline constexpr bool kAppleOffered = true;
#else
inline constexpr bool kAppleOffered = false;
#endif
constexpr bool offered(MotionEngine engine) {
  return kAppleOffered || engine != MotionEngine::Apple;
}
std::string_view name(MotionEngine engine);   // « rife », la clé de play.toml
std::string_view label(MotionEngine engine);  // « RIFE », pour le menu

// Les modèles d'interpolation que montre le menu (docs/interpolation.md), par leur nom
// : DistilDRBA v2 lite, RIFE 4.22 lite et RIFE 4.25 ; System : l'interpolation d'Apple, sur un Mac seulement.
// Avec le natif de GMFSS (le défaut, interp_retired.h : gmfss_kept), RIFE 4.25 avec GMFSS union au milieu
// des paires fortes, et GMFSS union seul. Les niveaux d'avant (4.25 lite, 4.26, GMFSS sous --no-gmfss-native)
// se relisent sur le plus proche (interp_retired.h).
enum class InterpLevel : std::uint8_t { DistilDrba, Rife422Lite, Rife425, GmfssMiddle, GmfssUnion, System };
inline constexpr std::string_view kRife422LiteFile = "rife_v4.22_lite.onnx";
inline constexpr std::string_view kRife425File = "rife_v4.25.onnx";
struct InterpChoice {
  InterpLevel level;
  MotionEngine engine;
  std::string_view rife_model;  // vide hors RIFE
  bool gmfss_middle = false;    // RIFE avec GMFSS union au milieu des paires fortes (Élevé)
};
inline constexpr std::array<InterpChoice, 6> kInterpLevels{{
    {.level = InterpLevel::DistilDrba, .engine = MotionEngine::DistilDrba, .rife_model = {}},
    {.level = InterpLevel::Rife422Lite, .engine = MotionEngine::Rife, .rife_model = kRife422LiteFile},
    {.level = InterpLevel::Rife425, .engine = MotionEngine::Rife, .rife_model = kRife425File},
    {.level = InterpLevel::GmfssMiddle,
     .engine = MotionEngine::Rife,
     .rife_model = kRife425File,
     .gmfss_middle = true},
    {.level = InterpLevel::GmfssUnion, .engine = MotionEngine::Gmfss, .rife_model = {}},
    {.level = InterpLevel::System, .engine = MotionEngine::Apple, .rife_model = {}},
}};
// Le choix d'un moteur et du modèle RIFE qu'il charge (le nom vide de play.toml déjà résolu) ; nullopt pour
// un autre modèle RIFE seul, ou un moteur débranché. GMFSS au milieu des paires fortes est GmfssMiddle quel
// que soit le modèle RIFE.
std::optional<InterpLevel> interp_level(MotionEngine engine, std::string_view rife_model,
                                        bool gmfss_middle = false);

// Les repères des listes (docs/ux-parametres.md, 3) : une pastille à côté du choix, le détail en infobulle.
enum class Badge : std::uint8_t { Recommended, Fastest, Finest, Experimental };
std::string_view badge_text(
    Badge badge);  // « Recommandé », « Le plus rapide », « Le plus fin », « Expérimental »
std::optional<Badge> badge_of(Downscale filter);
// « 720p » ; la hauteur d'image en français courant, jamais « lignes ».
std::string resolution_text(int height);

// Le warp de l'interpolation par rapport à SPAN (variante mouvement de RIFE ou de DistilDRBA, avec SPAN).
// Avant : SPAN agrandit chaque sortie, intermédiaires compris (×3 : trois passes par image source),
// et nettoie les défauts du warp. Après : SPAN agrandit chaque image source une fois, le warp
// déforme les images agrandies à la taille de sortie. Auto : après dès que la sortie dépasse
// kWarpAfterSpanAbove lignes, où SPAN coûte le plus (1080p → 4K : 14,7 ms par passe en FP16 sur une
// RTX 4070 Ti, docs/span-fp8-cutlass.md, contre 0,4 ms pour le warp), ou dès que le facteur atteint ×3 (D5 :
// deux images créées par paire, deux passes de SPAN de plus contre deux warps) : choisi à la main, ou en Auto
// le plus haut que l'écran lui laisse (order_factor, chain_shape.h). Qualité des deux ordres mesurée au banc
// (tools/bench/qualite/ordre.py, docs/banc-qualite.md, C09-02) : aussi proches de la vraie image l'un que
// l'autre (0,1 dB), l'image créée après SPAN 0,5 à 1,3 % moins nette et moins « SPAN » (2 à 3 dB d'écart de
// plus à SPAN de la vraie image) ; gain : une passe de SPAN par source au lieu d'une par image montrée.
enum class SpanOrder : std::uint8_t { Auto, Before, After };
inline constexpr int kSpanOrderCount = 3;
inline constexpr int kWarpAfterSpanAbove = 1080;
std::string_view name(SpanOrder order);   // « auto », la clé de play.toml
std::string_view label(SpanOrder order);  // « Auto (après SPAN au-dessus de 1080p ou dès ×3) », pour le menu
inline constexpr int kWarpAfterSpanFactor = 3;
// Le warp passe-t-il après SPAN pour une sortie de output_height lignes, au plus haut facteur de la chaîne
// (max_factor, arrondi au-dessous, order_factor ; 0 : inconnu) ?
constexpr bool warp_after_span(SpanOrder order, int output_height, int max_factor = 0) {
  return order == SpanOrder::After || (order == SpanOrder::Auto && (output_height > kWarpAfterSpanAbove ||
                                                                    max_factor >= kWarpAfterSpanFactor));
}

// La sortie HDR (Vidéo > Général) : Auto suit le système (HDR10 dès que l'écran en reçoit assez, host_facts.h
// : system_hdr), Activé la prend dès que le système est en HDR, même sous le seuil de luminosité, Désactivé
// garde le SDR gradé, comme --sdr. Section [hdr] output de play.toml.
enum class HdrMode : std::uint8_t { Auto, On, Off };
inline constexpr int kHdrModeCount = 3;
std::string_view name(HdrMode mode);   // « auto », la clé de play.toml
std::string_view label(HdrMode mode);  // « Auto (suit le système) », pour le menu

enum class OledBlack : std::uint8_t { Auto, On, Off };
inline constexpr int kOledBlackCount = 3;
std::string_view name(OledBlack mode);   // « auto », la clé de play.toml
std::string_view label(OledBlack mode);  // « Auto », pour le menu

// Statistiques à l'écran : toutes (Tab), ou mesure par mesure.
struct StatsSettings {
  bool all = false;
  bool fps = false;       // la cadence montrée seule, en haut à gauche (Vidéo > Général)
  bool chain = false;     // modèles, tailles, moteur
  bool frames = false;    // temps par image, σ, composition, images sautées
  bool rife = false;      // paires interpolées, dessins tenus, coupes
  bool tracks = false;    // pistes vidéo, audio, sous-titres
  bool output = false;    // sortie HDR10 ou SDR
  bool position = false;  // position à la milliseconde
  bool memory = false;    // mémoire vidéo
  // Débogage : chaque paire jugée tenue ou coupée teinte l'écran et va au journal (pair_marks.h). Le temps de
  // la session seulement : ni play.toml ni any() ne le connaissent.
  bool pairs = false;
  bool any() const { return all || fps || chain || frames || rife || tracks || output || position || memory; }
  friend bool operator==(const StatsSettings&, const StatsSettings&) = default;
};

// Le préréglage de la chaîne (auto_config.h) : Auto prend la tranche de la machine, Personnalisé laisse les
// réglages faits à la main. Priorité : ce que l'automatique préfère quand la machine ne tient pas tout.
enum class Preset : std::uint8_t { Auto, Economy, Fast, Normal, Quality, Ultra, Custom };
inline constexpr int kPresetCount = 7;
enum class Priority : std::uint8_t { Fluidity, Balance, Image };
inline constexpr int kPriorityCount = 3;

// Les bornes de auto_cap (i/s) : de quoi couper à 24 i/s ou laisser 480 i/s ; 0 prend l'écran.
inline constexpr int kAutoCapMin = 24;
inline constexpr int kAutoCapMax = 480;

// Les réglages par défaut du grade et du look : la configuration de référence, réglée à l'œil sur
// ses épisodes.
inline render::GradeSettings default_grade_values() {
  render::GradeSettings g;
  g.highlights_rtx = 1.0;
  g.area_rtx = 0.10;
  g.shadows = 0.50;  // le maximum des noirs profonds dynamiques
  return g;
}

// Une LUT de la pile du look au-delà de la première : son nom (comme PlaySettings::look) et son intensité de
// 0 à 1.
struct LookLayer {
  std::string key;
  double mix = 1.0;
  friend bool operator==(const LookLayer&, const LookLayer&) = default;
};

// play.toml se réécrit au plus une fois par seconde pendant qu'on règle (accueil et lecture).
inline constexpr std::int64_t kSettingsSaveEveryUs = 1'000'000;
// Après une écriture refusée de play.toml : le nouvel essai, plus lent pour ne pas remplir le journal
// (B36-01).
inline constexpr std::int64_t kSettingsRetryUs = 10'000'000;

struct PlaySettings {
  // Auto-configuration : Auto par défaut, y compris au premier double-clic sur une vidéo (l'assistant ne
  // s'ouvre pas) et après « Tout réinitialiser ». Un play.toml d'avant la section [auto] se lit en
  // Personnalisé : ses réglages ne bougent pas. Hors Personnalisé, la chaîne est refaite à chaque vidéo.
  Preset preset = Preset::Auto;
  Priority priority = Priority::Balance;
  bool upscale = true;
  // « Restaurer les traits » (P3.19) : une source plus haute que work_height est réduite puis réagrandie par
  // SPAN, qui corrige le flou des traits pour quatre fois moins de calcul. Coupé : une telle source garde
  // tout son détail et SPAN ne la touche pas.
  bool restore_lines = true;
  std::string model;  // fichier ONNX de SPAN (nom seul) ; vide : celui du lecteur actuel
  // La stabilisation temporelle avant SPAN (temporal_fix.h) ; éteinte par défaut. L'Auto n'y touche pas.
  // Recharge la chaîne.
  TemporalFix temporal_fix = TemporalFix::Off;
  bool temporal_fix_full = false;  // à pleine taille de travail ; sinon à demi-taille, le détail gardé
  int work_height = 720;
  Downscale downscale = Downscale::Hermite;
  int rife_factor = kDefaultRifeFactor;  // 0 : aucune interpolation
  int rife_factor_den = 1;               // 2 : ×rife_factor / 2 (×1,5 : 3 et 2)
  // Mode auto : le multiplicateur ×k suit ce que le GPU tient (core/auto_rate.h) ; en direct. En Stable,
  // jusqu'à rife_factor ; en Rapide (auto_free, le défaut), rife_factor ne plafonne rien : le facteur suit
  // la cadence de l'écran (top_factor, chain_shape.cpp) et seul auto_cap le borne (A22-10). L'Auto est le
  // défaut ; un multiplicateur choisi à la main (liste du menu) le coupe.
  bool rife_auto = true;
  // Plafond de la cadence de sortie du mode auto, en i/s (60, 120…) ; 0 : celle de l'écran, le maximum.
  int auto_cap = 0;
  // Mode d'Auto : « libre » (Rapide, le défaut depuis C6 : aucun palier, chaque image est calculée à
  // l'instant exact où elle sera vue, core/free_grid.h) ou « recommande » (Stable : paliers,
  // core/auto_rate.h). En direct ; Rapide se replie sans bruit sur Stable quand l'écran ou le moteur ne le
  // permettent pas. RIFE en retard laisse l'image d'origine à l'écran, sans repli.
  bool auto_free = true;
  // « Couper l'interpolation si la carte ne suit pas » (motion_cut.h) : en Rapide, RIFE qui ne produit plus
  // une image interpolée par paire de la source pendant une dizaine de secondes est coupé pour la vidéo en
  // cours, et la carte rendue à SPAN. Décoché : RIFE continue quoi qu'il arrive. En direct.
  bool auto_off = true;
  MotionEngine motion_engine = MotionEngine::Rife;
  std::string rife_model;  // fichier ONNX de RIFE (nom seul) ; vide : le défaut de la carte
  // Garder les poses (DRBA, kernels/spec/poses.h) : un personnage animé en deux garde son rythme
  // pendant que le fond s'interpole ; en direct.
  bool keep_poses = true;
  // « Lisser les dessins tenus » (upscale_smooth.cpp) : un dessin tenu de deux à six images, suivi d'un
  // dessin nouveau, se fond en un seul mouvement au lieu de rester figé puis de sauter ; une tenue plus
  // longue garde la copie, une coupe reste une coupe. RIFE seulement ; prend le pas sur keep_poses
  // (poses_kept), dont il est l'inverse. Éteint par défaut ; recharge la chaîne (l'étage lit cinq
  // images en avance).
  bool smooth_drawings = false;
  // Interpolation par calques (layer_stage.h, docs/calques.md) : là où la carte de la scène voit du décor qui
  // suit la caméra, l'image intermédiaire le reprend de a et b déplacées par la caméra au lieu du warp de
  // RIFE. Plaque de décor : chaque source reprend son décor de la sortie de SPAN d'une image clé par plan.
  // Allumés par défaut depuis les essais du 19 novembre, ils ne tournent qu'avec la carte : sans elle (AMD,
  // Intel, NVOF refusé, --scene-map absent) rien ne change. Rechargent la chaîne.
  bool layers = true;
  bool background_plate = true;
  // Échelle par paire (upscale_half.cpp) : une paire qui bouge fort, où RIFE passe à chaque instant, l'estime
  // sur une image deux fois plus petite que celle de RIFE, par un second moteur ; les autres paires ne
  // changent pas. Moins cher seulement, jamais plus grand. RIFE mouvement seulement. Essai, éteint par défaut
  // : la mesure image par image ne voit pas de perte, mais elle se voit en lecture (docs/interpolation.md) ;
  // recharge la chaîne (le second moteur se construit la première fois pour cette taille).
  bool pair_scale = false;
  // Moins de passes RIFE sur les paires qui bougent fort (rife_plan.h, middle_reaches) : les instants proches
  // du milieu s'en déduisent comme sur une paire calme, RIFE ne repasse qu'aux instants que le milieu
  // prolongé ne sert plus. Avec les poses (le défaut) ; en direct. Essai, éteint par défaut : à juger à l'œil
  // (docs/interpolation.md, Instants déduits sur les paires fortes).
  bool strong_middle = false;
  // RIFE mouvement estimé en 720p : au-delà de 720 lignes de travail, le flot s'estime sur une copie réduite
  // à kRifeMotionHeight, plus rapide. Éteint par défaut : le flot s'estime à la taille de travail, car sur un
  // panoramique de moins d'un pixel par image la copie 720p fait coller le milieu de 4.25 lite à la source et
  // le panoramique ondule (docs/interpolation.md). Recharge la chaîne.
  bool motion_720p = false;
  // Élevé (upscale_gmfss.cpp) : une paire dont le mouvement atteint gmfss_motion pixels (à 720 lignes) a son
  // image du milieu par GMFSS Fortuna Union, ses autres instants par RIFE sur chaque demi-paire ; les autres
  // paires restent à RIFE. TensorRT seulement ; recharge la chaîne (deux moteurs à construire).
  bool gmfss_middle = false;
  int gmfss_motion = kDefaultGmfssMotion;
  // La paire de GMFSS au demi-flot (gmfss_model.h, kGmfssHalfPairFile) : GMFlow sur les images réduites de
  // moitié, environ trois fois moins cher, un peu moins juste sur les petits mouvements. Élevé et Très élevé
  // ; recharge la chaîne (la paire a son propre moteur). Sans ce graphe, la paire au flot entier. Pris par
  // défaut : c'est lui qui fait tenir la cadence, pour une perte mineure.
  bool gmfss_half_flow = true;
  // Élevé en travail plus grand que RIFE (1080p) : l'image du milieu composée des sources de travail
  // splattées par le flot de GMFSS, corrigées du résidu de sa fusion (kernels/spec/gmfss.h, composition), au
  // lieu de la sortie 720p agrandie. Le détail de la source reste ; 4 splats et une composition de plus par
  // paire forte. Recharge la chaîne.
  bool gmfss_compose = false;
  // Très élevé : faux (le défaut), GMFSS fait l'image du milieu de chaque paire qui bouge, par la chaîne du
  // mélange (seuil 0) : à ×2 chaque image interpolée est de GMFSS, RIFE ne la remplace que si le temps
  // manque, ou fait les autres instants au-delà de ×2 depuis M. Vrai : la fusion à chaque instant, sans RIFE
  // (le Très élevé d'avant). Recharge la chaîne.
  bool gmfss_every_instant = false;
  // Élevé et Très élevé par le mélange : RIFE et GMFSS estiment en 540 lignes au lieu de 720
  // (kGmfssLightHeight), environ 1,8 fois moins de pixels ; le flot un peu moins juste, l'image du milieu
  // agrandie ou composée (gmfss_compose). Pour tenir ×2,5 au flot entier. Recharge la chaîne.
  bool gmfss_light = false;
  // Aucune clé GMFSS sous ce mouvement (kDefaultGmfssKeyFloor) : à juger à l'œil, les métriques de fidélité
  // donnent RIFE seul devant sur les petits mouvements. play.toml seulement ; à la prochaine ouverture.
  int gmfss_key_floor = kDefaultGmfssKeyFloor;
  // RIFE fusionné (kernels/spec/rife_fused.h) : le réseau coupé en six petits moteurs TensorRT reliés par des
  // noyaux CUDA, à la place du moteur RIFE entier (2,2 ms au lieu de 6,5 ms par passe à 768×1280 sur une RTX
  // 4070 Ti) ; seulement sous TensorRT et pour un RIFE dont la structure est reconnue (4.25, 4.22 lite), le
  // RIFE entier reprenant sinon, ou si un des six moteurs échoue. Recharge la chaîne.
  bool rife_fused = true;
  // Le warp final de l'interpolation en Catmull-Rom (kernels::RifeWarp : 4 × 4 points, a = −0,5) au lieu du
  // bilinéaire du réseau : des traits fins plus nets dans les images intermédiaires, pour un peu plus de
  // lectures. RIFE seulement (la fin de RIFE fusionné, et le warp de la variante mouvement). Allumé par
  // défaut ; recharge la chaîne.
  bool cubic_warp = true;
  // Mouvement d'ensemble (kernels/spec/warp_pan.h) : au warp de la variante mouvement de RIFE, la translation
  // de la paire, mesurée au centième de pixel, remplace le flot là où il est petit et proche d'elle, et le
  // mélange glisse vers un fondu sous un demi-pixel : moins d'ondulation et de saccade sur les panoramiques
  // lents (docs/interpolation.md, « Mouvement d'ensemble »). RIFE seulement, hors garder les poses. Éteint
  // par défaut ; recharge la chaîne.
  bool pan_lock = false;
  // DistilDRBA fusionné (kernels/spec/drba_fused.h) : le réseau coupé en quatre petits moteurs TensorRT
  // reliés par des noyaux CUDA, à la place du moteur DistilDRBA entier (0,92 ms au lieu de 2,10 ms par passe
  // à 1088×1920 sur une RTX 4070 Ti) ; seulement sous TensorRT, au cran Équilibre et pour un DistilDRBA v2
  // lite dont la structure est reconnue, le moteur entier reprenant sinon, ou si un des quatre moteurs
  // échoue. Recharge la chaîne.
  bool drba_fused = true;
  // Les crans de précision (precision.h). SPAN : Performance prend la variante FP8 du modèle, Qualité sa
  // variante FP32 (span_int8.h), Équilibre le FP16 tel que livré. Recharge la chaîne.
  PrecisionTier span_precision = PrecisionTier::Balanced;
  // RIFE et DistilDRBA : Performance charge le graphe « _fp8 » à côté du modèle (rife_v4.25_fp8.onnx),
  // Qualité le « _fp32 ». Recharge la chaîne.
  PrecisionTier motion_precision = PrecisionTier::Balanced;
  // Le warp avant ou après SPAN (SpanOrder) ; Auto par défaut. Recharge la chaîne.
  SpanOrder span_order = SpanOrder::Auto;
  // Smooth Motion du pilote NVIDIA (RTX 40 et 50, pilote ≥ 580) : sa couche Vulkan double les
  // images présentées, au-dessus de RIFE (×3 puis ×2 : 72 → 144 images par seconde). Posé au
  // lancement, avant l'instance Vulkan : vaut au prochain lancement.
  bool smooth_motion = false;
  // Le pic visé par le grade HDR, en nits (section hdr de play.toml) ; 0 : la crête que l'écran annonce.
  int true_hdr_peak = 0;
  // La sortie HDR voulue (HdrMode) ; hdr_native en découle à chaque ouverture et suit le système en Auto et
  // Activé. En direct.
  HdrMode hdr_mode = HdrMode::Auto;
  // La sortie en cours : HDR (le SDR passe en PQ par libplacebo, puis le grade anime guidé) ou SDR gradée.
  // Pas un réglage : décidée à chaque ouverture selon hdr_mode et le système (host_facts.h, hdr_wanted),
  // suivie si le HDR est allumé ou coupé pendant la lecture, jamais écrite dans play.toml.
  bool hdr_native = false;
  // Composition finale de libplacebo (toutes les routes) : filtre qui étire la chaîne vers la
  // fenêtre, et débandage des aplats et dégradés, sans effet sur l'export ; en direct.
  FinalScaler final_scaler = FinalScaler::Default;
  bool deband = false;
  // Netteté après SPAN (render/sharpen.h), de 0 à 100 % ; 0 : l'image n'est pas touchée. En direct,
  // [picture] sharpness de play.toml.
  int sharpness = 0;
  // À la fin de la vidéo, lire la suivante du dossier (ordre naturel des noms) ; vaut au prochain
  // fichier. Sans effet depuis l'accueil, qui ramène à la liste.
  // Mises à jour du lecteur (updater.h) : une notification, « notify » (Me prévenir, le défaut) ou « never »
  // ; la fréquence « launch », « daily », « weekly ».
  UpdateMode update_mode = UpdateMode::Notify;
  UpdateFrequency update_frequency = UpdateFrequency::Daily;
  // Langue de l'interface : « system » (le défaut, celle du système si l'interface l'a, l'anglais sinon :
  // interface_language), « en » ou « fr » ; prise en direct, à l'accueil comme en lecture (app/play/i18n.h).
  std::string language = std::string(kSystemInterfaceLanguage);
  // Taille de l'interface en pour cent (kUiScalePercent, ui_scale.h) ; 0 : automatique, l'échelle du bureau,
  // plus grande sur un grand écran que Linux laisse à 100 %. Prise en direct.
  int ui_scale = 0;
  // Le dossier où s'installent les moteurs téléchargés (TensorRT) et leurs téléchargements ; vide
  // : le dossier des données de l'utilisateur. Vaut au prochain lancement ; les moteurs déjà installés ne
  // sont pas déplacés.
  std::string engines_dir;
  // La carte graphique de la lecture (P3.16, portables à deux cartes) : son nom Vulkan ; vide, le choix de
  // libplacebo (la carte dédiée d'abord). Vaut au prochain lancement.
  std::string gpu;
  // « Vitesse avant précision » (Vidéo > Général) : le moteur Vulkan garde ses sommes en float16 quand la
  // perte ne se voit pas (au-dessus de 60 dB contre le calcul en float, gpu/vulkan/backend/net_gate.h), au
  // lieu de la règle stricte des 0,5 dB qui rendait SPAN deux fois plus lent sur une GeForce Ada. Décochée :
  // la règle stricte (gpu::kStrictPrecision). Vaut au prochain lancement ; les moteurs se reconstruisent.
  bool fast_precision = true;
  bool autoplay_next = true;  // l'épisode suivant du dossier à la fin, comme le README le promet (P3.84)
  int audio_delay_ms = 0;
  // Décalage des sous-titres (P3.72), de séance : z et x, ou le menu ; jamais écrit dans play.toml.
  int subtitle_delay_ms = 0;
  // L'appareil de sortie audio (vide : celui du système, suivi quand il change).
  std::string audio_device;
  // Le délai de chaque autre appareil, repris quand on y revient (P2.7) : un casque Bluetooth garde le sien,
  // que SDL ne mesure pas. Clé vide : la sortie du système. Celui de audio_device est audio_delay_ms.
  std::map<std::string, int> audio_device_delays_ms;
  // Synchro sur l'écran (SessionConfig::display_sync), en direct ; délai d'affichage de la vidéo
  // en ms, 0 : estimé (SessionConfig::video_latency_us).
  bool display_sync = true;
  int video_latency_ms = 0;
  float volume = 1.0F;
  float subtitle_scale = 1.0F;  // taille des sous-titres
  // Langues voulues à l'ouverture d'un fichier, de la plus voulue à la moins voulue (balises
  // ISO 639 : « jpn », « fre ») ; vide : les drapeaux du fichier. Une piste sans balise suit son
  // titre (« VOSTFR », « Japanese »). Valent au prochain fichier ; la barre change la piste en cours.
  // Audio et sous-titres voulus ensemble, du préréglage préféré au suivant : le premier dont l'audio
  // et les sous-titres existent dans le fichier est pris (media/selection.h). Audio vide : « selon le
  // fichier » ; sous-titres vides : aucun.
  std::vector<media::LanguagePreset> language_presets = default_language_presets();
  // Le choix simple qui a donné ces préférences (language_presets_for) : la langue de l'utilisateur et la
  // version ; Custom quand elles ont été réglées une à une dans Avancé.
  std::string own_language = std::string(system_language());
  LanguageVersion language_version = LanguageVersion::Original;
  // Grade HDR sur la sortie PQ (render/grade_params.h), sur les deux routes ; en direct.
  bool grade = true;
  render::GradeSettings grade_values = default_grade_values();
  // Le look de la méthode HDR anime (ADR 0012) : la LUT de base et « skin fix », fixes, et deux
  // curseurs de -100 à 100 sur les LUT de contraste et d'éclats ; 0 : la pile de référence (contraste ×7,
  // éclats ×1,5), le défaut. -100 : la LUT retirée ; 100 : son intensité doublée. Les clés look, look_mix,
  // looks et looks_mix d'un ancien play.toml ne sont plus lues.
  int look_contrast = 0;
  int look_highlight = 0;
  // Le blanc des surfaces en HDR, en % de celui que la pile porte (100 : vers 70 % du pic de l'écran ; 60, le
  // défaut : vers 40 %). Les éclats montent toujours jusqu'au pic : à 100 %, une corne ou un halo n'avaient
  // qu'un quart de diaphragme au-dessus d'une tête qu'ils éclairent (séance du 19/11), à 60 % ils en ont un.
  // Clé surface_level de play.toml : l'ancienne clé surface_white (100 par défaut) n'est plus lue.
  int surface_white = 60;
  // Limiteur de flashs (P3.51) : une suite de flashs rapides (UIT-R BT.1702) plafonne le grade HDR vers
  // 250 nits pendant une seconde. Coupé, les flashs gardent toute leur lumière.
  bool flash_guard = true;
  // Peau des visages (docs/hdr-anime.md) : la pré-analyse cherche les visages (cascade lbpcascade_animeface)
  // et le grade resserre sa plage de peau sur leurs teintes, pour rendre leur lumière aux lampes chaudes.
  // Éteinte par défaut, à juger à l'œil ; prise à la vidéo suivante (l'analyse des plans recommence).
  bool face_skin = false;
  // Vrai noir OLED (dernière passe du grade) : selon l'écran détecté (display_edid.h), forcé ou
  // coupé ; le pied finit `oled_black_steps` pas de code au-dessus du noir de la source.
  OledBlack oled_black = OledBlack::Auto;
  double oled_black_steps = 8.0;
  // Soins OLED, sur un écran détecté OLED (ou un vrai noir forcé) : en pause, la barre s'efface
  // après 20 s sans activité. Cadence fixe : l'image est représentée à chaque période de l'écran,
  // pour un VRR à fréquence constante (sans scintillement des scènes sombres), au prix d'une
  // composition par période.
  bool oled_care = true;
  bool steady_cadence = false;
  StatsSettings stats;
  friend bool operator==(const PlaySettings&, const PlaySettings&) = default;
};

// Les poses gardées en effet : « Lisser les dessins tenus » fait l'inverse, et l'emporte.
inline bool poses_kept(const PlaySettings& s) {
  return s.keep_poses && !s.smooth_drawings;
}

// Le curseur « Blanc des surfaces », en % (PlaySettings::surface_white).
inline constexpr core::Range<int> kSurfaceWhite{30, 100};

// La longueur d'un chemin de dossier d'installation que le menu et play.toml acceptent (A22-09).
inline constexpr std::size_t kMaxEnginesDirBytes = 4096;

namespace limits {
inline constexpr core::Range<int> kAudioDelayMs{-5000, 5000};
// Le curseur du délai audio va moins loin que la saisie au clavier (Ctrl+clic), qui va jusqu'à kAudioDelayMs.
inline constexpr core::Range<int> kAudioDelaySlider{-1000, 1000};
inline constexpr core::Range<int> kSubtitleDelaySlider{-5000, 5000};
inline constexpr core::Range<int> kVideoLatencyMs{0, 250};
inline constexpr core::Range<float> kVolume{0.0F, 1.5F};
inline constexpr core::Range<float> kSubtitleScale{0.5F, 2.0F};
inline constexpr std::size_t kLanguagePresets = 5;  // préréglages de langue
inline constexpr core::Range<int> kSubtitleDelayMs{-30'000, 30'000};
// Le bas de la plage est le seuil du HDR (kMinHdrPeakNits, host_facts.h, P3.66 ; tenu par un static_assert de
// host_facts.cpp) : un pic réglable fait toujours une sortie HDR.
// Le haut suit les téléviseurs HDR de 3000 à 4000 nits (B37-09) ; le grade accepte jusqu'à 10 000
// (render::grade_limits::kPeakNits).
inline constexpr core::Range<int> kTrueHdrPeak{350, 4000};  // ou 0 : automatique
inline constexpr core::Range<int> kSharpness{0, 100};
}  // namespace limits

// Le facteur d'interpolation des réglages.
inline Factor factor_of(const PlaySettings& s) {
  return {.num = s.rife_factor, .den = s.rife_factor_den};
}

// SPAN tourne-t-il, et à quelle taille la chaîne travaille-t-elle, pour une source de cette taille ? Les
// deux seules lectures de work_height pour la chaîne (A08-15) : chain_setup, l'anneau, l'étage et l'export.
inline bool span_runs(const PlaySettings& s, int source_height) {
  return span_runs(s.upscale, s.restore_lines, source_height, s.work_height);
}
inline WorkSize chain_work_size(const PlaySettings& s, int width, int height) {
  return chain_work_size(s.upscale, s.restore_lines, width, height, s.work_height);
}
// Hauteur où RIFE estime le mouvement quand la chaîne travaille plus haut (sortie 4K) : chain_setup et la
// note du panneau lisent la même (C23-04).
inline constexpr int kRifeMotionHeight = 720;
inline constexpr int kGmfssLightHeight = 540;
inline bool rife_motion_reduced(const WorkSize& work, bool at_720p) {
  return at_720p && work.height > kRifeMotionHeight;
}

// Refuse (FieldOutOfRange) le premier réglage hors plage, en le nommant.
std::expected<PlaySettings, Refusal> validate(const PlaySettings& settings);

// play.toml : une clé absente garde sa valeur par défaut, une clé inconnue est ignorée ; un type
// faux (FieldType) ou une valeur hors plage (FieldOutOfRange) est refusé, TOML invalide
// (ProfileUnreadable) aussi.
std::expected<PlaySettings, Refusal> parse_settings(std::string_view toml_text);
// Comme parse_settings, mais une valeur fautive n'invalide que sa clé : elle est écartée (le défaut la
// remplace), dite dans `dropped` (« section.clé : raison »), et le reste du fichier reste en vigueur. TOML
// invalide : refusé.
std::expected<PlaySettings, Refusal> parse_settings_lenient(std::string_view toml_text,
                                                            std::vector<std::string>* dropped = nullptr);
std::string format_settings(const PlaySettings& settings);
// Les réglages qui diffèrent du défaut, par section, une ligne « clé = valeur » chacun : ce que le rapport
// « Signaler un problème » joint (une option à l'essai allumée s'y voit). « (aucun) » si tout est au défaut.
std::string settings_changes_text(const PlaySettings& settings);
// Pose pour le rapport les faits « Préréglage » et « Réglages changés ».
void note_preset_facts(const PlaySettings& settings);

// Fichier absent : les réglages par défaut. Un fichier qui existe sans se lire (verrou passager réessayé,
// droits, dossier) est refusé (FileUnreadable), et ce processus ne le réécrit plus (B22-02). ignored reçoit
// les valeurs fautives écartées (« section.clé (raison) »), le défaut les remplace : l'appelant les dit
// (B22-05).
std::expected<PlaySettings, Refusal> load_settings(const std::filesystem::path& path,
                                                   std::vector<std::string>* ignored = nullptr);
// Comme load_settings, sans rien journaliser ni toucher à la base de fusion de save_settings : pour un fil
// qui relit un réglage sans reprendre les autres (B36-02, le fil des mises à jour). Une édition du fichier
// faite pendant que le lecteur tourne reste ainsi protégée.
std::expected<PlaySettings, Refusal> peek_settings(const std::filesystem::path& path);
// settings.toml (les moteurs choisis et installés) lu comme play.toml : un verrou passager (antivirus sous
// Windows) se réessaie avant de refuser, pour que l'assistant ne s'ouvre pas sur des défauts (C22-03).
std::expected<core::Settings, Refusal> load_engine_settings(const std::filesystem::path& path,
                                                            std::vector<std::string>* skipped = nullptr);
// Écrit à côté puis remplace (FileUnwritable si impossible) : jamais un fichier à moitié écrit. Un fichier
// existant qui ne se lit pas n'est pas écrasé (FileUnreadable) ; un TOML invalide est gardé en copie d'abord.
std::expected<void, Refusal> save_settings(const std::filesystem::path& path, const PlaySettings& settings);
// Le numéro d'un instantané, pris quand l'écriture est décidée (au post d'une écriture différée) : une
// écriture qui arrive après une plus récente du même fichier est sautée (C30-03).
std::uint64_t settings_ticket();
std::expected<void, Refusal> save_settings(const std::filesystem::path& path, const PlaySettings& settings,
                                           std::uint64_t ticket);

// L'avis des valeurs fautives que load_settings a écartées : les clés (« work_height, peak »), pour
// l'accueil. Vide s'il n'y en a pas (B22-05).
std::string ignored_values_note(const std::vector<std::string>& ignored);

// Les réglages de cette lecture : ceux de play.toml, remplacés par les options données.
PlaySettings settings_for(const PlaySettings& saved, const PlayOptions& options);
// Ce que play.toml doit garder : les réglages en cours, sauf ceux qu'une option a donnés pour cette
// lecture et que le menu n'a pas changés depuis (started : ceux du début de la lecture) ; ceux-là
// gardent la valeur du fichier. Une option d'essai n'y entre jamais.
PlaySettings settings_to_save(const PlaySettings& running, const PlaySettings& from_file,
                              const PlaySettings& started, const PlayOptions& options);
// Les options de lecture qui en découlent (modèle, hauteur, RIFE, HDR).
PlayOptions options_for(const PlayOptions& options, const PlaySettings& settings);

// Ce qu'un changement de réglages demande : rien, en direct, rouvrir la vidéo au même endroit, ou relancer le
// lecteur sur la vidéo au même endroit (la carte, le dossier ou la précision des moteurs, Smooth Motion :
// posés au lancement). Le panneau le dit de la même manière pour tous, avant d'appliquer.
enum class ChangeCost : std::uint8_t { None, Live, Reopen, Restart };
struct Change {
  ChangeCost cost = ChangeCost::None;
  std::string
      reason;  // pour Reopen et Restart : ce qui sera reconstruit ou rechargé, affiché avant d'appliquer
};
struct OpenedChain;  // chain_shape.h
// rife_loaded : un moteur RIFE est chargé (passer de 0 à ×N se fait alors en direct).
// opened : la chaîne ouverte ; un facteur, Auto, Rapide ou plafond qui demandent un anneau plus profond ou un
// autre ordre du warp la rouvrent (A08-03). nullptr : aucune chaîne ouverte, rien à comparer.
Change change_between(const PlaySettings& running, const PlaySettings& wanted, bool rife_loaded,
                      const OpenedChain* opened = nullptr);
// La part de wanted qui s'applique en direct : running, avec les réglages directs de wanted. Avec opened, le
// facteur, Auto, Rapide et le plafond de wanted ne passent que si la chaîne ouverte leur convient.
PlaySettings live_part(const PlaySettings& running, const PlaySettings& wanted, bool rife_loaded,
                       const OpenedChain* opened = nullptr);
// Le brouillon du panneau quand running a changé ailleurs (seen : running vu avant) : il prend les réglages
// directs de running, sans perdre un choix qui attend « Appliquer » (×8 choisi sans RIFE chargé, un moteur du
// mouvement changé sans interpolation) : un tel réglage ne suit running que si running l'a lui-même changé.
PlaySettings follow_running(const PlaySettings& draft, const PlaySettings& seen, const PlaySettings& running,
                            bool rife_loaded);

// Le délai que la sortie du système reprend quand l'appareil choisi disparaît (A16-02) : celui mémorisé pour
// elle, 0 sans mémoire. Rien quand on suit déjà le système : son délai est celui de la séance.
std::optional<std::int64_t> audio_fallback_delay_us(const PlaySettings& settings);

}  // namespace kirari::app::play
