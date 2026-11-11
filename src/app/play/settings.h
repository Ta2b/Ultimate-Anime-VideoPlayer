// Réglages de « ajp play » que le menu des paramètres change, gardés dans play.toml (dossier de
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
#include "app/play/update_choice.h"
#include "core/profile.h"
#include "core/types.h"
#include "media/selection.h"
#include "render/grade_params.h"

namespace ajp::app::play {

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

// Ce qui fait les images intermédiaires : RIFE (réseau, 4.25 ou 4.22 lite, avec ou sans garder les
// poses), ou DistilDRBA (kernels/spec/distil_drba.h : un RIFE à trois images
// entraîné à imiter DRBA, les poses gardées par le réseau lui-même, une image de retard), ou
// l'interpolation d'Apple (VTFrameRateConversion de VideoToolbox, macOS 15.4, sous le backend Core
// ML : gpu/system_interpolator.h). GMFSS Fortuna est retiré des sources (0.10.0) : un ancien play.toml qui
// le nomme (« gmfss », « gmfss_union ») se lit en RIFE (settings_toml.cpp). La valeur numérique n'est
// écrite nulle part : play.toml porte le nom.
enum class MotionEngine : std::uint8_t { Rife, DistilDrba, Apple };
inline constexpr int kMotionEngineCount = 3;
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

// Les niveaux de l'interpolation que montre le menu (docs/interpolation.md) : un nom que tout le monde
// comprend, le moteur et le modèle derrière, nommés seulement en mode avancé. System : l'interpolation
// d'Apple, sur un Mac seulement.
enum class InterpLevel : std::uint8_t { VeryFast, Fast, Balanced, System };
struct InterpChoice {
  InterpLevel level;
  MotionEngine engine;
  std::string_view rife_model;  // vide hors RIFE
};
inline constexpr std::array<InterpChoice, 4> kInterpLevels{{
    {.level = InterpLevel::VeryFast, .engine = MotionEngine::DistilDrba, .rife_model = {}},
    {.level = InterpLevel::Fast, .engine = MotionEngine::Rife, .rife_model = "rife_v4.22_lite.onnx"},
    {.level = InterpLevel::Balanced, .engine = MotionEngine::Rife, .rife_model = "rife_v4.25.onnx"},
    {.level = InterpLevel::System, .engine = MotionEngine::Apple, .rife_model = {}},
}};
// Le niveau d'un moteur et du modèle RIFE qu'il charge (le nom vide de play.toml déjà résolu) ; nullopt
// pour un autre modèle RIFE ou un moteur débranché.
std::optional<InterpLevel> interp_level(MotionEngine engine, std::string_view rife_model);

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
// kWarpAfterSpanAbove lignes, où SPAN coûte le plus (1080p → 4K : environ 8 ms par passe sur une
// RTX 4070 Ti, contre 0,4 ms pour le warp), ou dès que le facteur peut atteindre ×3 (D5 : deux images créées
// par paire, deux passes de SPAN de plus contre deux warps).
enum class SpanOrder : std::uint8_t { Auto, Before, After };
inline constexpr int kSpanOrderCount = 3;
inline constexpr int kWarpAfterSpanAbove = 1080;
std::string_view name(SpanOrder order);   // « auto », la clé de play.toml
std::string_view label(SpanOrder order);  // « Auto (après SPAN au-dessus de 1080p ou dès ×3) », pour le menu
inline constexpr int kWarpAfterSpanFactor = 3;
// Le warp passe-t-il après SPAN pour une sortie de output_height lignes, au facteur le plus haut que la
// lecture peut prendre (max_factor, arrondi au-dessous ; 0 : inconnu) ?
constexpr bool warp_after_span(SpanOrder order, int output_height, int max_factor = 0) {
  return order == SpanOrder::After || (order == SpanOrder::Auto && (output_height > kWarpAfterSpanAbove ||
                                                                    max_factor >= kWarpAfterSpanFactor));
}

enum class OledBlack : std::uint8_t { Auto, On, Off };
inline constexpr int kOledBlackCount = 3;
std::string_view name(OledBlack mode);   // « auto », la clé de play.toml
std::string_view label(OledBlack mode);  // « Auto », pour le menu

// Statistiques à l'écran : toutes (Tab), ou mesure par mesure.
struct StatsSettings {
  bool all = false;
  bool chain = false;     // modèles, tailles, moteur
  bool frames = false;    // temps par image, σ, composition, images sautées
  bool rife = false;      // paires interpolées, dessins tenus, coupes
  bool tracks = false;    // pistes vidéo, audio, sous-titres
  bool output = false;    // sortie HDR10 ou SDR
  bool position = false;  // position à la milliseconde
  bool memory = false;    // mémoire vidéo
  bool any() const { return all || chain || frames || rife || tracks || output || position || memory; }
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
  // permettent pas, et quand il perd trop d'images.
  bool auto_free = true;
  MotionEngine motion_engine = MotionEngine::Rife;
  std::string rife_model;  // fichier ONNX de RIFE (nom seul) ; vide : le défaut de la carte
  // Garder les poses (DRBA, kernels/spec/poses.h) : un personnage animé en deux garde son rythme
  // pendant que le fond s'interpole ; en direct.
  bool keep_poses = true;
  // RIFE fusionné (kernels/spec/rife_fused.h) : le réseau coupé en six petits moteurs TensorRT reliés par des
  // noyaux CUDA, à la place du moteur RIFE entier (2,2 ms au lieu de 6,5 ms par passe à 768×1280 sur une RTX
  // 4070 Ti) ; seulement sous TensorRT et pour un RIFE dont la structure est reconnue (4.25, 4.22 lite), le
  // RIFE entier reprenant sinon, ou si un des six moteurs échoue. Recharge la chaîne.
  bool rife_fused = true;
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
  // La sortie en cours : HDR (le SDR passe en PQ par libplacebo, puis le grade anime guidé) ou SDR gradée.
  // Pas un réglage : décidée à chaque ouverture selon le système (host_facts.h, system_hdr), suivie si le
  // HDR est allumé ou coupé pendant la lecture, jamais écrite dans play.toml.
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
  // Langue de l'interface : « en » ou « fr », celle du système par défaut (default_interface_language) ;
  // prise en direct (app/play/i18n.h).
  std::string language = default_interface_language();
  // Le dossier où s'installent les moteurs téléchargés (TensorRT) et leurs téléchargements ; vide
  // : le dossier des données de l'utilisateur. Vaut au prochain lancement ; les moteurs déjà installés ne
  // sont pas déplacés.
  std::string engines_dir;
  // La carte graphique de la lecture (P3.16, portables à deux cartes) : son nom Vulkan ; vide, le choix de
  // libplacebo (la carte dédiée d'abord). Vaut au prochain lancement.
  std::string gpu;
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
  // Grade HDR sur la sortie PQ (render/grade_params.h), sur les deux routes ; en direct.
  bool grade = true;
  render::GradeSettings grade_values = default_grade_values();
  // Le look de la méthode HDR anime (ADR 0012) : la LUT de base et « skin fix », fixes, et deux
  // curseurs de -100 à 100 sur les LUT de contraste et d'éclats ; 0 : la pile de référence (contraste ×7,
  // éclats ×1,5), le défaut. -100 : la LUT retirée ; 100 : son intensité doublée. Les clés look, look_mix,
  // looks et looks_mix d'un ancien play.toml ne sont plus lues.
  int look_contrast = 0;
  int look_highlight = 0;
  // Le blanc des surfaces en HDR, en % de celui que la pile porte (100 : vers 70 % du pic de l'écran ; 70 :
  // vers la moitié). Les éclats montent toujours jusqu'au pic.
  int surface_white = 100;
  // Limiteur de flashs (P3.51) : une suite de flashs rapides (UIT-R BT.1702) plafonne le grade HDR vers
  // 250 nits pendant une seconde. Coupé, les flashs gardent toute leur lumière.
  bool flash_guard = true;
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

// Le curseur « Blanc des surfaces », en % (PlaySettings::surface_white).
inline constexpr core::Range<int> kSurfaceWhite{50, 100};

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
inline constexpr std::size_t kLanguagePresets = 2;  // préréglages de langue
inline constexpr core::Range<int> kSubtitleDelayMs{-30'000, 30'000};
// Le bas de la plage est le seuil du HDR (kMinHdrPeakNits, host_facts.h, P3.66 ; tenu par un static_assert de
// host_facts.cpp) : un pic réglable fait toujours une sortie HDR.
inline constexpr core::Range<int> kTrueHdrPeak{350, 2000};  // ou 0 : automatique
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

// Fichier absent : les réglages par défaut.
std::expected<PlaySettings, Refusal> load_settings(const std::filesystem::path& path);
// Comme load_settings, sans rien journaliser ni toucher à la base de fusion de save_settings : pour un fil
// qui relit un réglage sans reprendre les autres (B36-02, le fil des mises à jour). Une édition du fichier
// faite pendant que le lecteur tourne reste ainsi protégée.
std::expected<PlaySettings, Refusal> peek_settings(const std::filesystem::path& path);
// Écrit à côté puis remplace (FileUnwritable si impossible) : jamais un fichier à moitié écrit.
std::expected<void, Refusal> save_settings(const std::filesystem::path& path, const PlaySettings& settings);

// Les réglages de cette lecture : ceux de play.toml, remplacés par les options données.
PlaySettings settings_for(const PlaySettings& saved, const PlayOptions& options);
// Ce que play.toml doit garder : les réglages en cours, sauf ceux qu'une option a donnés pour cette
// lecture et que le menu n'a pas changés depuis (started : ceux du début de la lecture) ; ceux-là
// gardent la valeur du fichier. Une option d'essai n'y entre jamais.
PlaySettings settings_to_save(const PlaySettings& running, const PlaySettings& from_file,
                              const PlaySettings& started, const PlayOptions& options);
// Les options de lecture qui en découlent (modèle, hauteur, RIFE, HDR).
PlayOptions options_for(const PlayOptions& options, const PlaySettings& settings);

// Ce qu'un changement de réglages demande.
enum class ChangeCost : std::uint8_t { None, Live, Reopen };
struct Change {
  ChangeCost cost = ChangeCost::None;
  std::string reason;  // pour Reopen : ce qui sera reconstruit ou rechargé, affiché avant d'appliquer
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

}  // namespace ajp::app::play
