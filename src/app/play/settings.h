// Réglages de « ajp play » que le menu des paramètres change, gardés dans play.toml (dossier de
// données du lecteur) d'une lecture à l'autre ; une option donnée sur la ligne de commande les
// remplace pour cette lecture seulement.
//
// Chaque réglage a sa plage ; une valeur hors plage est refusée en la nommant, jamais corrigée.
// Un changement s'applique en direct (réglages de RTX HDR, facteur RIFE, réduction, son,
// statistiques) ou demande de rouvrir la chaîne à la même position (modèle, hauteur de travail,
// RTX HDR allumé ou éteint, RIFE sans moteur chargé) : le menu le dit avant d'appliquer.
//
// Provisoire : les profils (core/profile.h, user.toml) remplacent ce fichier quand la coquille
// de l'application est fusionnée ; les clés gardent les mêmes noms et plages.
#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "app/play/final_scaler.h"
#include "app/play/options.h"
#include "app/play/precision.h"
#include "audio/mixer.h"
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

// Les valeurs du grade pour ce préréglage ; current pour « Personnalisé » et pour gamut_p3.
render::GradeSettings with_preset(render::GradePreset preset, const render::GradeSettings& current);

// Ce qui fait les images intermédiaires : RIFE (réseau, 4.25 ou 4.22 lite, avec ou sans garder les
// poses), GMFSS Fortuna (kernels/spec/gmfss.h : le plus fin sur l'animation, plusieurs fois plus
// lourd que RIFE ; 540 lignes de travail en ×2 tiennent sur une 4070 Ti), en base ou en union
// (GMFSS mêlé à l'image de RIFE), ou DistilDRBA (kernels/spec/distil_drba.h : un RIFE à trois images
// entraîné à imiter DRBA, les poses gardées par le réseau lui-même, une image de retard), ou
// l'interpolation d'Apple (VTFrameRateConversion de VideoToolbox, macOS 15.4, sous le backend Core
// ML : gpu/system_interpolator.h).
enum class MotionEngine : std::uint8_t { Rife, Gmfss, GmfssUnion, DistilDrba, Apple };
inline constexpr int kMotionEngineCount = 5;
constexpr bool is_gmfss(MotionEngine engine) {
  return engine == MotionEngine::Gmfss || engine == MotionEngine::GmfssUnion;
}
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

// Les repères des listes (docs/ux-parametres.md, 3) : une pastille à côté du choix, le détail en infobulle.
enum class Badge : std::uint8_t { Recommended, Fastest, Finest, Heaviest, Experimental };
std::string_view badge_text(
    Badge badge);  // « Recommandé », « Le plus rapide », « Le plus fin », « Le plus lent », « Expérimental »
std::optional<Badge> badge_of(MotionEngine engine);
std::optional<Badge> badge_of(Downscale filter);
// « 720p » ; la hauteur d'image en français courant, jamais « lignes ».
std::string resolution_text(int height);

// Le warp de l'interpolation par rapport à SPAN (variante mouvement de RIFE ou GMFSS, avec SPAN).
// Avant : SPAN agrandit chaque sortie, intermédiaires compris (×3 : trois passes par image source),
// et nettoie les défauts du warp. Après : SPAN agrandit chaque image source une fois, le warp
// déforme les images agrandies à la taille de sortie. Auto : après dès que la sortie dépasse
// kWarpAfterSpanAbove lignes, où SPAN coûte le plus (1080p → 4K : environ 8 ms par passe sur une
// RTX 4070 Ti, contre 0,4 ms pour le warp).
enum class SpanOrder : std::uint8_t { Auto, Before, After };
inline constexpr int kSpanOrderCount = 3;
inline constexpr int kWarpAfterSpanAbove = 1080;
std::string_view name(SpanOrder order);   // « auto », la clé de play.toml
std::string_view label(SpanOrder order);  // « Auto (après SPAN au-dessus de 1080p) », pour le menu
// Le warp passe-t-il après SPAN pour une sortie de output_height lignes ?
constexpr bool warp_after_span(SpanOrder order, int output_height) {
  return order == SpanOrder::After || (order == SpanOrder::Auto && output_height > kWarpAfterSpanAbove);
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
  bool output = false;    // sortie HDR10 ou SDR, pic de RTX HDR
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

struct PlaySettings {
  // Auto-configuration : Personnalisé par défaut, les réglages d'avant ne bougent pas ; le premier
  // lancement choisit Auto. Hors Personnalisé, la chaîne est refaite à chaque vidéo.
  Preset preset = Preset::Custom;
  Priority priority = Priority::Balance;
  bool upscale = true;
  std::string model;  // fichier ONNX de SPAN (nom seul) ; vide : celui du lecteur actuel
  int work_height = 720;
  Downscale downscale = Downscale::Area;
  int rife_factor = kDefaultRifeFactor;  // 0 : aucune interpolation
  int rife_factor_den = 1;               // 2 : ×rife_factor / 2 (×1,5 : 3 et 2)
  // Mode auto : le multiplicateur ×k (entier, jusqu'à rife_factor) suit ce que le GPU tient
  // (core/auto_rate.h) ; en direct. Par défaut (v0.6) : l'interpolation est en Auto, pour tous les profils et
  // après le benchmark ; un multiplicateur choisi à la main (liste du menu) la coupe.
  bool rife_auto = true;
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
  // GMFSS : Performance, la paire en FP8 hors du squelette de GMFlow (_s05fp8, 15 convolutions sur 37) ;
  // Équilibre, FP16 (l'ancien « Performance ») ; Qualité, GMFlow en FP32 (_s05q). Toujours au flot à
  // demi-échelle. Performance par défaut : mesuré −15 % sur la paire en ABAB, 47 dB au pire cas
  // (docs/fp8-interpolation.md) ; sans le graphe ou sans moteur FP8, Équilibre (journal : raison). Recharge
  // la chaîne.
  PrecisionTier gmfss_precision = PrecisionTier::Performance;
  // Le warp avant ou après SPAN (SpanOrder) ; Auto par défaut. Recharge la chaîne.
  SpanOrder span_order = SpanOrder::Auto;
  // Smooth Motion du pilote NVIDIA (RTX 40 et 50, pilote ≥ 580) : sa couche Vulkan double les
  // images présentées, au-dessus de RIFE (×3 puis ×2 : 72 → 144 images par seconde). Posé au
  // lancement, avant l'instance Vulkan : vaut au prochain lancement.
  bool smooth_motion = false;
  // Le mode HDR : son moteur (true_hdr : RTX HDR, NVIDIA ; hdr_native : HDR adaptatif, tous GPU ;
  // aucun : SDR) et ses réglages, communs aux deux moteurs (section hdr de play.toml).
  bool true_hdr = false;
  int true_hdr_contrast = 100;
  int true_hdr_saturation = 100;
  int true_hdr_gray = 50;
  int true_hdr_peak = 0;  // 0 : la crête vue par le contenu sur l'écran
  bool true_hdr_debanding = true;
  // HDR adaptatif : le SDR passe en PQ par libplacebo et le grade émule l'aspect TrueHDR en
  // suivant la scène (route native, mêmes contraste, saturation, gris moyen et pic).
  bool hdr_native = false;
  // Composition finale de libplacebo (toutes les routes) : filtre qui étire la chaîne vers la
  // fenêtre, et débandage des aplats et dégradés, sans effet sur l'export ; en direct.
  FinalScaler final_scaler = FinalScaler::Default;
  bool deband = false;
  // À la fin de la vidéo, lire la suivante du dossier (ordre naturel des noms) ; vaut au prochain
  // fichier. Sans effet depuis l'accueil, qui ramène à la liste.
  // Mises à jour du lecteur (docs/ux-parametres.md, 4.5) : « notify » (Me prévenir, le défaut décidé), « auto
  // », « never » ; le canal « stable » ou « preview » ; la fréquence « launch », « daily », « weekly ».
  std::string update_mode = "notify";
  std::string update_channel = "stable";
  std::string update_frequency = "daily";
  // Le dossier où s'installent les moteurs téléchargés (TensorRT, OpenVINO…) et leurs téléchargements ; vide
  // : le dossier des données de l'utilisateur. Vaut au prochain lancement ; les moteurs déjà installés ne
  // sont pas déplacés.
  std::string engines_dir;
  bool autoplay_next = false;
  bool default_player_dismissed =
      false;  // « Ne plus me le demander » : le bouton quitte Général, reste en bas d'Avancé
  // « Préparer en tâche de fond » (Image IA > Agrandissement) : après « Évaluer ma machine », sur une vidéo
  // ouverte, quand le niveau gardé est Rapide ou plus, dérive le paquet du mode Performance de SPAN
  // (span_derive.h). Éteint par défaut : la préparation prend de la mémoire vidéo et du GPU pendant la
  // lecture.
  bool span_prepare_background = false;
  int audio_delay_ms = 0;
  // Audio (audio/mixer.h, docs/audio-spatialisation.md) : l'appareil de sortie (vide : celui du système,
  // suivi quand il change), « Comment écoutez-vous ? » (headphones, speakers, cinema), le mode Fidèle (aucun
  // traitement), et les profils propres à un appareil (nom de l'appareil vers écoute).
  std::string audio_device;
  std::string listening = "speakers";
  bool audio_faithful = false;
  std::string audio_profile = "natural";  // clé de audio::mix_presets, ou « custom »
  audio::MixTuning audio_custom;          // les réglages du profil « custom »
  std::vector<std::pair<std::string, std::string>> device_listening;
  std::vector<std::pair<std::string, std::string>>
      device_profile;  // appareil -> profil de virtualisation (clé, ou « custom »)
  // « Son anime » (audio/anime_sound.h, docs/audio-anime.md), pistes stéréo vers sortie stéréo : le niveau
  // (clé de audio::anime_levels : natural, anime, strong ; ou « custom »), les réglages du niveau
  // « custom », et le niveau propre à un appareil (appareil -> niveau).
  std::string anime_sound = "anime";
  audio::AnimeTuning anime_custom = audio::find_anime_level("anime")->tuning;
  std::vector<std::pair<std::string, std::string>> device_anime;
  // Réponses impulsionnelles « HeSuVi » pour le casque (audio/hrir.h) : chemin d'un .wav (vide : aucune, le
  // modèle paramétrique), et celui propre à un appareil (appareil -> chemin).
  std::string hrir_path;
  std::vector<std::pair<std::string, std::string>> device_hrir;
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
  std::vector<media::LanguagePreset> language_presets{{"jpn", "fre"}, {"fre", ""}};
  // Grade HDR sur la sortie PQ (render/grade_params.h), sur les deux routes ; en direct. Hors
  // « Personnalisé », grade_values suit le préréglage (gamut_p3 excepté, propre à l'écran).
  bool grade = true;
  render::GradePreset grade_preset = render::GradePreset::Eclatant;
  render::GradeSettings grade_values = render::grade_preset(render::GradePreset::Eclatant);
  // Vrai noir OLED (dernière passe du grade) : selon l'écran détecté (display_edid.h), forcé ou
  // coupé ; le pied finit `oled_black_steps` pas de code au-dessus du noir de la source.
  OledBlack oled_black = OledBlack::Auto;
  double oled_black_steps = 2.0;
  // Soins OLED, sur un écran détecté OLED (ou un vrai noir forcé) : en pause, la barre s'efface
  // après 20 s sans activité. Cadence fixe : l'image est représentée à chaque période de l'écran,
  // pour un VRR à fréquence constante (sans scintillement des scènes sombres), au prix d'une
  // composition par période.
  bool oled_care = true;
  bool steady_cadence = false;
  StatsSettings stats;
  friend bool operator==(const PlaySettings&, const PlaySettings&) = default;
};

// Le moteur du mode HDR, clé hdr.engine de play.toml : « rtx » (RTX HDR, NVIDIA), « adaptive »
// (HDR adaptatif, tous GPU) ou « sdr ».
std::string_view hdr_engine_key(const PlaySettings& s);

namespace limits {
inline constexpr core::Range<int> kAudioDelayMs{-5000, 5000};
inline constexpr core::Range<int> kVideoLatencyMs{0, 250};
inline constexpr core::Range<float> kVolume{0.0F, 1.5F};
inline constexpr core::Range<float> kSubtitleScale{0.5F, 2.0F};
inline constexpr std::size_t kLanguagePresets = 2;  // préréglages de langue
inline constexpr core::Range<int> kTrueHdrContrast{0, 200};
inline constexpr core::Range<int> kTrueHdrSaturation{0, 200};
inline constexpr core::Range<int> kTrueHdrGray{10, 100};
inline constexpr core::Range<int> kTrueHdrPeak{400, 2000};  // ou 0 : automatique
}  // namespace limits

// Le facteur d'interpolation des réglages.
inline Factor factor_of(const PlaySettings& s) {
  return {.num = s.rife_factor, .den = s.rife_factor_den};
}

// Refuse (FieldOutOfRange) le premier réglage hors plage, en le nommant.
std::expected<PlaySettings, Refusal> validate(const PlaySettings& settings);

// play.toml : une clé absente garde sa valeur par défaut, une clé inconnue est ignorée ; un type
// faux (FieldType) ou une valeur hors plage (FieldOutOfRange) est refusé, TOML invalide
// (ProfileUnreadable) aussi.
std::expected<PlaySettings, Refusal> parse_settings(std::string_view toml_text);
std::string format_settings(const PlaySettings& settings);

// L'écoute en vigueur pour cet appareil : son profil propre s'il en a un, sinon le global (un appareil
// inconnu reprend le global sans rien demander). listen : 0 casque, 1 enceintes, 2 home cinéma.
struct EffectiveListening {
  int listen = 1;
  bool faithful = false;
  audio::MixTuning tuning;   // le profil de virtualisation (preset ou personnalisé)
  audio::AnimeTuning anime;  // le « Son anime » de l'appareil (enabled faux : Naturel)
  std::string hrir_path;     // les réponses impulsionnelles du casque ; vide : aucune
  bool operator==(const EffectiveListening&) const = default;
};
// Le mixage voulu : le profil choisi (clé de audio::mix_presets, « custom » : audio_custom), borné ; un
// profil qui ne va pas à l'écoute en cours (Voix claires au casque) retombe sur « Naturel ».
audio::MixTuning resolved_tuning(const PlaySettings& settings, int listen);
// Le mixage d'une clé de profil (ou « custom » : custom_values) pour cette écoute ; Naturel si le profil ne
// va pas à l'écoute.
audio::MixTuning tuning_of(std::string_view profile, const audio::MixTuning& custom_values, int listen);
// Les réglages du « Son anime » d'un niveau (clé de audio::anime_levels, ou « custom » : custom_values) ;
// Naturel pour une clé inconnue.
audio::AnimeTuning anime_tuning_of(std::string_view level, const audio::AnimeTuning& custom_values);
EffectiveListening effective_listening(const PlaySettings& settings, const std::string& device_name);
// Charge les réponses impulsionnelles d'un chemin (vide : aucune, sans refus). Un fichier illisible est
// refusé : l'appelant le dit et retombe sur le binaural paramétrique.
std::expected<std::shared_ptr<const audio::Hrir>, Refusal> load_listening_hrir(const std::string& path);
// « headphones » → 0, « speakers » → 1, « cinema » → 2 ; rien d'autre.
std::optional<int> listening_index(std::string_view key);
// Fichier absent : les réglages par défaut.
std::expected<PlaySettings, Refusal> load_settings(const std::filesystem::path& path);
// Écrit à côté puis remplace (FileUnwritable si impossible) : jamais un fichier à moitié écrit.
std::expected<void, Refusal> save_settings(const std::filesystem::path& path, const PlaySettings& settings);

// Les réglages de cette lecture : ceux de play.toml, remplacés par les options données.
PlaySettings settings_for(const PlaySettings& saved, const PlayOptions& options);
// Ce que play.toml doit garder : les réglages en cours, sauf ceux qu'une option a donnés pour cette
// lecture et que le menu n'a pas changés depuis (started : ceux du début de la lecture) ; ceux-là
// gardent la valeur du fichier. Une option d'essai n'y entre jamais.
PlaySettings settings_to_save(const PlaySettings& running, const PlaySettings& from_file,
                              const PlaySettings& started, const PlayOptions& options);
// Les options de lecture qui en découlent (modèle, hauteur, RIFE, RTX HDR).
PlayOptions options_for(const PlayOptions& options, const PlaySettings& settings);

// Ce qu'un changement de réglages demande.
enum class ChangeCost : std::uint8_t { None, Live, Reopen };
struct Change {
  ChangeCost cost = ChangeCost::None;
  std::string reason;  // pour Reopen : ce qui sera reconstruit ou rechargé, affiché avant d'appliquer
};
// rife_loaded : un moteur RIFE est chargé (passer de 0 à ×N se fait alors en direct).
Change change_between(const PlaySettings& running, const PlaySettings& wanted, bool rife_loaded);
// La part de wanted qui s'applique en direct : running, avec les réglages directs de wanted.
PlaySettings live_part(const PlaySettings& running, const PlaySettings& wanted, bool rife_loaded);

}  // namespace ajp::app::play
