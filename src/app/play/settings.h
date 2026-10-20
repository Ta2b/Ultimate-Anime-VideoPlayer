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
#include <string>
#include <string_view>
#include <vector>

#include "app/play/options.h"
#include "core/profile.h"
#include "core/types.h"
#include "render/grade_params.h"

namespace ajp::app::play {

// Filtre de la réduction avant SPAN (fusionnée à l'ingestion, en codes de la source) : celui des
// profils (core/profile.h), mêmes noms dans play.toml et user.toml.
using Downscale = core::DownscaleFilter;
inline constexpr int kDownscaleCount = 6;
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
// entraîné à imiter DRBA, les poses gardées par le réseau lui-même, une image de retard).
enum class MotionEngine : std::uint8_t { Rife, Gmfss, GmfssUnion, DistilDrba };
inline constexpr int kMotionEngineCount = 4;
constexpr bool is_gmfss(MotionEngine engine) {
  return engine == MotionEngine::Gmfss || engine == MotionEngine::GmfssUnion;
}
std::string_view name(MotionEngine engine);   // « rife », la clé de play.toml
std::string_view label(MotionEngine engine);  // « RIFE », pour le menu

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

struct PlaySettings {
  bool upscale = true;
  std::string model;  // fichier ONNX de SPAN (nom seul) ; vide : celui du lecteur actuel
  int work_height = 720;
  Downscale downscale = Downscale::Area;
  int rife_factor = kDefaultRifeFactor;  // 0 : aucune interpolation
  int rife_factor_den = 1;               // 2 : ×rife_factor / 2 (×1,5 : 3 et 2)
  MotionEngine motion_engine = MotionEngine::Rife;
  std::string rife_model;  // fichier ONNX de RIFE (nom seul) ; vide : le défaut de la carte
  // Garder les poses (DRBA, kernels/spec/poses.h) : un personnage animé en deux garde son rythme
  // pendant que le fond s'interpole ; en direct.
  bool keep_poses = false;
  // Warp après SPAN (variante mouvement de RIFE, avec SPAN) : SPAN agrandit chaque image source une
  // fois, et le warp déforme ces images agrandies à la taille de sortie ; SPAN ne repasse plus sur
  // les images intermédiaires (×3 : une passe au lieu de trois). Les défauts du warp ne sont plus
  // nettoyés par SPAN. Recharge la chaîne.
  bool warp_after_span = false;
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
  int audio_delay_ms = 0;
  // Synchro sur l'écran (SessionConfig::display_sync), en direct ; délai d'affichage de la vidéo
  // en ms, 0 : estimé (SessionConfig::video_latency_us).
  bool display_sync = true;
  int video_latency_ms = 0;
  float volume = 1.0F;
  float subtitle_scale = 1.0F;  // taille des sous-titres
  // Langues voulues à l'ouverture d'un fichier, de la plus voulue à la moins voulue (balises
  // ISO 639 : « jpn », « fre ») ; vide : les drapeaux du fichier. Une piste sans balise suit son
  // titre (« VOSTFR », « Japanese »). Valent au prochain fichier ; la barre change la piste en cours.
  std::vector<std::string> audio_languages{"jpn"};
  std::vector<std::string> subtitle_languages{"fre", "eng"};
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
inline constexpr std::size_t kMaxLanguages = 4;  // par liste
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
