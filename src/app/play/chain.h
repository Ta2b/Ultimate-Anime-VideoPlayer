// La chaîne ouverte par « ajp play » et ce que les actions de l'utilisateur y changent : touches,
// menu des paramètres en direct, pistes, profils, grade HDR et vrai noir OLED, soins OLED. Le
// thread main seul y touche (player.cpp ouvre la chaîne et tient la boucle).
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "app/play/chain_adaptive_log.h"
#include "app/play/display_edid.h"
#include "app/play/grade_learning.h"
#include "app/play/input.h"
#include "app/play/live_settings.h"
#include "app/play/pipeline.h"
#include "app/play/placebo_screen.h"
#include "app/play/player.h"
#include "app/play/profiles.h"
#include "app/play/session.h"
#include "app/play/settings.h"
#include "app/play/ui.h"
#include "core/types.h"
#include "gpu/kernel_ops.h"
#include "media/demuxer.h"
#include "render/grade_params.h"

namespace ajp::app::play {

struct ShotWork;  // chain_shots.h

// Les profils de la lecture : user.toml, le profil actif, la source pour choisir leur règle.
struct ProfileState {
  std::filesystem::path user_path;
  Profiles data;
  int active = 1;
  core::SourceInfo source;
  std::string default_model;  // le modèle SPAN par défaut, pour un réglage « modèle par défaut »
  // Le thread d'écriture de la lecture (1.11) : user.toml s'y écrit sans retenir l'affichage. Nul hors
  // d'une lecture : sur place.
  LatestWriter* writer = nullptr;
};

// Le mode HDR que le profil demande, quand un repli de l'environnement (sortie SDR) en fait tourner un
// autre : le repli n'est jamais mémorisé à la place de la préférence.
struct HdrIntent {
  bool hdr_native = false;
};

// Apprentissage du grade adaptatif (grade_learning.h) : la courbe vue au tour précédent, la correction en
// cours (écrite quand les curseurs se sont posés) et le temps passé sans rien changer.
struct LearnTrack {
  std::filesystem::path log;  // adaptive-learning.tsv ; vide : aucun apprentissage
  std::optional<GradeCurve> seen;
  std::optional<LearnSample> pending;
  SceneMix pending_mix;  // les poids appliqués à la scène de la correction
  std::int64_t pending_us = 0;
  std::int64_t still_us = 0;
  int corrections = -1;  // -1 : fichier pas encore lu
  int approvals = 0;
  // --hdr-contract-v2 (8.10) : le goût du grade v2 (grade_taste.h) s'apprend dans adaptive-learning-v2.tsv,
  // et les échantillons de `log` (l'autre grade) sont écartés. taste_set : le goût posé sur l'écran.
  bool contract_v2 = false;
  bool taste_set = false;
};

// Les LUT d'un look lues sur un fil de fond (1.11) : le résultat, posé sur l'écran au tour suivant.
struct LoadedLooks {
  std::array<std::optional<render::CubeLut>, render::kLookSlots> luts;
  std::vector<std::string> failed;  // les clés illisibles
};
struct LookLoad {
  struct Job {
    std::mutex mutex;
    std::optional<LoadedLooks> result;  // sous mutex
    std::atomic<bool> done{false};
  };
  std::shared_ptr<Job> job;
  std::jthread thread;
};

// Ce qui vit le temps d'une ouverture de la chaîne, et ce que les actions y changent.
struct Chain {
  Session& session;
  PlaceboScreen& screen;
  PlayerUi& ui;
  Pipeline& pipeline;
  LiveSettings& live;
  PlaySettings& running;  // réglages en cours, gardés dans play.toml
  ProfileState& profiles;
  bool rife_loaded = false;
  int auto_peak = 0;   // pic automatique de l'écran, en nits
  bool dirty = false;  // play.toml à réécrire
  // Raccourcis du clavier (Ctrl+G, interpolation, tout couper) : valables pour la séance, pas écrits dans
  // play.toml. Chaque champ garde la valeur du fichier jusqu'à ce que le menu change ce réglage.
  std::optional<bool> kept_grade;
  std::optional<std::pair<int, int>> kept_rife;    // facteur, dénominateur
  std::optional<PlaySettings> reopen;              // demandé par le menu des paramètres
  std::optional<EngineChange> engines;             // onglet Moteurs : à faire avant de rouvrir
  std::optional<ExportJob> export_job;             // onglet Export : à faire avant de rouvrir
  std::optional<std::filesystem::path> next_file;  // glissé sur la fenêtre, ou l'épisode voisin
  std::optional<std::int64_t> next_start_us;  // PlayFile : l'instant du fichier suivant (sinon la reprise)
  bool next_paused = false;                   // PlayFile : le fichier suivant s'ouvre en pause
  std::filesystem::path file;                 // la vidéo en cours
  const std::optional<DisplayFacts>& display;
  // Vrai noir OLED : les seuils en cours, et le recalage demandé (au plus 5 fois par seconde).
  std::optional<render::BlackThresholds> black;
  bool black_dirty = true;
  std::int64_t black_at_us = 0;
  // L'en-tête « chaîne » de l'OSD pour ces réglages, recalculé quand le facteur, la réduction ou le
  // moteur HDR changent à chaud ; vide : l'en-tête reste celui de l'ouverture.
  std::function<std::string(const PlaySettings&)> header_text;
  std::optional<HdrIntent> hdr_wanted;  // présent seulement pendant un repli HDR
  std::filesystem::path luts_dir;       // les .cube de l'utilisateur (look_catalog.h)
  bool source_hdr = false;              // la vidéo est PQ ou HLG : le grade ne la retouche pas comme du SDR
  // Grade adaptatif (render/grade_params.h : adaptive) : la luminance moyenne de la source, mesurée par le
  // moteur, lissée ici ; adaptive_t en est la position entre « scène sombre » (0) et « scène claire » (1).
  gpu::KernelOps* kernels = nullptr;
  double adaptive_luma = -1.0;   // lissée, 0 à 1 ; -1 : aucune mesure
  double adaptive_t = 0.0;       // noirs et gris moyen : suivent la luminosité
  double adaptive_blue = 0.0;    // « bleu » moyen lissé de la source
  double adaptive_bs = 0.0;      // série claire par nature : 0 (sombre ou ordinaire) à 1
  bool adaptive_cut = true;      // une coupe de scène (ou la première mesure) : le réglage peut changer
  bool adaptive_forced = false;  // la calibration tenait une scène forcée au tour précédent
  double series_luma = -1.0;     // luminosité moyenne de la série (épisode), lissée
  double series_us = 0.0;
  double adaptive_chroma = 0.0;     // chroma moyenne lissée
  double adaptive_tv = 0.0;         // vibrance : suit la couleur (bleu saturé), pas la luminosité
  double adaptive_glow = 0.0;       // scène sombre avec éclat : 0 à 1 (GradeSettings::adaptive_glow_*)
  double adaptive_glow_seen = 0.0;  // sa mesure, lissée comme la luminance
  std::int64_t adaptive_at_us = 0;
  std::uint64_t adaptive_cuts = 0;         // coupes de scène vues par RIFE (LiveSettings::scene_cuts)
  std::int64_t adaptive_cut_until_us = 0;  // la mesure suit l'image sans lissage jusque-là
  AdaptiveLog adaptive_log;                // adaptive-log.tsv (chain_adaptive_log.h)
  LearnTrack learn;
  // Pré-analyse des plans (chain_shots.h) : le grade adaptatif lit le plan de l'image qui vient au lieu
  // de la mesure en direct, dès qu'il est analysé. Nul : aucune.
  ShotWork* shots = nullptr;
  bool shot_cuts = false;  // --shot-cuts (8.7) : les plans donnent aussi les coupes de l'interpolation
  std::uint64_t adaptive_shot = 0;  // le plan dont le grade tient les valeurs ; 0 : la mesure en direct
  // La lumière de la scène que le grade en trois passes reçoit (render::SceneLight) ; gardée telle quelle
  // tant qu'aucune analyse ne dit mieux (un saut ne la perd pas). scene_light_at_us : son dernier envoi.
  std::optional<render::SceneLight> scene_light;
  std::int64_t scene_light_at_us = 0;
  std::int64_t shots_seek_us = 0;
  std::int64_t shots_newest_pts = kNoPts;
  std::int64_t shots_step_us = 41'667;
  std::size_t plan_shots_seen = 0;  // feed_plan_shots : les plans déjà passés à l'écran
  // Les lectures de LUT en cours (start_looks) ; seule la dernière sera posée. Jointes à la fin de la chaîne.
  std::vector<LookLoad> look_loads;
};

// Les réglages avec, pendant un repli HDR, le mode que le profil demande à la place du mode de
// repli : ce que remember_chain et play.toml retiennent.
PlaySettings with_intended_hdr(const Chain& to, PlaySettings settings);

// Les réglages avec, pour le grade adaptatif, noirs, couleurs et gris moyen glissés de la scène sombre (t =
// 0, les curseurs) vers la scène claire (t = 1). t = 0 ou adaptatif coupé : les réglages tels quels.
PlaySettings with_adaptive(const PlaySettings& s, double t, double tv, double bs = 0.0, double glow = 0.0);

// Le poids (0 à 1) de la scène sombre avec éclat d'une mesure de plan : un pic net (p99,5 loin au-dessus
// de la médiane) sur une petite part de points clairs, et un fond (la médiane) sombre pour les seuils
// adaptive_lo et adaptive_hi du grade.
double glow_weight(const gpu::KernelOps::SceneStats& stats, const render::GradeSettings& g);

// Une fois par tour : lit le plan analysé de l'image qui vient (chain_shots.h) et, si la position du grade
// adaptatif change de 2 % ou plus, pose le grade. Sans plan analysé, le shader suit la scène lui-même.
void adaptive_tick(Chain& to, std::int64_t now_us);

// Apprentissage, une fois par tour du grade adaptatif : une correction de la courbe posée depuis 3 s
// s'écrit avec sa scène ; 30 s de lecture sans rien changer s'écrivent en approbation.
void learn_tick(Chain& to, std::int64_t now_us);
// « Apprendre de mes corrections » : les réglages ajustés (rien si l'ajustement refuse, l'OSD dit pourquoi).
std::optional<PlaySettings> learned_settings(Chain& to, std::int64_t now_us);
// « Oublier mes corrections » : le fichier effacé.
void forget_learning(Chain& to, std::int64_t now_us);

// Charge les LUT du look (jusqu'à trois, enchaînées) dans le grade de l'écran. Rend les noms de celles qui
// manquent ou ne se lisent pas (une ligne de journal chacune) ; elles sont alors sautées.
std::vector<std::string> apply_looks(PlaceboScreen& screen, const PlaySettings& s,
                                     const std::filesystem::path& luts_dir);
// Les mêmes, lues sur un fil de fond (un changement de look pendant la lecture) : tend_looks, à chaque tour,
// pose la dernière demandée une fois lue et dit à l'OSD celles qui ne se lisent pas.
LoadedLooks load_looks(const PlaySettings& s, const std::filesystem::path& luts_dir);
std::vector<std::string> show_looks(PlaceboScreen& screen, LoadedLooks& looks);
void start_looks(Chain& to, const PlaySettings& s);
void tend_looks(Chain& to, std::int64_t now_us);

// Le grade de la chaîne pour chaque sortie : en PQ, route native si le HDR natif est choisi, aucun sinon ;
// en SDR, sur toutes les routes. Rien là où il est neutre.
render::GradeChoice grade_of(const PlaySettings& s, int auto_peak,
                             const std::optional<DisplayFacts>& display = {},
                             const std::optional<render::BlackThresholds>& black = {},
                             bool source_hdr = false);
// La sortie de l'écran, pour l'OSD et le journal.
std::string output_text(const PlaceboScreen& screen, const PlaySettings& s);
double fps_of(const media::Rational& rate);

ProfilesView view_of(const ProfileState& state);
// La chaîne en cours devient celle du profil actif, dans user.toml.
void remember_chain(ProfileState& state, const PlaySettings& s);
// La chaîne (modèle, hauteur, RIFE, HDR) diffère : le profil actif la retient.
bool chain_differs(const PlaySettings& a, const PlaySettings& b);

// Soins OLED, grade avec les seuils du vrai noir, et recalage de ces seuils sur la rampe mesurée.
void apply_oled_care(Chain& to, const PlaySettings& s);
// from_measured_frame : le grade adaptatif, posé à partir de l'image qui suit la dernière reçue du GPU.
void set_grade(Chain& to, const PlaySettings& s, bool from_measured_frame = false);
void refresh_black(Chain& to);
// Une action de l'utilisateur (touche, barre, menu) sur la chaîne.
void apply(const Input& input, Chain& to, std::int64_t now_us);

// --stats : une ligne par présentation ; le bilan de fin dans le journal.
void write_record(std::ofstream& stats, const PresentRecord& record);
void log_summary(const Session& session);

}  // namespace ajp::app::play
