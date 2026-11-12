// La chaîne ouverte par « ajp play » et ce que les actions de l'utilisateur y changent : touches,
// menu des paramètres en direct, pistes, grade HDR et vrai noir OLED, soins OLED. Le
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

#include "app/play/anime_shots.h"
#include "app/play/chain_shape.h"
#include "app/play/display_edid.h"
#include "app/play/input.h"
#include "app/play/live_settings.h"
#include "app/play/pipeline.h"
#include "app/play/placebo_screen.h"
#include "app/play/player.h"
#include "app/play/session.h"
#include "app/play/settings.h"
#include "app/play/stage_fallback.h"
#include "app/play/ui.h"
#include "core/types.h"
#include "gpu/kernel_ops.h"
#include "media/demuxer.h"
#include "render/grade_params.h"

namespace ajp::app::play {

struct ShotWork;  // chain_shots.h

// Les LUT d'un look lues sur un fil de fond (1.11) : le résultat, posé sur l'écran au tour suivant.
// Un fichier répété dans la pile (le contraste y est sept fois) n'est gardé qu'une fois, partagé (C41-05).
struct LoadedLooks {
  std::array<std::shared_ptr<const render::CubeLut>, render::kLookSlots> luts;
  std::vector<std::string> failed;  // les clés illisibles
};
// Un seul fil de lecture des LUT, qui ne garde que la dernière demande (P2.9) : cent allers-retours du
// curseur ne lancent plus cent fils. La destruction le joint (A30-02) : une lecture en cours, bornée,
// finit avant.
class LookLoader {
 public:
  LookLoader() = default;
  ~LookLoader();
  LookLoader(const LookLoader&) = delete;
  LookLoader& operator=(const LookLoader&) = delete;
  LookLoader(LookLoader&&) = delete;
  LookLoader& operator=(LookLoader&&) = delete;

  // Remplace la demande pas encore commencée ; lance le fil à la première.
  void request(const PlaySettings& s, const std::filesystem::path& luts_dir);
  // Le résultat de la dernière demande, une fois ; rien tant qu'elle n'est pas lue.
  std::optional<LoadedLooks> take();

  struct State;  // partagé avec le fil

 private:
  std::shared_ptr<State> state_;
  std::jthread thread_;
};

// Les valeurs de play.toml qu'un raccourci de séance (Ctrl+G, interpolation, tout couper) a remplacées dans
// la chaîne ; rien : la chaîne suit le fichier. Elles survivent aux réouvertures de la chaîne d'une vidéo
// (Player::kept, A08-04).
struct KeptChoices {
  std::optional<bool> grade;
  std::optional<std::pair<int, int>> rife;  // facteur, dénominateur
};

// Le facteur (et son dénominateur) après Ctrl+I : coupé s'il tournait ; sinon celui que la séance avait gardé
// (B08-01 : ×2 ou ×5/2 et non ×3), à défaut kDefaultRifeFactor.
std::pair<int, int> toggled_interpolation(int factor, int den,
                                          const std::optional<std::pair<int, int>>& kept);

// Ce qui vit le temps d'une ouverture de la chaîne, et ce que les actions y changent.
struct Chain {
  Session& session;
  PlaceboScreen& screen;
  PlayerUi& ui;
  Pipeline& pipeline;
  LiveSettings& live;
  PlaySettings& running;  // réglages en cours, gardés dans play.toml
  // La forme de cette ouverture et ses mesures (chain_shape.h) : un réglage qui en sort rouvre ;
  // opened.rife_loaded : un moteur d'interpolation est chargé.
  OpenedChain opened;
  int auto_peak = 0;   // pic automatique de l'écran, en nits
  bool dirty = false;  // play.toml à réécrire
  // Raccourcis du clavier (Ctrl+G, interpolation, tout couper) : valables pour la séance, pas écrits dans
  // play.toml. Chaque champ garde la valeur du fichier jusqu'à ce que le menu change ce réglage.
  KeptChoices kept;
  std::optional<PlaySettings> reopen;   // demandé par le menu des paramètres
  std::optional<EngineChange> engines;  // onglet Moteurs : à faire avant de rouvrir
  std::optional<ExportJob> export_job;  // onglet Export : à faire avant de rouvrir
  bool calibrate = false;               // « Refaire le banc » : à faire avant de rouvrir
  std::vector<std::filesystem::path>* dropped_subtitles = nullptr;  // Player::dropped_subtitles
  StageMemory* stages = nullptr;          // les étages refusés de la séance, oubliés par « Réessayer »
  ManualTracks* manual_tracks = nullptr;  // les pistes choisies à la main, pour les vidéos suivantes (B29-11)
  std::optional<std::filesystem::path> next_file;  // glissé sur la fenêtre, ou l'épisode voisin
  std::optional<std::int64_t> next_start_us;  // PlayFile : l'instant du fichier suivant (sinon la reprise)
  bool next_paused = false;                   // PlayFile : le fichier suivant s'ouvre en pause
  bool rederive = false;                      // B29-01 : autre écran, le préréglage refait la chaîne
  std::filesystem::path file;                 // la vidéo en cours
  const std::optional<DisplayFacts>& display;
  // Vrai noir OLED : les seuils en cours, et le recalage demandé (au plus 5 fois par seconde).
  std::optional<render::BlackThresholds> black;
  bool black_dirty = true;
  std::string display_seen;  // le connecteur de l'écran que la chaîne suit (follow_display, P3.58)
  std::int64_t black_at_us = 0;
  // L'en-tête « chaîne » de l'OSD pour ces réglages, recalculé quand le facteur, la réduction ou le
  // moteur HDR changent à chaud ; vide : l'en-tête reste celui de l'ouverture.
  std::function<std::string(const PlaySettings&)> header_text;
  std::filesystem::path luts_dir;  // les .cube de l'utilisateur (look_catalog.h)
  bool source_hdr = false;         // la vidéo est PQ ou HLG : le grade ne la retouche pas comme du SDR
  gpu::KernelOps* kernels = nullptr;
  // Pré-analyse des plans (chain_shots.h) : le grade lit la lumière du plan de l'image qui vient au lieu
  // de la mesure en direct, dès qu'il est analysé. Nul : aucune.
  ShotWork* shots = nullptr;
  bool shot_cuts = false;  // --shot-cuts (8.7) : les plans donnent aussi les coupes de l'interpolation
  // --anime-shots (méga plan A2) : la méthode « HDR anime » décide par plan (anime_shots.h) ; anime_glide, ce
  // qu'elle montre, anime_series_apl la médiane des APL des plans connus.
  bool anime_shots = false;
  AnimeGlide anime_glide;
  double anime_series_apl = 0.0;
  bool flash_limited = false;     // plafond des flashs dangereux tenu (next_flashing, P3.51)
  double anime_e = 0.0;           // l'exposition envoyée au grade (GradeSettings::anime_exposure)
  std::uint64_t shot_serial = 0;  // le plan analysé que le grade suit ; 0 : la mesure en direct
  std::int64_t shot_end_us = -1;  // sa fin, s'il vient de la pré-analyse (B42-01) ; -1 sinon
  double series_apl_sent = 0.0;   // la série des noirs profonds que le grade a reçue (B38-14)
  // La lumière de la scène que le grade en trois passes reçoit (render::SceneLight) ; gardée telle quelle
  // tant qu'aucune analyse ne dit mieux (un saut ne la perd pas). scene_light_at_us : son dernier envoi.
  std::optional<render::SceneLight> scene_light;
  // La peau des visages du plan montré (face_skin), envoyée au grade à la coupe, en sortie HDR comme SDR.
  render::SkinTones skin_tones;
  std::int64_t scene_light_at_us = 0;
  std::int64_t shots_seek_us = 0;
  std::int64_t shots_newest_pts = kNoPts;
  std::int64_t shots_step_us = 41'667;
  std::uint64_t plan_shots_seen = 0;  // feed_plan_shots : les plans déjà passés à l'écran
  // La lecture des LUT en cours (start_looks) ; seule la dernière demande sera posée.
  LookLoader looks;
};

// Les réglages que play.toml retient : le grade et l'interpolation choisis au menu, pas ceux d'un raccourci
// de séance (Chain::kept).
PlaySettings with_kept_choices(const KeptChoices& kept, PlaySettings settings);
PlaySettings with_kept_choices(const Chain& to, PlaySettings settings);

// Une fois par tour : pose le look lu sur un fil de fond, lit le plan analysé de l'image qui vient
// (chain_shots.h), sinon la scène vue en direct, et envoie sa lumière au grade quand elle change.
void grade_tick(Chain& to, std::int64_t now_us);

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
// Flashs dangereux (P3.51) : le grade PQ plafonné vers 250 nits, le plein champ d'une OLED ; son pic d'avant
// reste dans uncapped_peak pour l'annonce HDR (B41-03). Un pic déjà plus bas ne bouge pas.
void cap_flash_peak(render::GradeChoice& choice);
// La sortie de l'écran, pour l'OSD et le journal.
// La ligne « Sortie » des contrôles ; une source HDR sans « étalonnage des sources HDR » le dit : le grade
// coché n'y fait rien (mode Neutre).
std::string output_text(const PlaceboScreen& screen, const PlaySettings& s, bool source_hdr = false);
double fps_of(const media::Rational& rate);

// Soins OLED, grade avec les seuils du vrai noir, et recalage de ces seuils sur la rampe mesurée.
void apply_oled_care(Chain& to, const PlaySettings& s);
// from_measured_frame : le grade, posé à partir de l'image qui suit la dernière reçue du GPU.
void set_grade(Chain& to, const PlaySettings& s, bool from_measured_frame = false);
void refresh_black(Chain& to);
// L'écran a changé, ou son pic (P3.58) : le pic automatique, les soins OLED et le vrai noir suivent l'écran
// courant (Chain::display) et le pic relu, sans rouvrir la chaîne.
void follow_display(Chain& to, std::optional<double> peak_nits);
// Un fichier glissé pendant la lecture (P3.85) : une vidéo la remplace, des sous-titres s'ajoutent comme
// piste externe et sont choisis, le reste est refusé avec un message.
void take_drop(Chain& to, const Dropped& drop, std::int64_t now_us);
// Une action de l'utilisateur (touche, barre, menu) sur la chaîne.
void apply(const Input& input, Chain& to, std::int64_t now_us);

// --stats : le bilan de fin dans le journal (les lignes : StatsSink, present_stats.h).
void log_summary(const Session& session);

// Ce que la séance garde d'une chaîne à l'autre : les étages refusés (stages) et les sous-titres glissés
// pendant la vidéo (dropped), rajoutés à la chaîne rouverte (A29-03) ; wanted : la piste choisie avant la
// réouverture, reprise si c'est l'un d'eux ; manual : les pistes choisies à la main de la séance (B29-11).
void carry_session(Chain& to, StageMemory* stages, std::vector<std::filesystem::path>& dropped,
                   std::optional<std::optional<int>> wanted, ManualTracks* manual);

}  // namespace ajp::app::play
