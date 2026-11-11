// Le menu des paramètres : un panneau à droite de l'écran, un arbre de trois catégories (Vidéo, Audio et
// lecture, Général) et dix pages (settings_pages). Les réglages en direct
// s'appliquent pendant qu'on les change ; ceux qui demandent de rouvrir la chaîne attendent « Appliquer »,
// sous un encadré qui dit ce qui sera reconstruit ou rechargé (settings.h : change_between).
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "app/play/calibrated_chain.h"
#include "app/play/chain_shape.h"
#include "app/play/default_player.h"
#include "app/play/input.h"
#include "app/play/precision.h"
#include "app/play/settings.h"
#include "app/play/settings_system.h"
#include "core/engines.h"
#include "media/track.h"

namespace ajp::app::play {

struct ExportDraft;  // settings_export.h
struct MachineNote;  // settings_bench_results.h

// Un moteur du cache, pour l'onglet Moteurs.
struct EngineBadge {
  std::string name;    // « SPAN · HD V3.1 Sharp1 Performance »
  std::string detail;  // « entrée 1280x720 · construit en 60 s »
  bool loaded = false;
};

// Ce que le panneau propose, fixé par le lecteur.
struct SettingsChoices {
  std::filesystem::path luts_dir;   // les .cube de l'utilisateur (look_catalog.h)
  std::vector<std::string> models;  // fichiers SPAN du dossier des modèles
  std::string default_model;        // celui qu'un nom vide désigne
  int source_width = 0;             // la vidéo lue (onglet Export)
  int source_height = 0;            // la réduction n'a de sens que sous cette hauteur
  double source_fps = 0.0;          // images par seconde de la source ; 0 : inconnues
  media::Rational source_rate;      // la même, exacte (nom et fréquence de l'export)
  double display_hz = 0.0;          // fréquence de l'écran ; 0 : inconnue
  bool display_vrr = false;         // l'écran sait le VRR (vrr_capable)
  bool display_vrr_armed = false;   // le système l'a armé en ce moment
  bool window_fullscreen = false;   // KDE n'arme le VRR qu'en plein écran
  bool rife_available = false;      // une interpolation possible : RIFE, DistilDRBA, Apple (Mac)
  // Les crans de précision (precision.h) que le poste rend : Équilibre toujours ; Performance (FP8) et
  // Qualité (FP32) selon les graphes installés.
  TierSupport span_tiers;              // pour le modèle SPAN en cours (span_int8.h)
  TierSupport rife_tiers;              // les graphes « _fp8 » et « _fp32 » du modèle RIFE en cours
  TierSupport drba_tiers;              // ceux de DistilDRBA
  bool distil_drba_available = false;  // le graphe de DistilDRBA
  bool rife_loaded = false;            // moteur RIFE chargé : ×N en direct
  // La chaîne ouverte (chain_shape.h) : un facteur, Auto ou Rapide qui demandent un autre anneau ou un autre
  // ordre du warp attendent « Appliquer ». Rien : à l'accueil, aucune chaîne.
  std::optional<OpenedChain> opened;
  const OpenedChain* chain() const { return opened ? &*opened : nullptr; }
  std::vector<std::string> rife_models;  // fichiers RIFE du dossier du modèle par défaut
  std::string default_rife_model;        // celui qu'un nom vide désigne
  std::string rife_folder;               // le dossier des modèles RIFE (et de DistilDRBA)
  bool nvidia = false;                   // GPU NVIDIA : Smooth Motion n'est offert qu'à lui
  bool hdr_output = true;                // l'écran reçoit du PQ (sinon ni HDR natif ni grade)
  int auto_peak = 0;                     // pic automatique de l'écran, en nits
  std::vector<EngineBadge> engines;
  std::string chosen_engines;                   // « Vulkan, TensorRT » (settings.toml)
  std::vector<core::EngineId> switchable;       // moteurs de SPAN et RIFE installés (engine_switch.h)
  std::optional<core::EngineId> engine_in_use;  // celui de la bibliothèque chargée
  std::string engine_note;                      // le moteur préféré a refusé ce GPU : pourquoi
  std::string display_text;                     // l'écran selon son EDID (display_edid.h : describe)
  std::filesystem::path source_file;            // le fichier lu : l'export se propose à côté
  std::int64_t position_us = 0;                 // où en est sa lecture (PlayerUi::frame)
  bool paused = false;                          // sa lecture est en pause
  std::string black_status;                     // seuils du vrai noir en cours
  // Vidéo > Post-traitement (auto_config.h) : la machine telle que le préréglage Auto la voit.
  int machine_tier =
      2;  // la tranche (gpu_tier.h : Tier, 0 à 4), estimée ou mesurée par « Calibrer cette machine »
  int display_height = 0;  // hauteur de l'écran en pixels ; 0 : inconnue
  int display_width = 0;   // sa largeur ; 0 : inconnue (16:9 supposé)
  // Ce que la machine a mesuré, tel que le lecteur le passe à apply_preset à l'ouverture d'une vidéo
  // (A23-03) : choisir « Auto » au panneau doit donner la même chaîne. calibrated : Auto suit la calibration
  // de l'assistant (calibration.toml) sur cette carte et cet écran ; explored_* : ce que l'exploration du
  // benchmark a trouvé (bench.toml), pour cette carte et ce pilote.
  std::optional<CalibratedChain> calibrated;
  CalibrationHost calibration_host;  // la machine à laquelle le tableau du banc compare sa mesure
  int explored_work_height = 0;
  std::string explored_rife_model;
  std::string gpu_name;           // la carte
  std::vector<std::string> gpus;  // toutes les cartes du système : le choix n'est montré qu'à partir de deux
  struct AudioOut {
    std::string name;
    int channels = 0;
    bool is_default = false;
  };
  std::vector<AudioOut> audio_outputs;  // les appareils de sortie du système (audio/devices.h)
  std::string engines_dir_default;      // où s'installent les moteurs par défaut
  std::string engines_dir_free;         // la place libre à l'endroit choisi, « 2,1 Go »
  bool default_player_is_ajp = false;   // AJP est déjà le lecteur des vidéos (xdg-mime, Linux)
  std::string bench_text;  // « Dernière évaluation : 2026-10-28, niveau Rapide » ; vide : jamais faite
  PlayerPlace place;       // l'exécutable et le dossier de l'utilisateur (onglet Système)
  // Le mode Performance de SPAN (span_derive.h) : ce que la vidéo ouverte permet, pourquoi il est impossible
  // ici (vide : possible) et la tâche que la boucle du lecteur partage avec le panneau (span_auto.h ; vide :
  // celle du panneau seul).
  SpanPrepare span_prepare;
  std::string fp8_reason;
  std::shared_ptr<PrepareNote> span_job;
  // Les étages que la chaîne joue sans eux, ou à un autre cran, et pourquoi (stage_fallback.h) : la page
  // Post-traitement l'affiche avec « Réessayer ». Vide : la chaîne demandée tourne.
  std::string stage_note;
  std::filesystem::path settings_dir;  // le dossier de play.toml, bench.toml, settings.toml
  std::filesystem::path bench_clip;    // la séquence du banc (paths::bench_clip)
};

// Une page de l'arbre des réglages (settings_panel.cpp) : ce qu'elle dessine, sa catégorie, son nom, et si
// elle n'est montrée qu'avec la case « Réglages avancés ».
enum class Page : std::uint8_t {
  PostProcessing,
  Colors,
  Display,
  Sound,
  Playback,
  Application,
  Engines,
  Export,
  Statistics,
  About
};
struct SettingsPage {
  Page id;
  const char* category;
  const char* name;
};
// Toutes les pages, dans l'ordre de l'écran (les catégories se déduisent du changement de `category`).
std::span<const SettingsPage> settings_pages();

// Une liste de langues (settings_panel.cpp), partagée avec l'assistant de premier démarrage.
void language_choice(const char* label, const char* none_label, std::string& tag);

// Le champ « Dossier d'installation » de l'onglet Moteurs : le texte en cours de saisie, repris du réglage à
// chaque ouverture du menu (P5.27 : à SettingsPanel, plus un tampon statique).
struct FolderField {
  std::array<char, 4096>
      text{};             // la longueur d'un chemin (kMaxEnginesDirBytes) : rien n'est coupé en silence
  std::string committed;  // la valeur du réglage que le champ a reprise ou validée en dernier
  std::string problem;    // pourquoi la saisie n'a pas été prise ; vide : rien à dire
  bool read = false;
};

class SettingsPanel {
 public:
  SettingsPanel();
  ~SettingsPanel();
  SettingsPanel(const SettingsPanel&) = delete;
  SettingsPanel& operator=(const SettingsPanel&) = delete;

  void open(const PlaySettings& running) {
    open_ = true;
    draft_ = running;
    seen_ = running;
    stable_ = running;
    undo_.clear();
    redo_.clear();
    engine_draft_.reset();
    engines_folder_ = {};
  }
  void close() { open_ = false; }
  bool is_open() const { return open_; }
  // Une tâche tourne dans le panneau (optimisation, préparation) : des images régulières, même en pause.
  bool busy() const;
  // Il y a quelque chose à dessiner : le panneau ouvert.
  bool wants_draw() const { return open_; }

  struct Result {
    std::optional<PlaySettings> live;    // à appliquer tout de suite
    std::optional<PlaySettings> reopen;  // confirmé : rouvrir la chaîne avec ces réglages
  };
  // Dessine le panneau dans le cadre ImGui en cours (s'il est ouvert).
  Result draw(const PlaySettings& running, const SettingsChoices& choices, float scale,
              std::vector<Input>& actions);

 private:
  // Le corps de la page choisie, puis le pied du panneau (case avancée, annuler, appliquer) : draw les
  // assemble.
  void draw_page(const PlaySettings& running, const SettingsChoices& choices, std::vector<Input>& actions,
                 Result& result, bool& delay_held, bool& sub_delay_held);
  void draw_footer(const PlaySettings& running, const SettingsChoices& choices, const PlaySettings& before,
                   bool delay_held, bool sub_delay_held, Result& result);

  bool open_ = false;
  PlaySettings draft_;  // ce que montre le panneau ; diffère de running pour ce qui attend « Appliquer »
  PlaySettings seen_;   // running au tour précédent
  // Annuler / rétablir : stable_ est le dernier état posé (aucun réglage en cours de glissement), undo_ et
  // redo_ les états de part et d'autre.
  PlaySettings stable_;
  std::vector<PlaySettings> undo_;
  std::vector<PlaySettings> redo_;
  int page_ = 0;                         // la page de l'arbre affichée (settings_panel.cpp : kPages)
  std::array<bool, 8> category_open_{};  // les catégories ouvertes de l'arbre
  int shown_page_ = -1;         // la page de l'image précédente : sa catégorie s'ouvre quand elle change
  bool advanced_ = false;       // la case « Réglages avancés » : montre les sections avancées des pages
  bool pending_shown_ = false;  // le bandeau « Appliquer » était montré au tour précédent
  FolderField engines_folder_;
  std::optional<core::EngineId> engine_draft_;  // moteur coché dans l'onglet Moteurs, pas appliqué
  std::unique_ptr<ExportDraft> export_;         // l'onglet Export, gardé d'une ouverture à l'autre
  SystemNote system_;                           // l'onglet Système : sa dernière action
  UpdateNote updates_;                    // Général > Application : la vérification en cours ou son résultat
  std::unique_ptr<MachineNote> machine_;  // Vidéo > Post-traitement : les résultats du bench
  PrepareNote prepare_;  // « Préparer le mode Performance » : la tâche en cours ou son résultat
};

}  // namespace ajp::app::play
