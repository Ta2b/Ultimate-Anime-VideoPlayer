// Le menu des paramètres : un panneau à droite de l'écran, par onglets (Image, Fluidité, RTX HDR,
// Son, Statistiques, Moteurs : settings_engines.h, Système : settings_system.h). Les réglages en direct
// s'appliquent pendant qu'on les change ; ceux qui demandent de rouvrir la chaîne attendent « Appliquer »,
// sous un encadré qui dit ce qui sera reconstruit ou rechargé (settings.h : change_between).
#pragma once

#include <array>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "app/play/default_player.h"
#include "app/play/input.h"
#include "app/play/settings.h"
#include "app/play/settings_system.h"
#include "core/engines.h"
#include "media/track.h"

namespace ajp::app::play {

struct ExportDraft;  // settings_export.h

// Un moteur du cache, pour l'onglet Moteurs.
struct EngineBadge {
  std::string name;    // « SPAN · HD V3.1 Sharp1 Performance »
  std::string detail;  // « entrée 1280x720 · construit en 60 s »
  bool loaded = false;
};

// Les 9 profils tels que l'onglet Profils les montre.
struct ProfileRow {
  std::string name;
  std::string summary;  // la chaîne pour la source en cours
};
struct ProfilesView {
  std::array<std::optional<ProfileRow>, 9> rows;
  int active = 1;
  int start = 1;
};

// Ce que le panneau propose, fixé par le lecteur.
struct SettingsChoices {
  std::vector<std::string> models;       // fichiers SPAN du dossier des modèles
  std::string default_model;             // celui qu'un nom vide désigne
  int source_width = 0;                  // la vidéo lue (onglet Export)
  int source_height = 0;                 // la réduction n'a de sens que sous cette hauteur
  double source_fps = 0.0;               // images par seconde de la source ; 0 : inconnues
  media::Rational source_rate;           // la même, exacte (nom et fréquence de l'export)
  double display_hz = 0.0;               // fréquence de l'écran ; 0 : inconnue
  bool display_vrr = false;              // fréquence variable (VRR) active sur l'écran, d'après le système
  bool rife_available = false;           // une interpolation possible : RIFE, GMFSS, DistilDRBA, Apple (Mac)
  bool gmfss_available = false;          // graphes de GMFSS trouvés (gmfss_model.h)
  bool gmfss_union_available = false;    // ceux de GMFSS Union
  bool gmfss_quality_available = false;  // la paire à GMFlow FP32 (_s05q) de GMFSS base
  bool gmfss_union_quality_available = false;  // celle de GMFSS Union
  bool distil_drba_available = false;          // le graphe de DistilDRBA
  bool rife_loaded = false;                    // moteur RIFE chargé : ×N en direct
  std::vector<std::string> rife_models;        // fichiers RIFE du dossier du modèle par défaut
  std::string default_rife_model;              // celui qu'un nom vide désigne
  std::string rife_folder;                     // le dossier des modèles RIFE (et des graphes de GMFSS)
  bool nvidia = false;              // GPU NVIDIA : RTX HDR et Smooth Motion ne sont offerts qu'à lui
  bool true_hdr_available = false;  // SDK Maxine présent et moteur TensorRT en service
  bool true_hdr_backend = true;     // le moteur TensorRT, seul à ouvrir TrueHDR, est celui du lecteur
  bool hdr_output = true;           // l'écran reçoit du PQ (sinon ni HDR natif ni grade)
  int auto_peak = 0;                // pic automatique de RTX HDR, en nits
  std::vector<EngineBadge> engines;
  std::string chosen_engines;              // « Vulkan, TensorRT, NVIDIA Maxine (RTX HDR) » (settings.toml)
  std::vector<core::EngineId> switchable;  // moteurs de SPAN et RIFE installés (engine_switch.h)
  std::optional<core::EngineId> engine_in_use;  // celui de la bibliothèque chargée
  std::string engine_note;                      // le moteur préféré a refusé ce GPU : pourquoi
  std::string display_text;                     // l'écran selon son EDID (display_edid.h : describe)
  std::filesystem::path source_file;            // le fichier lu : l'export se propose à côté
  std::string black_status;                     // seuils du vrai noir en cours
  ProfilesView profiles;
  // Général > Configuration (auto_config.h) : la machine telle que le préréglage Auto la voit.
  int machine_tier = 2;              // la tranche (gpu_tier.h : Tier, 0 à 4), estimée ou mesurée par « Évaluer ma machine »
  int display_height = 0;            // hauteur de l'écran en pixels ; 0 : inconnue
  std::string gpu_name;              // la carte
  struct AudioOut {
    std::string name;
    int channels = 0;
    bool is_default = false;
  };
  std::vector<AudioOut> audio_outputs;  // les appareils de sortie du système (audio/devices.h)
  bool default_player_is_ajp = false; // AJP est déjà le lecteur des vidéos (xdg-mime, Linux)
  std::string bench_text;            // « Dernière évaluation : 2026-10-28, niveau Rapide » ; vide : jamais faite
  PlayerPlace place;  // l'exécutable et le dossier de l'utilisateur (onglet Système)
};

// Une liste de langues (settings_panel.cpp), partagée avec l'assistant de premier démarrage.
void language_choice(const char* label, const char* none_label, std::string& tag);

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
    engine_draft_.reset();
  }
  void close() { open_ = false; }
  bool is_open() const { return open_; }

  struct Result {
    std::optional<PlaySettings> live;    // à appliquer tout de suite
    std::optional<PlaySettings> reopen;  // confirmé : rouvrir la chaîne avec ces réglages
  };
  // Dessine le panneau dans le cadre ImGui en cours (s'il est ouvert).
  Result draw(const PlaySettings& running, const SettingsChoices& choices, float scale,
              std::vector<Input>& actions);

 private:
  bool open_ = false;
  PlaySettings draft_;  // ce que montre le panneau ; diffère de running pour ce qui attend « Appliquer »
  PlaySettings seen_;   // running au tour précédent
  int page_ = 0;                                // la page de l'arbre affichée (settings_panel.cpp : kPages)
  bool advanced_ = false;                       // la case « Réglages avancés » : montre la catégorie Avancé
  bool pending_shown_ = false;                  // le bandeau « Appliquer » était montré au tour précédent
  std::optional<core::EngineId> engine_draft_;  // moteur coché dans l'onglet Moteurs, pas appliqué
  std::unique_ptr<ExportDraft> export_;         // l'onglet Export, gardé d'une ouverture à l'autre
  SystemNote system_;                           // l'onglet Système : sa dernière action
  UpdateNote updates_;                          // Général > Mises à jour : la vérification en cours ou son résultat
  BenchNote bench_;                             // « Évaluer ma machine » : l'essai en cours ou son rapport
};

}  // namespace ajp::app::play
