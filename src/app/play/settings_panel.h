// Le menu des paramètres : un panneau à droite de l'écran, par onglets (Image, Fluidité, RTX HDR,
// Son, Statistiques, Moteurs : settings_engines.h). Les réglages en direct s'appliquent pendant qu'on les
// change ; ceux qui demandent de rouvrir la chaîne attendent « Appliquer », sous un encadré qui dit ce qui
// sera reconstruit ou rechargé (settings.h : change_between).
#pragma once

#include <array>
#include <optional>
#include <string>
#include <vector>

#include "app/play/input.h"
#include "app/play/settings.h"
#include "core/engines.h"

namespace ajp::app::play {

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
  int source_height = 0;                 // la réduction n'a de sens que sous cette hauteur
  double source_fps = 0.0;               // images par seconde de la source ; 0 : inconnues
  double display_hz = 0.0;               // fréquence de l'écran ; 0 : inconnue
  bool rife_available = false;           // une interpolation possible : modèle RIFE, flux optique ou GMFSS
  bool gmfss_available = false;          // graphes de GMFSS trouvés (gmfss_model.h)
  bool gmfss_union_available = false;    // ceux de GMFSS Union
  bool distil_drba_available = false;    // le graphe de DistilDRBA
  bool rife_loaded = false;              // moteur RIFE chargé : ×N en direct
  std::vector<std::string> rife_models;  // fichiers RIFE du dossier du modèle par défaut
  std::string default_rife_model;        // celui qu'un nom vide désigne
  std::string rife_folder;               // le dossier des modèles RIFE (et des graphes de GMFSS)
  bool true_hdr_available = false;       // SDK Maxine présent
  bool hdr_output = true;                // l'écran reçoit du PQ (sinon ni HDR natif ni grade)
  int auto_peak = 0;                     // pic automatique de RTX HDR, en nits
  std::vector<EngineBadge> engines;
  std::string chosen_engines;              // « Vulkan, TensorRT, NVIDIA Maxine (RTX HDR) » (settings.toml)
  std::vector<core::EngineId> switchable;  // moteurs de SPAN et RIFE installés (engine_switch.h)
  std::optional<core::EngineId> engine_in_use;  // celui de la bibliothèque chargée
  std::string engine_note;                      // le moteur préféré a refusé ce GPU : pourquoi
  std::string display_text;                     // l'écran selon son EDID (display_edid.h : describe)
  std::string black_status;                     // seuils du vrai noir en cours
  ProfilesView profiles;
};

class SettingsPanel {
 public:
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
  std::array<char, 64> rename_{};               // nouveau nom du profil actif, en cours de saisie
  int shown_profile_ = 0;                       // profil dont rename_ porte le nom
  std::optional<core::EngineId> engine_draft_;  // moteur coché dans l'onglet Moteurs, pas appliqué
};

}  // namespace ajp::app::play
