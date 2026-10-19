// Le menu des paramètres : un panneau à droite de l'écran, par onglets (Image, Fluidité, RTX HDR,
// Son, Statistiques, Moteurs). Les réglages en direct s'appliquent pendant qu'on les change ;
// ceux qui demandent de rouvrir la chaîne attendent « Appliquer », sous un encadré qui dit ce qui
// sera reconstruit ou rechargé (settings.h : change_between).
#pragma once

#include <array>
#include <optional>
#include <string>
#include <vector>

#include "app/play/input.h"
#include "app/play/settings.h"

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
  bool rife_available = false;           // modèle RIFE trouvé
  bool rife_loaded = false;              // moteur RIFE chargé : ×N en direct
  std::vector<std::string> rife_models;  // fichiers RIFE du dossier du modèle par défaut
  std::string default_rife_model;        // celui qu'un nom vide désigne
  bool true_hdr_available = false;       // SDK Maxine présent
  bool hdr_output = true;                // l'écran reçoit du PQ (sinon ni HDR natif ni grade)
  int auto_peak = 0;                     // pic automatique de RTX HDR, en nits
  std::vector<EngineBadge> engines;
  std::string chosen_engines;  // « Vulkan, TensorRT, NVIDIA Maxine (RTX HDR) » (settings.toml)
  std::string display_text;    // l'écran selon son EDID (display_edid.h : describe)
  std::string black_status;    // seuils du vrai noir en cours
  ProfilesView profiles;
};

class SettingsPanel {
 public:
  void open(const PlaySettings& running) {
    open_ = true;
    draft_ = running;
    seen_ = running;
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
  std::array<char, 64> rename_{};  // nouveau nom du profil actif, en cours de saisie
  int shown_profile_ = 0;          // profil dont rename_ porte le nom
};

}  // namespace ajp::app::play
