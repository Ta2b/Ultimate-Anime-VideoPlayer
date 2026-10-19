// Le menu des paramètres : un panneau à droite de l'écran, par onglets (Image, Fluidité, RTX HDR,
// Son, Statistiques, Moteurs). Les réglages en direct s'appliquent pendant qu'on les change ;
// ceux qui demandent de rouvrir la chaîne attendent « Appliquer », sous un encadré qui dit ce qui
// sera reconstruit ou rechargé (settings.h : change_between).
#pragma once

#include <optional>
#include <string>
#include <vector>

#include "app/play/settings.h"

namespace ajp::app::play {

// Un moteur du cache, pour l'onglet Moteurs.
struct EngineBadge {
  std::string name;    // « SPAN · HD V3.1 Sharp1 Performance »
  std::string detail;  // « entrée 1280x720 · construit en 60 s »
  bool loaded = false;
};

// Ce que le panneau propose, fixé par le lecteur.
struct SettingsChoices {
  std::vector<std::string> models;  // fichiers SPAN du dossier des modèles
  std::string default_model;        // celui qu'un nom vide désigne
  int source_height = 0;            // la réduction n'a de sens que sous cette hauteur
  bool rife_available = false;      // modèle RIFE trouvé
  bool rife_loaded = false;         // moteur RIFE chargé : ×N en direct
  bool true_hdr_available = false;  // SDK Maxine présent
  int auto_peak = 0;                // pic automatique de RTX HDR, en nits
  std::vector<EngineBadge> engines;
};

// Libellé lisible d'un modèle AnimeJaNai : « HD V3.1 Sharp1 Performance ».
std::string model_label(const std::string& file);

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
  Result draw(const PlaySettings& running, const SettingsChoices& choices, float scale);

 private:
  bool open_ = false;
  PlaySettings draft_;  // ce que montre le panneau ; diffère de running pour ce qui attend « Appliquer »
  PlaySettings seen_;   // running au tour précédent
};

}  // namespace ajp::app::play
