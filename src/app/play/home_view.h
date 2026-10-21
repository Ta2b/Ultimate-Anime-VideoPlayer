// L'image de l'écran d'accueil (home_screen.h) : « Ouvrir une vidéo », l'invite à glisser un
// fichier sur la fenêtre, et pourquoi la vidéo précédente n'a pas pu être lue. Le menu des
// paramètres se dessine à côté (PlayerUi::home).
#pragma once

#include <string>

namespace ajp::app::play {

struct HomeView {
  std::string problem;         // la vidéo choisie ne se lit pas : pourquoi ; vide sinon
  bool dialog_open = false;    // la boîte « Ouvrir une vidéo » est à l'écran
  bool open_file = false;      // rendu : « Ouvrir une vidéo » cliqué
  bool show_settings = false;  // rendu : « Paramètres » cliqué (menu fermé)
};

// Dans le cadre ImGui en cours ; au milieu de la place que le menu laisse à gauche s'il est ouvert.
void draw_home(HomeView& view, float scale, bool settings_open);

}  // namespace ajp::app::play
