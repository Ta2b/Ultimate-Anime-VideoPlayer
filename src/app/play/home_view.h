// L'image de l'écran d'accueil (home_screen.h) : « Ouvrir une vidéo », l'invite à glisser un
// fichier sur la fenêtre, et pourquoi la vidéo précédente n'a pas pu être lue. Le menu des
// paramètres se dessine à côté (PlayerUi::home).
#pragma once

#include <optional>
#include <string>

namespace kirari::app::play {

struct HomeView {
  std::string problem;         // la vidéo choisie ne se lit pas : pourquoi ; vide sinon
  bool dialog_open = false;    // la boîte « Ouvrir une vidéo » est à l'écran
  bool open_file = false;      // rendu : « Ouvrir une vidéo » cliqué
  bool show_settings = false;  // rendu : « Paramètres » cliqué (menu fermé)
};

// Le menu des paramètres ouvert, de son bord gauche à son bord droit, en pixels de la fenêtre.
struct PanelSpan {
  float left = 0.0F;
  float right = 0.0F;
};

// Dans le cadre ImGui en cours ; au milieu de la plus grande place que le menu laisse à côté de lui s'il est
// ouvert (il se déplace), sinon de la fenêtre.
void draw_home(HomeView& view, float scale, std::optional<PanelSpan> settings);

}  // namespace kirari::app::play
