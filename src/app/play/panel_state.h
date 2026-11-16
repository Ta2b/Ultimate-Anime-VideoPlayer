// L'état de mise en page du menu des paramètres (P5.27) : il suit le contexte ImGui, qui le porte dans
// io.UserData (PlayerUi le pose à la création du contexte), et non des variables statiques. Sans ImGui ici :
// ui.h l'inclut.
#pragma once

struct ImFont;

namespace ajp::app::play::panel {

struct State {
  bool page_open = false;   // entre begin_page et end_page
  bool group_open = false;  // un encadré de section ouvert dans la page
  bool dim = false;
  const char* group_title = nullptr;  // le titre de l'encadré ouvert, posé dans sa bordure à sa fermeture //
                                      // un curseur qui s'applique en direct est tenu : le panneau s'efface
  // La police grasse des titres d'infobulle (ui.cpp : load_fonts) ; nulle sans FreeType : le titre est alors
  // tracé deux fois, décalé d'un pixel.
  ImFont* bold = nullptr;
};

}  // namespace ajp::app::play::panel
