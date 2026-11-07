// L'état de mise en page du menu des paramètres (P5.27) : il suit le contexte ImGui, qui le porte dans
// io.UserData (PlayerUi le pose à la création du contexte), et non des variables statiques. Sans ImGui ici :
// ui.h l'inclut.
#pragma once

namespace ajp::app::play::panel {

struct State {
  bool advanced = false;    // la case « Réglages avancés » : les réglages rares ne s'affichent que cochée
  bool page_open = false;   // entre begin_page et end_page
  bool group_open = false;  // un encadré de section ouvert dans la page
};

}  // namespace ajp::app::play::panel
