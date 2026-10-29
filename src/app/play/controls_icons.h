// Couleurs et icônes de la barre de lecture (controls_draw.cpp), dessinées au trait avec Dear
// ImGui : aucune police d'icônes ni image à charger.
#pragma once

#include <imgui.h>

namespace ajp::app::play::icons {

// Couleurs sRGB : blanc pour le texte et les icônes, un bleu clair pour la progression.
ImU32 white(float a);
ImU32 black(float a);
ImU32 accent(float a);

// Ce que les dessins partagent : la liste, l'échelle, l'opacité de la barre.
struct Pen {
  ImDrawList* list = nullptr;
  float scale = 1.0F;
  float opacity = 1.0F;
};

// Chaque icône est centrée sur c, dans un carré de côté size.
void icon_play(const Pen& pen, ImVec2 c, float size);
void icon_pause(const Pen& pen, ImVec2 c, float size);
void icon_volume(const Pen& pen, ImVec2 c, float size, float volume, bool muted);
// Quatre équerres : tournées vers l'extérieur pour entrer en plein écran, vers le centre pour en
// sortir.
void icon_fullscreen(const Pen& pen, ImVec2 c, float size, bool leave);
// Sous-titres : un cadre arrondi et deux lignes de texte, soulignées quand une piste est choisie.
void icon_subtitles(const Pen& pen, ImVec2 c, float size, bool on);
// Piste audio : une note de musique.
void icon_audio_track(const Pen& pen, ImVec2 c, float size);
void icon_gear(const Pen& pen, ImVec2 c, float size);
// Chapitre précédent (direction −1) ou suivant (+1) : un triangle contre une barre.
void icon_chapter(const Pen& pen, ImVec2 c, float size, int direction);

}  // namespace ajp::app::play::icons
