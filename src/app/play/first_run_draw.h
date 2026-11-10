// Le dessin partagé par l'écran des moteurs (first_run.cpp) et l'assistant du premier lancement
// (first_run_wizard.cpp) : notes, titres de section, lignes de moteur, couleur d'accent.
#pragma once

#include <string>

#include <imgui.h>

#include "app/play/first_run.h"

namespace ajp::app::play {

inline constexpr ImVec4 kAccent{0.42F, 0.66F, 1.0F, 1.0F};

void note(const std::string& text);  // texte grisé, replié à la largeur
void section(const char* title);     // titre en couleur d'accent, souligné
void engine_row(EngineRow& row);     // case, nom, badges, état, licence à accepter

// Les écrans de l'assistant (wizard.h), centrés dans la fenêtre.
void draw_wizard(FirstRunView& view, float scale);

}  // namespace ajp::app::play
