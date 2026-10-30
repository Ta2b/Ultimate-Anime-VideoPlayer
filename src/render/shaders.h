// Les hooks mpv embarqués dans render (src/render/shaders/), octet pour octet. Le .cpp qui les
// définit est généré à la construction par embed_shaders.cmake.
#pragma once

#include <string_view>

namespace ajp::render::shaders {

// anime-hdr-grade du lecteur actuel (trois fichiers mis bout à bout) : neuf passes sur OUTPUT, en
// PQ BT.2020.
extern const std::string_view anime_hdr_grade;
// Le look : une LUT 3D .cube (cube_lut.h) mélangée à l'image après le grade, avant le vrai noir. Le bloc
// //!TEXTURE AJG_LUT s'ajoute derrière ce texte à l'exécution.
extern const std::string_view anime_hdr_look;
// Dixième passe, propre à ce dépôt : le vrai noir OLED (pied smoothstep sous black_hi).
extern const std::string_view anime_hdr_black;
// Mire d'étalonnage du lecteur actuel : patchs en luminance absolue, encodés en PQ.
extern const std::string_view anime_hdr_calibration;

}  // namespace ajp::render::shaders
