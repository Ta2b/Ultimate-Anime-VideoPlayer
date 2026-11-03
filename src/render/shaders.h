// Les hooks mpv embarqués dans render (src/render/shaders/), octet pour octet. Le .cpp qui les
// définit est généré à la construction par embed_shaders.cmake.
#pragma once

#include <string_view>

namespace ajp::render::shaders {

// Les //!PARAM du grade, communs à ses deux corps (chacun se lit derrière eux).
extern const std::string_view anime_hdr_params;
// Le grade d'avant (analyse et grade, mis bout à bout) : neuf passes sur OUTPUT, en PQ BT.2020, la scène
// mesurée par le shader. Gardé une version (--legacy-grade).
extern const std::string_view anime_hdr_grade;
// Le grade en trois passes (carte locale, zone floutée et lissée, grade) : la scène vient du lecteur.
extern const std::string_view anime_hdr_fused;
// La passe « HDR anime » (--hdr-anime) : se lit derrière les deux premières passes de anime_hdr_fused.
extern const std::string_view anime_hdr_anime;
// Le look : une LUT 3D .cube (cube_lut.h) mélangée à l'image après le grade, avant le vrai noir. Le bloc
// //!TEXTURE AJG_LUT s'ajoute derrière ce texte à l'exécution.
extern const std::string_view anime_hdr_look;
// Dixième passe, propre à ce dépôt : le vrai noir OLED (pied smoothstep sous black_hi).
extern const std::string_view anime_hdr_black;
// Mire d'étalonnage du lecteur actuel : patchs en luminance absolue, encodés en PQ.
extern const std::string_view anime_hdr_calibration;

}  // namespace ajp::render::shaders
