// Les hooks mpv embarqués dans render (src/render/shaders/), octet pour octet. Le .cpp qui les
// définit est généré à la construction par embed_shaders.cmake.
#pragma once

#include <string_view>

namespace ajp::render::shaders {

// Les //!PARAM du grade (la passe anime se lit derrière eux).
extern const std::string_view anime_hdr_params;
// Les trois passes du grade : carte locale 1/8, zone ~2,5 % lissée, puis la passe « anime » (surfaces
// ancrées, seuls les émetteurs montent, carte locale guidée par le pixel).
extern const std::string_view anime_hdr_anime;
// Le look : une LUT 3D .cube (cube_lut.h) mélangée à l'image après le grade, avant le vrai noir. Le bloc
// //!TEXTURE AJG_LUT s'ajoute derrière ce texte à l'exécution.
extern const std::string_view anime_hdr_look;
// Dixième passe, propre à ce dépôt : le vrai noir OLED (pied smoothstep sous black_hi).
extern const std::string_view anime_hdr_black;
// Mire d'étalonnage du lecteur actuel : patchs en luminance absolue, encodés en PQ.
extern const std::string_view anime_hdr_calibration;
// La netteté (render/sharpen.h) : un masque flou adaptatif sur SCALED.
extern const std::string_view sharpen;

}  // namespace ajp::render::shaders
