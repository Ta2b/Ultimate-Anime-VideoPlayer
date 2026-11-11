// Les shaders d'Anime4K (third_party/anime4k), octet pour octet : le .cpp qui les définit est généré à la
// construction par embed_shaders.cmake, à part des hooks du grade (render/shaders.h) pour ne pas recompiler
// les uns quand les autres changent. Chaque nom est celui du fichier de l'amont, sans « .glsl ».
#pragma once

#include <string_view>

namespace ajp::render::shaders {

extern const std::string_view Anime4K_Clamp_Highlights;
extern const std::string_view Anime4K_Restore_CNN_M;
extern const std::string_view Anime4K_Restore_CNN_Soft_M;
extern const std::string_view Anime4K_Upscale_CNN_x2_M;
extern const std::string_view Anime4K_Upscale_Denoise_CNN_x2_M;
extern const std::string_view Anime4K_Restore_CNN_VL;
extern const std::string_view Anime4K_Restore_CNN_Soft_VL;
extern const std::string_view Anime4K_Upscale_CNN_x2_VL;
extern const std::string_view Anime4K_Upscale_Denoise_CNN_x2_VL;

}  // namespace ajp::render::shaders
