# render : composition libplacebo, grade, sous-titres, OSD ; hors écran en export.
# Dans la tranche verticale : le grade HDR (hooks mpv embarqués, decide_grade), que l'écran de
# app/play pose sur OUTPUT.
ajp_add_module(render
  SOURCES cube_lut.cpp display_model.cpp anime4k.cpp grade.cpp grade_params.cpp output_tune.cpp sharpen.cpp "${CMAKE_CURRENT_BINARY_DIR}/render_shaders.cpp"
          "${CMAKE_CURRENT_BINARY_DIR}/anime4k_shaders.cpp"
  DEPS ajp::core ajp::dep::placebo
  PRIVATE_DEPS ajp::kernels)

# Les hooks mpv du grade sont embarqués dans la bibliothèque, octet pour octet. Le grade : ses paramètres
# (anime-hdr-params.hook), puis ses trois passes (anime-hdr-anime : carte locale, zone, passe anime), mises
# bout à bout par embed_shaders.cmake ; puis le look et le vrai noir.
file(GLOB render_hook_parts CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/anime-hdr-anime/*.hook")
set(render_hooks
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/anime-hdr-params.hook"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/anime-hdr-anime"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/anime-hdr-look.hook"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/anime-hdr-black.hook"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/sharpen.hook")
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/render_shaders.cpp"
  COMMAND "${CMAKE_COMMAND}" "-DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/render_shaders.cpp"
          "-DINPUTS=${render_hooks}" -P "${CMAKE_CURRENT_SOURCE_DIR}/embed_shaders.cmake"
  DEPENDS ${render_hooks} ${render_hook_parts} "${CMAKE_CURRENT_SOURCE_DIR}/embed_shaders.cmake"
  COMMENT "Embarque les hooks du grade"
  VERBATIM)

# Les shaders d'Anime4K (render/anime4k.h), copiés tels quels de l'amont dans third_party/anime4k : embarqués à
# part, avec leur en-tête (render/anime4k_shaders.h).
set(anime4k_dir "${CMAKE_SOURCE_DIR}/third_party/anime4k/glsl")
set(anime4k_shaders
  "${anime4k_dir}/Restore/Anime4K_Clamp_Highlights.glsl"
  "${anime4k_dir}/Restore/Anime4K_Restore_CNN_M.glsl"
  "${anime4k_dir}/Restore/Anime4K_Restore_CNN_Soft_M.glsl"
  "${anime4k_dir}/Restore/Anime4K_Restore_CNN_VL.glsl"
  "${anime4k_dir}/Restore/Anime4K_Restore_CNN_Soft_VL.glsl"
  "${anime4k_dir}/Upscale/Anime4K_Upscale_CNN_x2_M.glsl"
  "${anime4k_dir}/Upscale/Anime4K_Upscale_CNN_x2_VL.glsl"
  "${anime4k_dir}/Upscale+Denoise/Anime4K_Upscale_Denoise_CNN_x2_M.glsl"
  "${anime4k_dir}/Upscale+Denoise/Anime4K_Upscale_Denoise_CNN_x2_VL.glsl")
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/anime4k_shaders.cpp"
  COMMAND "${CMAKE_COMMAND}" "-DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/anime4k_shaders.cpp"
          "-DINPUTS=${anime4k_shaders}" "-DHEADER=render/anime4k_shaders.h"
          -P "${CMAKE_CURRENT_SOURCE_DIR}/embed_shaders.cmake"
  DEPENDS ${anime4k_shaders} "${CMAKE_CURRENT_SOURCE_DIR}/embed_shaders.cmake"
  COMMENT "Embarque les shaders d'Anime4K"
  VERBATIM)
