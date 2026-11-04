# render : composition libplacebo, grade, sous-titres, OSD ; hors écran en export.
# Dans la tranche verticale : le grade HDR (hooks mpv embarqués, decide_grade), que l'écran de
# app/play pose sur OUTPUT.
ajp_add_module(render
  SOURCES cube_lut.cpp curve.cpp faces.cpp display_model.cpp looks.cpp scene_hdr.cpp grade.cpp grade_input.cpp grade_params.cpp "${CMAKE_CURRENT_BINARY_DIR}/render_shaders.cpp"
  DEPS ajp::core ajp::subs ajp::dep::placebo)

# Les hooks mpv du grade sont embarqués dans la bibliothèque, octet pour octet. Le grade : ses paramètres
# (anime-hdr-params.hook), puis l'un des deux corps, mis bout à bout par embed_shaders.cmake : le grade en
# trois passes dont la scène vient du lecteur (anime-hdr-fused), ou celui d'avant en neuf passes
# (anime-hdr-grade : analyse dans le shader), gardé une version derrière --legacy-grade.
file(GLOB render_hook_parts CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/anime-hdr-grade/*.hook"
     "${CMAKE_CURRENT_SOURCE_DIR}/shaders/anime-hdr-fused/*.hook"
     "${CMAKE_CURRENT_SOURCE_DIR}/shaders/anime-hdr-anime/*.hook")
set(render_hooks
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/anime-hdr-params.hook"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/anime-hdr-funcs.glsl"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/anime-hdr-grade"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/anime-hdr-fused"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/anime-hdr-anime"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/anime-hdr-look.hook"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/anime-hdr-black.hook"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/anime-hdr-calibration.hook")
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/render_shaders.cpp"
  COMMAND "${CMAKE_COMMAND}" "-DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/render_shaders.cpp"
          "-DINPUTS=${render_hooks}" -P "${CMAKE_CURRENT_SOURCE_DIR}/embed_shaders.cmake"
  DEPENDS ${render_hooks} ${render_hook_parts} "${CMAKE_CURRENT_SOURCE_DIR}/embed_shaders.cmake"
  COMMENT "Embarque les hooks du grade"
  VERBATIM)
