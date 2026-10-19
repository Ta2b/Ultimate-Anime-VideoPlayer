# render : composition libplacebo, grade, sous-titres, OSD ; hors écran en export.
# Dans la tranche verticale : le grade HDR (hooks mpv embarqués, decide_grade), que l'écran de
# app/play pose sur OUTPUT.
ajp_add_module(render
  SOURCES grade.cpp grade_params.cpp "${CMAKE_CURRENT_BINARY_DIR}/render_shaders.cpp"
  DEPS ajp::core ajp::subs ajp::dep::placebo)

# Les hooks mpv du grade sont embarqués dans la bibliothèque, octet pour octet.
set(render_hooks
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/anime-hdr-grade.hook"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/anime-hdr-black.hook"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/anime-hdr-calibration.hook")
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/render_shaders.cpp"
  COMMAND "${CMAKE_COMMAND}" "-DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/render_shaders.cpp"
          "-DINPUTS=${render_hooks}" -P "${CMAKE_CURRENT_SOURCE_DIR}/embed_shaders.cmake"
  DEPENDS ${render_hooks} "${CMAKE_CURRENT_SOURCE_DIR}/embed_shaders.cmake"
  COMMENT "Embarque les hooks du grade"
  VERBATIM)
