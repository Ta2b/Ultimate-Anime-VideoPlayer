# render : composition libplacebo, grade, sous-titres, OSD ; hors écran en export. Thread main.
# libplacebo, Vulkan, ImGui et spdlog restent privés : les en-têtes publics ne font que
# déclarer pl_frame, pl_gpu_t, pl_tex_t et ImDrawData.
ajp_add_module(render
  SOURCES
    color.cpp compositor.cpp device_description.cpp grade.cpp grade_params.cpp imgui_layer.cpp
    offscreen.cpp output_ring.cpp render_device.cpp subtitle_layer.cpp
    "${CMAKE_CURRENT_BINARY_DIR}/render_shaders.cpp"
  DEPS ajp::core ajp::subs
  PRIVATE_DEPS ajp::dep::placebo ajp::dep::vulkan ajp::dep::imgui ajp::core_log)

# Aucun appel Vulkan lié directement : chaque fonction passe par vkGetInstanceProcAddr.
target_compile_definitions(ajp_render PRIVATE VK_NO_PROTOTYPES)

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
