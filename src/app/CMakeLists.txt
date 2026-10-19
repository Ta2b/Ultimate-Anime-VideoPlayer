# app : boucle SDL3, commandes, état de l'interface. Crée les threads et
# dépend de tous les modules, jamais l'inverse. core_log : app seul installe
# les puits du journal (install_log_sinks), depuis main.
#
# play/ : la tranche verticale « ajp play » (session du thread main, câblage des
# threads source, gpu, audio et subs, horloge audio maîtresse).
ajp_add_module(app
  SOURCES
    play/backend_library.cpp play/bridges.cpp play/controls.cpp play/controls_draw.cpp play/engine_build.cpp
    play/frame_queue.cpp play/frame_stats.cpp play/input.cpp play/options.cpp play/osd_text.cpp play/pipeline.cpp
    play/placebo_imgui.cpp play/placebo_overlays.cpp play/placebo_ring.cpp play/placebo_screen.cpp
    play/placebo_snapshot.cpp play/player.cpp
    play/ring_stage.cpp play/rife_plan.cpp play/session.cpp play/subtitle_queue.cpp play/tracks.cpp play/ui.cpp
    play/upscale_stage.cpp play/wiring.cpp
  DEPS ajp::core ajp::core_log ajp::media ajp::gpu ajp::prepare ajp::audio ajp::subs
       ajp::render ajp::present ajp::export
  PRIVATE_DEPS ajp::dep::ffmpeg ajp::dep::placebo ajp::dep::sdl3 ajp::dep::spdlog ajp::dep::vulkan
               ajp::dep::imgui ${CMAKE_DL_LIBS})
# Police de l'interface : celle du bureau par fontconfig quand il est là, sinon celle d'ImGui.
if(TARGET ajp::dep::fontconfig)
  target_link_libraries(ajp_app PRIVATE ajp::dep::fontconfig)
  target_compile_definitions(ajp_app PRIVATE AJP_WITH_FONTCONFIG=1)
endif()

# ajp : l'exécutable. Le backend TensorRT se cherche à côté de lui, puis dans l'arbre de
# construction quand il y est construit (AJP_WITH_CUDA).
add_executable(ajp main.cpp)
target_link_libraries(ajp PRIVATE ajp::app ajp::dep::ffmpeg ajp::dep::sdl3 ajp::dep::spdlog)
if(TARGET ajp_backend_tensorrt)
  target_compile_definitions(ajp PRIVATE AJP_BACKEND_TENSORRT="$<TARGET_FILE:ajp_backend_tensorrt>")
  add_dependencies(ajp ajp_backend_tensorrt)
endif()
ajp_set_warnings(ajp)
