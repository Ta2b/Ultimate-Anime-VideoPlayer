# app : boucle SDL3, commandes, état de l'interface. Crée les threads et
# dépend de tous les modules, jamais l'inverse. core_log : app seul installe
# les puits du journal (install_log_sinks), depuis main.
#
# play/ : la tranche verticale « ajp play » (session du thread main, câblage des
# threads source, gpu, audio et subs, horloge audio maîtresse).
ajp_add_module(app
  SOURCES
    play/backend_library.cpp play/bridges.cpp play/build_progress.cpp play/chain.cpp play/chain_stats.cpp play/model_choice.cpp play/chapters.cpp play/controls.cpp play/controls_draw.cpp play/controls_icons.cpp play/derived_model.cpp play/display_edid.cpp play/drba_model.cpp play/engine_build.cpp play/first_run.cpp play/host_facts.cpp
    play/frame_queue.cpp play/frame_stats.cpp play/gmfss_model.cpp play/input.cpp play/motion_model.cpp play/options.cpp play/osd_text.cpp play/pipeline.cpp
    play/placebo_imgui.cpp play/placebo_overlays.cpp play/placebo_ring.cpp play/placebo_screen.cpp
    play/placebo_snapshot.cpp play/player.cpp play/player_engines.cpp play/profiles.cpp play/resume.cpp
    play/ring_stage.cpp play/rife_plan.cpp play/scripted_input.cpp play/session.cpp play/settings.cpp play/settings_panel.cpp play/subtitle_queue.cpp play/tracks.cpp play/ui.cpp
    play/upscale_drba.cpp play/upscale_gmfss.cpp play/upscale_stage.cpp play/wiring.cpp
  DEPS ajp::core ajp::core_log ajp::media ajp::gpu ajp::prepare ajp::audio ajp::subs
       ajp::render ajp::present ajp::export
  PRIVATE_DEPS ajp::dep::ffmpeg ajp::dep::placebo ajp::dep::sdl3 ajp::dep::spdlog ajp::dep::vulkan
               ajp::dep::imgui ajp::dep::toml ${CMAKE_DL_LIBS})
# Police de l'interface : celle du bureau par fontconfig quand il est là, sinon celle d'ImGui.
if(TARGET ajp::dep::fontconfig)
  target_link_libraries(ajp_app PRIVATE ajp::dep::fontconfig)
  target_compile_definitions(ajp_app PRIVATE AJP_WITH_FONTCONFIG=1)
endif()

# ajp : l'exécutable. Les backends TensorRT, MIGraphX, OpenVINO et Vulkan se cherchent à côté de lui, puis dans
# l'arbre de construction quand ils y sont construits (TensorRT : AJP_WITH_CUDA).
add_executable(ajp main.cpp)
target_link_libraries(ajp PRIVATE ajp::app ajp::dep::ffmpeg ajp::dep::sdl3 ajp::dep::spdlog)
if(TARGET ajp_backend_tensorrt)
  target_compile_definitions(ajp PRIVATE AJP_BACKEND_TENSORRT="$<TARGET_FILE:ajp_backend_tensorrt>")
  add_dependencies(ajp ajp_backend_tensorrt)
endif()
foreach(runtime migraphx openvino)
  if(TARGET ajp_backend_${runtime})
    string(TOUPPER "${runtime}" upper)
    target_compile_definitions(ajp PRIVATE AJP_BACKEND_${upper}="$<TARGET_FILE:ajp_backend_${runtime}>")
    add_dependencies(ajp ajp_backend_${runtime})
  endif()
endforeach()
if(TARGET ajp_backend_vulkan)
  target_compile_definitions(ajp PRIVATE AJP_BACKEND_VULKAN="$<TARGET_FILE:ajp_backend_vulkan>")
  add_dependencies(ajp ajp_backend_vulkan)
endif()
ajp_set_warnings(ajp)
