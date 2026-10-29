# app : boucle SDL3, commandes, état de l'interface. Crée les threads et
# dépend de tous les modules, jamais l'inverse. core_log : app seul installe
# les puits du journal (install_log_sinks), depuis main.
#
# play/ : la tranche verticale « ajp play » (session du thread main, câblage des
# threads source, gpu, audio et subs, horloge audio maîtresse).
ajp_add_module(app
  SOURCES
    play/apple_model.cpp play/auto_bench.cpp play/auto_bench_run.cpp play/bench_steps.cpp play/span_auto.cpp play/auto_config.cpp play/backend_library.cpp play/bridges.cpp play/build_progress.cpp play/chain.cpp play/chain_hdr.cpp play/chain_setup.cpp play/chain_stats.cpp play/model_choice.cpp play/chapters.cpp play/controls.cpp play/controls_draw.cpp play/controls_icons.cpp play/default_player.cpp play/derived_model.cpp play/span_derive.cpp play/span_int8.cpp play/display_edid.cpp play/drba_model.cpp play/engine_build.cpp play/episodes.cpp play/engine_downloads.cpp play/engine_install.cpp play/engine_screen.cpp play/engine_switch.cpp play/ed25519.cpp play/engine_tools.cpp play/updater.cpp play/export_job.cpp play/export_run.cpp play/export_sink.cpp play/first_run.cpp play/wizard.cpp play/gpu_tier.cpp "${CMAKE_CURRENT_BINARY_DIR}/gpu_tiers_data.cpp" play/home_screen.cpp play/i18n.cpp play/i18n_fr.cpp play/i18n_fr_settings.cpp play/i18n_fr_image.cpp play/i18n_fr_sound.cpp play/i18n_fr_messages.cpp play/home_view.cpp play/maxine_install.cpp play/host_facts.cpp
    play/external_subtitles.cpp play/frame_queue.cpp play/frame_stats.cpp play/gmfss_model.cpp play/gpu_telemetry.cpp play/input.cpp play/motion_model.cpp play/options.cpp play/osd_text.cpp play/pipeline.cpp
    play/placebo_export.cpp play/placebo_export_vulkan.cpp play/placebo_grade.cpp play/placebo_imgui.cpp play/placebo_open.cpp play/placebo_overlays.cpp play/placebo_ring.cpp play/placebo_screen.cpp
    play/placebo_snapshot.cpp play/player.cpp play/player_engines.cpp play/player_run.cpp play/profiles.cpp play/resume.cpp
    play/ring_stage.cpp play/drba_fused_choice.cpp play/gmfss_fused_choice.cpp play/rife_fused_choice.cpp play/rife_plan.cpp play/scripted_input.cpp play/session.cpp play/settings.cpp play/settings_toml.cpp play/settings_chain_tabs.cpp play/settings_engines.cpp play/settings_export.cpp play/settings_panel.cpp play/settings_sound.cpp play/settings_system.cpp play/subtitle_queue.cpp play/tracks.cpp play/ui.cpp
    play/upscale_ahead.cpp play/upscale_drba.cpp play/upscale_gmfss.cpp play/upscale_motion.cpp play/upscale_stage.cpp play/vulkan_share.cpp play/wiring.cpp
  DEPS ajp::core ajp::core_log ajp::media ajp::gpu ajp::gpu_span_fp8 ajp::gpu_span_fp8_onnx ajp::prepare ajp::audio ajp::subs
       ajp::render ajp::present ajp::export
  PRIVATE_DEPS ajp::dep::ffmpeg ajp::dep::placebo ajp::dep::sdl3 ajp::dep::spdlog ajp::dep::vulkan
               ajp::dep::imgui ajp::dep::toml ${CMAKE_DL_LIBS})
# data/gpu_tiers.toml, embarqué octet pour octet (gpu_tier.h).
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/gpu_tiers_data.cpp"
  COMMAND "${CMAKE_COMMAND}" "-DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/gpu_tiers_data.cpp"
          "-DINPUT=${PROJECT_SOURCE_DIR}/data/gpu_tiers.toml" -DNAME=kGpuTiersToml -DNAMESPACE=ajp::app::play
          "-DHEADER=app/play/gpu_tier.h" -P "${CMAKE_CURRENT_SOURCE_DIR}/play/embed_text.cmake"
  DEPENDS "${PROJECT_SOURCE_DIR}/data/gpu_tiers.toml" "${CMAKE_CURRENT_SOURCE_DIR}/play/embed_text.cmake"
  COMMENT "Embarque data/gpu_tiers.toml"
  VERBATIM)
# L'EDID des écrans sous Windows (display_edid.cpp) : SetupAPI.
# Le registre et SHChangeNotify de l'inscription comme lecteur par défaut (default_player.cpp) :
# advapi32 et shell32.
if(WIN32)
  target_link_libraries(ajp_app PRIVATE setupapi dxgi advapi32 shell32 winmm user32)
endif()
# Police de l'interface : celle du bureau par fontconfig quand il est là, sinon celle d'ImGui.
if(TARGET ajp::dep::fontconfig)
  target_link_libraries(ajp_app PRIVATE ajp::dep::fontconfig)
  target_compile_definitions(ajp_app PRIVATE AJP_WITH_FONTCONFIG=1)
endif()

# La version, le dépôt des releases et la clé publique des mises à jour (updater.h) ; la clé se donne à la configuration
# (-DAJP_UPDATE_PUBKEY=<64 chiffres hexadécimaux> pour une autre clé). La clé privée correspondante vit hors du dépôt,
# sur le PC de Tatoub (docs/signature-des-releases.md).
set(AJP_UPDATE_PUBKEY "c50eeb3e63c327a087487cd3f485e66fbbf8ef2340a0b90d7ec99d4c2d3f0d58" CACHE STRING "Clé publique Ed25519 (hexadécimal) des mises à jour")
set(AJP_RELEASE_REPO "Ta2b/ajp-releases" CACHE STRING "Dépôt GitHub des releases")
target_compile_definitions(ajp_app PRIVATE AJP_VERSION="${PROJECT_VERSION}" AJP_RELEASE_REPO="${AJP_RELEASE_REPO}"
                           AJP_UPDATE_PUBKEY="${AJP_UPDATE_PUBKEY}")

# ajp : l'exécutable. Les backends TensorRT, MIGraphX, OpenVINO, Core ML et Vulkan se cherchent à côté de lui, puis dans
# l'arbre de construction quand ils y sont construits (TensorRT : AJP_WITH_CUDA).
add_executable(ajp main.cpp)
target_link_libraries(ajp PRIVATE ajp::app ajp::dep::ffmpeg ajp::dep::sdl3 ajp::dep::spdlog)
if(TARGET ajp_backend_tensorrt)
  target_compile_definitions(ajp PRIVATE AJP_BACKEND_TENSORRT="$<TARGET_FILE:ajp_backend_tensorrt>")
  add_dependencies(ajp ajp_backend_tensorrt)
endif()
foreach(runtime migraphx openvino coreml)
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
