# app : boucle SDL3, commandes, état de l'interface. Crée les threads et
# dépend de tous les modules, jamais l'inverse. core_log : app seul installe
# les puits du journal (install_log_sinks), depuis main.
#
# play/ : la tranche verticale « ajp play » (session du thread main, câblage des
# threads source, gpu, audio et subs, horloge audio maîtresse).
ajp_add_module(app
  SOURCES
    play/apple_model.cpp play/auto_bench.cpp play/auto_bench_run.cpp play/machine_calibration.cpp play/settings_bench_results.cpp play/bench_steps.cpp play/span_auto.cpp play/auto_config.cpp play/auto_watch.cpp play/backend_library.cpp play/bench_events.cpp play/pumped_task.cpp play/single_instance.cpp play/bridges.cpp play/build_progress.cpp play/chain.cpp play/chain_hdr.cpp play/chain_shots.cpp play/chain_setup.cpp play/chain_stats.cpp play/look_catalog.cpp play/model_choice.cpp play/chapters.cpp play/controls.cpp play/controls_draw.cpp play/controls_icons.cpp play/crash_dump.cpp play/problem_report.cpp play/chain_shape.cpp play/default_player.cpp play/data_dirs.cpp play/derived_model.cpp play/span_derive.cpp play/span_int8.cpp play/display_edid.cpp play/drba_model.cpp play/engine_build.cpp play/episodes.cpp play/engine_downloads.cpp play/engine_install.cpp play/engine_versions.cpp play/engine_online.cpp play/engine_screen.cpp play/engine_switch.cpp play/ed25519.cpp play/engine_tools.cpp play/updater.cpp play/export_job.cpp play/export_run.cpp play/export_sink.cpp play/first_run.cpp play/wizard.cpp play/gpu_tier.cpp "${CMAKE_CURRENT_BINARY_DIR}/gpu_tiers_data.cpp" play/home_screen.cpp play/host_choices.cpp play/i18n.cpp play/i18n_fr.cpp play/i18n_fr_settings.cpp play/i18n_fr_image.cpp play/i18n_fr_sound.cpp play/i18n_fr_system.cpp play/i18n_fr_wizard.cpp play/i18n_fr_messages.cpp play/i18n_fr_stats.cpp play/i18n_fr_visible.cpp play/home_view.cpp play/host_facts.cpp play/window_vrr.cpp play/system_peak.cpp play/system_language.cpp
    play/external_subtitles.cpp play/frame_queue.cpp play/frame_stats.cpp play/gpu_telemetry.cpp play/input.cpp play/motion_model.cpp play/options.cpp play/osd_text.cpp play/pipeline.cpp
    play/placebo_export.cpp play/placebo_export_vulkan.cpp play/placebo_grade.cpp play/placebo_imgui.cpp play/placebo_open.cpp play/placebo_present.cpp play/placebo_feedback.cpp play/placebo_wayland.cpp play/ring_watcher.cpp play/placebo_overlays.cpp play/placebo_ring.cpp play/placebo_offscreen.cpp play/placebo_screen.cpp
    play/placebo_snapshot.cpp play/player.cpp play/player_engines.cpp play/player_run.cpp play/latest_writer.cpp play/loop_wait.cpp
    play/anime_shots.cpp play/gpu_profile_out.cpp play/display_contract.cpp play/cut_book.cpp play/user_keys.cpp play/resume.cpp
    play/present_stats.cpp play/ring_stage.cpp play/drba_fused_choice.cpp play/rife_fused_choice.cpp play/rife_plan.cpp play/scripted_input.cpp play/session.cpp play/sleep_watch.cpp play/relaunch_target.cpp play/stage_fallback.cpp play/chain_choices.cpp play/session_subtitles.cpp play/session_osd.cpp play/session_display.cpp play/settings.cpp play/settings_toml.cpp play/settings_toml_lenient.cpp play/settings_merge.cpp play/settings_chain_tabs.cpp play/settings_engines.cpp play/settings_export.cpp play/settings_panel.cpp play/settings_sound.cpp play/settings_system.cpp play/subtitle_queue.cpp play/tracks.cpp play/ui.cpp play/mapped_file.cpp
    play/upscale_ahead.cpp play/upscale_drba.cpp play/upscale_late.cpp play/upscale_motion.cpp play/upscale_smooth.cpp play/upscale_stage.cpp play/vulkan_share.cpp play/wiring.cpp
  DEPS ajp::core ajp::core_log ajp::core_os ajp::media ajp::gpu ajp::gpu_span_fp8 ajp::gpu_span_fp8_onnx ajp::prepare ajp::audio ajp::subs
       ajp::render ajp::export
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
# advapi32 et shell32. Les minidumps de plantage (crash_dump.cpp) : dbghelp.
if(WIN32)
  target_link_libraries(ajp_app PRIVATE setupapi dxgi advapi32 shell32 winmm user32 dbghelp winhttp ole32)
endif()
# wp_presentation (placebo_wayland.cpp) : le code client du protocole, produit par wayland-scanner depuis
# third_party/wayland-protocols/presentation-time.xml.
if(TARGET ajp::dep::wayland)
  set(ajp_wl_xml "${PROJECT_SOURCE_DIR}/third_party/wayland-protocols/presentation-time.xml")
  set(ajp_wl_dir "${CMAKE_CURRENT_BINARY_DIR}/wayland")
  add_custom_command(
    OUTPUT "${ajp_wl_dir}/presentation-time-client.h" "${ajp_wl_dir}/presentation-time-protocol.c"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${ajp_wl_dir}"
    COMMAND "${AJP_WAYLAND_SCANNER}" client-header "${ajp_wl_xml}" "${ajp_wl_dir}/presentation-time-client.h"
    COMMAND "${AJP_WAYLAND_SCANNER}" private-code "${ajp_wl_xml}" "${ajp_wl_dir}/presentation-time-protocol.c"
    DEPENDS "${ajp_wl_xml}"
    COMMENT "wayland-scanner : presentation-time"
    VERBATIM)
  add_library(ajp_wayland_presentation STATIC "${ajp_wl_dir}/presentation-time-protocol.c")
  set_target_properties(ajp_wayland_presentation PROPERTIES LINKER_LANGUAGE C POSITION_INDEPENDENT_CODE ON)
  target_include_directories(ajp_wayland_presentation SYSTEM PUBLIC "${ajp_wl_dir}")
  target_link_libraries(ajp_wayland_presentation PUBLIC ajp::dep::wayland)
  target_sources(ajp_app PRIVATE "${ajp_wl_dir}/presentation-time-client.h")
  target_link_libraries(ajp_app PRIVATE ajp_wayland_presentation)
  target_compile_definitions(ajp_app PRIVATE AJP_WITH_WAYLAND_PRESENTATION=1)
endif()
# Police de l'interface : celle du bureau par fontconfig quand il est là, sinon celle d'ImGui.
if(TARGET ajp::dep::fontconfig)
  target_link_libraries(ajp_app PRIVATE ajp::dep::fontconfig)
  target_compile_definitions(ajp_app PRIVATE AJP_WITH_FONTCONFIG=1)
endif()

# La version, le dépôt des releases et la clé publique des mises à jour (updater.h) ; la clé se donne à la configuration
# (-DAJP_UPDATE_PUBKEY=<64 chiffres hexadécimaux> pour une autre clé). La clé privée correspondante vit hors du dépôt,
# sur le poste de signature (docs/signature-des-releases.md).
set(AJP_UPDATE_PUBKEY "c50eeb3e63c327a087487cd3f485e66fbbf8ef2340a0b90d7ec99d4c2d3f0d58" CACHE STRING "Clé publique Ed25519 (hexadécimal) des mises à jour")
# La clé de secours (P1.14) : un manifeste signé par elle est cru aussi, et peut révoquer l'active (revoked_keys).
# Vide tant qu'elle n'est pas créée (docs/signature-des-releases.md).
set(AJP_UPDATE_PUBKEY_BACKUP "" CACHE STRING "Clé publique Ed25519 de secours (hexadécimal) des mises à jour")
# Le dépôt des releases par défaut vient de tools/release/versions.env, la seule source des workflows et des scripts
# de paquet (A28-05).
file(STRINGS "${PROJECT_SOURCE_DIR}/tools/release/versions.env" ajp_release_repo_line REGEX "^AJP_RELEASE_REPO=")
string(REGEX REPLACE "^AJP_RELEASE_REPO=" "" ajp_release_repo_default "${ajp_release_repo_line}")
if(NOT ajp_release_repo_default)
  message(FATAL_ERROR "AJP_RELEASE_REPO absent de tools/release/versions.env")
endif()
set(AJP_RELEASE_REPO "${ajp_release_repo_default}" CACHE STRING "Dépôt GitHub des releases")
# Le suffixe de préversion de l'étiquette (« -rc.1 »), posé par la CI de release (A26-04) : le binaire dit alors la même
# version que l'étiquette. Vide pour une release finale et hors CI.
set(AJP_VERSION_SUFFIX "" CACHE STRING "Suffixe de préversion de la version (-rc.N), tiré de l'étiquette")
if(AJP_VERSION_SUFFIX AND NOT AJP_VERSION_SUFFIX MATCHES "^-[0-9A-Za-z.]+$")
  message(FATAL_ERROR "AJP_VERSION_SUFFIX « ${AJP_VERSION_SUFFIX} » : attendu -<lettres, chiffres, points>")
endif()
target_compile_definitions(ajp_app PRIVATE AJP_VERSION="${PROJECT_VERSION}${AJP_VERSION_SUFFIX}"
                           AJP_RELEASE_REPO="${AJP_RELEASE_REPO}"
                           AJP_UPDATE_PUBKEY="${AJP_UPDATE_PUBKEY}"
                           AJP_UPDATE_PUBKEY_BACKUP="${AJP_UPDATE_PUBKEY_BACKUP}")

# ajp : l'exécutable. Les backends TensorRT, Core ML et Vulkan se cherchent à côté de lui, puis dans
# l'arbre de construction quand ils y sont construits (TensorRT : AJP_WITH_CUDA).
add_executable(ajp main.cpp)
# Windows : une application fenêtrée, sans la console qu'un double clic ouvrait un instant (le lecteur
# semblait s'ouvrir deux fois). main() reste le point d'entrée ; d'un terminal, leave_lone_console() reprend
# sa console.
if(WIN32)
  set_target_properties(ajp PROPERTIES WIN32_EXECUTABLE TRUE)
  if(MSVC)
    target_link_options(ajp PRIVATE /ENTRY:mainCRTStartup)
  endif()
endif()
target_link_libraries(ajp PRIVATE ajp::app ajp::dep::ffmpeg ajp::dep::sdl3 ajp::dep::spdlog)
if(TARGET ajp_backend_tensorrt)
  target_compile_definitions(ajp PRIVATE AJP_BACKEND_TENSORRT="$<TARGET_FILE:ajp_backend_tensorrt>")
  add_dependencies(ajp ajp_backend_tensorrt)
endif()
foreach(runtime coreml)
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
