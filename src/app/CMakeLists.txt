# app : boucle SDL3, commandes, état de l'interface. Crée les threads et
# dépend de tous les modules, jamais l'inverse. core_log : app seul installe
# les puits du journal (install_log_sinks), depuis main.
#
# play/ : la tranche verticale « kirari play » (session du thread main, câblage des
# threads source, gpu, audio et subs, horloge audio maîtresse).
kirari_add_module(app
  SOURCES
    play/apple_model.cpp play/auto_bench.cpp play/auto_bench_run.cpp play/machine_calibration.cpp play/settings_bench_results.cpp play/bench_steps.cpp play/span_auto.cpp play/auto_config.cpp play/auto_watch.cpp play/backend_library.cpp play/bench_events.cpp play/pumped_task.cpp play/single_instance.cpp play/bridges.cpp play/subtitle_lead.cpp play/edition_plan.cpp play/build_progress.cpp play/chain.cpp play/chain_hdr.cpp play/chain_shots.cpp play/chain_setup.cpp play/chain_half.cpp play/chain_rife.cpp play/chain_gmfss.cpp play/chain_temporal.cpp play/temporal_fix_model.cpp play/chain_stats.cpp play/look_catalog.cpp play/model_choice.cpp play/anime4k_choice.cpp play/chain_upscaler.cpp play/chapters.cpp play/controls.cpp play/controls_draw.cpp play/controls_icons.cpp play/crash_dump.cpp play/problem_report.cpp play/settings_changes.cpp play/chain_shape.cpp play/default_player.cpp play/data_dirs.cpp play/legacy_dirs.cpp play/derived_model.cpp play/span_derive.cpp play/span_commit.cpp play/span_fp16_engine.cpp play/span_int8.cpp play/display_edid.cpp play/drba_model.cpp play/gmfss_model.cpp play/engine_build.cpp play/episodes.cpp play/engine_downloads.cpp play/engine_install.cpp play/engine_cuda_arch.cpp play/engine_space.cpp play/engine_retry.cpp play/engine_vendor.cpp play/engine_versions.cpp play/engine_online.cpp play/engine_screen.cpp play/engine_switch.cpp play/ed25519.cpp play/engine_tools.cpp play/updater.cpp play/updater_state.cpp play/export_job.cpp play/export_run.cpp play/export_sink.cpp play/first_run.cpp play/first_run_wizard.cpp play/wizard.cpp play/gpu_tier.cpp "${CMAKE_CURRENT_BINARY_DIR}/gpu_tiers_data.cpp" play/home_screen.cpp play/host_choices.cpp play/i18n.cpp play/i18n_fr.cpp play/i18n_fr_settings.cpp play/i18n_fr_image.cpp play/i18n_fr_sound.cpp play/i18n_fr_menu.cpp play/i18n_fr_system.cpp play/i18n_fr_wizard.cpp play/i18n_fr_messages.cpp play/i18n_fr_stats.cpp play/i18n_fr_visible.cpp play/home_view.cpp play/host_facts.cpp play/window_vrr.cpp play/system_peak.cpp play/system_language.cpp play/release_refresh.cpp play/resource_usage.cpp play/display_probe.cpp play/display_follow.cpp play/display_gpu.cpp play/player_probe.cpp play/invalid_copies.cpp play/chain_retreat.cpp play/player_notices.cpp


    play/external_subtitles.cpp play/frame_queue.cpp play/frame_stats.cpp play/gpu_telemetry.cpp play/input.cpp play/motion_model.cpp play/motion_cut.cpp play/options.cpp play/osd_text.cpp play/pipeline.cpp
    play/placebo_export.cpp play/placebo_export_vulkan.cpp play/placebo_grade.cpp play/placebo_imgui.cpp play/placebo_open.cpp play/placebo_cache.cpp play/placebo_present.cpp play/placebo_feedback.cpp play/placebo_wayland.cpp play/wayland_activation.cpp play/wayland_output.cpp play/ring_watcher.cpp play/placebo_overlays.cpp play/placebo_ring.cpp play/placebo_source_grade.cpp play/placebo_offscreen.cpp play/placebo_screen.cpp
    play/placebo_snapshot.cpp play/present_watch.cpp play/player.cpp play/player_engines.cpp play/player_run.cpp play/player_after.cpp play/latest_writer.cpp play/loop_wait.cpp

    play/anime_shots.cpp play/gpu_profile_out.cpp play/display_contract.cpp play/cut_book.cpp play/pair_marks.cpp play/user_keys.cpp play/resume.cpp
    play/present_stats.cpp play/ring_stage.cpp play/drba_fused_choice.cpp play/rife_fused_choice.cpp play/rife_plan.cpp play/scripted_input.cpp play/session.cpp play/session_steps.cpp play/sleep_watch.cpp play/relaunch_target.cpp play/update_thread.cpp play/stage_fallback.cpp play/chain_choices.cpp play/session_subtitles.cpp play/session_osd.cpp play/session_display.cpp play/settings.cpp play/settings_toml.cpp play/interp_retired.cpp play/settings_format.cpp play/settings_history.cpp play/settings_toml_lenient.cpp play/settings_merge.cpp play/panel_tip.cpp play/loading_settings.cpp play/settings_pages.cpp play/settings_motion_tab.cpp play/settings_chain_tabs.cpp play/settings_upscale_level.cpp play/settings_color_tabs.cpp play/settings_engines.cpp play/settings_export.cpp play/settings_languages.cpp play/card_advice.cpp play/panel_size.cpp play/settings_panel.cpp play/settings_sound.cpp play/settings_system.cpp play/subtitle_queue.cpp play/tracks.cpp play/ui.cpp play/ui_fonts.cpp "${CMAKE_CURRENT_BINARY_DIR}/ui_font_data.cpp" play/subtitle_fonts.cpp "${CMAKE_CURRENT_BINARY_DIR}/gandhi_sans_regular.cpp" "${CMAKE_CURRENT_BINARY_DIR}/gandhi_sans_italic.cpp" "${CMAKE_CURRENT_BINARY_DIR}/gandhi_sans_bold.cpp" "${CMAKE_CURRENT_BINARY_DIR}/gandhi_sans_bold_italic.cpp" play/mapped_file.cpp play/relaunch_args.cpp play/launch_data.cpp
    play/upscale_ahead.cpp play/upscale_gmfss.cpp play/upscale_half.cpp play/upscale_drba.cpp play/upscale_early.cpp play/upscale_late.cpp play/upscale_fate.cpp play/upscale_motion.cpp play/upscale_smooth.cpp play/upscale_temporal.cpp play/temporal_window.cpp play/upscale_stage.cpp play/upscale_start.cpp play/upscale_span.cpp play/vulkan_share.cpp play/wiring.cpp play/sleep_notice.cpp play/after_gap.cpp play/player_machine.cpp play/cadence_watch.cpp
  DEPS kirari::core kirari::core_log kirari::core_os kirari::media kirari::gpu kirari::gpu_span_fp8 kirari::gpu_span_fp8_onnx kirari::prepare kirari::audio kirari::subs
       kirari::render kirari::export
  PRIVATE_DEPS kirari::av kirari::dep::ffmpeg kirari::dep::placebo kirari::dep::sdl3 kirari::dep::spdlog kirari::dep::vulkan
               kirari::dep::imgui kirari::dep::toml ${CMAKE_DL_LIBS})
# data/gpu_tiers.toml, embarqué octet pour octet (gpu_tier.h).
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/gpu_tiers_data.cpp"
  COMMAND "${CMAKE_COMMAND}" "-DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/gpu_tiers_data.cpp"
          "-DINPUT=${PROJECT_SOURCE_DIR}/data/gpu_tiers.toml" -DNAME=kGpuTiersToml -DNAMESPACE=kirari::app::play
          "-DHEADER=app/play/gpu_tier.h" -P "${CMAKE_CURRENT_SOURCE_DIR}/play/embed_text.cmake"
  DEPENDS "${PROJECT_SOURCE_DIR}/data/gpu_tiers.toml" "${CMAKE_CURRENT_SOURCE_DIR}/play/embed_text.cmake"
  COMMENT "Embarque data/gpu_tiers.toml"
  VERBATIM)
# La police de l'interface (ui_fonts.h), embarquée octet pour octet.
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/ui_font_data.cpp"
  COMMAND "${CMAKE_COMMAND}" "-DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/ui_font_data.cpp"
          "-DINPUT=${PROJECT_SOURCE_DIR}/third_party/inter/Inter-Regular.ttf" -DNAME=kUiFontTtf
          -DNAMESPACE=kirari::app::play "-DHEADER=app/play/ui_fonts.h" -P "${CMAKE_CURRENT_SOURCE_DIR}/play/embed_text.cmake"
  DEPENDS "${PROJECT_SOURCE_DIR}/third_party/inter/Inter-Regular.ttf" "${CMAKE_CURRENT_SOURCE_DIR}/play/embed_text.cmake"
  COMMENT "Embarque third_party/inter/Inter-Regular.ttf"
  VERBATIM)
# La police des sous-titres (subtitle_fonts.h), ses quatre styles embarqués octet pour octet.
foreach(kirari_style IN ITEMS "regular;Regular" "italic;Italic" "bold;Bold" "bold_italic;BoldItalic")
  list(GET kirari_style 0 kirari_file)
  list(GET kirari_style 1 kirari_name)
  add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/gandhi_sans_${kirari_file}.cpp"
    COMMAND "${CMAKE_COMMAND}" "-DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/gandhi_sans_${kirari_file}.cpp"
            "-DINPUT=${PROJECT_SOURCE_DIR}/third_party/gandhi-sans/GandhiSans-${kirari_name}.otf"
            -DNAME=kGandhiSans${kirari_name} -DNAMESPACE=kirari::app::play "-DHEADER=app/play/subtitle_fonts.h"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/play/embed_text.cmake"
    DEPENDS "${PROJECT_SOURCE_DIR}/third_party/gandhi-sans/GandhiSans-${kirari_name}.otf"
            "${CMAKE_CURRENT_SOURCE_DIR}/play/embed_text.cmake"
    COMMENT "Embarque third_party/gandhi-sans/GandhiSans-${kirari_name}.otf"
    VERBATIM)
endforeach()
# L'EDID des écrans sous Windows (display_edid.cpp) : SetupAPI.
# Le registre et SHChangeNotify de l'inscription comme lecteur par défaut (default_player.cpp) :
# advapi32 et shell32. Les minidumps de plantage (crash_dump.cpp) : dbghelp.
if(WIN32)
  target_link_libraries(kirari_app PRIVATE setupapi dxgi advapi32 shell32 shlwapi winmm user32 dbghelp winhttp ole32)
endif()
# wp_presentation (placebo_wayland.cpp) et xdg_activation_v1 (wayland_activation.cpp, C34-15) : le code
# client des protocoles, produit par wayland-scanner depuis third_party/wayland-protocols/.
if(TARGET kirari::dep::wayland)
  set(kirari_wl_dir "${CMAKE_CURRENT_BINARY_DIR}/wayland")
  set(kirari_wl_code "")
  set(kirari_wl_headers "")
  foreach(kirari_wl_name IN ITEMS presentation-time xdg-activation-v1)
    set(kirari_wl_xml "${PROJECT_SOURCE_DIR}/third_party/wayland-protocols/${kirari_wl_name}.xml")
    add_custom_command(
      OUTPUT "${kirari_wl_dir}/${kirari_wl_name}-client.h" "${kirari_wl_dir}/${kirari_wl_name}-protocol.c"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${kirari_wl_dir}"
      COMMAND "${KIRARI_WAYLAND_SCANNER}" client-header "${kirari_wl_xml}" "${kirari_wl_dir}/${kirari_wl_name}-client.h"
      COMMAND "${KIRARI_WAYLAND_SCANNER}" private-code "${kirari_wl_xml}" "${kirari_wl_dir}/${kirari_wl_name}-protocol.c"
      DEPENDS "${kirari_wl_xml}"
      COMMENT "wayland-scanner : ${kirari_wl_name}"
      VERBATIM)
    list(APPEND kirari_wl_code "${kirari_wl_dir}/${kirari_wl_name}-protocol.c")
    list(APPEND kirari_wl_headers "${kirari_wl_dir}/${kirari_wl_name}-client.h")
  endforeach()
  add_library(kirari_wayland_presentation STATIC ${kirari_wl_code})
  set_target_properties(kirari_wayland_presentation PROPERTIES LINKER_LANGUAGE C POSITION_INDEPENDENT_CODE ON)
  target_include_directories(kirari_wayland_presentation SYSTEM PUBLIC "${kirari_wl_dir}")
  target_link_libraries(kirari_wayland_presentation PUBLIC kirari::dep::wayland)
  target_sources(kirari_app PRIVATE ${kirari_wl_headers})
  target_link_libraries(kirari_app PRIVATE kirari_wayland_presentation)
  target_compile_definitions(kirari_app PRIVATE KIRARI_WITH_WAYLAND_PRESENTATION=1)
endif()
# Police de l'interface : celle du bureau par fontconfig quand il est là, sinon celle d'ImGui.
if(TARGET kirari::dep::fontconfig)
  target_link_libraries(kirari_app PRIVATE kirari::dep::fontconfig)
  target_compile_definitions(kirari_app PRIVATE KIRARI_WITH_FONTCONFIG=1)
endif()

# La version, le dépôt des releases et la clé publique des mises à jour (updater.h) ; la clé se donne à la configuration
# (-DKIRARI_UPDATE_PUBKEY=<64 chiffres hexadécimaux> pour une autre clé). La clé privée correspondante vit hors du dépôt,
# sur le poste de signature (docs/signature-des-releases.md).
set(KIRARI_UPDATE_PUBKEY "c09703350d91e8a6ebdb815bf27ec8b662e301c954aed52568da164258d45088" CACHE STRING "Clé publique Ed25519 (hexadécimal) des mises à jour")
# La clé de secours (P1.14) : un manifeste signé par elle est cru aussi, et peut révoquer l'active (revoked_keys).
# Sa clé privée reste hors ligne, jamais dans la CI (docs/signature-des-releases.md).
set(KIRARI_UPDATE_PUBKEY_BACKUP "5300ff802baa8ad547045edf4a02bf469e5837cc7ae7eee1b39980c876880e2a" CACHE STRING "Clé publique Ed25519 de secours (hexadécimal) des mises à jour")
# Les clés sont des hexadécimaux de 64 chiffres, ramenés en minuscules : le lecteur les compare à celles que les
# manifestes révoquent, écrites en minuscules (B26-04). La clé de secours peut rester vide.
foreach(kirari_key_var KIRARI_UPDATE_PUBKEY KIRARI_UPDATE_PUBKEY_BACKUP)
  string(TOLOWER "${${kirari_key_var}}" ${kirari_key_var})
  if(${kirari_key_var} AND NOT ${kirari_key_var} MATCHES "^[0-9a-f]+$")
    message(FATAL_ERROR "${kirari_key_var} : attendu 64 chiffres hexadécimaux")
  endif()
  string(LENGTH "${${kirari_key_var}}" kirari_key_length)
  if(kirari_key_length AND NOT kirari_key_length EQUAL 64)
    message(FATAL_ERROR "${kirari_key_var} : attendu 64 chiffres hexadécimaux, ${kirari_key_length} reçus")
  endif()
endforeach()
# Le dépôt des releases par défaut vient de tools/release/versions.env, la seule source des workflows et des scripts
# de paquet (A28-05).
file(STRINGS "${PROJECT_SOURCE_DIR}/tools/release/versions.env" kirari_release_repo_line REGEX "^KIRARI_RELEASE_REPO=")
# Un changement de versions.env refait la configuration : sans cela, le binaire garderait l'ancien dépôt (B18-08).
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/tools/release/versions.env")
string(REGEX REPLACE "^KIRARI_RELEASE_REPO=" "" kirari_release_repo_default "${kirari_release_repo_line}")
if(NOT kirari_release_repo_default)
  message(FATAL_ERROR "KIRARI_RELEASE_REPO absent de tools/release/versions.env")
endif()
set(KIRARI_RELEASE_REPO "${kirari_release_repo_default}" CACHE STRING "Dépôt GitHub des releases")
# Le suffixe de préversion de l'étiquette (« -rc.1 »), posé par la CI de release (A26-04) : le binaire dit alors la même
# version que l'étiquette. Vide pour une release finale et hors CI.
set(KIRARI_VERSION_SUFFIX "" CACHE STRING "Suffixe de préversion de la version (-rc.N), tiré de l'étiquette")
if(KIRARI_VERSION_SUFFIX AND NOT KIRARI_VERSION_SUFFIX MATCHES "^-[0-9A-Za-z.]+$")
  message(FATAL_ERROR "KIRARI_VERSION_SUFFIX « ${KIRARI_VERSION_SUFFIX} » : attendu -<lettres, chiffres, points>")
endif()
target_compile_definitions(kirari_app PRIVATE KIRARI_VERSION="${PROJECT_VERSION}${KIRARI_VERSION_SUFFIX}"
                           KIRARI_RELEASE_REPO="${KIRARI_RELEASE_REPO}"
                           KIRARI_UPDATE_PUBKEY="${KIRARI_UPDATE_PUBKEY}"
                           KIRARI_UPDATE_PUBKEY_BACKUP="${KIRARI_UPDATE_PUBKEY_BACKUP}"
                           KIRARI_SYSTEM_LIBS="${KIRARI_SYSTEM_LIBS_TEXT}")

# kirari : l'exécutable. Les backends TensorRT, Core ML et Vulkan se cherchent à côté de lui, puis dans
# l'arbre de construction quand ils y sont construits (TensorRT : KIRARI_WITH_CUDA).
add_executable(kirari main.cpp)
# Windows : une application fenêtrée, sans la console qu'un double clic ouvrait un instant (le lecteur
# semblait s'ouvrir deux fois). main() reste le point d'entrée ; d'un terminal, leave_lone_console() reprend
# sa console.
if(WIN32)
  target_sources(kirari PRIVATE kirari.rc)
  set_target_properties(kirari PROPERTIES WIN32_EXECUTABLE TRUE)
  if(MSVC)
    target_link_options(kirari PRIVATE /ENTRY:mainCRTStartup)
  endif()
endif()
target_link_libraries(kirari PRIVATE kirari::app kirari::dep::ffmpeg kirari::dep::sdl3 kirari::dep::spdlog)
# C18-06 : les chemins des backends dans l'arbre de construction ne servent qu'aux postes de développement ; le
# binaire livré (Release) ne les porte plus, ni ne les essaie en repli.
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(kirari_dev_backend_paths_default OFF)
else()
  set(kirari_dev_backend_paths_default ON)
endif()
option(KIRARI_DEV_BACKEND_PATHS "Chercher aussi les backends dans l'arbre de construction"
       ${kirari_dev_backend_paths_default})
foreach(runtime tensorrt coreml vulkan)
  if(TARGET kirari_backend_${runtime})
    string(TOUPPER "${runtime}" upper)
    if(KIRARI_DEV_BACKEND_PATHS)
      target_compile_definitions(kirari PRIVATE KIRARI_BACKEND_${upper}="$<TARGET_FILE:kirari_backend_${runtime}>")
    endif()
    add_dependencies(kirari kirari_backend_${runtime})
  endif()
endforeach()
kirari_set_warnings(kirari)
