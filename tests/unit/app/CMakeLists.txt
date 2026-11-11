# La session se teste avec des doublures (écran, boîtes, horloge, thread gpu) ; la boucle
# complète avec les vrais threads source, audio (pilote SDL « dummy ») et subs, sur un
# clip généré à l'exécution par l'écrivain de clips des tests de media.
# Test caché [.span_compat] de span_derive_test.cpp : le SPAN d'origine et le paquet (avec son reçu à côté) que
# tools/span/derive_span_fp8.py a écrits ; vides, il se SKIP.
set(AJP_SPAN_FP8_MODEL "" CACHE FILEPATH "SPAN d'origine du paquet (tests cachés du SPAN FP8)")
set(AJP_SPAN_FP8_PYTHON_PACKAGE "" CACHE FILEPATH "Paquet .span8 écrit par derive_span_fp8.py (test caché [.span_compat])")
# Test caché [.span_prepare] : la préparation de bout en bout par le vrai backend TensorRT, sur une vraie vidéo
# (AJP_SPAN_DERIVE_VIDEO) ; le paquet et son reçu s'écrivent dans AJP_SPAN_DERIVE_OUT (sinon un dossier jetable).
set(AJP_SPAN_DERIVE_VIDEO "" CACHE FILEPATH "Vidéo pour le test caché [.span_prepare]")
set(AJP_SPAN_DERIVE_OUT "" CACHE PATH "Dossier des modèles où le test caché [.span_prepare] écrit")
ajp_add_unit_tests(app
  SOURCES live_settings_test.cpp settings_persist_test.cpp chain_stats_test.cpp look_loader_test.cpp present_stats_test.cpp vrr_watch_test.cpp chain_shots_test.cpp chain_setup_test.cpp apple_model_test.cpp episodes_test.cpp gpu_telemetry_test.cpp build_progress_test.cpp chapters_test.cpp controls_test.cpp crash_dump_test.cpp problem_report_test.cpp overlay_hold_test.cpp input_test.cpp home_idle_test.cpp close_watch_test.cpp single_instance_test.cpp pumped_task_test.cpp e2e_play_test.cpp decoder_options_test.cpp default_player_test.cpp display_edid_test.cpp external_subtitles_test.cpp engine_install_test.cpp engine_online_test.cpp engine_versions_test.cpp ed25519_test.cpp updater_test.cpp engine_switch_test.cpp export_job_test.cpp export_sink_test.cpp first_run_test.cpp wizard_test.cpp machine_calibration_test.cpp gpu_tier_test.cpp auto_bench_test.cpp bench_steps_test.cpp span_auto_test.cpp auto_config_test.cpp frame_latency_test.cpp model_choice_test.cpp osd_text_test.cpp motion_model_test.cpp resume_test.cpp settings_test.cpp settings_applied_test.cpp settings_pages_test.cpp settings_history_test.cpp paths_test.cpp auto_watch_test.cpp i18n_test.cpp span_derive_test.cpp span_speed_test.cpp span_int8_test.cpp span_fp8_cutlass_test.cpp rife_fused_choice_test.cpp drba_fused_choice_test.cpp stage_rig.cpp tracks_test.cpp system_language_test.cpp upscale_stage_test.cpp upscale_stage_drba_test.cpp upscale_stage_branches_test.cpp upscale_stage_passes_test.cpp late_fate_stage_test.cpp steady_stage_test.cpp smooth_stage_test.cpp span_warp_stage_test.cpp present_level_test.cpp window_size_test.cpp doubles.cpp options_test.cpp pipeline_test.cpp rife_plan_test.cpp ring_stage_test.cpp ring_watcher_test.cpp ring_interop_test.cpp compose_key_test.cpp session_period_test.cpp session_start_test.cpp session_test.cpp session_refusal_test.cpp stage_fallback_test.cpp sleep_watch_test.cpp session_seek_test.cpp session_watch_test.cpp loop_wait_test.cpp
    anime_shots_test.cpp wayland_stamp_test.cpp look_v2_test.cpp cut_book_test.cpp subtitle_queue_test.cpp sample_demand_test.cpp ui_fonts_test.cpp session_subtitles_test.cpp latest_writer_test.cpp video_rate_test.cpp wiring_test.cpp attached_fonts_test.cpp subtitle_lead_test.cpp edition_plan_test.cpp ../media/clip_writer.cpp bridges_test.cpp engine_space_test.cpp
  DEPS ajp::dep::ffmpeg ajp::dep::sdl3)
target_include_directories(ajp_app_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../media")
if(TARGET ajp_backend_tensorrt)
  target_compile_definitions(ajp_app_tests PRIVATE AJP_BACKEND_TENSORRT="$<TARGET_FILE:ajp_backend_tensorrt>")
  add_dependencies(ajp_app_tests ajp_backend_tensorrt)
else()
  target_compile_definitions(ajp_app_tests PRIVATE AJP_BACKEND_TENSORRT="")
endif()
if(TARGET ajp_backend_vulkan)
  target_compile_definitions(ajp_app_tests PRIVATE AJP_BACKEND_VULKAN="$<TARGET_FILE:ajp_backend_vulkan>")
  add_dependencies(ajp_app_tests ajp_backend_vulkan)
else()
  target_compile_definitions(ajp_app_tests PRIVATE AJP_BACKEND_VULKAN="")
endif()
target_compile_definitions(ajp_app_tests PRIVATE
  AJP_SPAN_DERIVE_VIDEO="${AJP_SPAN_DERIVE_VIDEO}" AJP_SPAN_DERIVE_OUT="${AJP_SPAN_DERIVE_OUT}"
  AJP_SPAN_FP8_MODEL="${AJP_SPAN_FP8_MODEL}" AJP_SPAN_FP8_PYTHON_PACKAGE="${AJP_SPAN_FP8_PYTHON_PACKAGE}")
target_compile_definitions(ajp_app_tests PRIVATE AJP_APP_SOURCE_DIR="${CMAKE_SOURCE_DIR}/src/app")
target_compile_definitions(ajp_app_tests PRIVATE AJP_MEDIA_CLIP_DIR="${CMAKE_CURRENT_BINARY_DIR}/clips")

# De bout en bout (P2.38) : le vrai lecteur, sous Vulkan logiciel et Xvfb (job lavapipe). Le cas Catch est caché
# ([.e2e]) : seul ce test le lance, hors des préréglages filtrés (Windows, macOS sans GPU).
target_compile_definitions(ajp_app_tests PRIVATE AJP_PLAYER_EXE="$<TARGET_FILE:ajp>")
add_dependencies(ajp_app_tests ajp)
add_test(NAME "e2e: lecture de bout en bout et scénarios de recette" COMMAND ajp_app_tests "[e2e]")
set_tests_properties("e2e: lecture de bout en bout et scénarios de recette" PROPERTIES TIMEOUT 400 RUN_SERIAL TRUE)


# L'anneau CUDA vers Vulkan et la dérivation du SPAN FP8 sur la vraie carte, cachés dans leur binaire (B40-02) :
# `ctest -L nvidia` (voir tests/CMakeLists.txt). Sans backend TensorRT construit, il n'y a rien à lancer.
if(TARGET ajp_backend_tensorrt)
  ajp_add_hardware_tests(ajp_app_tests nvidia 1800 ring_interop span_compat span_prepare)
endif()
