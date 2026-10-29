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
  SOURCES apple_model_test.cpp episodes_test.cpp gpu_telemetry_test.cpp build_progress_test.cpp chapters_test.cpp controls_test.cpp decoder_options_test.cpp default_player_test.cpp display_edid_test.cpp external_subtitles_test.cpp engine_install_test.cpp ed25519_test.cpp updater_test.cpp engine_switch_test.cpp export_job_test.cpp export_sink_test.cpp first_run_test.cpp wizard_test.cpp gpu_tier_test.cpp auto_bench_test.cpp bench_steps_test.cpp span_auto_test.cpp auto_config_test.cpp frame_latency_test.cpp model_choice_test.cpp osd_text_test.cpp motion_model_test.cpp profiles_test.cpp resume_test.cpp settings_test.cpp span_derive_test.cpp span_int8_test.cpp span_fp8_cutlass_test.cpp rife_fused_choice_test.cpp drba_fused_choice_test.cpp gmfss_fused_choice_test.cpp stage_rig.cpp tracks_test.cpp upscale_stage_test.cpp window_size_test.cpp doubles.cpp options_test.cpp pipeline_test.cpp rife_plan_test.cpp ring_stage_test.cpp session_period_test.cpp session_test.cpp subtitle_queue_test.cpp wiring_test.cpp ../media/clip_writer.cpp ../vendor_hdr/support.cpp
  DEPS ajp::vendor_hdr ajp::dep::ffmpeg ajp::dep::sdl3)
# Le faux SDK Maxine des tests vendor_hdr (support.cpp) sert au test d'installation de RTX HDR ;
# ses bibliothèques factices chargeables ne servent pas ici.
target_compile_definitions(ajp_app_tests PRIVATE AJP_FAKE_MAXINE_PATH="" AJP_FAKE_DEP_PATH="")
target_include_directories(ajp_app_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../media")
if(TARGET ajp_backend_tensorrt)
  target_compile_definitions(ajp_app_tests PRIVATE AJP_BACKEND_TENSORRT="$<TARGET_FILE:ajp_backend_tensorrt>")
  add_dependencies(ajp_app_tests ajp_backend_tensorrt)
else()
  target_compile_definitions(ajp_app_tests PRIVATE AJP_BACKEND_TENSORRT="")
endif()
target_compile_definitions(ajp_app_tests PRIVATE
  AJP_SPAN_DERIVE_VIDEO="${AJP_SPAN_DERIVE_VIDEO}" AJP_SPAN_DERIVE_OUT="${AJP_SPAN_DERIVE_OUT}"
  AJP_SPAN_FP8_MODEL="${AJP_SPAN_FP8_MODEL}" AJP_SPAN_FP8_PYTHON_PACKAGE="${AJP_SPAN_FP8_PYTHON_PACKAGE}")
target_compile_definitions(ajp_app_tests PRIVATE AJP_MEDIA_CLIP_DIR="${CMAKE_CURRENT_BINARY_DIR}/clips")
