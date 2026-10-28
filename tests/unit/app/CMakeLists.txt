# La session se teste avec des doublures (écran, boîtes, horloge, thread gpu) ; la boucle
# complète avec les vrais threads source, audio (pilote SDL « dummy ») et subs, sur un
# clip généré à l'exécution par l'écrivain de clips des tests de media.
ajp_add_unit_tests(app
  SOURCES apple_model_test.cpp episodes_test.cpp gpu_telemetry_test.cpp build_progress_test.cpp chapters_test.cpp controls_test.cpp decoder_options_test.cpp default_player_test.cpp display_edid_test.cpp external_subtitles_test.cpp engine_install_test.cpp engine_switch_test.cpp export_job_test.cpp export_sink_test.cpp first_run_test.cpp gpu_tier_test.cpp frame_latency_test.cpp model_choice_test.cpp osd_text_test.cpp motion_model_test.cpp profiles_test.cpp resume_test.cpp settings_test.cpp span_int8_test.cpp stage_rig.cpp tracks_test.cpp upscale_stage_test.cpp window_size_test.cpp doubles.cpp options_test.cpp pipeline_test.cpp rife_plan_test.cpp ring_stage_test.cpp session_period_test.cpp session_test.cpp subtitle_queue_test.cpp wiring_test.cpp ../media/clip_writer.cpp ../vendor_hdr/support.cpp
  DEPS ajp::vendor_hdr ajp::dep::ffmpeg ajp::dep::sdl3)
# Le faux SDK Maxine des tests vendor_hdr (support.cpp) sert au test d'installation de RTX HDR ;
# ses bibliothèques factices chargeables ne servent pas ici.
target_compile_definitions(ajp_app_tests PRIVATE AJP_FAKE_MAXINE_PATH="" AJP_FAKE_DEP_PATH="")
target_include_directories(ajp_app_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../media")
target_compile_definitions(ajp_app_tests PRIVATE AJP_MEDIA_CLIP_DIR="${CMAKE_CURRENT_BINARY_DIR}/clips")
