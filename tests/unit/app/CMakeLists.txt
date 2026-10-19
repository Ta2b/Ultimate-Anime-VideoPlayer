# La session se teste avec des doublures (écran, boîtes, horloge, thread gpu) ; la boucle
# complète avec les vrais threads source, audio (pilote SDL « dummy ») et subs, sur un
# clip généré à l'exécution par l'écrivain de clips des tests de media.
ajp_add_unit_tests(app
  SOURCES controls_test.cpp tracks_test.cpp doubles.cpp options_test.cpp pipeline_test.cpp rife_plan_test.cpp session_test.cpp subtitle_queue_test.cpp ../media/clip_writer.cpp
  DEPS ajp::dep::ffmpeg ajp::dep::sdl3)
target_include_directories(ajp_app_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../media")
target_compile_definitions(ajp_app_tests PRIVATE AJP_MEDIA_CLIP_DIR="${CMAKE_CURRENT_BINARY_DIR}/clips")
