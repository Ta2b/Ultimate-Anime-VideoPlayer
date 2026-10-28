# Les données (sinus encodés, trames synthétiques) se fabriquent dans les tests ;
# la sortie SDL tourne sur le pilote « dummy », choisi dans le test.
ajp_add_unit_tests(audio
  SOURCES clock_test.cpp converter_test.cpp decoder_test.cpp drift_test.cpp output_test.cpp hrir_test.cpp mixer_test.cpp pcm_test.cpp
          player_test.cpp sine.cpp sync_test.cpp thread_test.cpp
  DEPS ajp::dep::ffmpeg ajp::dep::sdl3)
