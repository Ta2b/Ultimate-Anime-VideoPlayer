# Le choix de l'encodeur, logique pure, et l'écriture avec un encodeur factice (la file, le thread
# et le mux sur un clip généré) : l'encodage matériel lui-même demande un GPU et n'est pas testé ici.
ajp_add_unit_tests(export
  SOURCES encoder_choice_test.cpp export_writer_test.cpp ../media/clip_writer.cpp
  DEPS ajp::dep::ffmpeg)
target_include_directories(ajp_export_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../media")
target_compile_definitions(ajp_export_tests PRIVATE AJP_MEDIA_CLIP_DIR="${CMAKE_CURRENT_BINARY_DIR}/clips")
