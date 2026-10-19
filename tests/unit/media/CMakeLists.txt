# Les clips de test sont générés à l'exécution dans le dossier de build :
# aucun fichier binaire n'entre dans le dépôt.
ajp_add_unit_tests(media
  SOURCES
    clip_writer.cpp clips.cpp color_test.cpp decoder_test.cpp demuxer_test.cpp hw_test.cpp
    language_test.cpp packet_test.cpp seek_test.cpp selection_test.cpp source_test.cpp
  DEPS ajp::dep::ffmpeg ajp::dep::vulkan)
target_compile_definitions(ajp_media_tests PRIVATE AJP_MEDIA_CLIP_DIR="${CMAKE_CURRENT_BINARY_DIR}/clips")

