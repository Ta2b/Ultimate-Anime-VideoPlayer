# Les fichiers source et les sorties des tests se créent à l'exécution dans le dossier de
# build : aucun fichier binaire n'entre dans le dépôt.
ajp_add_unit_tests(export
  SOURCES
    export_test.cpp framemd5_thread_test.cpp hardware_test.cpp hdr10_progress_test.cpp refusal_test.cpp
    settings_test.cpp support.cpp
  DEPS ajp::dep::ffmpeg)
target_compile_definitions(ajp_export_tests PRIVATE AJP_EXPORT_WORK_DIR="${CMAKE_CURRENT_BINARY_DIR}/work")
