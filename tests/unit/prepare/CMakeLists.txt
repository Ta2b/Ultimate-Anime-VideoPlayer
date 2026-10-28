# Exécutable d'aide : faux trtexec, et processus concurrents pour le cache.
add_executable(ajp_prepare_helper helper.cpp)
target_link_libraries(ajp_prepare_helper PRIVATE ajp::prepare)
ajp_set_warnings(ajp_prepare_helper)

ajp_add_unit_tests(prepare
  SOURCES
    builder_test.cpp cache_test.cpp drba_split_test.cpp identity_test.cpp onnx_dims_test.cpp onnx_metadata_test.cpp onnx_outputs_test.cpp onnx_scan_test.cpp
    preparer_test.cpp rife_split_test.cpp
    process_test.cpp receipt_test.cpp trtexec_command_test.cpp utf8_test.cpp)
target_compile_definitions(ajp_prepare_tests PRIVATE AJP_PREPARE_HELPER="$<TARGET_FILE:ajp_prepare_helper>")
add_dependencies(ajp_prepare_tests ajp_prepare_helper)
# AJP_RIFE_MODEL_DIR : le dossier des RIFE du lecteur (rife_v4.25.onnx, rife_v4.22_lite.onnx), lus en place pour
# le test de découpe ; absent, ce test se SKIP.
set(AJP_RIFE_MODEL_DIR "$ENV{HOME}/ajp/models/rife" CACHE PATH "Dossier des RIFE ONNX du lecteur (test de découpe)")
target_compile_definitions(ajp_prepare_tests PRIVATE AJP_RIFE_MODEL_DIR="${AJP_RIFE_MODEL_DIR}")
# AJP_DRBA_MODEL_DIR : le dossier des modèles du lecteur (distil_drba_v2_lite.onnx, distil_drba_v2_lite_s05.onnx), lus en
# place pour le test de découpe ; absent, ce test se SKIP.
set(AJP_DRBA_MODEL_DIR "$ENV{HOME}/ajp/models/rife" CACHE PATH "Dossier des DistilDRBA ONNX du lecteur (test de découpe)")
target_compile_definitions(ajp_prepare_tests PRIVATE AJP_DRBA_MODEL_DIR="${AJP_DRBA_MODEL_DIR}")
