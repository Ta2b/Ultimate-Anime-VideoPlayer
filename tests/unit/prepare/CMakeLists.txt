# Exécutable d'aide : faux trtexec, et processus concurrents pour le cache.
add_executable(kirari_prepare_helper helper.cpp)
target_link_libraries(kirari_prepare_helper PRIVATE kirari::prepare)
kirari_set_warnings(kirari_prepare_helper)

kirari_add_unit_tests(prepare
  SOURCES
    builder_test.cpp cache_purge_test.cpp cache_test.cpp engine_cache_test.cpp drba_split_test.cpp identity_test.cpp onnx_dims_test.cpp onnx_frames_test.cpp onnx_outputs_test.cpp onnx_scan_test.cpp
    gmflow_mlp_test.cpp gmflow_native_test.cpp gmfss_native_test.cpp gmfss_pair_test.cpp onnx_permute_test.cpp rife_native_test.cpp rife_split_test.cpp rife_split_guards_test.cpp tfix_split_test.cpp
    gmflow_attention_test.cpp
    process_test.cpp trtexec_command_test.cpp utf8_test.cpp)
target_compile_definitions(kirari_prepare_tests PRIVATE KIRARI_PREPARE_HELPER="$<TARGET_FILE:kirari_prepare_helper>")
add_dependencies(kirari_prepare_tests kirari_prepare_helper)
# KIRARI_RIFE_MODEL_DIR : le dossier des RIFE du lecteur (rife_v4.25.onnx, rife_v4.22_lite.onnx), lus en place pour
# le test de découpe ; absent, ce test se SKIP.
set(KIRARI_RIFE_MODEL_DIR "${KIRARI_MODELS_DIR}" CACHE PATH "Dossier des RIFE ONNX du lecteur (test de découpe)")
target_compile_definitions(kirari_prepare_tests PRIVATE KIRARI_RIFE_MODEL_DIR="${KIRARI_RIFE_MODEL_DIR}")
# KIRARI_DRBA_MODEL_DIR : le dossier des modèles du lecteur (distil_drba_v2_lite.onnx, distil_drba_v2_lite_s05.onnx), lus en
# place pour le test de découpe ; absent, ce test se SKIP.
set(KIRARI_DRBA_MODEL_DIR "${KIRARI_MODELS_DIR}" CACHE PATH "Dossier des DistilDRBA ONNX du lecteur (test de découpe)")
target_compile_definitions(kirari_prepare_tests PRIVATE KIRARI_DRBA_MODEL_DIR="${KIRARI_DRBA_MODEL_DIR}")
