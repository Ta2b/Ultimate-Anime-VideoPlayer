# Exécutable d'aide : faux trtexec, et processus concurrents pour le cache.
add_executable(ajp_prepare_helper helper.cpp)
target_link_libraries(ajp_prepare_helper PRIVATE ajp::prepare)
ajp_set_warnings(ajp_prepare_helper)

ajp_add_unit_tests(prepare
  SOURCES
    builder_test.cpp cache_test.cpp identity_test.cpp onnx_dims_test.cpp onnx_metadata_test.cpp onnx_outputs_test.cpp onnx_scan_test.cpp
    preparer_test.cpp
    process_test.cpp receipt_test.cpp trtexec_command_test.cpp utf8_test.cpp)
target_compile_definitions(ajp_prepare_tests PRIVATE AJP_PREPARE_HELPER="$<TARGET_FILE:ajp_prepare_helper>")
add_dependencies(ajp_prepare_tests ajp_prepare_helper)
