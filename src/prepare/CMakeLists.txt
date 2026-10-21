# prepare : construction des moteurs, reçus de qualification. Thread preparer.
# Le lanceur de processus a une implémentation par système ; le reste est portable.
if(WIN32)
  set(ajp_prepare_process process_win.cpp)
else()
  set(ajp_prepare_process process_posix.cpp)
endif()

ajp_add_module(prepare
  SOURCES
    builder.cpp cache.cpp cache_io.cpp identity.cpp key_lock.cpp model_source.cpp
    onnx_blend.cpp onnx_dims.cpp onnx_outputs.cpp onnx_scan.cpp preparer.cpp process_text.cpp ${ajp_prepare_process} receipt.cpp
    receipt_fields.cpp sha256.cpp trtexec_command.cpp trtexec_progress.cpp utf8.cpp
  DEPS ajp::core ajp::gpu
  PRIVATE_DEPS ajp::core_log ajp::dep::toml)
