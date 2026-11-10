# prepare : construction des moteurs (trtexec, cache, découpes ONNX). Le lecteur la lance par EngineBuild
# (app/play/engine_build.h).
# Le lanceur de processus a une implémentation par système ; le reste est portable.
if(WIN32)
  set(ajp_prepare_process process_win.cpp)
else()
  set(ajp_prepare_process process_posix.cpp)
endif()

ajp_add_module(prepare
  SOURCES
    builder.cpp cache.cpp cache_io.cpp cache_purge.cpp identity.cpp key_lock.cpp model_source.cpp
    drba_split.cpp onnx_blend.cpp onnx_dims.cpp onnx_outputs.cpp onnx_scan.cpp onnx_split.cpp rife_split.cpp process_text.cpp ${ajp_prepare_process}
    sha256.cpp trtexec_command.cpp trtexec_progress.cpp utf8.cpp
  DEPS ajp::core ajp::gpu ajp::kernels
  PRIVATE_DEPS ajp::core_log ajp::dep::toml ajp::gpu_ir)

# Lié aussi dans les bibliothèques chargées à l'exécution (ajp_backend_vulkan : RIFE fusionné).
set_target_properties(ajp_prepare PROPERTIES POSITION_INDEPENDENT_CODE ON)
