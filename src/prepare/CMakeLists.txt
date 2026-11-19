# prepare : construction des moteurs (trtexec, cache, découpes ONNX). Le lecteur la lance par EngineBuild
# (app/play/engine_build.h).
# Le lanceur de processus a une implémentation par système ; le reste est portable.
if(WIN32)
  set(kirari_prepare_process process_win.cpp)
else()
  set(kirari_prepare_process process_posix.cpp)
endif()

kirari_add_module(prepare
  SOURCES
    builder.cpp cache.cpp cache_io.cpp cache_purge.cpp identity.cpp key_lock.cpp model_source.cpp
    drba_split.cpp gmflow_native.cpp gmfss_fused.cpp gmfss_native.cpp gmfss_pair_native.cpp gmfss_pair_split.cpp onnx_blend.cpp onnx_cache.cpp onnx_dims.cpp onnx_frames.cpp onnx_outputs.cpp onnx_scan.cpp onnx_split.cpp rife_native.cpp rife_split.cpp tfix_split.cpp process_text.cpp ${kirari_prepare_process}
    sha256.cpp trtexec_command.cpp trtexec_progress.cpp utf8.cpp
    gmflow_attention_check.cpp gmflow_attention_native.cpp gmflow_onnx_view.cpp
  DEPS kirari::core kirari::gpu kirari::kernels
  PRIVATE_DEPS kirari::core_log kirari::dep::toml kirari::gpu_ir)

# Lié aussi dans les bibliothèques chargées à l'exécution (kirari_backend_vulkan : RIFE fusionné).
set_target_properties(kirari_prepare PROPERTIES POSITION_INDEPENDENT_CODE ON)
