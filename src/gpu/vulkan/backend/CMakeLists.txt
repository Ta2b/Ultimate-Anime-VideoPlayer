# vk_backend : le backend Vulkan (gpu/backend.h, ABI 2) et ses noyaux de la chaîne
# (gpu/kernel_ops.h) sur le socle vk_compute, plus le moteur de réseaux : abaissement des graphes
# gpu_ir vers les shaders de ../shaders, artefacts, autotuner. Bibliothèque statique, testée
# directement ; kirari_backend_vulkan l'enveloppe en bibliothèque chargée à l'exécution.
include(KirariSpirv)

kirari_add_module(vk_backend
  SOURCES
    deferred_steps.cpp
    drba_fused_engine.cpp
    device_profile.cpp
    family.cpp
    host_tensor.cpp
    kernel_frames.cpp
    kernel_image.cpp
    kernel_ops.cpp
    kernel_ring.cpp
    kernel_slice.cpp
    kernel_stats.cpp
    net_alias.cpp
    net_artifact.cpp
    net_autotune.cpp
    net_presets.cpp
    net_build.cpp
    net_gate.cpp
    net_engine.cpp
    net_hoist.cpp
    net_lower.cpp
    net_conv_low_bit.cpp
    net_conv_m8.cpp
    net_lower_conv.cpp
    net_lower_copy.cpp
    net_measure.cpp
    net_params.cpp
    net_profile.cpp
    net_query.cpp
    net_runtime.cpp
    net_span8.cpp
    net_weights.cpp
    queue.cpp
    reduce_weights.cpp
    rife_fused_engine.cpp
    stage_clock.cpp
    vulkan_backend.cpp
    vulkan_backend_runtime.cpp
    vulkan_drba_fused.cpp
    vulkan_rife_fused.cpp
    vulkan_span8.cpp
    vulkan_span_derive.cpp
  DEPS kirari::vk_compute kirari::gpu kirari::gpu_ir kirari::core_types
  PRIVATE_DEPS kirari::core kirari::kernels kirari::prepare kirari::gpu_span_fp8_onnx kirari::dep::spdlog)

set(shaders "${PROJECT_SOURCE_DIR}/src/gpu/vulkan/shaders")
# Une variante « _f16 » (tenseurs float16 permis) de chaque shader de réseau.
set(net_shaders
  "${shaders}/net_avg_pool.comp"
  "${shaders}/net_binary.comp"
  "${shaders}/net_conv.comp"
  "${shaders}/net_conv_reduce.comp"
  "${shaders}/net_conv_transpose.comp"
  "${shaders}/net_copy.comp"
  "${shaders}/net_depth_space.comp"
  "${shaders}/net_grid_sample.comp"
  "${shaders}/net_pad.comp"
  "${shaders}/net_reduce_mean.comp"
  "${shaders}/net_resize.comp"
  "${shaders}/net_unary.comp"
  "${shaders}/ring_load.comp"
  "${shaders}/ring_store.comp")
# Le noyau à matrices coopératives FP8 exige GL_EXT_float_e4m3 (glslang 15.2 ou plus récent) : il
# n'entre dans la construction que si le compilateur le connaît ; sans lui, la variante n'est pas
# proposée (net_conv_low_bit.cpp, low_bit_eligible).
set(fp8_probe "${CMAKE_CURRENT_BINARY_DIR}/fp8_probe.comp")
file(WRITE "${fp8_probe}" "#version 450
#extension GL_EXT_float_e4m3 : require
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
layout(local_size_x = 32) in;
layout(std430, binding = 0) buffer B { uvec4 v[]; } b;
void main() {
  coopmat<floate4m3_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> a;
  coopMatLoad(a, b.v, 0, 1, gl_CooperativeMatrixLayoutRowMajor);
}
")
execute_process(COMMAND "${KIRARI_GLSLC}" --target-env=${KIRARI_SPIRV_TARGET_ENV} -o "${CMAKE_CURRENT_BINARY_DIR}/fp8_probe.spv"
                "${fp8_probe}" RESULT_VARIABLE fp8_probe_result OUTPUT_QUIET ERROR_QUIET)
set(fp8_shaders "")
if(fp8_probe_result EQUAL 0)
  set(fp8_shaders "${shaders}/net_conv_coopmat_fp8_f16.comp")
else()
  message(STATUS "glslc sans GL_EXT_float_e4m3 : noyau Vulkan FP8 non construit")
endif()
kirari_add_spirv(kirari_vk_backend
  NAMESPACE kirari::gpu::vk::net::spirv
  SHADERS
    ${net_shaders}
    ${fp8_shaders}
    "${shaders}/net_conv_f16p.comp"
    "${shaders}/net_conv_coopmat_f16.comp"
    "${shaders}/net_conv_coopmat_acc16.comp"
    "${shaders}/net_conv_coopmat_m8_f16.comp"
    "${shaders}/net_conv_coopmat_int8_f16.comp"
    "${shaders}/net_conv_wino_f16.comp"
    "${shaders}/net_conv_wino_f16p.comp"
  VARIANT f16
  DEFINES KIRARI_NET_F16
  VARIANT_SHADERS ${net_shaders})

# La bibliothèque chargée à l'exécution lie ces bibliothèques statiques : code indépendant de
# la position.
set_target_properties(kirari_vk_backend kirari_vk_compute kirari_gpu_ir kirari_kernels kirari_core_log kirari_core_os kirari_core
  kirari_gpu_span_fp8 kirari_gpu_span_fp8_onnx PROPERTIES POSITION_INDEPENDENT_CODE ON)

# La bibliothèque chargée à l'exécution : kirari_backend_vulkan.so (kirari_backend_<nom>, sans
# préfixe lib), qui n'exporte que les points d'entrée de gpu/backend.h. Les bibliothèques
# statiques liées (spdlog, vk_compute, gpu_ir…) restent privées à la bibliothèque.
add_library(kirari_backend_vulkan MODULE entry.cpp)
target_link_libraries(kirari_backend_vulkan PRIVATE kirari::vk_backend kirari::core_log kirari::dep::spdlog)
set_target_properties(kirari_backend_vulkan PROPERTIES
  PREFIX ""
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_options(kirari_backend_vulkan PRIVATE "LINKER:--exclude-libs,ALL" "LINKER:--no-undefined")
endif()
kirari_set_warnings(kirari_backend_vulkan)

# Inspection d'un modèle : opérateurs, formes et mémoire après les passes ; --vulkan décrit le graphe que
# le moteur Vulkan abaisse (prepare_graph), d'où sa place ici plutôt que dans gpu_ir (A44-10).
add_executable(kirari_onnx_info onnx_info_main.cpp)
target_link_libraries(kirari_onnx_info PRIVATE kirari::vk_backend kirari::gpu_ir kirari::kernels)
kirari_set_warnings(kirari_onnx_info)
