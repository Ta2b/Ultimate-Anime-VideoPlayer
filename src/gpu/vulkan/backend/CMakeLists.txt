# vk_backend : le backend Vulkan (gpu/backend.h, ABI 2) et ses noyaux de la chaîne
# (gpu/kernel_ops.h) sur le socle vk_compute, plus le moteur de réseaux : abaissement des graphes
# gpu_ir vers les shaders de ../shaders, artefacts, autotuner. Bibliothèque statique, testée
# directement ; ajp_backend_vulkan l'enveloppe en bibliothèque chargée à l'exécution.
include(AjpSpirv)

ajp_add_module(vk_backend
  SOURCES
    deferred_steps.cpp
    device_profile.cpp
    family.cpp
    host_tensor.cpp
    kernel_frames.cpp
    kernel_image.cpp
    kernel_ops.cpp
    kernel_pair.cpp
    kernel_ring.cpp
    kernel_slice.cpp
    net_alias.cpp
    net_artifact.cpp
    net_autotune.cpp
    net_presets.cpp
    net_build.cpp
    net_engine.cpp
    net_lower.cpp
    net_lower_conv.cpp
    net_lower_copy.cpp
    net_measure.cpp
    net_params.cpp
    net_runtime.cpp
    net_weights.cpp
    queue.cpp
    reduce_weights.cpp
    shared_tensor.cpp
    vulkan_backend.cpp
    vulkan_backend_runtime.cpp
  DEPS ajp::vk_compute ajp::gpu_ir ajp::core_types
  PRIVATE_DEPS ajp::kernels ajp::dep::spdlog)

set(shaders "${PROJECT_SOURCE_DIR}/src/gpu/vulkan/shaders")
# Une variante « _f16 » (tenseurs float16 permis) de chaque shader de réseau.
set(net_shaders
  "${shaders}/net_avg_pool.comp"
  "${shaders}/net_binary.comp"
  "${shaders}/net_conv.comp"
  "${shaders}/net_conv_transpose.comp"
  "${shaders}/net_copy.comp"
  "${shaders}/net_depth_space.comp"
  "${shaders}/net_grid_sample.comp"
  "${shaders}/net_pad.comp"
  "${shaders}/net_reduce_mean.comp"
  "${shaders}/net_resize.comp"
  "${shaders}/net_unary.comp"
  "${shaders}/ring_store.comp")
# Le noyau à matrices coopératives FP8 exige GL_EXT_float_e4m3 (glslang 15.2 ou plus récent) : il
# n'entre dans la construction que si le compilateur le connaît ; sans lui, la variante n'est pas
# proposée (net_lower_conv.cpp, fp8_eligible).
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
execute_process(COMMAND "${AJP_GLSLC}" --target-env=${AJP_SPIRV_TARGET_ENV} -o "${CMAKE_CURRENT_BINARY_DIR}/fp8_probe.spv"
                "${fp8_probe}" RESULT_VARIABLE fp8_probe_result OUTPUT_QUIET ERROR_QUIET)
set(fp8_shaders "")
if(fp8_probe_result EQUAL 0)
  set(fp8_shaders "${shaders}/net_conv_coopmat_fp8_f16.comp")
else()
  message(STATUS "glslc sans GL_EXT_float_e4m3 : noyau Vulkan FP8 non construit")
endif()
ajp_add_spirv(ajp_vk_backend
  NAMESPACE ajp::gpu::vk::net::spirv
  SHADERS
    ${net_shaders}
    ${fp8_shaders}
    "${shaders}/net_conv_f16p.comp"
    "${shaders}/net_conv_coopmat_f16.comp"
    "${shaders}/net_conv_coopmat_acc16.comp"
    "${shaders}/net_conv_wino_f16.comp"
    "${shaders}/net_conv_wino_f16p.comp"
  VARIANT f16
  DEFINES AJP_NET_F16
  VARIANT_SHADERS ${net_shaders})

# La bibliothèque chargée à l'exécution lie ces bibliothèques statiques : code indépendant de
# la position.
set_target_properties(ajp_vk_backend ajp_vk_compute ajp_gpu_ir ajp_kernels ajp_core_log
  PROPERTIES POSITION_INDEPENDENT_CODE ON)

# La bibliothèque chargée à l'exécution : ajp_backend_vulkan.so (ajp_backend_<nom>, sans
# préfixe lib), qui n'exporte que les points d'entrée de gpu/backend.h. Les bibliothèques
# statiques liées (spdlog, vk_compute, gpu_ir…) restent privées à la bibliothèque.
add_library(ajp_backend_vulkan MODULE entry.cpp)
target_link_libraries(ajp_backend_vulkan PRIVATE ajp::vk_backend ajp::core_log ajp::dep::spdlog)
set_target_properties(ajp_backend_vulkan PROPERTIES
  PREFIX ""
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_options(ajp_backend_vulkan PRIVATE "LINKER:--exclude-libs,ALL" "LINKER:--no-undefined")
endif()
ajp_set_warnings(ajp_backend_vulkan)
