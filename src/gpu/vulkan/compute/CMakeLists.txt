# vk_compute : socle de calcul Vulkan. Chargement sans liaison (VK_NO_PROTOTYPES, tables remplies
# depuis un vkGetInstanceProcAddr), contexte adopté ou créé, sous-allocateur borné, pipelines
# depuis le SPIR-V embarqué, enregistrements rejouables sur sémaphore timeline, et les noyaux
# GLSL de src/kernels/glsl/ (KernelRunner). Bibliothèque statique : le moteur Vulkan la liera.
include(AjpSpirv)

ajp_add_module(vk_compute
  SOURCES
    buffer.cpp
    caps.cpp
    context.cpp
    context_create.cpp
    descriptors.cpp
    functions.cpp
    image.cpp
    kernel_dispatch.cpp
    kernel_runner.cpp
    kernel_types.cpp
    kernels_color.cpp
    kernels_window.cpp
    library.cpp
    log.cpp
    memory.cpp
    pipeline.cpp
    queue.cpp
    recording.cpp
    refuse.cpp
    sync.cpp
    weights.cpp
  DEPS ajp::core_types ajp::kernels ajp::dep::vulkan
  PRIVATE_DEPS ajp::dep::spdlog ${CMAKE_DL_LIBS})

# Aucun prototype Vulkan, ici comme chez ceux qui incluent nos en-têtes.
target_compile_definitions(ajp_vk_compute PUBLIC VK_NO_PROTOTYPES)

set(glsl "${PROJECT_SOURCE_DIR}/src/kernels/glsl")
# Les tenseurs des réseaux en float16 : variante « _f16 » des shaders qui en lisent ou écrivent.
set(tensor_shaders
  "${glsl}/ingest_combine.comp"
  "${glsl}/output_matrix.comp"
  "${glsl}/resample_h.comp"
  "${glsl}/resample_v.comp"
  "${glsl}/rgb_window.comp"
  "${glsl}/rgba16_store.comp"
  "${glsl}/rife_constants.comp"
  "${glsl}/transfer.comp"
  "${glsl}/warp_blend.comp")
ajp_add_spirv(ajp_vk_compute
  NAMESPACE ajp::gpu::vk::spirv
  SHADERS
    ${tensor_shaders}
    "${glsl}/block_diff.comp"
    "${glsl}/ingest_chroma_h.comp"
    "${glsl}/output_chroma_v.comp"
    "${glsl}/scene_diff.comp"
    "${glsl}/yuv_window.comp"
  VARIANT f16
  DEFINES AJP_TENSOR_F16
  VARIANT_SHADERS ${tensor_shaders})
