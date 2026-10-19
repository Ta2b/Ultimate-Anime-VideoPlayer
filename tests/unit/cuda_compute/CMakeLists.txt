# Tests du socle CUDA : chaque noyau contre la référence CPU de kernels/ref sur l'appareil, et
# l'import d'images et de sémaphores Vulkan. Tous [cuda] : sans GPU NVIDIA, ils se SKIP avec la
# raison. Sur ce PC : ajp-gpu ctest --preset linux-prefix-cuda -R cuda_compute.
ajp_add_unit_tests(cuda_compute
  SOURCES
    bench_test.cpp
    context_test.cpp
    detect_test.cpp
    resize_test.cpp
    rife_test.cpp
    rig.cpp
    transfer_test.cpp
    warp_test.cpp
    yuv_test.cpp
  DEPS ajp::kernels)

# Interop Vulkan : descripteurs de fichier opaques et chargeur ouvert par dlopen, Linux seulement
# pour l'instant (les HANDLE Win32 viendront avec la construction Windows).
if(UNIX)
  target_sources(ajp_cuda_compute_tests PRIVATE interop_test.cpp vulkan_rig.cpp)
  target_link_libraries(ajp_cuda_compute_tests PRIVATE ajp::dep::vulkan ${CMAKE_DL_LIBS})
endif()
