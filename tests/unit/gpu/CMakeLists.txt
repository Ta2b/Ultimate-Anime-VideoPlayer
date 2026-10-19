# gpu : contrats des backends, ordonnanceur des niveaux, ring, Program, exécution de bout en
# bout et thread, sur le backend CPU de référence (src/gpu/cpu), sans GPU.
ajp_add_unit_tests(gpu
  SOURCES
    backend_test.cpp cpu_backend_test.cpp executor_test.cpp gpu_test_support.cpp gpu_thread_test.cpp
    level_scheduler_test.cpp loader_test.cpp output_ring_test.cpp program_test.cpp worker_test.cpp
  DEPS ajp::media ajp::backend_cpu_impl ajp::gpu_ir)
# gpu_ir/onnx_builder.h : les modèles ONNX des tests naissent dans le test.
target_include_directories(ajp_gpu_tests PRIVATE "${PROJECT_SOURCE_DIR}/tests/unit")

# Backends factices du chargeur : ABI ancienne, autre empreinte, symbole absent, appareil refusé.
set(ajp_fake_backends "${CMAKE_CURRENT_BINARY_DIR}/backends")
foreach(fake oldabi badlayout nocreate refuses)
  add_library(ajp_backend_${fake} MODULE fake_backend.cpp)
  target_link_libraries(ajp_backend_${fake} PRIVATE ajp::core_types)
  set_target_properties(ajp_backend_${fake} PROPERTIES PREFIX "" LIBRARY_OUTPUT_DIRECTORY "${ajp_fake_backends}")
  ajp_set_warnings(ajp_backend_${fake})
  add_dependencies(ajp_gpu_tests ajp_backend_${fake})
endforeach()
target_compile_definitions(ajp_backend_oldabi PRIVATE AJP_FAKE_OLD_ABI)
target_compile_definitions(ajp_backend_badlayout PRIVATE AJP_FAKE_BAD_LAYOUT)
target_compile_definitions(ajp_backend_nocreate PRIVATE AJP_FAKE_NO_CREATE)

add_dependencies(ajp_gpu_tests ajp_backend_cpu)
target_compile_definitions(ajp_gpu_tests PRIVATE
  AJP_TEST_CPU_BACKEND_DIR="$<TARGET_FILE_DIR:ajp_backend_cpu>"
  AJP_TEST_FAKE_BACKEND_DIR="${ajp_fake_backends}")
