# cuda_trt : runtime TensorRT du backend NVIDIA. Désérialise les moteurs que
# prepare construit avec trtexec, et les exécute par enqueueV3 sur un flux
# CUDA non bloquant. Seulement avec AJP_WITH_CUDA.
ajp_add_module(cuda_trt
  SOURCES
    cuda_objects.cpp engine.cpp engine_io.cpp log.cpp runtime.cpp trt_log.cpp types.cpp
  DEPS ajp::core_types ajp::dep::cudart
  PRIVATE_DEPS ajp::dep::tensorrt ajp::dep::spdlog)

# Banc d'un moteur construit : temps GPU par exécution, médiane et écart type.
add_executable(ajp_trt_bench bench_main.cpp)
target_link_libraries(ajp_trt_bench PRIVATE ajp::cuda_trt ajp::dep::spdlog)
ajp_set_warnings(ajp_trt_bench)

# Construction d'un moteur réel par le constructeur trtexec de prepare, vérifié
# par désérialisation : la chaîne complète d'un modèle livré, hors lecteur.
add_executable(ajp_trt_build build_main.cpp)
target_link_libraries(ajp_trt_build PRIVATE ajp::cuda_trt ajp::prepare ajp::dep::spdlog)
ajp_set_warnings(ajp_trt_build)
