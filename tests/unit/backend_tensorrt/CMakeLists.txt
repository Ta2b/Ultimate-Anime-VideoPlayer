# Le backend se charge par dlopen, comme le lecteur le charge : le test ne le lie pas.
# AJP_SPAN_MODEL : le modèle SPAN d'AnimeJaNai installé sur le poste, lu en place (jamais
# copié dans le dépôt) ; absent, le test qui l'exécute se SKIP.
set(AJP_SPAN_MODEL "$ENV{HOME}/.local/share/AnimeJaNai-HDR-linux/animejanai/onnx/2x_AnimeJaNai_HD_V3.1Sharp1_Performance_SPANF3_b5f48_unshuffle_fp16.onnx"
    CACHE FILEPATH "Modèle SPAN ×2 d'AnimeJaNai pour le test du backend TensorRT")
# Test caché [.gmfss_split] : la paire GMFSS d'avant (67 canaux), celle d'après (sans Concat) et la fusion, aux
# tailles fixées (1280x768) ; vides, le test se SKIP.
set(AJP_GMFSS_SPLIT_OLD_PAIR "" CACHE FILEPATH "Paire GMFSS d'avant la version 2 des graphes, 1280x768")
set(AJP_GMFSS_SPLIT_NEW_PAIR "" CACHE FILEPATH "Paire GMFSS de la version 2 des graphes, 1280x768")
set(AJP_GMFSS_SPLIT_FUSION "" CACHE FILEPATH "Fusion GMFSS, 1280x768")
add_executable(ajp_backend_tensorrt_tests backend_fixture.cpp backend_motion_test.cpp backend_test.cpp export_test.cpp gmfss_test.cpp)
target_link_libraries(ajp_backend_tensorrt_tests PRIVATE
  ajp::gpu ajp::kernels ajp::prepare ajp::dep::cudart ${CMAKE_DL_LIBS} Catch2::Catch2WithMain)
target_compile_definitions(ajp_backend_tensorrt_tests PRIVATE
  AJP_BACKEND_TENSORRT="$<TARGET_FILE:ajp_backend_tensorrt>"
  AJP_TEST_CACHE_DIR="${CMAKE_CURRENT_BINARY_DIR}/engines"
  AJP_SPAN_MODEL="${AJP_SPAN_MODEL}"
  AJP_GMFSS_SPLIT_OLD_PAIR="${AJP_GMFSS_SPLIT_OLD_PAIR}"
  AJP_GMFSS_SPLIT_NEW_PAIR="${AJP_GMFSS_SPLIT_NEW_PAIR}"
  AJP_GMFSS_SPLIT_FUSION="${AJP_GMFSS_SPLIT_FUSION}")
add_dependencies(ajp_backend_tensorrt_tests ajp_backend_tensorrt)
ajp_set_warnings(ajp_backend_tensorrt_tests)
catch_discover_tests(ajp_backend_tensorrt_tests TEST_PREFIX "backend_tensorrt: " DISCOVERY_MODE PRE_TEST)
