# Le backend se charge par dlopen, comme le lecteur le charge : le test ne le lie pas.
# AJP_SPAN_MODEL : le modèle SPAN d'AnimeJaNai installé sur le poste, lu en place (jamais
# copié dans le dépôt) ; absent, le test qui l'exécute se SKIP.
set(AJP_SPAN_MODEL "$ENV{HOME}/.local/share/AnimeJaNai-HDR-linux/animejanai/onnx/2x_AnimeJaNai_HD_V3.1Sharp1_Performance_SPANF3_b5f48_unshuffle_fp16.onnx"
    CACHE FILEPATH "Modèle SPAN ×2 d'AnimeJaNai pour le test du backend TensorRT")
# AJP_SPAN_FP8_PACKAGE, AJP_SPAN_FP8_MODEL : le paquet dérivé par tools/span/derive_span_fp8.py et le SPAN dont il
# vient, pour le test caché [.span_fp8] (docs/span-fp8-cutlass.md) ; vides, il se SKIP.
set(AJP_SPAN_FP8_PACKAGE "" CACHE FILEPATH "Paquet .span8 du SPAN FP8 (test caché [.span_fp8])")
set(AJP_SPAN_FP8_MODEL "" CACHE FILEPATH "SPAN d'origine du paquet (test caché [.span_fp8])")
# AJP_SPAN_DERIVE_IMAGES : des PPM P6 d anime de 1920×1088 au moins, séparés par des virgules, pour le test caché
# [.span_derive] (dérivation native du paquet).
set(AJP_SPAN_DERIVE_IMAGES "" CACHE STRING "Images PPM séparées par des virgules (test caché [.span_derive])")
set(AJP_SPAN_FP8_IMAGE "" CACHE FILEPATH "Image PPM d'anime pour le test caché [.span_fp8] (sinon un dessin synthétique)")
# AJP_RIFE_MODEL_DIR : le dossier des RIFE du lecteur (rife_v4.25.onnx, rife_v4.22_lite.onnx), lus en place par le test
# caché [.rife_fused] ; AJP_RIFE_IMAGE : une image d'anime en PPM pour sa paire (sinon un dessin synthétique).
set(AJP_RIFE_MODEL_DIR "$ENV{HOME}/ajp/models/rife" CACHE PATH "Dossier des RIFE ONNX du lecteur (test caché [.rife_fused])")
set(AJP_RIFE_IMAGE "" CACHE FILEPATH "Image PPM d'anime pour le test caché [.rife_fused]")
# AJP_DRBA_MODEL_DIR : le dossier des DistilDRBA du lecteur (distil_drba_v2_lite.onnx, distil_drba_v2_lite_s05.onnx), lus en
# place par le test caché [.drba_fused] ; AJP_DRBA_IMAGE : une image d'anime en PPM pour ses trois images (sinon un dessin).
set(AJP_DRBA_MODEL_DIR "$ENV{HOME}/ajp/models/rife" CACHE PATH "Dossier des DistilDRBA ONNX du lecteur (test caché [.drba_fused])")
set(AJP_DRBA_IMAGE "" CACHE FILEPATH "Image PPM d'anime pour le test caché [.drba_fused]")
add_executable(ajp_backend_tensorrt_tests backend_fixture.cpp backend_motion_test.cpp backend_test.cpp drba_fused_test.cpp export_test.cpp gmfss_test.cpp
  rife_fused_test.cpp span_derive_test.cpp span_fp8_test.cpp)
# Test caché [.gmfss_split] : la paire GMFSS d'avant (67 canaux), celle d'après (sans Concat) et la fusion, aux
# tailles fixées (1280x768) ; vides, le test se SKIP.
set(AJP_GMFSS_SPLIT_OLD_PAIR "" CACHE FILEPATH "Paire GMFSS d'avant la version 2 des graphes, 1280x768")
set(AJP_GMFSS_SPLIT_NEW_PAIR "" CACHE FILEPATH "Paire GMFSS de la version 2 des graphes, 1280x768")
set(AJP_GMFSS_SPLIT_FUSION "" CACHE FILEPATH "Fusion GMFSS, 1280x768")
target_link_libraries(ajp_backend_tensorrt_tests PRIVATE
  ajp::gpu ajp::gpu_span_fp8 ajp::gpu_span_fp8_onnx ajp::kernels ajp::prepare ajp_backend_tensorrt_kernels ajp::dep::cudart ${CMAKE_DL_LIBS} Catch2::Catch2WithMain)
target_compile_definitions(ajp_backend_tensorrt_tests PRIVATE
  AJP_BACKEND_TENSORRT="$<TARGET_FILE:ajp_backend_tensorrt>"
  AJP_TEST_CACHE_DIR="${CMAKE_CURRENT_BINARY_DIR}/engines"
  AJP_SPAN_MODEL="${AJP_SPAN_MODEL}"
  AJP_SPAN_FP8_PACKAGE="${AJP_SPAN_FP8_PACKAGE}"
  AJP_SPAN_FP8_MODEL="${AJP_SPAN_FP8_MODEL}"
  AJP_SPAN_FP8_IMAGE="${AJP_SPAN_FP8_IMAGE}"
  AJP_RIFE_MODEL_DIR="${AJP_RIFE_MODEL_DIR}"
  AJP_RIFE_IMAGE="${AJP_RIFE_IMAGE}"
  AJP_GMFSS_SPLIT_OLD_PAIR="${AJP_GMFSS_SPLIT_OLD_PAIR}"
  AJP_GMFSS_SPLIT_NEW_PAIR="${AJP_GMFSS_SPLIT_NEW_PAIR}"
  AJP_GMFSS_SPLIT_FUSION="${AJP_GMFSS_SPLIT_FUSION}"
  AJP_DRBA_MODEL_DIR="${AJP_DRBA_MODEL_DIR}"
  AJP_DRBA_IMAGE="${AJP_DRBA_IMAGE}"
  AJP_SPAN_DERIVE_IMAGES="${AJP_SPAN_DERIVE_IMAGES}")
add_dependencies(ajp_backend_tensorrt_tests ajp_backend_tensorrt)
ajp_set_warnings(ajp_backend_tensorrt_tests)
catch_discover_tests(ajp_backend_tensorrt_tests TEST_PREFIX "backend_tensorrt: " DISCOVERY_MODE PRE_TEST)
