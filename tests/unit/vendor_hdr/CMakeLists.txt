# Faux SDK des tests des chargeurs : de vraies bibliothèques partagées, sans SONAME pour que
# chaque copie se charge par son chemin. support.cpp les copie dans les faux dossiers de SDK.
set(fakes fake_maxine fake_rtx_video fake_sdk_dep)
foreach(fake IN LISTS fakes)
  add_library(ajp_vendor_hdr_${fake} SHARED fakes/${fake}.cpp)
  set_target_properties(ajp_vendor_hdr_${fake} PROPERTIES NO_SONAME ON WINDOWS_EXPORT_ALL_SYMBOLS ON)
  target_include_directories(ajp_vendor_hdr_${fake} PRIVATE "${PROJECT_SOURCE_DIR}/src")
  ajp_set_warnings(ajp_vendor_hdr_${fake})
endforeach()

ajp_add_unit_tests(vendor_hdr
  SOURCES
    abi_check_test.cpp
    binary_probe_test.cpp
    blockers_test.cpp
    cuda_driver.cpp
    loader_test.cpp
    manifest_test.cpp
    params_test.cpp
    real_sdk.cpp
    real_sdk_test.cpp
    real_true_hdr_bench.cpp
    real_true_hdr_test.cpp
    runtime_test.cpp
    sdk_store_test.cpp
    stage_test.cpp
    support.cpp
    true_hdr_test.cpp
  # core : le câblage des bloqueurs vers resolve(), tel que app le fait ; kernels : la vraie
  # conversion YUV vers RGB (spec/color.h) des entrées des tests [nvidia].
  DEPS ajp::core ajp::kernels ${CMAKE_DL_LIBS})

# Tests [nvidia] sur le vrai SDK : ils SKIP quand ce dossier n'a pas les en-têtes du SDK.
set(AJP_TEST_MAXINE_SDK "/usr/local/VideoFX" CACHE PATH "SDK Maxine VFX installé, lu par les tests [nvidia] de vendor_hdr")

target_compile_definitions(ajp_vendor_hdr_tests PRIVATE
  AJP_FAKE_MAXINE_PATH="$<TARGET_FILE:ajp_vendor_hdr_fake_maxine>"
  AJP_FAKE_RTX_PATH="$<TARGET_FILE:ajp_vendor_hdr_fake_rtx_video>"
  AJP_FAKE_DEP_PATH="$<TARGET_FILE:ajp_vendor_hdr_fake_sdk_dep>"
  AJP_MAXINE_SDK_DIR="${AJP_TEST_MAXINE_SDK}")
add_dependencies(ajp_vendor_hdr_tests
  ajp_vendor_hdr_fake_maxine ajp_vendor_hdr_fake_rtx_video ajp_vendor_hdr_fake_sdk_dep)
