# Faux SDK des tests des chargeurs : de vraies bibliothèques partagées, sans SONAME pour que
# chaque copie se charge par son chemin. support.cpp les copie dans les faux dossiers de SDK.
set(fakes fake_maxine fake_rtx_video fake_sdk_dep)
foreach(fake IN LISTS fakes)
  add_library(ajp_vendor_hdr_${fake} SHARED fakes/${fake}.cpp)
  set_target_properties(ajp_vendor_hdr_${fake} PROPERTIES NO_SONAME ON)
  target_include_directories(ajp_vendor_hdr_${fake} PRIVATE "${PROJECT_SOURCE_DIR}/src")
  ajp_set_warnings(ajp_vendor_hdr_${fake})
endforeach()

ajp_add_unit_tests(vendor_hdr
  SOURCES
    abi_check_test.cpp
    binary_probe_test.cpp
    blockers_test.cpp
    loader_test.cpp
    manifest_test.cpp
    params_test.cpp
    sdk_store_test.cpp
    stage_test.cpp
    support.cpp
    true_hdr_test.cpp)

target_compile_definitions(ajp_vendor_hdr_tests PRIVATE
  AJP_FAKE_MAXINE_PATH="$<TARGET_FILE:ajp_vendor_hdr_fake_maxine>"
  AJP_FAKE_RTX_PATH="$<TARGET_FILE:ajp_vendor_hdr_fake_rtx_video>"
  AJP_FAKE_DEP_PATH="$<TARGET_FILE:ajp_vendor_hdr_fake_sdk_dep>")
add_dependencies(ajp_vendor_hdr_tests
  ajp_vendor_hdr_fake_maxine ajp_vendor_hdr_fake_rtx_video ajp_vendor_hdr_fake_sdk_dep)
