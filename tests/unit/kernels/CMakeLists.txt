ajp_add_unit_tests(kernels
  SOURCES
    color_test.cpp
    detect_test.cpp
    dither_test.cpp
    distil_drba_test.cpp
    glsl_test.cpp
    gmfss_test.cpp
    half_test.cpp
    parity_test.cpp
    resample_test.cpp
    rife_test.cpp
    tolerance_test.cpp
    transfer_test.cpp
    warp_test.cpp)

# Les tests évaluent les formules constexpr de spec.h avec les arrondis de la référence :
# ajp::kernels leur transmet -ffp-contract=off. Les ports FMA simulés appellent std::fma.

# L'en-tête GLSL généré se compile avec glslc, trouvé quand le lecteur est construit.
if(AJP_GLSLC)
  add_test(NAME "kernels: l'en-tête GLSL de spec.h se compile avec glslc"
    COMMAND "${AJP_GLSLC}" -fshader-stage=compute -I "${AJP_KERNELS_GLSL_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/spec_check.comp" -o "${CMAKE_CURRENT_BINARY_DIR}/spec_check.spv")
endif()
