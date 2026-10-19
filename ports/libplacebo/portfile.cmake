# libplacebo v7.360.1, la version du fork. Les sous-modules dont la
# construction a besoin (jinja, markupsafe, fast_float) sont épinglés aux
# commits que référence ce tag ; glad et Vulkan-Headers ne servent pas ici
# (OpenGL coupé, registre Vulkan fourni par vulkan-headers).
vcpkg_from_git(
  OUT_SOURCE_PATH SOURCE_PATH
  URL https://github.com/haasn/libplacebo.git
  REF cee9b076f2c63104ccfd497fa79c39a867293ec4
  FETCH_REF v7.360.1
)

function(ajp_fetch_submodule dir url ref)
  vcpkg_from_git(OUT_SOURCE_PATH sub_path URL "${url}" REF "${ref}")
  file(REMOVE_RECURSE "${SOURCE_PATH}/3rdparty/${dir}")
  file(COPY "${sub_path}/" DESTINATION "${SOURCE_PATH}/3rdparty/${dir}")
endfunction()
ajp_fetch_submodule(jinja https://github.com/pallets/jinja.git 15206881c006c79667fe5154fe80c01c65410679)
ajp_fetch_submodule(markupsafe https://github.com/pallets/markupsafe.git 297fc8e356e6836a62087949245d09a28e9f1b13)
ajp_fetch_submodule(fast_float https://github.com/fastfloat/fast_float.git 97b54ca9e75f5303507699d27c6b4f4efe4641a1)

# Triplet MSVC : clang-cl compile ce port (voir clang-cl-toolchain.cmake.in). Un triplet
# qui fournit déjà sa chaîne (clang-cl, MinGW) la garde.
if(VCPKG_TARGET_IS_WINDOWS AND NOT VCPKG_TARGET_IS_MINGW AND NOT VCPKG_CHAINLOAD_TOOLCHAIN_FILE)
  configure_file("${CMAKE_CURRENT_LIST_DIR}/clang-cl-toolchain.cmake.in"
    "${CURRENT_BUILDTREES_DIR}/clang-cl-toolchain.cmake" @ONLY)
  set(VCPKG_CHAINLOAD_TOOLCHAIN_FILE "${CURRENT_BUILDTREES_DIR}/clang-cl-toolchain.cmake")
endif()

vcpkg_find_acquire_program(PYTHON3)
get_filename_component(PYTHON3_DIR "${PYTHON3}" DIRECTORY)
vcpkg_add_to_path("${PYTHON3_DIR}")

vcpkg_configure_meson(
  SOURCE_PATH "${SOURCE_PATH}"
  OPTIONS
    -Dvulkan=enabled
    -Dvk-proc-addr=disabled
    "-Dvulkan-registry=${CURRENT_INSTALLED_DIR}/share/vulkan/registry/vk.xml"
    -Dshaderc=enabled
    -Dglslang=disabled
    -Dlcms=enabled
    -Dopengl=disabled
    -Dd3d11=disabled
    -Dlibdovi=disabled
    -Dunwind=disabled
    -Dxxhash=disabled
    -Ddemos=false
    -Dtests=false
)
vcpkg_install_meson()
vcpkg_fixup_pkgconfig()
vcpkg_copy_pdbs()
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
