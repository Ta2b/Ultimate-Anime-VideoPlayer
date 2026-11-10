# libplacebo v7.387.0, la version du fork. Les sous-modules dont la
# construction a besoin (jinja, markupsafe, fast_float) sont épinglés aux
# commits que référence ce tag ; glad et Vulkan-Headers ne servent pas ici
# (OpenGL coupé, registre Vulkan fourni par vulkan-headers).
vcpkg_from_git(
  OUT_SOURCE_PATH SOURCE_PATH
  URL https://github.com/haasn/libplacebo.git
  REF 275f7dea27924cdd34ab2e6c20e448b6bc4fab65
  FETCH_REF v7.387.0
  # Présentation en boucle fermée (chantier 6.1, docs/adr/0011-presentation-en-boucle-fermee.md) :
  # drapeaux et chaîne de création de la swapchain, VkPresentId2KHR et VkPresentTimingsInfoEXT à la
  # présentation, swapchain courante et recréations, QUEUE_FULL sans fence attendue à l'infini. Sans
  # present_flags ni pl_vulkan_swapchain_submit, libplacebo se comporte comme avant. Le verrou « recording »
  # tenu du vol de la commande à sa soumission, que le correctif ajoutait sur 7.360.1, n'y est plus : l'amont
  # soumet désormais par CMD_BEGIN et CMD_SUBMIT, qui tiennent ce verrou d'un bout à l'autre.
  PATCHES 0001-ajp-present-timing.patch
)

function(ajp_fetch_submodule dir url ref)
  vcpkg_from_git(OUT_SOURCE_PATH sub_path URL "${url}" REF "${ref}")
  file(REMOVE_RECURSE "${SOURCE_PATH}/3rdparty/${dir}")
  file(COPY "${sub_path}/" DESTINATION "${SOURCE_PATH}/3rdparty/${dir}")
endfunction()
ajp_fetch_submodule(jinja https://github.com/pallets/jinja.git 15206881c006c79667fe5154fe80c01c65410679)
ajp_fetch_submodule(markupsafe https://github.com/pallets/markupsafe.git 297fc8e356e6836a62087949245d09a28e9f1b13)
ajp_fetch_submodule(fast_float https://github.com/fastfloat/fast_float.git 97b54ca9e75f5303507699d27c6b4f4efe4641a1)

# Sous Windows, clang-cl au lieu de cl (clang-cl.cmake) : vcpkg_configure_meson lit le
# compilateur de cette chaîne, pour ce seul port. Ses <stdatomic.h> se passent de drapeau.
if(VCPKG_TARGET_IS_WINDOWS)
  set(VCPKG_CHAINLOAD_TOOLCHAIN_FILE "${CMAKE_CURRENT_LIST_DIR}/clang-cl.cmake")
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
