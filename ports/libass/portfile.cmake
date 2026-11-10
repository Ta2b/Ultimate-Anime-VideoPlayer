# libass de l'amont (0.17.5), plus le rendu sur plusieurs fils du fork the-database/libass
# (ass_set_render_thread_count, la seule fonction hors de l'amont que le lecteur appelle), en un correctif :
# rendu-sur-plusieurs-fils.patch, le même que tools/deps/linux-from-source.sh. Un correctif de sécurité de
# l'amont arrive donc en changeant REF (A35-01, tools/ci/deps_vulns.py le signale). Les modes de rendu différé
# pour GPU du fork, inutilisés ici, n'y sont pas.
#
# Régénérer le correctif sur une nouvelle version de l'amont : cloner libass, partir de l'étiquette, y
# appliquer le correctif actuel (ou, s'il ne s'applique plus, reprendre du fork les commits 8354eaf 7180665
# 218cf48 4a32722 3b4f4b9 4e6e7d1 a26dc9a bc2e43e par git cherry-pick), puis « git diff <étiquette> HEAD »
# sous l'en-tête du fichier.
#
# Assembleur (NASM sur x64), libunibreak et fils exigés, comme le port officiel.
vcpkg_from_git(
  OUT_SOURCE_PATH SOURCE_PATH
  URL https://github.com/libass/libass.git
  REF 4a05d8127f525943ebf45fdc6497c9e665947f0d
  PATCHES rendu-sur-plusieurs-fils.patch
)

if(VCPKG_TARGET_IS_WINDOWS)
  set(font_options -Dfontconfig=disabled -Ddirectwrite=enabled -Dcoretext=disabled)
  # Le rendu sur plusieurs fils utilise <stdatomic.h> : MSVC le refuse sous gnu99 (le défaut du meson.build) et le
  # garde derrière /experimental:c11atomics. Il déclare aussi une variable par fil en __thread
  # (GCC), que MSVC écrit __declspec(thread).
  set(std_options -Dc_std=c11)
  string(APPEND VCPKG_C_FLAGS " /experimental:c11atomics /D__thread=__declspec(thread)")
  string(APPEND VCPKG_CXX_FLAGS " ")  # vcpkg_cmake_configure veut les deux drapeaux, ou aucun
elseif(VCPKG_TARGET_IS_OSX)
  set(font_options -Dfontconfig=disabled -Ddirectwrite=disabled -Dcoretext=enabled)
else()
  set(font_options -Dfontconfig=enabled -Ddirectwrite=disabled -Dcoretext=disabled)
endif()

if(VCPKG_TARGET_ARCHITECTURE STREQUAL "x64")
  vcpkg_find_acquire_program(NASM)
  get_filename_component(nasm_dir "${NASM}" DIRECTORY)
  vcpkg_add_to_path("${nasm_dir}")
endif()

vcpkg_configure_meson(
  SOURCE_PATH "${SOURCE_PATH}"
  OPTIONS ${font_options} ${std_options} -Dasm=enabled -Dtest=disabled -Dcompare=disabled -Dprofile=disabled -Dfuzz=disabled -Dcheckasm=disabled -Dlibunibreak=enabled -Dthreads=enabled
)
vcpkg_install_meson()
vcpkg_fixup_pkgconfig()
vcpkg_copy_pdbs()
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/COPYING")
