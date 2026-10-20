# libass du fork the-database, commit 896614fa, celui du lecteur actuel.
vcpkg_from_git(
  OUT_SOURCE_PATH SOURCE_PATH
  URL https://github.com/the-database/libass.git
  REF 896614fa263f9f64944f7b8650ff6806d5328c4d
)

if(VCPKG_TARGET_IS_WINDOWS)
  set(font_options -Dfontconfig=disabled -Ddirectwrite=enabled -Dcoretext=disabled)
  # Le fork utilise <stdatomic.h> : MSVC le refuse sous gnu99 (le défaut du meson.build) et le
  # garde derrière /experimental:c11atomics.
  set(std_options -Dc_std=c11)
  string(APPEND VCPKG_C_FLAGS " /experimental:c11atomics")
  string(APPEND VCPKG_CXX_FLAGS " ")  # vcpkg_cmake_configure veut les deux drapeaux, ou aucun
elseif(VCPKG_TARGET_IS_OSX)
  set(font_options -Dfontconfig=disabled -Ddirectwrite=disabled -Dcoretext=enabled)
else()
  set(font_options -Dfontconfig=enabled -Ddirectwrite=disabled -Dcoretext=disabled)
endif()

vcpkg_configure_meson(
  SOURCE_PATH "${SOURCE_PATH}"
  OPTIONS ${font_options} ${std_options} -Dasm=disabled -Dtest=disabled -Dcompare=disabled -Dprofile=disabled -Dfuzz=disabled -Dcheckasm=disabled -Dlibunibreak=disabled
)
vcpkg_install_meson()
vcpkg_fixup_pkgconfig()
vcpkg_copy_pdbs()
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/COPYING")
