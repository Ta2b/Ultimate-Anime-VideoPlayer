# libass du fork the-database, commit 896614fa, celui du lecteur actuel.
vcpkg_from_git(
  OUT_SOURCE_PATH SOURCE_PATH
  URL https://github.com/the-database/libass.git
  REF 896614fa263f9f64944f7b8650ff6806d5328c4d
)

if(VCPKG_TARGET_IS_WINDOWS)
  set(font_options -Dfontconfig=disabled -Ddirectwrite=enabled -Dcoretext=disabled)
elseif(VCPKG_TARGET_IS_OSX)
  set(font_options -Dfontconfig=disabled -Ddirectwrite=disabled -Dcoretext=enabled)
else()
  set(font_options -Dfontconfig=enabled -Ddirectwrite=disabled -Dcoretext=disabled)
endif()

# Le fork ajoute des atomiques (libass/ass_threading.h) : <stdatomic.h>, ou les
# Interlocked de MSVC quand __STDC_NO_ATOMICS__ est défini. cl.exe ne définit cette
# macro qu'en /std:c11 ou /std:c17 ; avec le gnu99 du projet, il n'a aucun drapeau
# /std et son <stdatomic.h> refuse de compiler du C.
set(std_options "")
if(VCPKG_TARGET_IS_WINDOWS AND NOT VCPKG_TARGET_IS_MINGW)
  set(std_options -Dc_std=c11)
endif()

vcpkg_configure_meson(
  SOURCE_PATH "${SOURCE_PATH}"
  OPTIONS ${font_options} ${std_options} -Dasm=disabled -Dtest=disabled -Dcompare=disabled -Dprofile=disabled -Dfuzz=disabled -Dcheckasm=disabled -Dlibunibreak=disabled
)
vcpkg_install_meson()
vcpkg_fixup_pkgconfig()
vcpkg_copy_pdbs()
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/COPYING")
