# Windows : chaque exécutable du projet prend la page de code UTF-8 (manifeste utf8.manifest,
# activeCodePage, Windows 10 1903 et plus). Sans elle, argv arrive dans la page ANSI (1252 sur les
# runners) : le nom d'un test Catch2 que CTest passe en argument (« 144 Hz fixe divise le
# rafraîchissement ») ne correspond plus aux octets UTF-8 du programme (/utf-8), et un chemin de
# vidéo non ASCII, ouvert depuis l'Explorateur, se lit de travers dans std::filesystem::path et
# FFmpeg.
include_guard(GLOBAL)

function(_ajp_utf8_manifest_in dir)
  get_property(targets DIRECTORY "${dir}" PROPERTY BUILDSYSTEM_TARGETS)
  foreach(target IN LISTS targets)
    get_target_property(type ${target} TYPE)
    if(type STREQUAL "EXECUTABLE")
      target_sources(${target} PRIVATE "${CMAKE_SOURCE_DIR}/cmake/utf8.manifest")
    endif()
  endforeach()
  get_property(subdirs DIRECTORY "${dir}" PROPERTY SUBDIRECTORIES)
  foreach(subdir IN LISTS subdirs)
    _ajp_utf8_manifest_in("${subdir}")
  endforeach()
endfunction()

# À appeler à la fin du CMakeLists.txt racine, quand toutes les cibles existent. MSVC seul : son
# éditeur de liens (et CMake avec lui) intègre les fichiers .manifest des sources.
function(ajp_utf8_manifests)
  if(MSVC)
    _ajp_utf8_manifest_in("${CMAKE_SOURCE_DIR}")
  endif()
endfunction()
