find_package(Catch2 3 CONFIG REQUIRED)
include(Catch)

# Support commun aux tests qui créent un appareil Vulkan (couche de validation, main).
add_subdirectory(support)

# Chaque module construit ajoute ses tests ; un dossier par module.
file(GLOB module_dirs LIST_DIRECTORIES true "${CMAKE_CURRENT_SOURCE_DIR}/unit/*")
foreach(dir IN LISTS module_dirs)
  get_filename_component(module "${dir}" NAME)
  if(IS_DIRECTORY "${dir}" AND TARGET ajp_${module})
    add_subdirectory("${dir}")
  endif()
endforeach()

# Fuzz des entrées venues de l'extérieur (tests/fuzz), avec le lecteur ; POSIX seulement.
if(TARGET ajp_app AND NOT WIN32)
  add_subdirectory(fuzz)
endif()
