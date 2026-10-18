find_package(Catch2 3 CONFIG REQUIRED)
include(Catch)

# Chaque module construit ajoute ses tests ; un dossier par module.
file(GLOB module_dirs LIST_DIRECTORIES true "${CMAKE_CURRENT_SOURCE_DIR}/unit/*")
foreach(dir IN LISTS module_dirs)
  get_filename_component(module "${dir}" NAME)
  if(IS_DIRECTORY "${dir}" AND TARGET ajp_${module})
    add_subdirectory("${dir}")
  endif()
endforeach()
