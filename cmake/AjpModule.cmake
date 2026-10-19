# ajp_add_module(<nom> [SOURCES ...] [DEPS ...] [PRIVATE_DEPS ...])
#
# Un dossier de src/ = un module = une cible ajp_<nom>, alias ajp::<nom>.
# Un module sans source reste une cible INTERFACE : ses dépendances sont
# déjà déclarées, et le graphe se vérifie avant d'écrire la première ligne.
include_guard(GLOBAL)

function(ajp_set_warnings target)
  if(WIN32)
    # windows.h sans les macros min et max, ni les en-têtes rarement utiles (winsock).
    target_compile_definitions(${target} PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
  endif()
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /permissive- /utf-8)
    if(AJP_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE /WX)
    endif()
  else()
    target_compile_options(${target} PRIVATE
      -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion
      -Wnon-virtual-dtor -Wold-style-cast -Woverloaded-virtual
      # Les initialiseurs désignés omettent volontairement les champs par défaut.
      -Wno-missing-field-initializers)
    if(AJP_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE -Werror)
    endif()
  endif()
endfunction()

function(ajp_add_module name)
  cmake_parse_arguments(ARG "" "" "SOURCES;DEPS;PRIVATE_DEPS" ${ARGN})
  set(target ajp_${name})

  if(ARG_SOURCES)
    add_library(${target} STATIC ${ARG_SOURCES})
    target_include_directories(${target} PUBLIC "${PROJECT_SOURCE_DIR}/src")
    target_link_libraries(${target} PUBLIC ${ARG_DEPS} PRIVATE ${ARG_PRIVATE_DEPS})
    ajp_set_warnings(${target})
  else()
    add_library(${target} INTERFACE)
    target_include_directories(${target} INTERFACE "${PROJECT_SOURCE_DIR}/src")
    target_link_libraries(${target} INTERFACE ${ARG_DEPS})
  endif()

  add_library(ajp::${name} ALIAS ${target})
endfunction()

# ajp_add_unit_tests(<module> SOURCES ... [DEPS ...])
#
# Un exécutable Catch2 par module, ajp_<module>_tests, découvert par CTest.
# Les tests vivent dans tests/unit/<module>/, à côté de rien d'autre.
function(ajp_add_unit_tests module)
  cmake_parse_arguments(ARG "" "" "SOURCES;DEPS" ${ARGN})
  set(target ajp_${module}_tests)
  add_executable(${target} ${ARG_SOURCES})
  target_link_libraries(${target} PRIVATE ajp::${module} ${ARG_DEPS} Catch2::Catch2WithMain)
  ajp_set_warnings(${target})
  catch_discover_tests(${target} TEST_PREFIX "${module}: " DISCOVERY_MODE PRE_TEST)
endfunction()
