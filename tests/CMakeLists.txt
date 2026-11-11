find_package(Catch2 3 CONFIG REQUIRED)
include(Catch)

# Les tests cachés qui jugent sur la vraie carte ou sur un modèle installé (B40-02), un test CTest par groupe Catch
# `[.groupe]`, étiqueté pour `ctest -L <label>` : nvidia (TensorRT, noyaux CUDA, NVDEC), gpu (repli Vulkan sur une carte
# matérielle), models (SPAN installé sur le poste). Hors AJP_HARDWARE_TESTS ils ne sont pas enregistrés : la CI sous
# lavapipe, la garde de release et le plafond de tests sautés ne les voient pas. Le préréglage linux-prefix-cuda l'allume.
option(AJP_HARDWARE_TESTS "Enregistrer les tests cachés qui demandent une vraie carte ou des modèles (ctest -L nvidia|gpu|models)" OFF)
function(ajp_add_hardware_tests target label timeout)
  if(NOT AJP_HARDWARE_TESTS)
    return()
  endif()
  foreach(group IN LISTS ARGN)
    set(name "${label}: ${target} ${group}")
    add_test(NAME "${name}" COMMAND ${target} "[.${group}]")
    set_tests_properties("${name}" PROPERTIES LABELS ${label} TIMEOUT ${timeout} RUN_SERIAL TRUE SKIP_RETURN_CODE 4)
  endforeach()
endfunction()

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
