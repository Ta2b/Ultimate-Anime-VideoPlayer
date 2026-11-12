find_package(Catch2 3 CONFIG REQUIRED)
include(Catch)

# Les tests cachés qui jugent sur la vraie carte ou sur un modèle installé (B40-02), un test CTest par groupe Catch
# `[.groupe]`, étiqueté pour `ctest -L <label>` : nvidia (TensorRT, noyaux CUDA, NVDEC), gpu (repli Vulkan sur une carte
# matérielle), models (SPAN installé sur le poste). Hors AJP_HARDWARE_TESTS ils ne sont pas enregistrés : la CI sous
# lavapipe, la garde de release et le plafond de tests sautés ne les voient pas. Le préréglage linux-prefix-cuda l'allume.
option(AJP_HARDWARE_TESTS "Enregistrer les tests cachés qui demandent une vraie carte ou des modèles (ctest -L nvidia|gpu|models)" OFF)
# Sur le PC de référence (windows-rtx), un test matériel qui se saute (pilote absent, modèle introuvable) est un
# échec, pas un « Skipped » vert : sans cela, tout passe au vert avec les tests sautés (C40-04).
option(AJP_HARDWARE_TESTS_STRICT "Un test matériel sauté échoue (le PC de référence)" OFF)
function(ajp_add_hardware_tests target label timeout)
  if(NOT AJP_HARDWARE_TESTS)
    return()
  endif()
  foreach(group IN LISTS ARGN)
    set(name "${label}: ${target} ${group}")
    add_test(NAME "${name}" COMMAND ${target} "[.${group}]")
    set_tests_properties("${name}" PROPERTIES LABELS ${label} TIMEOUT ${timeout} RUN_SERIAL TRUE)
    if(NOT AJP_HARDWARE_TESTS_STRICT)
      set_tests_properties("${name}" PROPERTIES SKIP_RETURN_CODE 4)
    endif()
  endforeach()
endfunction()

# Où les tests de moteurs cherchent les modèles que le poste a déjà (jamais copiés dans le dépôt). HOME n'existe pas
# sous Windows : sans ce repli, les chemins devenaient « /.local/... », et les tests SPAN, RIFE et DistilDRBA du
# backend TensorRT se sautaient en silence sur le PC de Tatoub (B28-10). Deux réglages à poser d'un coup pour un autre
# rangement : AJP_MODELS_DIR (rife_v4.25.onnx, distil_drba_v2_lite.onnx…, le dossier que tools/release/fetch-models.sh
# remplit) et AJP_ANIMEJANAI_ONNX_DIR (les SPAN d'AnimeJaNai) ; chaque test garde son réglage précis (AJP_SPAN_MODEL…).
if(WIN32)
  # Des « / » : le chemin entre dans des définitions de compilation (une chaîne C, où « \U » est une erreur).
  file(TO_CMAKE_PATH "$ENV{USERPROFILE}" AJP_TEST_HOME)
else()
  set(AJP_TEST_HOME "$ENV{HOME}")
endif()
set(AJP_MODELS_DIR "${AJP_TEST_HOME}/ajp/models/rife" CACHE PATH "Dossier des RIFE et DistilDRBA ONNX pour les tests de moteurs")
set(AJP_ANIMEJANAI_ONNX_DIR "${AJP_TEST_HOME}/.local/share/AnimeJaNai-HDR-linux/animejanai/onnx"
    CACHE PATH "Dossier des SPAN ONNX d'AnimeJaNai pour les tests de moteurs")

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
