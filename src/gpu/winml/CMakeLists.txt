# ajp_backend_migraphx sous Windows : le backend Vulkan (chaîne sur l'appareil de rendu) dont les
# réseaux passent par MIGraphX de Windows ML, sur les GPU AMD RDNA 3 et plus récents. AJP_WITH_WINML ;
# AJP_WINML_ROOT : le paquet NuGet Microsoft.Windows.AI.MachineLearning extrait (include/,
# runtimes/win-x64/native/), à la version de tools/release/versions.env
# (tools/deps/windows-winml.ps1).
#
# Rien n'est lié à l'import : onnxruntime.dll et Microsoft.Windows.AI.MachineLearning.dll se
# chargent par leur chemin, à côté du module (winml_library.h), et seuls leurs en-têtes servent
# ici. Hors Windows, le module se construit pour vérifier son code, et refuse à l'exécution.
set(AJP_WINML_ROOT "" CACHE PATH "Paquet NuGet Microsoft.Windows.AI.MachineLearning extrait")
find_path(AJP_WINML_INCLUDE WinMLEpCatalog.h HINTS "${AJP_WINML_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
set(AJP_WINML_RUNTIME "${AJP_WINML_ROOT}/runtimes/win-x64/native")

# Le runtime à part de ses points d'entrée : les tests du backend Vulkan le lient pour comparer
# ses sorties à l'interpréteur sur le vrai matériel (vendor_runtime_test.cpp).
ajp_add_module(winml_rt SOURCES winml_library.cpp winml_manifest.cpp winml_runtime.cpp
  DEPS ajp::vk_backend ajp::core_log PRIVATE_DEPS ajp::dep::spdlog)
target_include_directories(ajp_winml_rt SYSTEM PRIVATE "${AJP_WINML_INCLUDE}" "${AJP_WINML_INCLUDE}/winml")
# L'API d'ONNX Runtime vient de onnxruntime.dll chargé à l'exécution (Ort::InitApi), jamais d'une
# bibliothèque d'import.
target_compile_definitions(ajp_winml_rt PRIVATE ORT_API_MANUAL_INIT)
set_target_properties(ajp_winml_rt PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)

add_library(ajp_backend_migraphx MODULE entry.cpp)
target_link_libraries(ajp_backend_migraphx PRIVATE ajp::winml_rt ajp::core_log ajp::dep::spdlog)
set_target_properties(ajp_backend_migraphx PROPERTIES
  PREFIX ""
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_options(ajp_backend_migraphx PRIVATE "LINKER:--exclude-libs,ALL" "LINKER:--no-undefined")
endif()
ajp_set_warnings(ajp_backend_migraphx)
# Les bibliothèques de Windows ML à côté du module dans l'arbre de construction, comme dans le
# paquet du moteur (tools/release/package-windows.ps1).
if(WIN32)
  add_custom_command(TARGET ajp_backend_migraphx POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${AJP_WINML_RUNTIME}/onnxruntime.dll"
      "${AJP_WINML_RUNTIME}/Microsoft.Windows.AI.MachineLearning.dll" "$<TARGET_FILE_DIR:ajp_backend_migraphx>"
    VERBATIM)
endif()
