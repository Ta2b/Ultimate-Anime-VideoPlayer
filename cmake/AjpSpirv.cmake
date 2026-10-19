# ajp_add_spirv(<cible> NAMESPACE <espace> SHADERS <fichier.comp>...
#               [VARIANT <suffixe> DEFINES <MACRO[=valeur]>... VARIANT_SHADERS <fichier.comp>...])
#
# Compile chaque shader compute en SPIR-V hors ligne avec AJP_GLSLC, le valide par spirv-val
# s'il est trouvé, puis embarque ses mots dans la cible : un .cpp et un .h générés dans le
# dossier de construction, rien n'entre dans le dépôt. Le shader « dossier/nom.comp » s'appelle
# « nom » ; une variante se compile avec -D<MACRO> et s'appelle « nom_<suffixe> ».
#
# L'en-tête généré, <cible>_spirv.h, déclare dans <espace> :
#   std::span<const SpirvShader> shaders();          tous les shaders, dans l'ordre déclaré
#   std::span<const std::uint32_t> words(name);      les mots du shader nommé, vides s'il manque
#
# Les shaders incluent ajp_kernels_spec.glsl (constantes de kernels/spec.h, généré par
# src/kernels) et les .glsl de leur propre dossier.
#
# Ce fichier sert aussi de script : cmake -DAJP_SPIRV_IN=<.spv> -DAJP_SPIRV_OUT=<.inc>
# -P AjpSpirv.cmake écrit les mots d'un module SPIR-V en liste C++.

if(CMAKE_SCRIPT_MODE_FILE AND DEFINED AJP_SPIRV_IN)
  file(READ "${AJP_SPIRV_IN}" hex HEX)
  string(LENGTH "${hex}" digits)
  math(EXPR remainder "${digits} % 8")
  # Un module commence par le nombre magique 0x07230203, rangé en petit-boutiste par glslc.
  string(SUBSTRING "${hex}" 0 8 magic)
  if(digits EQUAL 0 OR NOT remainder EQUAL 0 OR NOT magic STREQUAL "03022307")
    message(FATAL_ERROR "${AJP_SPIRV_IN} : module SPIR-V invalide (${digits} chiffres, en-tête ${magic})")
  endif()
  string(REGEX REPLACE "(..)(..)(..)(..)" "0x\\4\\3\\2\\1u,\n" words "${hex}")
  file(WRITE "${AJP_SPIRV_OUT}" "${words}")
  return()
endif()

include_guard(GLOBAL)

find_program(AJP_SPIRV_VAL NAMES spirv-val HINTS "$ENV{VULKAN_SDK}/bin")
set(AJP_SPIRV_SCRIPT "${CMAKE_CURRENT_LIST_FILE}")
set(AJP_SPIRV_TARGET_ENV "vulkan1.2" CACHE STRING "Environnement cible de glslc et spirv-val")
# spirv-val vient de SPIRV-Tools, souvent empaqueté à part de glslc. Facultatif en local ; la CI
# le rend obligatoire (-DAJP_SPIRV_VAL_REQUIRED=ON) pour qu'aucun module ne soit embarqué sans
# validation. Son absence n'est jamais silencieuse.
option(AJP_SPIRV_VAL_REQUIRED "Échouer à la configuration quand spirv-val est introuvable" OFF)
if(NOT AJP_SPIRV_VAL)
  if(AJP_SPIRV_VAL_REQUIRED)
    message(FATAL_ERROR "spirv-val introuvable (SPIRV-Tools) et AJP_SPIRV_VAL_REQUIRED est activé")
  endif()
  message(WARNING "spirv-val introuvable (SPIRV-Tools) : le SPIR-V embarqué ne sera pas validé à la "
                  "construction, seulement lu par vk_compute (spirv_reflect) à la création des pipelines")
endif()

# Une commande par shader : glslc vers un .spv temporaire, spirv-val, puis la liste de mots.
# Le .inc n'est écrit qu'après une validation réussie : un module refusé ne reste jamais à jour.
function(_ajp_spirv_command source name defines out_dir out_inc)
  set(spv "${out_dir}/${name}.spv")
  set(inc "${out_dir}/${name}.spv.inc")
  set(dep "${out_dir}/${name}.d")
  set(define_flags "")
  foreach(definition IN LISTS defines)
    list(APPEND define_flags "-D${definition}")
  endforeach()
  set(validate "")
  if(AJP_SPIRV_VAL)
    set(validate COMMAND "${AJP_SPIRV_VAL}" --target-env "${AJP_SPIRV_TARGET_ENV}" "${spv}")
  endif()
  get_filename_component(source_dir "${source}" DIRECTORY)
  add_custom_command(
    OUTPUT "${inc}"
    BYPRODUCTS "${spv}"
    COMMAND "${AJP_GLSLC}" --target-env=${AJP_SPIRV_TARGET_ENV} -Werror
            -I "${AJP_KERNELS_GLSL_DIR}" -I "${source_dir}" ${define_flags}
            -MD -MF "${dep}" -MT "${inc}" -o "${spv}" "${source}"
    ${validate}
    COMMAND "${CMAKE_COMMAND}" -DAJP_SPIRV_IN=${spv} -DAJP_SPIRV_OUT=${inc} -P "${AJP_SPIRV_SCRIPT}"
    DEPENDS "${source}" "${AJP_KERNELS_GLSL_DIR}/ajp_kernels_spec.glsl" ajp_kernels_glsl_header
            "${AJP_SPIRV_SCRIPT}"
    DEPFILE "${dep}"
    COMMENT "SPIR-V ${name}"
    VERBATIM)
  set(${out_inc} "${inc}" PARENT_SCOPE)
endfunction()

function(ajp_add_spirv target)
  cmake_parse_arguments(ARG "" "NAMESPACE;VARIANT" "SHADERS;DEFINES;VARIANT_SHADERS" ${ARGN})
  if(NOT ARG_NAMESPACE OR NOT ARG_SHADERS)
    message(FATAL_ERROR "ajp_add_spirv(${target}) : NAMESPACE et SHADERS sont obligatoires")
  endif()
  if(ARG_VARIANT_SHADERS AND NOT ARG_VARIANT)
    message(FATAL_ERROR "ajp_add_spirv(${target}) : VARIANT_SHADERS sans VARIANT")
  endif()

  set(out_dir "${CMAKE_CURRENT_BINARY_DIR}/${target}_spirv")
  file(MAKE_DIRECTORY "${out_dir}")
  set(names "")
  set(includes "")

  # Chaque entrée : <source>|<nom>|<définitions séparées par des virgules>.
  set(entries "")
  foreach(source IN LISTS ARG_SHADERS)
    get_filename_component(stem "${source}" NAME_WE)
    list(APPEND entries "${source}|${stem}|")
  endforeach()
  string(REPLACE ";" "," variant_defines "${ARG_DEFINES}")
  foreach(source IN LISTS ARG_VARIANT_SHADERS)
    get_filename_component(stem "${source}" NAME_WE)
    list(APPEND entries "${source}|${stem}_${ARG_VARIANT}|${variant_defines}")
  endforeach()

  foreach(entry IN LISTS entries)
    string(REPLACE "|" ";" fields "${entry}")
    list(GET fields 0 source)
    list(GET fields 1 name)
    list(LENGTH fields field_count)
    set(defines "")
    if(field_count GREATER 2)
      list(GET fields 2 packed)
      string(REPLACE "," ";" defines "${packed}")
    endif()
    if(NOT IS_ABSOLUTE "${source}")
      set(source "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
    endif()
    if(name IN_LIST names)
      message(FATAL_ERROR "ajp_add_spirv(${target}) : deux shaders s'appellent ${name}")
    endif()
    _ajp_spirv_command("${source}" "${name}" "${defines}" "${out_dir}" inc)
    list(APPEND names "${name}")
    list(APPEND includes "${inc}")
  endforeach()

  # En-tête et table générés : une liste constexpr par shader, incluse depuis son .inc.
  set(header "${out_dir}/${target}_spirv.h")
  set(table "${out_dir}/${target}_spirv.cpp")
  set(header_text "// Généré par ajp_add_spirv(${target}) : ne pas modifier.
#pragma once

#include <cstdint>
#include <span>
#include <string_view>

namespace ${ARG_NAMESPACE} {

struct SpirvShader {
  std::string_view name;
  std::span<const std::uint32_t> words;
};

// Tous les shaders de ${target}, dans l'ordre de leur déclaration.
std::span<const SpirvShader> shaders();

// Les mots du shader nommé, vides s'il n'existe pas.
std::span<const std::uint32_t> words(std::string_view name);

}  // namespace ${ARG_NAMESPACE}
")
  set(arrays "")
  set(rows "")
  foreach(name IN LISTS names)
    string(APPEND arrays "constexpr std::uint32_t k_${name}[] = {\n#include \"${name}.spv.inc\"\n};\n")
    string(APPEND rows "    SpirvShader{\"${name}\", k_${name}},\n")
  endforeach()
  set(table_text "// Généré par ajp_add_spirv(${target}) : ne pas modifier.
#include \"${target}_spirv.h\"

#include <array>

namespace ${ARG_NAMESPACE} {
namespace {

${arrays}
constexpr std::array kShaders{
${rows}};

}  // namespace

std::span<const SpirvShader> shaders() {
  return kShaders;
}

std::span<const std::uint32_t> words(std::string_view name) {
  for (const SpirvShader& shader : kShaders) {
    if (shader.name == name) return shader.words;
  }
  return {};
}

}  // namespace ${ARG_NAMESPACE}
")
  file(CONFIGURE OUTPUT "${header}" CONTENT "${header_text}" @ONLY)
  file(CONFIGURE OUTPUT "${table}" CONTENT "${table_text}" @ONLY)

  target_sources(${target} PRIVATE "${table}" ${includes})
  target_include_directories(${target} PUBLIC "${out_dir}")
  add_dependencies(${target} ajp_kernels_glsl_header)
endfunction()
