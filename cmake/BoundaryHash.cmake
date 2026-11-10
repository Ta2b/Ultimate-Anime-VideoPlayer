# L'empreinte des en-têtes de la frontière lecteur ⇄ backend (P5.19) : leur texte sans commentaires ni blancs,
# haché. Une virtuelle ajoutée, retirée ou déplacée la change ; un commentaire ne la change pas. Elle entre dans
# kBackendLayout (gpu/backend.h) par AJP_BOUNDARY_HASH : un backend construit contre d'autres en-têtes est
# refusé au chargement même si kBackendAbi n'a pas bougé. Test : tests/unit/gpu/boundary_hash_test.cmake.
include_guard(GLOBAL)

# Tout ce que backend.h inclut du projet, de proche en proche (A21-01) : kernel_ops.h tient kernels/spec.h
# (FrameStats, Filter, constantes que les noyaux recopient) et core/moving_area.h, ring.h tient messages.h. Les
# en-têtes de kernels/spec/ sont tous pris, triés : un nouveau y entre sans retouche.
file(GLOB ajp_kernel_spec_headers RELATIVE "${CMAKE_CURRENT_LIST_DIR}/.."
     "${CMAKE_CURRENT_LIST_DIR}/../src/kernels/spec/*.h")
list(SORT ajp_kernel_spec_headers)
set(AJP_BOUNDARY_HEADERS
  src/core/frame.h src/core/gpu_device.h src/core/messages.h src/core/moving_area.h src/core/ring.h
  src/core/types.h
  src/gpu/backend.h src/gpu/gpu_profile.h src/gpu/kernel_ops.h src/gpu/lanes.h src/gpu/tensor.h
  src/kernels/spec.h ${ajp_kernel_spec_headers})

# out : seize chiffres hexadécimaux, le début du SHA-256 du texte réduit des fichiers, dans l'ordre donné.
function(ajp_boundary_hash out)
  set(all "")
  foreach(file IN LISTS ARGN)
    file(READ "${file}" text)
    string(REGEX REPLACE "/\\*([^*]|\\*+[^*/])*\\*+/" " " text "${text}")
    string(REGEX REPLACE "//[^\n]*" " " text "${text}")
    string(REGEX REPLACE "[ \t\r\n]+" " " text "${text}")
    string(APPEND all "${text}\n")
  endforeach()
  string(SHA256 digest "${all}")
  string(SUBSTRING "${digest}" 0 16 digest)
  set(${out} "${digest}" PARENT_SCOPE)
endfunction()
