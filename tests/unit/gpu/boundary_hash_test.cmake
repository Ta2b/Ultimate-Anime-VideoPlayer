# L'empreinte de la frontière (cmake/BoundaryHash.cmake) : une virtuelle ajoutée la change, un commentaire ou des
# blancs non. Lancé par CTest : cmake -DSOURCE=<dépôt> -DWORK=<dossier> -P boundary_hash_test.cmake
include("${SOURCE}/cmake/BoundaryHash.cmake")
set(header "${SOURCE}/src/gpu/backend.h")
file(READ "${header}" text)
ajp_boundary_hash(base "${header}")

file(MAKE_DIRECTORY "${WORK}")
file(WRITE "${WORK}/commented.h" "// une note de plus\n${text}\n/* et un bloc */  \n")
ajp_boundary_hash(commented "${WORK}/commented.h")
if(NOT commented STREQUAL base)
  message(FATAL_ERROR "un commentaire change l'empreinte : ${base} puis ${commented}")
endif()

string(REPLACE "virtual ~Backend() = default;" "virtual ~Backend() = default;\n  virtual void added();" virtual_text
       "${text}")
if(virtual_text STREQUAL text)
  message(FATAL_ERROR "le destructeur de Backend est introuvable dans backend.h")
endif()
file(WRITE "${WORK}/virtual.h" "${virtual_text}")
ajp_boundary_hash(added "${WORK}/virtual.h")
if(added STREQUAL base)
  message(FATAL_ERROR "une virtuelle ajoutée ne change pas l'empreinte (${base})")
endif()
message(STATUS "empreinte ${base} ; virtuelle ajoutée : ${added}")

# Les en-têtes de kernels/spec sont de la frontière (A21-01) : un champ de FrameStats la change.
list(FIND AJP_BOUNDARY_HEADERS "src/kernels/spec/frame_stats.h" spec_found)
list(FIND AJP_BOUNDARY_HEADERS "src/core/moving_area.h" moving_found)
if(spec_found EQUAL -1 OR moving_found EQUAL -1)
  message(FATAL_ERROR "kernels/spec/frame_stats.h ou core/moving_area.h hors de l'empreinte : ${AJP_BOUNDARY_HEADERS}")
endif()
