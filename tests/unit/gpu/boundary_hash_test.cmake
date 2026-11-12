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

# Un commentaire en première ou en dernière position, sur un en-tête qui n'en avait pas (B27-12).
file(WRITE "${WORK}/bare.h" "#pragma once\nint x;\n")
file(WRITE "${WORK}/licensed.h" "// licence\n#pragma once\nint x;\n// fin")
ajp_boundary_hash(bare "${WORK}/bare.h")
ajp_boundary_hash(licensed "${WORK}/licensed.h")
if(NOT bare STREQUAL licensed)
  message(FATAL_ERROR "un commentaire de tête ou de queue change l'empreinte : ${bare} puis ${licensed}")
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

# Un commentaire en première ou en dernière position (blanc de tête ou de queue laissé par sa suppression) ne change
# pas l'empreinte : retirer la ligne de licence d'un en-tête ne doit pas faire refuser le module déjà construit.
file(WRITE "${WORK}/bare.h" "#pragma once\nint x;\n")
file(WRITE "${WORK}/head_comment.h" "// licence\n#pragma once\nint x;\n")
file(WRITE "${WORK}/tail_comment.h" "#pragma once\nint x;\n// fin\n")
file(WRITE "${WORK}/block_comment.h" "/* licence */\n#pragma once\nint x;\n/* fin */\n")
ajp_boundary_hash(bare "${WORK}/bare.h")
foreach(variant head_comment tail_comment block_comment)
  ajp_boundary_hash(with_comment "${WORK}/${variant}.h")
  if(NOT with_comment STREQUAL bare)
    message(FATAL_ERROR "${variant} change l'empreinte : ${bare} puis ${with_comment}")
  endif()
endforeach()

# C21-01 : un commentaire de ligne qui contient « /* » n'ouvre pas de bloc ; le code qui le suit reste dans
# l'empreinte.
file(WRITE "${WORK}/glob_comment.h" "int a; // voir src/*.h\nint b;\nint c; /* x */\n")
file(WRITE "${WORK}/glob_comment_changed.h" "int a; // voir src/*.h\nint b2;\nint c; /* x */\n")
ajp_boundary_hash(glob_before "${WORK}/glob_comment.h")
ajp_boundary_hash(glob_after "${WORK}/glob_comment_changed.h")
if(glob_before STREQUAL glob_after)
  message(FATAL_ERROR "un champ après « // …/*… » ne change pas l'empreinte (${glob_before})")
endif()

# C21-08 : la liste couvre la fermeture des inclusions du projet, de proche en proche, depuis ses en-têtes : un
# « #include "core/…" » ajouté à l'un d'eux sans l'ajouter à la liste ferait sortir ses types de l'empreinte.
set(pending ${AJP_BOUNDARY_HEADERS})
set(seen "")
while(pending)
  list(POP_FRONT pending header)
  list(FIND seen "${header}" known)
  if(NOT known EQUAL -1)
    continue()
  endif()
  list(APPEND seen "${header}")
  file(STRINGS "${SOURCE}/${header}" includes REGEX "^[ \t]*#[ \t]*include[ \t]*\"")
  foreach(line IN LISTS includes)
    string(REGEX REPLACE "^[ \t]*#[ \t]*include[ \t]*\"([^\"]+)\".*$" "\\1" included "${line}")
    if(NOT EXISTS "${SOURCE}/src/${included}")
      continue()  # un en-tête de tiers ou relatif : hors du projet
    endif()
    list(FIND AJP_BOUNDARY_HEADERS "src/${included}" listed)
    if(listed EQUAL -1)
      message(FATAL_ERROR "${header} inclut src/${included}, absent de AJP_BOUNDARY_HEADERS (cmake/BoundaryHash.cmake)")
    endif()
    list(APPEND pending "src/${included}")
  endforeach()
endwhile()
list(LENGTH seen closure)
message(STATUS "fermeture de la frontière : ${closure} en-têtes")
