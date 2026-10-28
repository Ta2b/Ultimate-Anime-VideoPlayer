# cmake -DOUTPUT=<fichier.cpp> -DINPUT=<fichier> -DNAME=<identifiant> -DNAMESPACE=<espace> -DHEADER=<en-tête>
#       -P embed_text.cmake
#
# Écrit un .cpp qui définit `const std::string_view NAME` sur les octets exacts du fichier (en
# hexadécimal : aucun échappement, aucune limite de longueur de littéral).
file(READ "${INPUT}" hex HEX)
string(LENGTH "${hex}" hex_length)
math(EXPR size "${hex_length} / 2")
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "'\\\\x\\1'," bytes "${hex}")
string(REGEX REPLACE "(('\\\\x[0-9a-f][0-9a-f]',){16})" "\\1\n  " bytes "${bytes}")
set(body "// Généré par src/app/play/embed_text.cmake : ne pas modifier.\n#include \"${HEADER}\"\n\n")
string(APPEND body "namespace ${NAMESPACE} {\nnamespace {\nconstexpr char k_${NAME}[${size}] = {\n  ${bytes}\n};\n}  // namespace\n\n")
string(APPEND body "const std::string_view ${NAME}{k_${NAME}, sizeof(k_${NAME})};\n\n}  // namespace ${NAMESPACE}\n")
file(WRITE "${OUTPUT}.tmp" "${body}")
file(COPY_FILE "${OUTPUT}.tmp" "${OUTPUT}" ONLY_IF_DIFFERENT)
file(REMOVE "${OUTPUT}.tmp")
