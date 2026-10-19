# cmake -DOUTPUT=<fichier.cpp> -DINPUTS=<a.hook;b.hook> -P embed_shaders.cmake
#
# Écrit un .cpp qui définit, pour chaque hook, un std::string_view sur ses octets exacts
# (shaders.h). Les octets passent en hexadécimal : aucun échappement, aucune limite de
# longueur de littéral (MSVC refuse les chaînes de plus de 16 Kio).
set(body "// Généré par src/render/embed_shaders.cmake : ne pas modifier.\n")
string(APPEND body "#include \"render/shaders.h\"\n\nnamespace ajp::render::shaders {\nnamespace {\n")
set(views "")
foreach(input IN LISTS INPUTS)
  get_filename_component(stem "${input}" NAME_WE)
  string(REPLACE "-" "_" ident "${stem}")
  file(READ "${input}" hex HEX)
  string(LENGTH "${hex}" hex_length)
  math(EXPR size "${hex_length} / 2")
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "'\\\\x\\1'," bytes "${hex}")
  string(REGEX REPLACE "(('\\\\x[0-9a-f][0-9a-f]',){16})" "\\1\n  " bytes "${bytes}")
  string(APPEND body "constexpr char k_${ident}[${size}] = {\n  ${bytes}\n};\n")
  string(APPEND views "const std::string_view ${ident}{k_${ident}, sizeof(k_${ident})};\n")
endforeach()
string(APPEND body "}  // namespace\n\n${views}\n}  // namespace ajp::render::shaders\n")
file(WRITE "${OUTPUT}.tmp" "${body}")
file(COPY_FILE "${OUTPUT}.tmp" "${OUTPUT}" ONLY_IF_DIFFERENT)
file(REMOVE "${OUTPUT}.tmp")
