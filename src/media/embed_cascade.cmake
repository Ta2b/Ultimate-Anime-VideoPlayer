# cmake -DOUTPUT=<fichier.cpp> -DINPUT=<cascade.xml> -P embed_cascade.cmake
#
# Écrit un .cpp qui définit media::kAnimeFaceCascade (media/face_cascade.h), un std::string_view sur les octets
# exacts de la cascade de visages (third_party/lbpcascade_animeface). Les octets passent en hexadécimal, comme
# les hooks du grade (src/render/embed_shaders.cmake) : MSVC refuse les littéraux de plus de 16 Kio.
file(READ "${INPUT}" hex HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "'\\\\x\\1'," bytes "${hex}")
string(LENGTH "${bytes}" bytes_length)
math(EXPR size "${bytes_length} / 7")
string(REGEX REPLACE "(('\\\\x[0-9a-f][0-9a-f]',){16})" "\\1\n  " bytes "${bytes}")
set(body "// Généré par src/media/embed_cascade.cmake : ne pas modifier.\n")
string(APPEND body "#include \"media/face_cascade.h\"\n\nnamespace ajp::media {\nnamespace {\n")
string(APPEND body "constexpr char kBytes[${size}] = {\n  ${bytes}\n};\n}  // namespace\n\n")
string(APPEND body "const std::string_view kAnimeFaceCascade{kBytes, sizeof(kBytes)};\n\n}  // namespace ajp::media\n")
file(WRITE "${OUTPUT}.tmp" "${body}")
file(COPY_FILE "${OUTPUT}.tmp" "${OUTPUT}" ONLY_IF_DIFFERENT)
file(REMOVE "${OUTPUT}.tmp")
