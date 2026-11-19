# gmflow_squelette_vide : le squelette de GMFlow tel que le lit prepare/gmflow_native.h (poids et plan), vidé pour
# gmflow_squelette_emule.py, qui le rejoue avec les arrondis des noyaux contre ONNX Runtime. Hors du lecteur, sans
# GPU ni dépendance : la lecture et le décodeur ONNX sont compilés tels quels depuis src/.
#
#   cmake -S tools/gmfss/gmflow_squelette -B build/gmflow_squelette -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/gmflow_squelette
cmake_minimum_required(VERSION 3.24)
project(gmflow_squelette_vide LANGUAGES CXX)
set(KIRARI_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../../../src")
add_executable(gmflow_squelette_vide gmflow_squelette_vide.cpp "${KIRARI_SRC}/prepare/gmflow_native.cpp"
               "${KIRARI_SRC}/gpu/ir/onnx_reader.cpp" "${KIRARI_SRC}/gpu/ir/onnx_tensor.cpp"
               "${KIRARI_SRC}/gpu/ir/protobuf.cpp" "${KIRARI_SRC}/kernels/ref/half.cpp")
target_compile_features(gmflow_squelette_vide PRIVATE cxx_std_23)
target_include_directories(gmflow_squelette_vide PRIVATE "${KIRARI_SRC}")
