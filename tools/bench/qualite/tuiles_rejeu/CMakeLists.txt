# tuiles_rejeu : les tuiles de SPAN du lecteur (app/play/upscale_tiles.cpp) et sa reprise entière
# (upscale_reuse.cpp) rejouées sur processeur, sur la carte de la scène tirée des flux NVOF de carte_rejeu.
# Hors du lecteur, sans GPU ni dépendance : le code de la carte et des deux décisions est compilé tel quel.
#
#   cmake -S tools/bench/qualite/tuiles_rejeu -B build/tuiles_rejeu -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/tuiles_rejeu
cmake_minimum_required(VERSION 3.24)
project(tuiles_rejeu LANGUAGES CXX)
set(KIRARI_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../../../../src")
add_executable(tuiles_rejeu tuiles_rejeu.cpp
               "${KIRARI_SRC}/app/play/upscale_tiles.cpp" "${KIRARI_SRC}/app/play/upscale_reuse.cpp"
               "${KIRARI_SRC}/kernels/ref/scene_flow.cpp" "${KIRARI_SRC}/kernels/ref/scene_camera.cpp"
               "${KIRARI_SRC}/kernels/ref/scene_fit.cpp" "${KIRARI_SRC}/kernels/ref/color.cpp"
               "${KIRARI_SRC}/kernels/ref/resample.cpp")
target_compile_features(tuiles_rejeu PRIVATE cxx_std_23)
target_include_directories(tuiles_rejeu PRIVATE "${KIRARI_SRC}")
# upscale_reuse.h passe par gpu/backend.h, qui demande l'empreinte de la frontière : rien ne traverse ici
# de frontière de backend, une valeur nulle suffit.
target_compile_definitions(tuiles_rejeu PRIVATE KIRARI_BOUNDARY_HASH=0ULL)
