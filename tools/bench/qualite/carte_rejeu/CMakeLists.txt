# carte_rejeu : la carte de la scène du lecteur (kernels::ref::scene_motion_of, lissage compris) rejouée sur
# des paires de lumas et leurs flux NVOF écrits par nvof_dump, pour le banc carte_lecteur.py, et avec --calques
# les calques et la plaque de décor (calques_rejeu.py). Hors du lecteur, sans GPU ni dépendance : le code de la
# carte et des calques est compilé tel quel depuis src/kernels/ref et src/app/play/layer_plan.cpp.
#
#   cmake -S tools/bench/qualite/carte_rejeu -B build/carte_rejeu -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/carte_rejeu
cmake_minimum_required(VERSION 3.24)
project(carte_rejeu LANGUAGES CXX)
set(KIRARI_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../../../../src")
add_executable(carte_rejeu carte_rejeu.cpp calques.cpp "${KIRARI_SRC}/kernels/ref/scene_flow.cpp"
                           "${KIRARI_SRC}/kernels/ref/scene_camera.cpp" "${KIRARI_SRC}/kernels/ref/scene_fit.cpp"
                           "${KIRARI_SRC}/kernels/ref/color.cpp" "${KIRARI_SRC}/kernels/ref/resample.cpp"
                           "${KIRARI_SRC}/kernels/ref/layers.cpp" "${KIRARI_SRC}/kernels/ref/warp.cpp"
                           "${KIRARI_SRC}/kernels/ref/warp_pan.cpp"
                           "${KIRARI_SRC}/app/play/layer_plan.cpp")
target_compile_features(carte_rejeu PRIVATE cxx_std_23)
target_include_directories(carte_rejeu PRIVATE "${KIRARI_SRC}")
