// Le problème du mélange du RIFE natif (rife_native_mix.cpp, docs/rife-natif.md) sur le choix commun de
// native_mix.h. La base est TensorRT : l'option 0 de chaque morceau, sans erreur contre la référence, qui est
// sa sortie. Un morceau prend TensorRT ou un réseau natif plus rapide que lui ; pas de graine. En-tête sans
// CUDA : les tests sur hôte le lisent (rife_mix_test.cpp rejoue le choix d'avant native_mix.h contre lui).
#pragma once

#include <cstddef>
#include <vector>

#include "gpu/cuda/backend/native_mix.h"

namespace kirari::gpu::cuda::rife_mix {

// Les options qu'un morceau peut prendre (times : [morceau][option], 0 pour TensorRT) : TensorRT, et chaque
// réseau plus rapide que lui (un réseau plus lent ne gagne rien sur TensorRT, la référence).
std::vector<std::vector<int>> candidates(const native_mix::Grid& times);
// Le problème : peak, le pic de chaque groupe de la sortie de TensorRT.
native_mix::Problem problem(native_mix::Grid times, std::vector<double> peak, double min_db);
// Ce que la chaîne rend pour un mélange : l'erreur de chaque groupe contre TensorRT, qui est aussi la base.
native_mix::Eval eval(const std::vector<double>& mse);
// Les options gardées : celles du mélange vérifié, sinon tout à TensorRT.
native_mix::Options kept(const native_mix::Mix& mix, std::size_t pieces);

}  // namespace kirari::gpu::cuda::rife_mix
