// Les clés d'Artifact des moteurs fusionnés du backend TensorRT (RIFE, DistilDRBA) : ce que load() doit
// savoir, écrit et relu sans CUDA (ajp::cuda_fused_keys, testé sur toutes les plateformes, B03-09).
#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "kernels/spec/drba_fused.h"
#include "kernels/spec/rife_fused.h"

namespace ajp::gpu::cuda {

// La clé d'un Artifact de ce moteur : « rife-fused:<h>x<w>:<image|motion|image-cubic>:<d0,d1…>:<moteur0>,… »,
// les moteurs étant les fichiers du cache, dans l'ordre encodeur, blocs 0 à N-1. Artifact ne porte que la clé
// et un chemin : la clé dit tout ce que load() doit savoir.
struct RifeFusedKey {
  int height = 0;
  int width = 0;
  bool motion = false;
  kernels::RifeWarp warp = kernels::RifeWarp::Bilinear;  // le warp de l'image (sortie « image-cubic »)
  kernels::RifeFusedPlan plan;
  std::vector<std::string> engines;
};
std::string rife_fused_artifact_key(const RifeFusedKey& key);
// Rien si la clé est celle d'un autre moteur ou illisible.
std::optional<RifeFusedKey> rife_fused_artifact_key_parse(std::string_view text);

// La clé d'un Artifact de ce moteur : « drba-fused:<h>x<w>:<t>:<moteur0>,<moteur1>,<moteur2>,<moteur3> », les
// moteurs étant les fichiers du cache, dans l'ordre encodeur, blocs 0 à 2. Artifact ne porte que la clé et un
// chemin : la clé dit tout ce que load() doit savoir.
struct DrbaFusedKey {
  int height = 0;
  int width = 0;
  kernels::DrbaFusedPlan plan;
  std::vector<std::string> engines;
};
std::string drba_fused_artifact_key(const DrbaFusedKey& key);
// Rien si la clé est celle d'un autre moteur ou illisible.
std::optional<DrbaFusedKey> drba_fused_artifact_key_parse(std::string_view text);

}  // namespace ajp::gpu::cuda
