// Runtimes d'inférence présents sur le système, sondés par leurs bibliothèques
// sans les initialiser : le pilote CUDA (et sa version), ROCm et MIGraphX,
// OpenVINO. Le premier lancement s'en sert pour dire ce qui manque avant
// d'installer un moteur, par exemple un pilote NVIDIA trop ancien pour TensorRT.
#pragma once

#include <optional>
#include <string>
#include <vector>

#include "app/library_loader.h"
#include "core/engines.h"

namespace ajp::app {

// Version de CUDA prise en charge par le pilote, telle que cuDriverGetVersion la rend.
struct CudaDriver {
  std::string library;  // nom qui s'est chargé : « libcuda.so.1 »
  int version = 0;      // 1000 × majeur + 10 × mineur : 13040 pour CUDA 13.4
  int major() const { return version / 1000; }
  int minor() const { return (version % 1000) / 10; }
};

// TensorRT 11 exige CUDA 13, donc un pilote NVIDIA R580 ou plus récent.
inline constexpr int kTensorRtMinCuda = 13000;

struct RuntimeReport {
  std::optional<CudaDriver> cuda;
  std::optional<std::string> hip;       // ROCm : bibliothèque HIP chargée
  std::optional<std::string> migraphx;  // bibliothèque C de MIGraphX
  std::optional<std::string> openvino;
  std::vector<std::string> notes;  // pourquoi une bibliothèque trouvée n'a pas été retenue
};

// Noms essayés, dans l'ordre, pour chaque runtime.
struct RuntimeNames {
  std::vector<std::string> cuda;
  std::vector<std::string> hip;
  std::vector<std::string> migraphx;
  std::vector<std::string> openvino;
};
RuntimeNames runtime_names(core::Os os);

RuntimeReport probe_runtimes(const LibraryLoader& loader, core::Os os);

}  // namespace ajp::app
