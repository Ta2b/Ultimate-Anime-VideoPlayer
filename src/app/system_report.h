// Ce que le lecteur sait du système au démarrage : les GPU vus par Vulkan, les
// runtimes d'inférence présents, l'OS. Il en tire core::SystemInfo pour
// core::recommend_engines, par des règles écrites ici et nulle part ailleurs.
#pragma once

#include <optional>
#include <string>
#include <vector>

#include "app/library_loader.h"
#include "app/runtime_probe.h"
#include "app/vulkan_probe.h"
#include "core/engines.h"
#include "core/types.h"

namespace ajp::app {

struct SystemReport {
  core::Os os = core::Os::Linux;
  std::vector<VulkanDevice> devices;      // ordre de l'énumération
  std::optional<Refusal> vulkan_refusal;  // pourquoi aucun appareil n'a pu être lu
  RuntimeReport runtimes;
};

SystemReport detect_system(const LibraryLoader& loader, core::Os os);

// Index du GPU de rendu dans devices : un GPU dédié avant un intégré, puis le plus
// de mémoire, puis l'ordre de l'énumération ; un rendu logiciel (CPU) en dernier.
// std::nullopt sans appareil.
std::optional<std::size_t> render_device(const std::vector<VulkanDevice>& devices);

// Règles de la présélection, faute de mieux qu'une détection par Vulkan :
//   tensor_cores    NVIDIA avec unités matricielles exposées (VK_*_cooperative_matrix),
//                   soit une RTX, Turing ou plus récente ;
//   rocm_supported  AMD avec unités matricielles (WMMA), soit RDNA3 ou plus récent ;
//   openvino_gpu    Intel avec unités matricielles (XMX : Arc et Xe2), ou dédié.
// Le GPU de rendu vient en premier, les autres suivent dans l'ordre de l'énumération.
core::SystemInfo to_system_info(const SystemReport& report);

core::GpuInfo gpu_info(const VulkanDevice& device);

// Le pilote NVIDIA permet-il TensorRT 11 (CUDA 13) ? std::nullopt sans pilote CUDA.
std::optional<bool> cuda_fits_tensorrt(const RuntimeReport& runtimes);

// Description en français, une ligne par GPU puis par runtime, pour « ajp engines » et le journal.
std::vector<std::string> describe(const SystemReport& report);

}  // namespace ajp::app
