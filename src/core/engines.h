// Choix des moteurs au premier lancement, et depuis Paramètres > Moteurs.
//
// Seul Vulkan est livré partout ; les autres moteurs s'installent à la demande.
// recommend_engines() rend la présélection, les badges et la raison de chaque
// ligne. Fonction pure : la détection des GPU (par Vulkan) se fait ailleurs.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/gpu_device.h"

namespace ajp::core {

enum class Os : std::uint8_t { Windows, Linux, MacOs };
// Le système pour lequel ce lecteur est compilé.
#if defined(_WIN32)
inline constexpr Os kHostOs = Os::Windows;
#elif defined(__APPLE__)
inline constexpr Os kHostOs = Os::MacOs;
#else
inline constexpr Os kHostOs = Os::Linux;
#endif
using GpuVendor = ajp::GpuVendor;  // déduit de l'identifiant PCI (core/gpu_device.h)

struct GpuInfo {
  GpuVendor vendor = GpuVendor::Other;
  std::string name;
  std::uint64_t vram_bytes = 0;
  bool tensor_cores = false;    // NVIDIA RTX (Turing et après)
  bool rocm_supported = false;  // AMD reconnu par MIGraphX de ROCm (Linux)
  bool winml_migraphx = false;  // AMD RDNA 3 et après, que MIGraphX de Windows ML sert (Windows)
  bool openvino_gpu = false;    // Intel Arc ou intégré récent reconnu par OpenVINO
};

struct SystemInfo {
  Os os = Os::Windows;
  std::vector<GpuInfo> gpus;  // le premier est le GPU de rendu
};

enum class EngineId : std::uint8_t {
  Vulkan,
  TensorRt,
  RtxVideoSdk,  // RTX HDR (TrueHDR) sous Windows
  MaxineVfx,    // TrueHDR sous Linux
  MiGraphX,
  OpenVino,
  DirectMl,
  Metal,
};

enum class Delivery : std::uint8_t {
  Bundled,   // livré avec le lecteur, non désinstallable
  Download,  // téléchargé depuis l'index haché
  Import,    // SDK propriétaire importé depuis un dossier choisi
};

enum class Badge : std::uint8_t {
  Recommended,
  Compatibility,
  NvidiaOnly,
  AmdOnly,
  IntelOnly,
  AppleOnly,
  WindowsOnly,
  LinuxOnly,
};

// Pourquoi une ligne est présélectionnée, proposée ou grisée.
enum class Reason : std::uint8_t {
  AlwaysIncluded,
  FastestForThisGpu,
  RtxHdrForThisGpu,
  CompareWithVulkan,  // DirectML, MIGraphX de Windows ML : à l'essai, utile si Vulkan mesure moins bien
  NoMatchingGpu,
};

struct EngineOffer {
  EngineId id;
  Delivery delivery;
  std::vector<Badge> badges;
  Reason reason;
  bool relevant = false;     // false : grisé, installable en mode avancé
  bool preselected = false;  // coché par défaut
};

// Une ligne par moteur connu, dans l'ordre d'affichage.
std::vector<EngineOffer> recommend_engines(const SystemInfo& system);

}  // namespace ajp::core
