#include "app/engine_text.h"

#include <algorithm>
#include <array>
#include <format>

namespace ajp::app {

std::string_view engine_title(core::EngineId engine) {
  switch (engine) {
    case core::EngineId::Vulkan:
      return "Vulkan";
    case core::EngineId::TensorRt:
      return "TensorRT";
    case core::EngineId::RtxVideoSdk:
      return "RTX Video SDK (RTX HDR)";
    case core::EngineId::MaxineVfx:
      return "Maxine VFX (TrueHDR)";
    case core::EngineId::MiGraphX:
      return "MIGraphX";
    case core::EngineId::OpenVino:
      return "OpenVINO";
    case core::EngineId::DirectMl:
      return "DirectML";
    case core::EngineId::Metal:
      return "Metal";
  }
  return "?";
}

std::string_view engine_summary(core::EngineId engine) {
  switch (engine) {
    case core::EngineId::Vulkan:
      return "notre moteur, livré avec le lecteur : fonctionne sur tous les GPU";
    case core::EngineId::TensorRt:
      return "upscaling et interpolation les plus rapides sur les RTX";
    case core::EngineId::RtxVideoSdk:
    case core::EngineId::MaxineVfx:
      return "RTX HDR : HDR depuis une vidéo SDR, SDK NVIDIA importé depuis son dossier";
    case core::EngineId::MiGraphX:
      return "moteur d'AMD (ROCm) pour les Radeon récentes";
    case core::EngineId::OpenVino:
      return "moteur d'Intel pour les Arc et les GPU intégrés récents";
    case core::EngineId::DirectMl:
      return "moteur de Windows, plus lent : seulement pour les anciennes Radeon";
    case core::EngineId::Metal:
      return "moteur des Mac, livré avec le lecteur";
  }
  return "";
}

std::string_view badge_label(core::Badge badge) {
  switch (badge) {
    case core::Badge::Recommended:
      return "Recommandé";
    case core::Badge::Compatibility:
      return "Compatibilité";
    case core::Badge::NvidiaOnly:
      return "Seulement NVIDIA";
    case core::Badge::AmdOnly:
      return "Seulement AMD";
    case core::Badge::IntelOnly:
      return "Seulement Intel";
    case core::Badge::AppleOnly:
      return "Seulement Apple";
    case core::Badge::WindowsOnly:
      return "Seulement Windows";
    case core::Badge::LinuxOnly:
      return "Seulement Linux";
  }
  return "?";
}

std::string_view reason_text(core::Reason reason) {
  switch (reason) {
    case core::Reason::AlwaysIncluded:
      return "toujours inclus";
    case core::Reason::FastestForThisGpu:
      return "le plus rapide sur ce GPU";
    case core::Reason::RtxHdrForThisGpu:
      return "RTX HDR possible sur ce GPU";
    case core::Reason::CompareWithVulkan:
      return "à essayer si Vulkan est lent sur ce GPU";
    case core::Reason::NoMatchingGpu:
      return "aucun GPU compatible détecté";
  }
  return "";
}

std::string size_text(std::uint64_t bytes) {
  constexpr std::uint64_t kKib = 1024;
  if (bytes < kKib) return std::format("{} octets", bytes);
  constexpr std::array<std::string_view, 4> kUnits{"Kio", "Mio", "Gio", "Tio"};
  double value = static_cast<double>(bytes) / static_cast<double>(kKib);
  std::size_t unit = 0;
  // 1023,96 Kio s'arrondit à « 1,0 Mio », pas à « 1024,0 Kio ».
  while (value >= 1023.95 && unit + 1 < kUnits.size()) {
    value /= 1024.0;
    ++unit;
  }
  std::string number = std::format("{:.1f}", value);
  std::ranges::replace(number, '.', ',');
  return std::format("{} {}", number, kUnits[unit]);
}

}  // namespace ajp::app
