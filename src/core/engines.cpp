#include "core/engines.h"

namespace ajp::core {
namespace {

// Moteur d'inférence propre à un fabricant : badge, et la condition qui le rend utile.
EngineOffer vendor_engine(EngineId id, Badge only, bool fits) {
  EngineOffer offer{.id = id,
                    .delivery = Delivery::Download,
                    .badges = {only},
                    .reason = fits ? Reason::FastestForThisGpu : Reason::NoMatchingGpu,
                    .relevant = fits,
                    .preselected = fits};
  if (fits) offer.badges.insert(offer.badges.begin(), Badge::Recommended);
  return offer;
}

EngineOffer hdr_sdk(EngineId id, Badge os_badge, bool fits) {
  EngineOffer offer{.id = id,
                    .delivery = Delivery::Import,
                    .badges = {Badge::NvidiaOnly, os_badge},
                    .reason = fits ? Reason::RtxHdrForThisGpu : Reason::NoMatchingGpu,
                    .relevant = fits,
                    .preselected = fits};
  if (fits) offer.badges.insert(offer.badges.begin(), Badge::Recommended);
  return offer;
}

}  // namespace

std::vector<EngineOffer> recommend_engines(const SystemInfo& system) {
  const GpuInfo none;
  const GpuInfo& gpu = system.gpus.empty() ? none : system.gpus.front();
  const bool rtx = gpu.vendor == GpuVendor::Nvidia && gpu.tensor_cores;
  const bool mac = system.os == Os::MacOs;

  std::vector<EngineOffer> offers;
  offers.push_back(EngineOffer{.id = EngineId::Vulkan,
                               .delivery = Delivery::Bundled,
                               .badges = {},
                               .reason = Reason::AlwaysIncluded,
                               .relevant = true,
                               .preselected = true});

  if (mac) {
    offers.push_back(EngineOffer{.id = EngineId::Metal,
                                 .delivery = Delivery::Bundled,
                                 .badges = {Badge::Recommended, Badge::AppleOnly},
                                 .reason = Reason::AlwaysIncluded,
                                 .relevant = true,
                                 .preselected = true});
  } else {
    offers.push_back(vendor_engine(EngineId::TensorRt, Badge::NvidiaOnly, rtx));
    if (system.os == Os::Windows) {
      offers.push_back(hdr_sdk(EngineId::RtxVideoSdk, Badge::WindowsOnly, rtx));
    } else {
      offers.push_back(hdr_sdk(EngineId::MaxineVfx, Badge::LinuxOnly, rtx));
    }
    if (system.os == Os::Windows) {
      // MIGraphX de Windows ML (sans ROCm), à l'essai sur les AMD RDNA 3 et après : proposé, jamais
      // coché d'office, à comparer avec Vulkan.
      const bool fits = gpu.vendor == GpuVendor::Amd && gpu.winml_migraphx;
      offers.push_back(EngineOffer{.id = EngineId::MiGraphX,
                                   .delivery = Delivery::Download,
                                   .badges = {Badge::AmdOnly},
                                   .reason = fits ? Reason::CompareWithVulkan : Reason::NoMatchingGpu,
                                   .relevant = fits,
                                   .preselected = false});
    } else {
      offers.push_back(vendor_engine(EngineId::MiGraphX, Badge::AmdOnly,
                                     gpu.vendor == GpuVendor::Amd && gpu.rocm_supported));
    }
    offers.push_back(vendor_engine(EngineId::OpenVino, Badge::IntelOnly,
                                   gpu.vendor == GpuVendor::Intel && gpu.openvino_gpu));
  }

  if (system.os == Os::Windows) {
    // Seulement pour les anciens AMD sans MIGraphX de Windows ML, et seulement si Vulkan fait
    // moins bien : proposé, jamais coché d'office, en attendant la mesure.
    const bool old_amd = gpu.vendor == GpuVendor::Amd && !gpu.winml_migraphx;
    offers.push_back(EngineOffer{.id = EngineId::DirectMl,
                                 .delivery = Delivery::Download,
                                 .badges = {Badge::Compatibility, Badge::WindowsOnly},
                                 .reason = old_amd ? Reason::CompareWithVulkan : Reason::NoMatchingGpu,
                                 .relevant = old_amd,
                                 .preselected = false});
  }

  // Vulkan est recommandé quand aucun moteur plus rapide ne convient à ce GPU.
  bool faster_engine = false;
  for (const EngineOffer& offer : offers) {
    faster_engine |= offer.id != EngineId::Vulkan && offer.reason == Reason::FastestForThisGpu;
    faster_engine |= offer.id == EngineId::Metal;
  }
  offers.front().badges = {faster_engine ? Badge::Compatibility : Badge::Recommended};
  return offers;
}

}  // namespace ajp::core
