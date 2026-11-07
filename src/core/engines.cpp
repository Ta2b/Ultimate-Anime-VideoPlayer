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
    // MIGraphX et OpenVINO ne sont pas proposés (engine_plugged) : AMD et Intel passent par Vulkan.
    // TensorRT seulement sur une RTX (P3.5) : ailleurs, ni case ni écran Moteurs, Vulkan seul.
    if (rtx) offers.push_back(vendor_engine(EngineId::TensorRt, Badge::NvidiaOnly, rtx));
  }

  // Pas d'offre DirectML : aucun moteur DirectML n'existe dans cette version (les anciens AMD
  // restent sur Vulkan).

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
