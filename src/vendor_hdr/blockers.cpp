#include "vendor_hdr/blockers.h"

#include <format>
#include <utility>

namespace ajp::vendor_hdr {

std::optional<Blocked> first_blocker(const TrueHdrEnvironment& environment) {
  const std::optional<SdkKind> sdk = sdk_for(environment.os);
  if (!sdk) {
    return Blocked{TrueHdrBlocker::UnsupportedOs,
                   "TrueHDR n'existe pas sous macOS : la route native convertit en HDR."};
  }
  if (!environment.nvidia_gpu) {
    return Blocked{TrueHdrBlocker::NotNvidiaGpu, "TrueHDR demande un GPU NVIDIA RTX."};
  }
  if (!environment.tensorrt_engine) {
    return Blocked{TrueHdrBlocker::EngineNotTensorRt,
                   "TrueHDR demande le moteur TensorRT (Paramètres > Moteurs)."};
  }
  if (!environment.sdk_ready) {
    std::string reason = environment.sdk_reason.empty()
                             ? std::format("Importer le {} (Paramètres > Moteurs).", display_name(*sdk))
                             : environment.sdk_reason;
    return Blocked{TrueHdrBlocker::SdkMissing, std::move(reason)};
  }
  if (!environment.module_present) {
    return Blocked{TrueHdrBlocker::ModuleMissing,
                   "Le moteur TensorRT n'est pas installé : l'ajouter dans Paramètres > Moteurs."};
  }
  if (!environment.adapter_present) {
    return Blocked{TrueHdrBlocker::AdapterMissing,
                   "Ce moteur TensorRT n'a pas l'adaptateur TrueHDR : mettre à jour le lecteur."};
  }
  if (!environment.output_pq) {
    return Blocked{TrueHdrBlocker::OutputNotPq,
                   "TrueHDR n'agit qu'en sortie HDR PQ : activer le HDR de l'écran."};
  }
  return std::nullopt;
}

Refusal to_refusal(const Blocked& blocked) {
  return Refusal{RefusalCode::HdrUnavailable, blocked.reason};
}

std::string first_blocker_reason(const TrueHdrEnvironment& environment) {
  std::optional<Blocked> blocked = first_blocker(environment);
  return blocked ? std::move(blocked->reason) : std::string();
}

}  // namespace ajp::vendor_hdr
