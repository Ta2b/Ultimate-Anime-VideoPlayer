// Chargeur du NVIDIA RTX Video SDK 1.1 (Windows), chemin CUDA : la DLL du SDK s'ouvre par
// LoadLibraryExW depuis la copie du magasin, TrueHDR seul (sans Super Resolution) sur le flux 0.
// Le pont aji_rtx.dll du fork n'a plus de source : on appelle le SDK directement (plan,
// « Reprise du fork »). Prototypes et réserves dans rtx_video_abi.h.
//
// Le chargeur se compile partout ; hors Windows, la DLL ne s'ouvre pas et open refuse.
#pragma once

#include <expected>
#include <filesystem>
#include <memory>

#include "core/types.h"
#include "vendor_hdr/sdk_store.h"
#include "vendor_hdr/true_hdr.h"

namespace ajp::vendor_hdr {

// Refuse (LibraryUnavailable, VendorCallFailed) quand la DLL, un symbole ou la création
// manquent, et (InputMismatch) sans config.cuda_context. Une seule instance à la fois : l'API
// du SDK n'a pas de poignée, et la destruction d'une instance arrête le SDK pour tout le
// processus. gpu détruit donc l'ancienne avant d'ouvrir la suivante (changement de taille).
std::expected<std::unique_ptr<TrueHdr>, Refusal> open_rtx_video(const InstalledSdk& sdk,
                                                                const TrueHdrConfig& config);

namespace detail {
// Même chargement depuis une bibliothèque que le magasin n'a pas vérifiée : tests seulement.
std::expected<std::unique_ptr<TrueHdr>, Refusal> open_rtx_video_at(const std::filesystem::path& library,
                                                                   const TrueHdrConfig& config);
}  // namespace detail

}  // namespace ajp::vendor_hdr
