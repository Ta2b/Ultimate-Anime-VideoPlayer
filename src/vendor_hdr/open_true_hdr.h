// Point d'entrée de gpu : ouvre TrueHDR depuis le SDK que le magasin vient de vérifier.
//
//   SdkStore store(data, *sdk_for(os));
//   SdkStatus sdk = store.status();          // revérifie la copie juste avant de charger
//   TrueHdrStage stage(sdk.installed ? open_true_hdr(*sdk.installed, config)
//                                    : std::unexpected(Refusal{RefusalCode::HdrUnavailable, sdk.reason}));
//
// Pour une autre taille, l'étage précédent est détruit avant d'ouvrir le suivant : RTX Video
// n'admet qu'une instance par processus (rtx_video.h).
#pragma once

#include <expected>
#include <memory>

#include "core/types.h"
#include "vendor_hdr/sdk_store.h"
#include "vendor_hdr/true_hdr.h"

namespace ajp::vendor_hdr {

// RTX Video ou Maxine selon sdk.kind().
std::expected<std::unique_ptr<TrueHdr>, Refusal> open_true_hdr(const InstalledSdk& sdk,
                                                               const TrueHdrConfig& config);

}  // namespace ajp::vendor_hdr
