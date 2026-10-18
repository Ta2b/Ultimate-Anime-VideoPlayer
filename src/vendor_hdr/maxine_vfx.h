// Chargeur du SDK NVIDIA Maxine VFX 1.3 (Linux) : bibliothèques ouvertes par chemin, sans
// LD_LIBRARY_PATH (maxine_runtime.h), et effet TrueHDR à descripteurs NvCVImage persistants.
//
// Repris de animejanai-inference-rtx@118c272 src/pipeline_hdr.c:1-86 (adaptateur qualifié) et
// aji-rtx-linux/aji_rtx_vfx.cpp : même ordre d'appels, mêmes formats (RGBA 8 bits en entrée,
// RGB10A2 en sortie), flux 0 hérité, modèle chargé au premier appel. Le SDK garde le pointeur
// passé à NvVFX_SetImage : les descripteurs vivent donc dans l'effet, jamais sur la pile
// (docs/LINUX-INFERENCE.md du lecteur, « descripteurs NvCVImage locaux »).
//
// Le contexte CUDA du thread gpu doit être courant pendant open et process.
#pragma once

#include <expected>
#include <filesystem>
#include <memory>

#include "core/types.h"
#include "vendor_hdr/sdk_store.h"
#include "vendor_hdr/true_hdr.h"

namespace ajp::vendor_hdr {

// Refuse (LibraryUnavailable, VendorCallFailed) quand une bibliothèque, un symbole ou un appel
// du SDK manque ; rien n'est gardé ouvert après un refus.
std::expected<std::unique_ptr<TrueHdr>, Refusal> open_maxine(const InstalledSdk& sdk,
                                                             const TrueHdrConfig& config);

namespace detail {
// Même chargement depuis une racine que le magasin n'a pas vérifiée : tests du chargeur seulement.
std::expected<std::unique_ptr<TrueHdr>, Refusal> open_maxine_at(const std::filesystem::path& root,
                                                                const TrueHdrConfig& config);
}  // namespace detail

}  // namespace ajp::vendor_hdr
