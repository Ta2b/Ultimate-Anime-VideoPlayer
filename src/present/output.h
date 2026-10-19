// Espace de sortie de la swapchain : ce qu'on demande, ce qu'on obtient, et ce qu'il en coûte.
// libplacebo ne répond pas à pl_swapchain_colorspace_hint : seul l'espace de l'image suivante
// (pl_swapchain_frame::color_space) dit ce que le compositeur a accepté.
//
// HDR actif : on demande HDR10 (PQ, BT.2020) avec max_luma égal au pic du grade. Sans PQ, repli
// scRGB (linéaire étendu, BT.709) et grade coupé, comme le veut le plan ; sans scRGB non plus,
// SDR. L'OSD affiche ce qui a été perdu.
#pragma once

#include <cstdint>
#include <optional>

#include "core/resolve.h"

namespace ajp::present {

// Ce qu'on demande à la swapchain.
enum class OutputHint : std::uint8_t { Sdr, Pq, ScRgb };

// La courbe de l'image que la swapchain a rendue.
enum class SurfaceTransfer : std::uint8_t { Pq, Linear, Other };

struct OutputState {
  core::OutputSpace space = core::OutputSpace::Srgb;
  bool grade = true;           // false : repli scRGB, le grade est coupé
  bool hdr_requested = false;  // DisplayInfo::hdr au moment de la demande
  bool hdr_lost = false;       // HDR demandé, PQ non obtenu
  friend bool operator==(const OutputState&, const OutputState&) = default;
};

// Ce qu'on demande d'abord pour cet écran.
OutputHint first_hint(const core::DisplayInfo& display);

OutputState classify_output(OutputHint requested, SurfaceTransfer obtained);

// Ce qu'il faut demander ensuite quand la demande n'a pas été tenue : scRGB après PQ, rien de
// plus après scRGB (on reste en SDR).
std::optional<OutputHint> fallback_hint(OutputHint requested, SurfaceTransfer obtained);

}  // namespace ajp::present
