// Espaces de couleur : le contenu d'un slot du ring et la sortie décrite par DisplayInfo, dans
// le vocabulaire de core (frame.h, resolve.h) puis en pl_color_space.
#pragma once

#include <optional>

#include "core/frame.h"
#include "core/profile.h"
#include "core/resolve.h"

struct pl_color_space;

namespace ajp::render {

// Ce que contient un slot : du RGB plein, dans les primaires et le transfert de la source, ou
// en PQ BT.2020 au pic de l'écran quand TrueHDR l'a converti.
struct SlotColor {
  Primaries primaries = Primaries::Bt709;
  Transfer transfer = Transfer::Bt1886;
  std::optional<HdrMetadata> hdr;  // métadonnées de mastering d'une source HDR
};

// La couleur du slot pour une source et la route HDR du Plan : TrueHDR écrit du PQ BT.2020
// étiqueté au pic de l'écran, qui doit passer tel quel (hdr-grade.lua, « hdr-compute-peak »).
SlotColor slot_color(const ColorInfo& source, core::HdrMode route, double display_peak_nits);

// Source PQ ou HLG.
constexpr bool is_hdr(Transfer transfer) {
  return transfer == Transfer::Pq || transfer == Transfer::Hlg;
}

namespace detail {

// sdr_white_nits : où libplacebo place le blanc d'une source SDR (hdr-reference-white de mpv).
pl_color_space to_placebo(const SlotColor& color, double sdr_white_nits);
// PQ BT.2020 au pic de l'écran si HDR actif en PQ ; scRGB : linéaire étendu BT.709 ; sRGB sinon.
pl_color_space to_placebo(const core::DisplayInfo& display);

}  // namespace detail

}  // namespace ajp::render
