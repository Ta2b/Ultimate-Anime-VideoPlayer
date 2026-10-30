// Les looks maison : des LUT 3D produites par le calcul (aucun fichier, aucune licence), au domaine d'une LUT
// .cube ordinaire (image SDR BT.709 encodée en gamma, blanc = 1), et le repérage des LUT d'HDR.
#pragma once

#include <span>
#include <string>
#include <string_view>

#include "render/cube_lut.h"

namespace ajp::render {

struct BuiltinLook {
  std::string_view key;    // « maison:vif » dans play.toml
  std::string_view label;  // anglais ; la traduction est dans l'interface
};

inline constexpr std::string_view kBuiltinPrefix = "maison:";

std::span<const BuiltinLook> builtin_looks();

// La LUT d'un look maison ; refus si la clé est inconnue.
std::expected<CubeLut, Refusal> builtin_look(std::string_view key, int size = 33);

// Une LUT dont le nom ou le titre dit « pq », « hdr » ou « 2020 » attend du PQ BT.2020 (étalonnée pour l'HDR)
// ; toute autre est lue comme une LUT SDR BT.709.
bool look_is_pq(std::string_view name, std::string_view title);

}  // namespace ajp::render
