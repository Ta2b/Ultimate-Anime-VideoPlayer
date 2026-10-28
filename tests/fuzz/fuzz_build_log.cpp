// Journal de construction d'un moteur (*.build.log du cache, écrit par trtexec) : texte
// quelconque ; une construction lue a une taille et une durée plausibles.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include "app/play/build_progress.h"
#include "fuzz_check.h"
#include "fuzz_targets.h"

int ajp::fuzz::build_log(const std::uint8_t* data, std::size_t size) {
  const std::string_view text(reinterpret_cast<const char*>(data), size);
  const auto build = ajp::app::play::read_build_log(text);
  if (!build) return 0;
  ajp::fuzz::require(build->width > 0 && build->height > 0, "une taille lue est positive");
  ajp::fuzz::require(build->pixels == std::int64_t{build->width} * build->height,
                     "pixels = largeur × hauteur");
  ajp::fuzz::require(std::isfinite(build->seconds) && build->seconds > 0.0,
                     "une durée lue est finie et positive");
  return 0;
}
