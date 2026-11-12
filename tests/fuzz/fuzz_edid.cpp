// EDID de l'écran (lu par le pilote, tel que l'écran le donne) : octets quelconques, blocs
// d'extension CTA-861 et DisplayID compris, lus sans plantage ni lecture hors du tampon.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "app/play/display_edid.h"
#include "fuzz_check.h"
#include "fuzz_targets.h"

int ajp::fuzz::edid(const std::uint8_t* data, std::size_t size) {
  // Une copie à la taille exacte : ASan voit toute lecture au-delà.
  const std::vector<std::uint8_t> edid(data, data + size);
  const ajp::app::play::DisplayFacts facts = ajp::app::play::parse_edid(edid);
  // Ce pic règle le grade HDR : fini et borné (le plus haut octet de CTA-861 donne 12 540).
  if (facts.peak_nits) {
    ajp::fuzz::require(std::isfinite(*facts.peak_nits), "une luminance de crête lue est finie");
    ajp::fuzz::require(*facts.peak_nits > 0.0 && *facts.peak_nits <= 12'600.0,
                       "une luminance de crête lue est positive et bornée");
  }
  return 0;
}
