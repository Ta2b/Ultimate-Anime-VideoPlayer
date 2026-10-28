// Raccourcis du module pour construire un refus nommé.
#pragma once

#include <expected>
#include <string>
#include <utility>

#include "core/types.h"

namespace ajp::vendor_hdr {

// Résultat d'une opération qui ne rend rien d'autre qu'un succès ou un refus.
using Done = std::expected<void, Refusal>;

inline std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

}  // namespace ajp::vendor_hdr
