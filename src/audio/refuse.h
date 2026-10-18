// Refus nommés du module audio. En-tête interne, inclus par les .cpp seulement.
#pragma once

#include <expected>
#include <string>
#include <utility>

#include "core/types.h"

namespace ajp::audio {

inline std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

}  // namespace ajp::audio
