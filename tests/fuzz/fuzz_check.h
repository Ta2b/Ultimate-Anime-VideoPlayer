// Ce qu'une cible de fuzz vérifie au-delà de « ni plantage ni comportement indéfini » : une
// propriété fausse arrête le processus, comme un rapport d'ASan, et le pilote de rejeu dit quelle
// entrée la casse.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <source_location>

namespace ajp::fuzz {

inline void require(bool ok, const char* what, std::source_location where = std::source_location::current()) {
  if (ok) return;
  std::fprintf(stderr, "fuzz : propriété fausse : %s (%s:%u)\n", what, where.file_name(),
               static_cast<unsigned>(where.line()));
  std::abort();
}

}  // namespace ajp::fuzz
