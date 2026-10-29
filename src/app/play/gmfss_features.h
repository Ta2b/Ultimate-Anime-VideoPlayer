// GMFSS fusionné (prepare/gmfss_split.h) : de quelle image sont les caractéristiques que chaque tranche
// garde. Une paire (a, b) lit les caractéristiques des deux images ; celles de b servent encore à la paire
// suivante (b, c), où elles sont celles du côté a : une image dont la tranche porte déjà les caractéristiques
// n'est pas refaite. Une image se reconnaît à son identité et à son époque (un seek, une coupe, une image
// sautée ne laissent jamais une tranche passer pour une autre image).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

#include "core/types.h"

namespace ajp::app::play {

struct GmfssFrameKey {
  SourceId id;
  std::uint32_t epoch = 0;
  friend bool operator==(const GmfssFrameKey&, const GmfssFrameKey&) = default;
};

class GmfssFeatureSlots {
 public:
  // La tranche slot porte-t-elle déjà les caractéristiques de cette image ? Sinon elle est tenue pour vide :
  // fill dit quand le moteur a fini d'y écrire.
  bool reuse(std::size_t slot, const GmfssFrameKey& key) {
    if (of_[slot] == key) {
      ++reused_;
      return true;
    }
    of_[slot].reset();
    return false;
  }
  void fill(std::size_t slot, const GmfssFrameKey& key) {
    of_[slot] = key;
    ++computed_;
  }
  // Tout oublier (les tampons changent de mains).
  void clear() { of_ = {}; }

  std::uint64_t computed() const { return computed_; }
  std::uint64_t reused() const { return reused_; }

 private:
  std::array<std::optional<GmfssFrameKey>, 2> of_;
  std::uint64_t computed_ = 0;
  std::uint64_t reused_ = 0;
};

}  // namespace ajp::app::play
