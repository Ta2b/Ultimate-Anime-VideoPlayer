// Voies de calcul d'un backend (étude 2 des optimisations GPU) : la mesure, l'agrandissement et
// l'interpolation enfilées chacune sur sa file, qui se recouvrent sur l'appareil.
//
// Backend::use_lane choisit la voie des appels suivants du même thread (enqueue, noyaux,
// write_ring). L'appelant ne pose aucune dépendance : le backend la déduit des ressources que
// chaque opération lit et écrit (LaneHazards). Un mauvais choix de voie coûte du recouvrement,
// jamais un résultat faux.
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <unordered_map>

namespace ajp::gpu {

// Mesure : ingestion, mesures de la paire, relectures. Agrandissement : SPAN et l'écriture de son
// slot. Interpolation : RIFE, DistilDRBA, warp et l'écriture des images intermédiaires.
enum class Lane : std::uint8_t { Measure, Upscale, Interpolate };
inline constexpr std::size_t kLanes = 3;

constexpr std::size_t lane_index(Lane lane) {
  return static_cast<std::size_t>(lane);
}

// Le rang de priorité d'une voie, 0 la plus pressée. La mesure d'abord : ses noyaux font 0,02 à 0,3 ms et
// l'hôte attend ses relectures pour décider de la paire ; à la plus basse priorité, elle attendait que SPAN
// rende les SM (p95 de l'attente hôte : 14 ms, la durée d'un SPAN). Puis l'agrandissement, qui porte
// l'échéance d'affichage, puis l'interpolation.
constexpr int lane_rank(Lane lane) {
  switch (lane) {
    case Lane::Measure:
      return 0;
    case Lane::Upscale:
      return 1;
    case Lane::Interpolate:
      return 2;
  }
  return 2;
}

// La priorité de file de la voie dans la plage d'un appareil, least (la plus basse) et greatest (la plus
// haute), comme CUDA les donne : un nombre plus petit est plus prioritaire. Une plage trop courte serre
// les rangs vers le bas.
constexpr int lane_priority(Lane lane, int least, int greatest) {
  const int step = greatest <= least ? 1 : -1;
  const int priority = greatest + step * lane_rank(lane);
  return greatest <= least ? std::min(priority, least) : std::max(priority, least);
}

// Les dépendances entre voies. Pour chaque ressource (une adresse : tenseur, zone d'activation,
// contexte d'un moteur, sommes d'un noyau), la dernière écriture et les lectures faites depuis,
// par voie, chacune datée par le rang de son opération sur sa voie. Une opération attend, sur
// les autres voies, la dernière écriture de ce qu'elle lit ; si elle écrit, aussi les lectures
// faites depuis cette écriture. Sans GPU : le backend fait d'une attente (voie, rang) l'attente
// de l'événement posé après cette opération sur la file de la voie. Un seul thread l'appelle.
class LaneHazards {
 public:
  struct Wait {
    Lane lane = Lane::Measure;
    std::uint64_t mark = 0;  // rang de l'opération attendue sur lane, 1 pour la première
    friend bool operator==(const Wait&, const Wait&) = default;
  };
  // Au plus une attente par autre voie : kLanes - 1 places sans allocation (C21-06, un accès par opération
  // de chaque image, sur le thread qui enfile le GPU). S'itère comme un vecteur.
  class Waits {
   public:
    Waits() = default;
    Waits(std::initializer_list<Wait> waits) {
      for (const Wait& wait : waits) push_back(wait);
    }
    void push_back(const Wait& wait) { items_.at(count_++) = wait; }
    const Wait* begin() const { return items_.data(); }
    const Wait* end() const { return items_.data() + count_; }
    bool empty() const { return count_ == 0; }
    std::size_t size() const { return count_; }
    friend bool operator==(const Waits& a, const Waits& b) { return std::ranges::equal(a, b); }

   private:
    std::array<Wait, kLanes - 1> items_{};
    std::size_t count_ = 0;
  };
  struct Op {
    std::uint64_t mark = 0;  // rang de cette opération sur sa voie
    Waits waits;             // au plus une par autre voie, la plus récente
  };

  // L'opération suivante de lane, qui lit reads et écrit writes (une ressource lue et écrite va
  // dans les deux) : son rang, et les attentes à poser avant elle. Une attente déjà posée sur la
  // file de lane (un rang au moins aussi récent) n'est pas répétée.
  Op access(Lane lane, std::span<const std::uintptr_t> reads, std::span<const std::uintptr_t> writes) {
    const std::size_t self = lane_index(lane);
    std::array<std::uint64_t, kLanes> need{};
    for (const std::uintptr_t resource : reads) {
      const auto found = states_.find(resource);
      if (found != states_.end()) need_write(found->second, self, need);
    }
    for (const std::uintptr_t resource : writes) {
      const auto found = states_.find(resource);
      if (found == states_.end()) continue;
      need_write(found->second, self, need);
      for (std::size_t other = 0; other < kLanes; ++other) {
        if (other != self) need[other] = std::max(need[other], found->second.read[other]);
      }
    }
    Op op = settle(self, need);
    for (const std::uintptr_t resource : reads) states_[resource].read[self] = op.mark;
    for (const std::uintptr_t resource : writes) {
      State& state = states_[resource];
      state.writer = self;
      state.written = op.mark;
      state.read = {};
    }
    if (states_.size() > prune_above_) prune();
    return op;
  }

  // Une opération de lane qui suit tout le travail déjà enfilé sur les autres voies : une
  // jointure (signal(), la fin d'une génération du ring, la destruction).
  Op join(Lane lane) {
    const std::size_t self = lane_index(lane);
    std::array<std::uint64_t, kLanes> need = marks_;
    need[self] = 0;
    Op op = settle(self, need);
    prune();
    return op;
  }

  // Une mémoire rendue (libérée derrière toutes les voies) : son adresse n'a plus d'histoire. Reprise par
  // un autre tampon, elle n'hérite pas des lectures de l'ancien (A21-09).
  void forget(std::uintptr_t resource) { states_.erase(resource); }

  std::size_t tracked() const { return states_.size(); }  // adresses suivies (tests)

  // Rang de la dernière opération de lane ; 0 si aucune.
  std::uint64_t last(Lane lane) const { return marks_[lane_index(lane)]; }

 private:
  struct State {
    std::size_t writer = 0;
    std::uint64_t written = 0;                 // rang de la dernière écriture ; 0 : jamais écrite
    std::array<std::uint64_t, kLanes> read{};  // dernière lecture par voie depuis l'écriture
  };

  static void need_write(const State& state, std::size_t self, std::array<std::uint64_t, kLanes>& need) {
    if (state.written != 0 && state.writer != self)
      need[state.writer] = std::max(need[state.writer], state.written);
  }

  // Une adresse dont chaque accès est déjà attendu par toutes les autres voies n'imposera plus aucune
  // attente : l'oublier ne change aucun résultat. Sans cela la table gardait toutes les adresses vues
  // (tampons libérés, anciennes générations de l'anneau) pour toute la vie du backend (A21-09).
  bool settled(const State& state) const {
    for (std::size_t other = 0; other < kLanes; ++other) {
      if (state.written != 0 && other != state.writer && waited_[other][state.writer] < state.written)
        return false;
      for (std::size_t reader = 0; reader < kLanes; ++reader) {
        if (other != reader && waited_[other][reader] < state.read[reader]) return false;
      }
    }
    return true;
  }
  void prune() {
    std::erase_if(states_, [this](const auto& entry) { return settled(entry.second); });
    // Ce qui reste est encore utile : le seuil suit, pour ne pas tout reparcourir à chaque accès.
    prune_above_ = std::max(kPruneAbove, states_.size() * 2);
  }

  Op settle(std::size_t self, const std::array<std::uint64_t, kLanes>& need) {
    Op op{.mark = ++marks_[self], .waits = {}};
    for (std::size_t other = 0; other < kLanes; ++other) {
      if (other == self || need[other] <= waited_[self][other]) continue;
      op.waits.push_back(Wait{.lane = static_cast<Lane>(other), .mark = need[other]});
      waited_[self][other] = need[other];
    }
    return op;
  }

  std::array<std::uint64_t, kLanes> marks_{};
  // waited_[a][b] : rang de b que la file de a attend déjà.
  std::array<std::array<std::uint64_t, kLanes>, kLanes> waited_{};
  // Les adresses que le backend a vues et qui peuvent encore imposer une attente (prune) ; une mémoire
  // libérée puis rendue à la même adresse ne coûte que des attentes de trop.
  std::unordered_map<std::uintptr_t, State> states_;
  static constexpr std::size_t kPruneAbove = 256;
  std::size_t prune_above_ = kPruneAbove;
};

}  // namespace ajp::gpu
