// Le registre des graphes CUDA d'un moteur (chantier 0.4) : par jeu d'adresses, un lancement chaud, une
// capture, puis des rejeux. Pur (sans CUDA) pour se tester ; le moteur garde les graphes eux-mêmes, rangés
// par l'indice que next() rend. Le dépassement (plus de place pour un nouveau jeu d'adresses) et les échecs
// sont comptés et dits, au lieu d'un lancement direct muet.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ajp::gpu {

enum class GraphStep : std::uint8_t {
  Direct,    // graphes coupés : lancement direct
  Warm,      // premier lancement du jeu d'adresses (allocations hors capture)
  Capture,   // deuxième : capturer pendant le lancement
  Replay,    // le graphe existe : le rejouer
  Overflow,  // plus de place : lancement direct, compté
};

struct GraphCounters {
  std::uint64_t warm = 0;
  std::uint64_t captures = 0;
  std::uint64_t capture_failures = 0;
  std::uint64_t replays = 0;
  std::uint64_t overflow = 0;
  std::uint64_t launch_failures = 0;
  friend bool operator==(const GraphCounters&, const GraphCounters&) = default;
};

template <class Key>
class GraphBook {
 public:
  struct Pick {
    GraphStep step = GraphStep::Direct;
    std::size_t index = 0;  // l'emplacement du graphe, pour Warm, Capture et Replay
  };

  GraphBook(std::size_t capacity, int capture_attempts) : capacity_(capacity), attempts_(capture_attempts) {}

  Pick next(const Key& key) {
    if (off_) return {};
    auto found = std::ranges::find(entries_, key, &Entry::key);
    if (found == entries_.end()) {
      if (entries_.size() >= capacity_) {
        ++counters_.overflow;
        if (!overflow_said_) {
          overflow_said_ = true;
          notice_ = std::format("graphes CUDA : plus de place pour un {}e jeu d'adresses, lancements directs",
                                capacity_ + 1);
        }
        return {.step = GraphStep::Overflow};
      }
      entries_.push_back(Entry{.key = key});
      found = entries_.end() - 1;
    }
    const std::size_t index = static_cast<std::size_t>(found - entries_.begin());
    if (found->ready) {
      ++counters_.replays;
      return {.step = GraphStep::Replay, .index = index};
    }
    last_ = index;
    if (!found->warmed) {
      found->warmed = true;
      ++counters_.warm;
      return {.step = GraphStep::Warm, .index = index};
    }
    return {.step = GraphStep::Capture, .index = index};
  }

  // Le résultat de la capture que next() vient de demander. Après `attempts` refus, les graphes se coupent.
  void captured(bool ok, std::string_view reason = {}) {
    if (ok) {
      ++counters_.captures;
      entries_[last_].ready = true;
      return;
    }
    ++counters_.capture_failures;
    if (static_cast<int>(counters_.capture_failures) >= attempts_) {
      off_ = true;
      notice_ = std::format("graphes CUDA coupés : capture refusée {} fois, dernière erreur : {}",
                            counters_.capture_failures, reason);
    }
  }

  // Un rejeu ou un lancement qui échoue : compté (l'appelant refuse l'image).
  void launch_failed() { ++counters_.launch_failures; }

  bool off() const { return off_; }
  const GraphCounters& counters() const { return counters_; }
  std::size_t size() const { return entries_.size(); }
  // Le mot pour le journal du backend, une fois ; vide sinon.
  std::string take_notice() { return std::exchange(notice_, std::string()); }
  // Oublie tous les graphes (les adresses ont changé) ; les compteurs restent.
  void clear() {
    entries_.clear();
    overflow_said_ = false;
  }

 private:
  struct Entry {
    Key key;
    bool warmed = false;
    bool ready = false;
  };
  std::size_t capacity_;
  int attempts_;
  std::vector<Entry> entries_;
  std::size_t last_ = 0;
  GraphCounters counters_;
  bool off_ = false;
  bool overflow_said_ = false;
  std::string notice_;
};

}  // namespace ajp::gpu
