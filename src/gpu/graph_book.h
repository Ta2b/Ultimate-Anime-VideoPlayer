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

  // Une place au moins : un registre vide n'aurait aucune entrée à chasser (next() la déréférence).
  GraphBook(std::size_t capacity, int capture_attempts)
      : capacity_(std::max<std::size_t>(capacity, 1)), attempts_(capture_attempts) {}

  // Graphes coupés pour ce lancement : l'appelant lance en direct sans tirer de next(). Après
  // kRetryAfterDirect lancements directs, les graphes se réessaient (C21-07) : une cause passagère (mémoire
  // basse au début de la lecture) ne les coupe plus pour toute la séance.
  bool skip() { return still_off(); }

  Pick next(const Key& key) {
    if (still_off()) return {};
    ++tick_;
    auto found = std::ranges::find(entries_, key, &Entry::key);
    if (found == entries_.end()) {
      if (entries_.size() >= capacity_) {
        // Plein : une entrée qui n'a plus servi depuis kStaleAfter × capacité tirages (adresses d'une
        // génération d'anneau passée) laisse sa place ; un jeu d'adresses vivant n'est jamais chassé, un
        // cycle plus large que le registre resterait sinon en captures sans fin (A21-08).
        const auto oldest = std::ranges::min_element(entries_, {}, &Entry::used);
        if (tick_ - oldest->used <= kStaleAfter * capacity_) {
          ++counters_.overflow;
          if (!overflow_said_) {
            overflow_said_ = true;
            notice_ = std::format(
                "graphes CUDA : plus de place pour un {}e jeu d'adresses, lancements directs", capacity_ + 1);
          }
          return {.step = GraphStep::Overflow};
        }
        // L'appelant détruit le graphe de cet emplacement à la capture qui suit (Capture sur un index
        // qui en a déjà un).
        *oldest = Entry{.key = key};
        found = oldest;
      } else {
        entries_.push_back(Entry{.key = key});
        found = entries_.end() - 1;
      }
    }
    found->used = tick_;
    const std::size_t index = static_cast<std::size_t>(found - entries_.begin());
    if (found->ready) {
      ++counters_.replays;
      // Assez de rejeux de suite sans refus : les échecs passés (un retour de veille, des heures plus tôt)
      // ne comptent plus pour couper les graphes (A21-08).
      if (++healthy_ >= kHealAfterReplays) {
        capture_streak_ = 0;
        launch_streak_ = 0;
      }
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

  // Le résultat de la capture que next() vient de demander. Après `attempts` refus de suite, les graphes se
  // coupent ; une capture réussie remet la série à zéro (des refus isolés, chacun suivi d'une réussite, ne
  // coupent rien).
  void captured(bool ok, std::string_view reason = {}) {
    if (ok) {
      ++counters_.captures;
      capture_streak_ = 0;
      if (last_ < entries_.size()) entries_[last_].ready = true;
      return;
    }
    ++counters_.capture_failures;
    healthy_ = 0;
    if (++capture_streak_ >= attempts_) {
      off_ = true;
      notice_ = std::format("graphes CUDA coupés : capture refusée {} fois, dernière erreur : {}",
                            capture_streak_, reason);
    }
  }

  // Un rejeu ou un lancement qui échoue sans erreur collante : compté ; le graphe de cet emplacement est
  // oublié (l'appelant le détruit et lance en direct), il se recapture au passage suivant. Après
  // kMaxLaunchFailures échecs sans kHealAfterReplays rejeux réussis entre eux, les graphes se coupent avec
  // une notice.
  static constexpr int kMaxLaunchFailures = 3;
  static constexpr std::uint64_t kHealAfterReplays = 1000;  // une quarantaine de secondes à 24 i/s
  static constexpr std::uint64_t kStaleAfter = 64;          // tirages par place du registre
  static constexpr std::uint64_t kRetryAfterDirect = 2000;  // près d'une minute et demie à 24 i/s
  void launch_failed(std::size_t index) {
    ++counters_.launch_failures;
    healthy_ = 0;
    if (index < entries_.size()) entries_[index].ready = false;
    if (++launch_streak_ >= kMaxLaunchFailures && !off_) {
      off_ = true;
      notice_ = std::format("graphes CUDA coupés : rejeu refusé {} fois, lancements directs", launch_streak_);
    }
  }

  bool off() const { return off_; }
  const GraphCounters& counters() const { return counters_; }
  std::size_t size() const { return entries_.size(); }
  // Le mot pour le journal du backend, une fois ; vide sinon.
  std::string take_notice() { return std::exchange(notice_, std::string()); }
  // Oublie tous les graphes (les adresses ont changé) ; les compteurs restent, les séries d'échecs repartent
  // de zéro et des graphes coupés se réessaient (A21-08).
  void clear() {
    entries_.clear();
    overflow_said_ = false;
    capture_streak_ = 0;
    launch_streak_ = 0;
    healthy_ = 0;
    off_ = false;
    direct_ = 0;
  }

 private:
  bool still_off() {
    if (!off_) return false;
    if (++direct_ < kRetryAfterDirect) return true;
    off_ = false;
    direct_ = 0;
    capture_streak_ = 0;
    launch_streak_ = 0;
    healthy_ = 0;
    notice_ = std::format("graphes CUDA réessayés après {} lancements directs", kRetryAfterDirect);
    return false;
  }

  struct Entry {
    Key key;
    bool warmed = false;
    bool ready = false;
    std::uint64_t used = 0;  // tick_ du dernier tirage
  };
  std::size_t capacity_;
  int attempts_;
  std::vector<Entry> entries_;
  std::size_t last_ = 0;
  GraphCounters counters_;
  bool off_ = false;
  bool overflow_said_ = false;
  std::uint64_t tick_ = 0;
  std::uint64_t healthy_ = 0;  // rejeux de suite sans refus
  int capture_streak_ = 0;     // refus de capture de suite
  int launch_streak_ = 0;      // refus de rejeu depuis la dernière remise à zéro
  std::uint64_t direct_ = 0;   // lancements directs depuis que les graphes sont coupés
  std::string notice_;
};

}  // namespace ajp::gpu
