// Le registre des présentations en vol et de leur retour (chantier 6.3). Pur : aucune E/S, aucune horloge
// propre ; placebo_feedback.cpp le nourrit de ce que Vulkan relit.
//
// Chaque présentation reçoit un identifiant croissant (VkPresentId2KHR). Le retour (date où l'image est
// devenue visible) arrive plus tard, dans le désordre et parfois en partie : chaque identifiant n'est
// rendu qu'une fois, avec la date visée et l'heure de soumission gardées à l'envoi. La file de retour du
// pilote est bornée : quand elle déborde (VK_ERROR_PRESENT_TIMING_QUEUE_FULL_EXT), les demandes de date
// sont suspendues jusqu'au retour suivant. Une présentation restée sans retour derrière kLost plus
// récentes est perdue.
//
// Le domaine de temps du retour : CLOCK_MONOTONIC (celui de host_now_us) tel quel, sinon un domaine
// propre à la swapchain recalé sur l'hôte toutes les kRecalibrateUs (TimeDomainMap).
#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>

namespace ajp::core {

struct PresentSample {
  std::uint64_t id = 0;
  std::int64_t submit_us = 0;   // heure hôte de la soumission
  std::int64_t target_us = 0;   // date visée, heure hôte (0 : aucune)
  std::int64_t visible_us = 0;  // date où l'image est apparue, heure hôte
};

class PresentLedger {
 public:
  static constexpr std::size_t kLost = 64;

  struct Issue {
    std::uint64_t id = 0;
    bool timing = false;  // demander la date de cette présentation
  };

  // Une présentation part : son identifiant, et s'il faut demander sa date (wanted, sauf file suspendue).
  // Seules les présentations datées attendent un retour.
  Issue issue(std::int64_t submit_us, std::int64_t target_us, bool wanted = true);
  // La dernière présentation émise n'a pas eu de place dans la file de retour : plus de demande jusqu'au
  // retour suivant (elle-même n'en aura pas).
  void queue_full();
  // Le retour d'une présentation ; rien pour un identifiant inconnu, déjà rendu ou perdu.
  std::optional<PresentSample> report(std::uint64_t id, std::int64_t visible_us);
  // La plus ancienne présentation en vol (niveau Wait : celle à sonder), 0 s'il n'y en a pas.
  std::uint64_t oldest() const { return pending_.empty() ? 0 : pending_.front().id; }
  // Une nouvelle swapchain : les présentations en vol sur l'ancienne ne reviendront pas.
  void forget();

  bool suspended() const { return suspended_; }
  std::uint64_t issued() const { return next_id_ - 1; }
  std::uint64_t returned() const { return returned_; }
  std::uint64_t lost() const { return lost_; }        // datées restées sans retour
  std::uint64_t untimed() const { return untimed_; }  // présentations parties sans demande de date
  std::size_t in_flight() const { return pending_.size(); }

 private:
  struct Pending {
    std::uint64_t id;
    std::int64_t submit_us;
    std::int64_t target_us;
  };
  std::deque<Pending> pending_;
  std::uint64_t next_id_ = 1;
  std::uint64_t returned_ = 0;
  std::uint64_t lost_ = 0;
  std::uint64_t untimed_ = 0;
  bool suspended_ = false;
};

// L'écart entre un domaine de temps de la swapchain et l'heure hôte, relevé par des horodatages calibrés.
class TimeDomainMap {
 public:
  static constexpr std::int64_t kRecalibrateUs = 2'000'000;

  // Domaine CLOCK_MONOTONIC : aucun écart, jamais de recalage.
  void set_monotonic() {
    monotonic_ = true;
    known_ = true;
    offset_ns_ = 0;
  }
  // Un relevé simultané des deux horloges (en ns), à l'heure hôte now_us.
  void calibrate(std::uint64_t domain_ns, std::uint64_t monotonic_ns, std::int64_t now_us);
  bool due(std::int64_t now_us) const {
    return !monotonic_ && (!known_ || now_us - taken_us_ >= kRecalibrateUs);
  }
  bool known() const { return known_; }
  // Une date du domaine (ns) en heure hôte (µs).
  std::int64_t to_host_us(std::uint64_t domain_ns) const;
  // Une heure hôte (µs) en date du domaine (ns) ; 0 pour 0.
  std::uint64_t to_domain_ns(std::int64_t host_us) const;
  void reset() { *this = TimeDomainMap{}; }

 private:
  std::int64_t offset_ns_ = 0;  // monotone − domaine
  std::int64_t taken_us_ = 0;
  bool monotonic_ = false;
  bool known_ = false;
};

}  // namespace ajp::core
