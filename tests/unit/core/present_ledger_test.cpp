// Le registre du retour de présentation (core/present_ledger.h), avec un pilote simulé : file de retour
// bornée, dates rendues en retard et dans le désordre.
#include "core/present_ledger.h"

#include <cstdint>
#include <deque>
#include <random>
#include <set>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "core/vblank_clock.h"

using Catch::Approx;
using namespace ajp::core;

namespace {

// Le pilote : chaque présentation datée paraît au vblank suivant sa soumission ; sa date se relit deux
// présentations plus tard, au plus `queue` en attente à la fois.
struct Driver {
  std::size_t queue = 16;
  std::deque<std::pair<std::uint64_t, std::int64_t>> waiting;  // id, date d'affichage
  bool full() const { return waiting.size() >= queue; }
};

}  // namespace

TEST_CASE("retour de présentation : 600 images, chacune revient avec son vblank", "[core][present]") {
  const double period = 1e6 / 164.78;
  std::mt19937 rng(11);
  std::uniform_real_distribution<double> jitter(-300.0, 300.0);
  PresentLedger ledger;
  VblankClock clock;
  clock.set_nominal_period(1e6 / 165.0);  // l'écran s'annonce à 165 Hz
  Driver driver;
  std::set<std::uint64_t> seen;
  for (int i = 0; i < 600; ++i) {
    const auto now = static_cast<std::int64_t>(1'000'000.0 + period * static_cast<double>(i));
    // La relecture, avant la présentation : ce qui a paru, deux par deux et dans le désordre.
    while (driver.waiting.size() > 2) {
      auto first = driver.waiting.front();
      driver.waiting.pop_front();
      if (!driver.waiting.empty() && i % 5 == 0) std::swap(first, driver.waiting.front());
      const auto sample = ledger.report(first.first, first.second);
      REQUIRE(sample.has_value());
      CHECK(seen.insert(sample->id).second);
      clock.observe(sample->visible_us, sample->id, sample->target_us);
    }
    const PresentLedger::Issue issue = ledger.issue(now, 0);
    REQUIRE(issue.timing);
    driver.waiting.emplace_back(issue.id,
                                static_cast<std::int64_t>(static_cast<double>(now) + period + jitter(rng)));
  }
  for (const auto& [id, at] : driver.waiting) CHECK(ledger.report(id, at).has_value());
  CHECK(ledger.issued() == 600);
  CHECK(ledger.returned() == 600);
  CHECK(ledger.lost() == 0);
  CHECK(ledger.in_flight() == 0);
  CHECK_FALSE(ledger.report(5, 0).has_value());  // déjà rendue
  CHECK(clock.hz() == Approx(164.78).margin(0.01));
}

TEST_CASE("retour de présentation : file pleine forcée, les demandes reprennent", "[core][present]") {
  PresentLedger ledger;
  Driver driver{.queue = 1};
  std::uint64_t timed = 0;
  for (int i = 0; i < 300; ++i) {
    const std::int64_t now = 1'000'000 + std::int64_t{i} * 6'069;
    // Une date sur deux seulement est prête au tour suivant.
    if (!driver.waiting.empty() && i % 2 == 0) {
      CHECK(ledger.report(driver.waiting.front().first, now - 3'000).has_value());
      driver.waiting.pop_front();
    }
    const PresentLedger::Issue issue = ledger.issue(now, 0);
    if (!issue.timing) continue;
    if (driver.full()) {
      ledger.queue_full();  // VK_ERROR_PRESENT_TIMING_QUEUE_FULL_EXT : l'image part quand même
      CHECK(ledger.suspended());
      continue;
    }
    ++timed;
    driver.waiting.emplace_back(issue.id, now);
  }
  CHECK(ledger.issued() == 300);
  CHECK(timed >= 100);  // jamais bloqué : une demande reprend à chaque retour
  CHECK(ledger.returned() + ledger.untimed() + ledger.in_flight() == ledger.issued());
  CHECK(ledger.lost() == 0);
}

TEST_CASE("retour de présentation : une nouvelle swapchain perd ce qui était en vol", "[core][present]") {
  PresentLedger ledger;
  (void)ledger.issue(10, 0);
  (void)ledger.issue(20, 0);
  ledger.forget();
  CHECK(ledger.lost() == 2);
  CHECK_FALSE(ledger.report(1, 30).has_value());
  CHECK(ledger.issue(40, 0).id == 3);  // les identifiants restent croissants
  CHECK_FALSE(ledger.issue(50, 0, false).timing);
  CHECK(ledger.untimed() == 1);
}

TEST_CASE("retour de présentation : domaine de temps recalé sur l'hôte", "[core][present]") {
  TimeDomainMap map;
  CHECK(map.due(0));
  map.calibrate(5'000'000'000ULL, 7'000'000'000ULL, 7'000'000);
  CHECK(map.known());
  CHECK(map.to_host_us(5'000'500'000ULL) == 7'000'500);
  CHECK(map.to_domain_ns(7'000'500) == 5'000'500'000ULL);
  CHECK(map.to_domain_ns(0) == 0);
  CHECK_FALSE(map.due(8'000'000));
  CHECK(map.due(9'000'000));
  map.set_monotonic();
  CHECK_FALSE(map.due(100'000'000));
  CHECK(map.to_host_us(3'000'000) == 3'000);
}
