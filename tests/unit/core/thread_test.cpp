#include "core/thread.h"

#include <atomic>
#include <chrono>
#include <future>
#include <mutex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include <catch2/catch_test_macros.hpp>

using ajp::Refusal;
using ajp::RefusalCode;
using ajp::start_thread;
using ajp::ThreadReport;
using namespace std::chrono_literals;

namespace {

// Les refus reçus par le rapporteur, depuis le thread lancé.
class Reports {
 public:
  ThreadReport report() {
    return ThreadReport([this](const Refusal& refusal) {
      const std::scoped_lock lock(mutex_);
      received_.push_back(refusal);
    });
  }
  std::vector<Refusal> received() const {
    const std::scoped_lock lock(mutex_);
    return received_;
  }

 private:
  mutable std::mutex mutex_;
  std::vector<Refusal> received_;
};

// Un corps qui lève en se déplaçant : le démarrage échoue avant tout appel.
struct ThrowsOnMove {
  ThrowsOnMove() = default;
  ThrowsOnMove(ThrowsOnMove&& /*other*/) {  // NOLINT(performance-noexcept-move-constructor)
    throw std::system_error(std::make_error_code(std::errc::resource_unavailable_try_again), "thread refusé");
  }
  ThrowsOnMove(const ThrowsOnMove&) = delete;
  ThrowsOnMove& operator=(const ThrowsOnMove&) = delete;
  ThrowsOnMove& operator=(ThrowsOnMove&&) = delete;
  ~ThrowsOnMove() = default;
  void operator()(const std::stop_token& /*stop*/, const ThreadReport& /*report*/) const {}
};

}  // namespace

TEST_CASE("start_thread : le corps tourne jusqu'à la demande d'arrêt", "[thread]") {
  Reports reports;
  std::atomic<bool> stopped = false;
  std::promise<void> started;
  {
    auto thread = start_thread("essai", reports.report(),
                               [&](const std::stop_token& stop, const ThreadReport& /*report*/) {
                                 started.set_value();
                                 while (!stop.stop_requested()) std::this_thread::sleep_for(1ms);
                                 stopped = true;
                               });
    REQUIRE(thread);
    CHECK(started.get_future().wait_for(2s) == std::future_status::ready);
  }  // le std::jthread demande l'arrêt et attend
  CHECK(stopped);
  CHECK(reports.received().empty());
}

TEST_CASE("start_thread : une exception du corps devient un refus ThreadFailed", "[thread]") {
  Reports reports;
  SECTION("std::exception : son texte entre dans le détail") {
    auto thread = start_thread("source", reports.report(), [](const std::stop_token&, const ThreadReport&) {
      throw std::runtime_error("panne simulée");
    });
    REQUIRE(thread);
    thread->join();
    const std::vector<Refusal> received = reports.received();
    REQUIRE(received.size() == 1);
    CHECK(received[0].code == RefusalCode::ThreadFailed);
    CHECK(received[0].detail == "thread source : panne simulée");
  }
  SECTION("une exception qui ne dérive pas de std::exception") {
    auto thread =
        start_thread("subs", reports.report(), [](const std::stop_token&, const ThreadReport&) { throw 42; });
    REQUIRE(thread);
    thread->join();
    const std::vector<Refusal> received = reports.received();
    REQUIRE(received.size() == 1);
    CHECK(received[0].code == RefusalCode::ThreadFailed);
    CHECK(received[0].detail == "thread subs : exception inconnue");
  }
}

TEST_CASE("start_thread : le corps signale ses propres refus par le même rapporteur", "[thread]") {
  Reports reports;
  auto thread =
      start_thread("audio", reports.report(), [](const std::stop_token&, const ThreadReport& report) {
        report(Refusal{RefusalCode::AudioDecodeFailed, "paquet 3"});
      });
  REQUIRE(thread);
  thread->join();
  const std::vector<Refusal> received = reports.received();
  REQUIRE(received.size() == 1);
  CHECK(received[0].code == RefusalCode::AudioDecodeFailed);
}

TEST_CASE("start_thread : un rapporteur qui lève ne fait pas sortir l'exception", "[thread]") {
  std::atomic<int> calls = 0;
  const ThreadReport throwing([&](const Refusal&) {
    ++calls;
    throw std::runtime_error("rapporteur en panne");
  });
  auto thread = start_thread("essai", throwing, [](const std::stop_token&, const ThreadReport&) {
    throw std::logic_error("corps en panne");
  });
  REQUIRE(thread);
  thread->join();  // pas de std::terminate
  CHECK(calls == 1);
}

TEST_CASE("start_thread : refus de démarrage, sans appel du corps", "[thread]") {
  bool called = false;
  SECTION("rapporteur absent") {
    auto thread = start_thread("essai", ThreadReport{},
                               [&](const std::stop_token&, const ThreadReport&) { called = true; });
    REQUIRE_FALSE(thread);
    CHECK(thread.error().code == RefusalCode::FieldMissing);
  }
  SECTION("std::system_error au démarrage devient ThreadUnavailable") {
    Reports reports;
    auto thread = start_thread("preparer", reports.report(), ThrowsOnMove{});
    REQUIRE_FALSE(thread);
    CHECK(thread.error().code == RefusalCode::ThreadUnavailable);
    CHECK(thread.error().detail.starts_with("thread preparer : thread refusé"));
    CHECK(reports.received().empty());
  }
  CHECK_FALSE(called);
}

TEST_CASE("refusal_from_exception : le code demandé et le texte de l'exception", "[thread]") {
  Refusal refusal{RefusalCode::FieldMissing, {}};
  try {
    throw std::out_of_range("indice 9");
  } catch (...) {
    refusal = ajp::refusal_from_exception(RefusalCode::DecodeFailed, "thread source");
  }
  CHECK(refusal.code == RefusalCode::DecodeFailed);
  CHECK(refusal.detail == "thread source : indice 9");
}
