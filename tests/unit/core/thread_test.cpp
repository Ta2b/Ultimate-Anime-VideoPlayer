#include "core/thread.h"

#include <array>
#include <atomic>
#include <chrono>
#include <future>
#include <mutex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/thread_role.h"

#if defined(__linux__)
#include <pthread.h>
#endif

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

#if defined(__linux__)
TEST_CASE("start_thread : le système voit le nom du thread, coupé à 15 octets (A20-11)", "[thread][core]") {
  ajp::install_system_thread_names();
  const auto seen = [](std::string_view name) {
    std::promise<std::string> named;
    auto future = named.get_future();
    auto thread = ajp::start_thread(name, ajp::ThreadReport([](const ajp::Refusal&) {}),
                                    [&named](const std::stop_token&, const ajp::ThreadReport&) {
                                      std::array<char, 16> got{};
                                      pthread_getname_np(pthread_self(), got.data(), got.size());
                                      named.set_value(got.data());
                                    });
    REQUIRE(thread);
    return future.get();
  };
  CHECK(seen("source") == "source");
  CHECK(seen("pre-analyse-des-plans") == "pre-analyse-des");
  ajp::thread_namer().store(nullptr);
}

TEST_CASE("Priorité : le crochet relève la lecture et l'audio, pas le fond", "[thread][core]") {
  static std::atomic<int> raised{0};
  const bool accepted = ajp::set_thread_priority_raiser([]() noexcept {
    ++raised;
    return true;
  });
  CHECK(accepted);  // le thread appelant est relevé tout de suite
  const int base = raised.load();
  std::thread([] {
    ajp::set_thread_role(ajp::ThreadRole::Playback);
    ajp::set_thread_role(ajp::ThreadRole::Audio);
    ajp::set_thread_role(ajp::ThreadRole::Background);
  }).join();
  CHECK(raised.load() == base + 2);
  CHECK(ajp::set_thread_priority_raiser(nullptr) == false);
}
#endif

TEST_CASE("start_task : l'exception du corps devient ThreadFailed", "[thread]") {
  auto task = ajp::start_task<int>("essai", []() -> int { throw std::runtime_error("levée"); });
  REQUIRE(task != nullptr);
  const auto result = task->get();
  REQUIRE_FALSE(result.has_value());
  CHECK(result.error().code == ajp::RefusalCode::ThreadFailed);
  CHECK(ajp::start_task<int>("essai", [] { return 3; })->get() == 3);
}
