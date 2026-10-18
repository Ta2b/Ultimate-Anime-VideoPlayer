// Cache atomique : verrou de clé, dossiers de travail, publication unique,
// avec des écrivains concurrents en threads et en processus.
#include "prepare/cache.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "prepare/cache_io.h"
#include "prepare/key_lock.h"
#include "prepare/process.h"
#include "support.h"

using namespace ajp::prepare;
using ajp::RefusalCode;
using namespace std::chrono_literals;

namespace {

std::expected<void, ajp::Refusal> write_output(const test::fs::path& output, std::string_view content) {
  test::write_file(output, content);
  return {};
}

}  // namespace

TEST_CASE("write_atomic : contenu remplacé d'un coup, aucun temporaire laissé", "[prepare][cache]") {
  const test::TempDir dir;
  REQUIRE(write_atomic(dir / "recu.toml", "premier"));
  REQUIRE(write_atomic(dir / "recu.toml", "second"));
  CHECK(test::read_file(dir / "recu.toml") == "second");
  CHECK(std::distance(test::fs::directory_iterator(dir.path()), test::fs::directory_iterator{}) == 1);
  CHECK(test::refusal(write_atomic(dir / "absent/recu.toml", "x")) == RefusalCode::CacheIo);
}

TEST_CASE("KeyLock : exclusif entre deux ouvertures, rendu à la destruction", "[prepare][cache]") {
  const test::TempDir dir;
  const auto file = dir / "cle.key-lock";
  {
    auto first = KeyLock::acquire(file);
    REQUIRE(first);
    CHECK(test::refusal(KeyLock::acquire(file, {}, LockWait::TryOnly)) == RefusalCode::CacheBusy);
    std::stop_source stop;
    stop.request_stop();
    CHECK(test::refusal(KeyLock::acquire(file, stop.get_token())) == RefusalCode::Cancelled);
  }
  CHECK(KeyLock::acquire(file, {}, LockWait::TryOnly));
  CHECK(test::fs::exists(file));  // jamais supprimé
}

TEST_CASE("KeyLock : tenu par un autre processus", "[prepare][cache]") {
  const test::TempDir dir;
  const auto file = dir / "cle.key-lock";
  std::stop_source checked;
  std::atomic<bool> busy_seen = false;
  const auto exit = run_process(
      {.program = test::helper_path(), .args = {"hold-lock", file.string(), "300"}, .timeout = 10s},
      [&](std::string_view line) {
        if (line != "verrou pris") return;
        busy_seen = test::refusal(KeyLock::acquire(file, {}, LockWait::TryOnly)) == RefusalCode::CacheBusy;
      },
      checked.get_token());
  REQUIRE(exit);
  CHECK(exit->code == 0);
  CHECK(busy_seen);
  CHECK(KeyLock::acquire(file, {}, LockWait::TryOnly));  // rendu à la mort du processus
}

TEST_CASE("Dossiers de travail : marqueur du fork, propriétaire exact", "[prepare][cache]") {
  const test::TempDir dir;
  auto stage = reserve_stage(dir.path(), kStagePrefix, "engine:a.engine");
  REQUIRE(stage);
  CHECK(test::read_file(*stage / ".aji-stage-owner") == "aji-stage-owner-v1\nengine:a.engine\n");
  CHECK(owns_stage(*stage, "engine:a.engine"));
  CHECK_FALSE(owns_stage(*stage, "engine:b.engine"));
  auto other = reserve_stage(dir.path(), kStagePrefix, "engine:b.engine");
  REQUIRE(other);
  CHECK(has_owned_stages(dir.path(), kStagePrefix, "engine:a.engine"));
  REQUIRE(recover_owned_stages(dir.path(), kStagePrefix, "engine:a.engine"));
  CHECK_FALSE(test::fs::exists(*stage));
  CHECK(test::fs::exists(*other));  // le dossier d'une autre clé reste
}

TEST_CASE("produce_once : une production, puis réutilisation sans verrou", "[prepare][cache]") {
  const test::TempDir dir;
  const CacheEntry entry{.destination = dir / "moteurs/a.engine"};
  int calls = 0;
  const auto producer = [&](const test::fs::path& output) {
    ++calls;
    return write_output(output, "moteur");
  };
  CHECK(produce_once(entry, producer).value() == Production::Produced);
  CHECK(produce_once(entry, producer).value() == Production::Reused);
  CHECK(calls == 1);
  CHECK(test::read_file(entry.destination) == "moteur");
  CHECK(test::stage_count(entry.destination.parent_path()) == 0);
}

TEST_CASE("produce_once : un échec ne publie rien et ne laisse rien", "[prepare][cache]") {
  const test::TempDir dir;
  const CacheEntry entry{.destination = dir / "a.engine"};
  const auto failing = [](const test::fs::path& output) -> std::expected<void, ajp::Refusal> {
    test::write_file(output, "moitié");
    return std::unexpected(ajp::Refusal{RefusalCode::BuilderFailed, "échec"});
  };
  CHECK(test::refusal(produce_once(entry, failing)) == RefusalCode::BuilderFailed);
  const auto empty = [](const test::fs::path& output) { return write_output(output, ""); };
  CHECK(test::refusal(produce_once(entry, empty)) == RefusalCode::BuilderFailed);
  CHECK_FALSE(test::fs::exists(entry.destination));
  CHECK(test::stage_count(dir.path()) == 0);
}

TEST_CASE("produce_once : huit threads, une seule production", "[prepare][cache]") {
  const test::TempDir dir;
  const CacheEntry entry{.destination = dir / "a.engine"};
  std::atomic<int> calls = 0;
  std::atomic<int> produced = 0;
  std::atomic<int> failures = 0;
  {
    std::vector<std::jthread> writers;
    for (int i = 0; i < 8; ++i) {
      writers.emplace_back([&] {
        auto result = produce_once(entry, [&](const test::fs::path& output) {
          ++calls;
          std::this_thread::sleep_for(50ms);
          return write_output(output, "contenu unique");
        });
        if (!result) ++failures;
        if (result && *result == Production::Produced) ++produced;
      });
    }
  }
  CHECK(failures == 0);
  CHECK(calls == 1);
  CHECK(produced == 1);
  CHECK(test::read_file(entry.destination) == "contenu unique");
  CHECK(test::stage_count(dir.path()) == 0);
}

TEST_CASE("produce_once : quatre processus, une seule production", "[prepare][cache]") {
  const test::TempDir dir;
  const auto destination = dir / "a.engine";
  const auto calls = dir / "productions.txt";
  std::vector<std::string> outcomes(4);
  {
    std::vector<std::jthread> writers;
    for (std::size_t i = 0; i < outcomes.size(); ++i) {
      writers.emplace_back([&, i] {
        const auto exit = run_process({.program = test::helper_path(),
                                       .args = {"cache-writer", destination.string(), calls.string(), "200"},
                                       .timeout = 30s},
                                      [&](std::string_view line) { outcomes[i] = line; }, {});
        if (!exit || exit->code != 0) outcomes[i] = "échec";
      });
    }
  }
  CHECK(test::read_lines(calls).size() == 1);
  CHECK(std::ranges::count(outcomes, "produit") == 1);
  CHECK(std::ranges::count(outcomes, "repris") == 3);
  CHECK(test::read_file(destination) == "contenu unique");
  CHECK(test::stage_count(dir.path()) == 0);
}

TEST_CASE("produce_once : le reste d'un constructeur mort est nettoyé", "[prepare][cache]") {
  const test::TempDir dir;
  const CacheEntry entry{.destination = dir / "a.engine"};
  const auto exit = run_process(
      {.program = test::helper_path(), .args = {"cache-orphan", entry.destination.string()}, .timeout = 10s},
      {}, {});
  REQUIRE(exit);
  REQUIRE(has_owned_stages(dir.path(), kStagePrefix, entry.owner()));
  CHECK(produce_once(entry, [](const test::fs::path& output) {
          return write_output(output, "neuf");
        }).value() == Production::Produced);
  CHECK_FALSE(has_owned_stages(dir.path(), kStagePrefix, entry.owner()));
  CHECK(test::read_file(entry.destination) == "neuf");
}

TEST_CASE("io_message : conseil selon l'erreur", "[prepare][cache]") {
  const auto full = io_message("écriture", "/cache/x", std::make_error_code(std::errc::no_space_on_device));
  CHECK(full.find("Libérez de la place") != std::string::npos);
  const auto denied = io_message("écriture", "/cache/x", std::make_error_code(std::errc::permission_denied));
  CHECK(denied.find("droits d'écriture") != std::string::npos);
  CHECK(io_message("écriture", "/cache/x", {}).starts_with("écriture : /cache/x : "));
}
