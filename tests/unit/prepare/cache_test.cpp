// Cache atomique : verrou de clé, dossiers de travail, publication unique,
// avec des écrivains concurrents en threads et en processus.
#include "prepare/cache.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <optional>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "prepare/cache_io.h"
#include "prepare/key_lock.h"
#include "prepare/process.h"
#include "prepare/sha256.h"
#include "support.h"

using namespace ajp::prepare;
using ajp::RefusalCode;
using namespace std::chrono_literals;

namespace {

std::expected<void, ajp::Refusal> write_output(const test::fs::path& output, std::string_view content) {
  test::write_file(output, content);
  return {};
}

// Un dossier de travail de ce propriétaire reste-t-il dans parent ?
bool has_owned_stages(const test::fs::path& parent, std::string_view prefix, std::string_view owner) {
  std::error_code error;
  for (const auto& entry : test::fs::directory_iterator(parent, error)) {
    if (entry.path().filename().string().starts_with(prefix) && owns_stage(entry.path(), owner)) return true;
  }
  return false;
}

}  // namespace

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

TEST_CASE("KeyLock : l'attente s'arrête au délai maximum (P2.31)", "[prepare][cache]") {
  const test::TempDir dir;
  const auto file = dir / "cle.key-lock";
  auto first = KeyLock::acquire(file);
  REQUIRE(first);
  CHECK(test::refusal(KeyLock::acquire(file, {}, LockWait::Block, 60ms)) == RefusalCode::CacheBusy);
}

#ifndef _WIN32
// C34-08 : la purge retire un verrou orphelin pendant qu'elle le tient. Un lancement qui attendait sur
// l'ancien fichier l'obtient ensuite ; il doit voir que ce n'est plus le verrou du chemin et prendre le
// nouveau, sinon un troisième lancement prend le nouveau fichier et deux constructions se croient seules.
TEST_CASE("KeyLock : un verrou retiré pendant l'attente est rouvert (C34-08)", "[prepare][cache]") {
  const test::TempDir dir;
  const auto file = dir / "aji-v3-x.engine.key-lock";
  auto purge = KeyLock::acquire(file);
  REQUIRE(purge);
  std::atomic<bool> waiting_got{false};
  std::optional<KeyLock> waiter;
  std::thread second([&] {
    auto got = KeyLock::acquire(file, {}, LockWait::Block, 5s);
    if (got) waiter.emplace(std::move(*got));
    waiting_got = true;
  });
  std::this_thread::sleep_for(100ms);  // le second attend, ouvert sur l'ancien fichier
  test::fs::remove(file);              // ce que fait remove_orphan_lock, verrou tenu
  purge = std::unexpected(ajp::Refusal{});
  second.join();
  REQUIRE(waiting_got);
  REQUIRE(waiter);
  CHECK(test::fs::exists(file));  // le chemin porte le verrou du second
  // Un troisième lancement trouve le verrou tenu : avant, il en créait un nouveau et le prenait.
  CHECK(test::refusal(KeyLock::acquire(file, {}, LockWait::TryOnly)) == RefusalCode::CacheBusy);
}
#endif

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

TEST_CASE("produce_once : un arrêt demandé après la production n'en jette pas le moteur (B05-04)",
          "[prepare][cache]") {
  const test::TempDir dir;
  const CacheEntry entry{.destination = dir / "moteur.engine"};
  std::stop_source stop;
  auto produced = produce_once(
      entry,
      [&](const test::fs::path& output) {
        auto written = write_output(output, "moteur complet");
        stop.request_stop();  // « Annuler et regarder » pendant la vérification qui suit trtexec
        return written;
      },
      stop.get_token());
  REQUIRE(produced);
  CHECK(*produced == Production::Produced);
  CHECK(is_published(entry.destination));
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

TEST_CASE("produce_once : disque plein en pleine écriture, rien de publié ni de laissé (A28-22)",
          "[prepare][cache]") {
  const test::TempDir dir;
  const CacheEntry entry{.destination = dir / "moteurs/a.engine"};
  // Le constructeur écrit un moteur tronqué puis dit que le disque est plein.
  const auto full = [](const test::fs::path& output) -> std::expected<void, ajp::Refusal> {
    test::write_file(output, "moteur tron");
    return std::unexpected(ajp::Refusal{RefusalCode::CacheIo, "No space left on device"});
  };
  CHECK(test::refusal(produce_once(entry, full)) == RefusalCode::CacheIo);
  CHECK_FALSE(test::fs::exists(entry.destination));
  CHECK(test::stage_count(entry.destination.parent_path()) == 0);
  // La place revenue, la production suivante publie le moteur entier.
  CHECK(produce_once(entry, [](const test::fs::path& output) {
          return write_output(output, "moteur");
        }).value() == Production::Produced);
  CHECK(test::read_file(entry.destination) == "moteur");
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

TEST_CASE("sync_file et publish_file : écrit sur disque, puis publié", "[prepare][cache]") {
  const test::TempDir dir;
  test::write_file(dir / "travail", "octets");
  REQUIRE(sync_file(dir / "travail"));
  CHECK(test::refusal(sync_file(dir / "absent")) == RefusalCode::CacheIo);
  test::write_file(dir / "final", "ancien");
  REQUIRE(publish_file(dir / "travail", dir / "final"));
  CHECK(test::read_file(dir / "final") == "octets");
  CHECK_FALSE(test::fs::exists(dir / "travail"));
  CHECK(test::refusal(publish_file(dir / "absent", dir / "final")) == RefusalCode::CacheIo);
  CHECK(test::read_file(dir / "final") == "octets");
}

TEST_CASE("io_message : conseil selon l'erreur", "[prepare][cache]") {
  const auto full = io_message("écriture", "/cache/x", std::make_error_code(std::errc::no_space_on_device));
  CHECK(full.find("Libérez de la place") != std::string::npos);
  const auto denied = io_message("écriture", "/cache/x", std::make_error_code(std::errc::permission_denied));
  CHECK(denied.find("droits d'écriture") != std::string::npos);
  CHECK(io_message("écriture", "/cache/x", {}).starts_with("écriture : /cache/x : "));
}

TEST_CASE("produce_once : une entrée abîmée depuis sa publication est refaite (P2.31)", "[prepare][cache]") {
  const test::TempDir dir;
  const CacheEntry entry{.destination = dir / "a.engine"};
  int made = 0;
  const auto produce = [&](const test::fs::path& output) {
    ++made;
    return write_output(output, "moteur");
  };
  REQUIRE(produce_once(entry, produce) == Production::Produced);
  // L'empreinte porte aussi la taille et la date du moteur : pas de relecture tant qu'elles ne bougent pas
  // (B36-06).
  CHECK(test::read_file(entry.digest_file()).starts_with(sha256_hex("moteur") + " 6 "));
  REQUIRE(produce_once(entry, produce) == Production::Reused);
  // Mêmes taille et date : crue sans relire (la limite voulue de la mémoire).
  const auto stamp = test::fs::last_write_time(entry.destination);
  test::write_file(entry.destination, "moteuR");  // même taille, autres octets
  test::fs::last_write_time(entry.destination, stamp);
  CHECK(produce_once(entry, produce) == Production::Reused);
  // Une autre date : relu, haché, refait. (Les horloges de fichiers sont grossières : la date est forcée.)
  test::fs::last_write_time(entry.destination, stamp + std::chrono::seconds(5));
  CHECK(produce_once(entry, produce) == Production::Produced);
  CHECK(made == 2);
  CHECK(test::read_file(entry.destination) == "moteur");
  // Une entrée d'avant l'empreinte se reprend et la reçoit ; une empreinte seule, sans taille ni date (ancien
  // format), se lit toujours et force le hachage.
  test::write_file(entry.digest_file(), sha256_hex("moteur") + "\n");
  CHECK(produce_once(entry, produce) == Production::Reused);
  test::write_file(entry.digest_file(), sha256_hex("autre") + "\n");
  CHECK(produce_once(entry, produce) == Production::Produced);
  test::fs::remove(entry.digest_file());
  CHECK(produce_once(entry, produce) == Production::Reused);
  CHECK(test::fs::exists(entry.digest_file()));
}
