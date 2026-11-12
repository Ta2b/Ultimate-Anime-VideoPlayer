// Cache des moteurs : reprise d'un moteur refusé au chargement, empreinte, attente du verrou, purge, version
// de TensorRT chargée par trtexec, clé de contenu mémorisée.
#include <algorithm>
#include <chrono>
#include <format>
#include <string>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "bench.h"
#include "prepare/builder.h"
#include "prepare/cache.h"
#include "prepare/cache_purge.h"
#include "prepare/key_lock.h"
#include "prepare/model_source.h"
#include "prepare/sha256.h"
#include "prepare/trtexec_command.h"
#include "prepare/trtexec_progress.h"
#include "support.h"

using namespace ajp::prepare;
using namespace ajp::prepare::test_bench;
using ajp::RefusalCode;

// C05-01 : un moteur repris du cache dit qu'il l'est ; le lecteur, s'il le voit refusé au chargement, le met
// de côté (sans son empreinte) et la construction suivante en refait un.
TEST_CASE("Un moteur repris du cache refusé au chargement est mis de côté puis reconstruit",
          "[prepare][builder][cache]") {
  Bench bench("ok");
  Steps steps;
  auto progress = steps.progress();
  const TrtexecBuilder builder(bench.config);
  const auto first = builder.build(bench.model, kHd, progress);
  REQUIRE(first);
  CHECK_FALSE(first->reused);
  const auto again = builder.build(bench.model, kHd, progress);
  REQUIRE(again);
  CHECK(again->reused);
  CHECK(bench.builder_calls() == 1);

  const CacheEntry entry{.destination = again->path};
  REQUIRE(set_aside_entry(entry));
  CHECK_FALSE(test::fs::exists(again->path));
  CHECK_FALSE(test::fs::exists(entry.digest_file()));
  test::fs::path bad = again->path;
  bad += ".bad";
  CHECK(test::fs::exists(bad));

  const auto rebuilt = builder.build(bench.model, kHd, progress);
  REQUIRE(rebuilt);
  CHECK_FALSE(rebuilt->reused);
  CHECK(bench.builder_calls() == 2);
  CHECK(test::fs::exists(rebuilt->path));
}

TEST_CASE("Mettre de côté une entrée absente n'est pas une erreur", "[prepare][cache]") {
  const test::TempDir dir;
  CHECK(set_aside_entry(CacheEntry{.destination = dir / "absent.engine"}));
}

// C05-04 : un moteur intact dont seule la date a bougé reprend la nouvelle estampille dans son empreinte.
TEST_CASE("Une date changée sur un moteur intact est reprise dans l'empreinte", "[prepare][cache]") {
  const test::TempDir dir;
  const CacheEntry entry{.destination = dir / "a.engine"};
  const auto produce = [](const test::fs::path& output) -> std::expected<void, ajp::Refusal> {
    test::write_file(output, "moteur");
    return {};
  };
  REQUIRE(produce_once(entry, produce) == Production::Produced);
  const auto moved = test::fs::last_write_time(entry.destination) + std::chrono::seconds(5);
  test::fs::last_write_time(entry.destination, moved);
  REQUIRE(produce_once(entry, produce) == Production::Reused);
  const auto ticks = static_cast<std::int64_t>(moved.time_since_epoch().count());
  CHECK(test::read_file(entry.digest_file()) == std::format("{} 6 {}\n", sha256_hex("moteur"), ticks));
}

// C05-05 : l'attente du verrou de la clé se dit, comme celle du GPU.
TEST_CASE("L'attente du verrou de la clé se dit", "[prepare][cache]") {
  const test::TempDir dir;
  int told = 0;
  CacheEntry entry{.destination = dir / "a.engine", .on_wait = [&told] { ++told; }};
  auto held = KeyLock::acquire(entry.lock_file());
  REQUIRE(held);
  std::stop_source stop;
  std::jthread cancel([&] {
    std::this_thread::sleep_for(200ms);
    stop.request_stop();
  });
  const auto produced = produce_once(
      entry, [](const test::fs::path&) -> std::expected<void, ajp::Refusal> { return {}; }, stop.get_token());
  CHECK(test::refusal(produced) == RefusalCode::Cancelled);
  CHECK(told == 1);
}

TEST_CASE("Une entrée libre ne dit rien d'attente", "[prepare][cache]") {
  const test::TempDir dir;
  int told = 0;
  CacheEntry entry{.destination = dir / "a.engine", .on_wait = [&told] { ++told; }};
  REQUIRE(produce_once(entry, [](const test::fs::path& output) -> std::expected<void, ajp::Refusal> {
    test::write_file(output, "moteur");
    return {};
  }));
  CHECK(told == 0);
}

// C05-06 : un timing cache dont personne n'a refait la date reste tant qu'un moteur de ce cache a servi.
TEST_CASE("Le timing cache d'un GPU en service n'est pas purgé", "[prepare][cache]") {
  const test::TempDir dir;
  const auto now = test::fs::file_time_type::clock::now();
  const test::fs::path timing = dir / "aji-timing-0123456789abcdef.cache";
  test::write_file(timing, "mesures");
  test::fs::last_write_time(timing, now - std::chrono::hours(24 * 60));
  const test::fs::path engine = dir / "aji-v3-cle.engine";
  test::write_file(engine, "moteur");
  test::write_file(dir / "aji-v3-cle.engine.sha256", "x");  // repris aujourd'hui (mark_used)
  purge_engine_cache(dir.path(), {}, now);
  CHECK(test::fs::exists(engine));
  CHECK(test::fs::exists(timing));
}

// C05-11 : sans moteur gardé, le timing cache part avec son verrou et son cache écarté.
TEST_CASE("Le timing cache d'un GPU sans moteur part avec ses verrous et ses restes", "[prepare][cache]") {
  const test::TempDir dir;
  const auto now = test::fs::file_time_type::clock::now();
  const auto old = now - std::chrono::hours(24 * 60);
  for (const char* name :
       {"aji-timing-0123456789abcdef.cache", "aji-timing-0123456789abcdef.cache.bad",
        "aji-timing-0123456789abcdef.cache.suspect", "aji-timing-0123456789abcdef.cache.build-lock"}) {
    test::write_file(dir / name, "x");
    test::fs::last_write_time(dir / name, old);
  }
  test::write_file(dir / "aji-timing-ffffffffffffffff.cache", "frais");  // d'un autre GPU, en service
  const CachePurgeResult result = purge_engine_cache(dir.path(), {}, now);
  CHECK_FALSE(test::fs::exists(dir / "aji-timing-0123456789abcdef.cache"));
  CHECK_FALSE(test::fs::exists(dir / "aji-timing-0123456789abcdef.cache.bad"));
  CHECK_FALSE(test::fs::exists(dir / "aji-timing-0123456789abcdef.cache.suspect"));
  CHECK_FALSE(test::fs::exists(dir / "aji-timing-0123456789abcdef.cache.build-lock"));
  CHECK(test::fs::exists(dir / "aji-timing-ffffffffffffffff.cache"));
  CHECK(result.removed_locks == 1);
}

// C05-10 : trtexec qui charge un autre TensorRT que le lecteur est refusé avec un message qui le dit.
TEST_CASE("Un trtexec qui charge un autre TensorRT est refusé", "[prepare][builder]") {
  Bench bench("old-tensorrt");
  Steps steps;
  auto progress = steps.progress();
  const auto result = TrtexecBuilder(bench.config).build(bench.model, kHd, progress);
  REQUIRE_FALSE(result);
  CHECK(result.error().code == RefusalCode::BuilderFailed);
  CHECK(result.error().detail.find("11.1.0") != std::string::npos);
  CHECK(result.error().detail.find("11.3.0") != std::string::npos);
  CHECK_FALSE(test::fs::exists(bench.plan(kHd).engine));
}

TEST_CASE("La version de TensorRT se lit sur la ligne RUNNING de trtexec", "[prepare][builder]") {
  CHECK(TrtexecProgress::reported_version("&&&& RUNNING TensorRT.trtexec [TensorRT v110300] [b99] # x") ==
        "v110300");
  CHECK(TrtexecProgress::reported_version("[I] [TensorRT v110300]").empty());
  CHECK(TrtexecProgress::reported_version("&&&& RUNNING TensorRT.trtexec [TensorRT vXYZ]").empty());
  CHECK(TrtexecProgress::reported_version("&&&& RUNNING sans version").empty());
}

// C05-02 : la clé de contenu d'un ONNX à poids externes est mémorisée, et suit un poids qui change.
TEST_CASE("La clé de contenu suit les poids externes, mémorisée entre deux lectures", "[prepare][builder]") {
  Bench bench("ok", {.external = {"poids.bin"}});
  const test::fs::path weights = bench.model.onnx_path.parent_path() / "poids.bin";
  test::write_file(weights, "poids un");
  const auto first = read_model(bench.model, "input");
  const auto second = read_model(bench.model, "input");
  REQUIRE(first);
  REQUIRE(second);
  CHECK(first->content_key == second->content_key);
  CHECK(first->content_key ==
        onnx_source_key(first->graph, weights.parent_path(), first->summary.external_locations).value());
  test::write_file(weights, "poids deux, plus longs");
  const auto changed = read_model(bench.model, "input");
  REQUIRE(changed);
  CHECK(changed->content_key != first->content_key);
  CHECK(changed->content_key ==
        onnx_source_key(changed->graph, weights.parent_path(), changed->summary.external_locations).value());
}
