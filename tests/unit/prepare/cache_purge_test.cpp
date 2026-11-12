// Purge du cache des moteurs TensorRT (A33-04) : par âge d'usage, puis par taille, les plus anciens d'abord.
#include "prepare/cache_purge.h"

#include <chrono>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "prepare/cache.h"
#include "prepare/cache_io.h"
#include "support.h"

using namespace ajp::prepare;
using namespace std::chrono_literals;

namespace {

namespace fs = std::filesystem;

// Un moteur publié de bytes octets, avec ses voisins, dont l'empreinte date de age.
fs::path engine(const test::TempDir& dir, const std::string& key, std::size_t bytes,
                fs::file_time_type::duration age) {
  const fs::path file = dir / ("aji-v3-" + key + ".engine");
  test::write_file(file, std::string(bytes, 'e'));
  for (const char* suffix : {".sha256", ".build.log", ".layers.json", ".key-lock"}) {
    fs::path sibling = file;
    sibling += suffix;
    test::write_file(sibling, "x");
  }
  const auto when = fs::file_time_type::clock::now() - age;
  fs::last_write_time(file, when);
  fs::path digest = file;
  digest += ".sha256";
  fs::last_write_time(digest, when);
  return file;
}

}  // namespace

TEST_CASE("Cache des moteurs : un moteur sans usage depuis longtemps part, avec ses voisins",
          "[prepare][cache]") {
  const test::TempDir dir;
  const fs::path old = engine(dir, "old", 100, 24h * 90);
  const fs::path fresh = engine(dir, "fresh", 100, 1h);
  test::write_file(dir / "aji-timing-0123.cache", "t");
  fs::last_write_time(dir / "aji-timing-0123.cache", fs::file_time_type::clock::now() - 24h * 90);
  test::write_file(dir / "autre.bin", "à garder");

  const CachePurgeResult result = purge_engine_cache(dir.path());
  CHECK(result.removed == 1);
  CHECK(result.kept == 1);
  CHECK_FALSE(fs::exists(old));
  CHECK_FALSE(fs::exists(dir / "aji-v3-old.engine.sha256"));
  CHECK_FALSE(fs::exists(dir / "aji-v3-old.engine.build.log"));
  CHECK_FALSE(fs::exists(dir / "aji-v3-old.engine.layers.json"));
  CHECK(fs::exists(dir / "aji-v3-old.engine.key-lock"));  // les verrous restent
  // Un moteur gardé a servi depuis peu : le timing cache de ce GPU reste, même s'il n'a pas bougé (C05-06).
  CHECK(fs::exists(dir / "aji-timing-0123.cache"));
  CHECK(fs::exists(fresh));
  CHECK(fs::exists(dir / "autre.bin"));
}

TEST_CASE("Cache des moteurs : au-delà de la taille, les moins récents partent, pas ceux du jour",
          "[prepare][cache]") {
  const test::TempDir dir;
  const fs::path a = engine(dir, "a", 1000, 24h * 10);
  const fs::path b = engine(dir, "b", 1000, 24h * 3);
  const fs::path c = engine(dir, "c", 1000, 2h);
  const fs::path d = engine(dir, "d", 1000, 1h);
  const CachePurgeLimits limits{.max_bytes = 1500};
  const CachePurgeResult result = purge_engine_cache(dir.path(), limits);
  // c et d ont servi aujourd'hui : gardés même au-delà ; a et b, plus anciens, partent.
  CHECK(fs::exists(d));
  CHECK(fs::exists(c));
  CHECK_FALSE(fs::exists(b));
  CHECK_FALSE(fs::exists(a));
  CHECK(result.removed == 2);
  CHECK(result.kept == 2);
}

TEST_CASE("Cache des moteurs : un moteur repris rafraîchit sa date d'usage", "[prepare][cache]") {
  const test::TempDir dir;
  const fs::path file = engine(dir, "reused", 10, 24h * 90);
  const CacheEntry entry{.destination = file};
  fs::remove(entry.digest_file());  // une empreinte d'avant : refaite à la reprise
  fs::last_write_time(file, fs::file_time_type::clock::now() - 24h * 90);
  auto produced = produce_once(entry, [](const fs::path&) -> std::expected<void, ajp::Refusal> {
    FAIL("le moteur publié doit se reprendre");
    return {};
  });
  REQUIRE(produced);
  CHECK(*produced == Production::Reused);
  const CachePurgeResult result = purge_engine_cache(dir.path());
  CHECK(result.removed == 0);
  CHECK(fs::exists(file));

  // Une empreinte déjà là est rafraîchie aussi.
  fs::last_write_time(entry.digest_file(), fs::file_time_type::clock::now() - 24h * 90);
  REQUIRE(produce_once(entry, [](const fs::path&) -> std::expected<void, ajp::Refusal> { return {}; }));
  CHECK(purge_engine_cache(dir.path()).removed == 0);
}

TEST_CASE("Cache des moteurs : un dossier absent ne lève rien", "[prepare][cache]") {
  const CachePurgeResult result =
      purge_engine_cache(std::filesystem::temp_directory_path() / "ajp-absent-xyz");
  CHECK(result.removed == 0);
  CHECK(result.kept == 0);
}

TEST_CASE("Cache des moteurs : le dossier de travail d'une construction tuée est retiré (B05-01, B33-02)",
          "[prepare][cache]") {
  const test::TempDir dir;
  // Orphelin d'une clé qui ne reviendra jamais (le pilote a changé) : daté de 400 jours.
  auto old = reserve_stage(dir.path(), kStagePrefix, "engine:aji-v3-disparu.engine");
  REQUIRE(old);
  test::write_file(*old / "model.engine", std::string(1000, 'm'));
  fs::last_write_time(*old, fs::file_time_type::clock::now() - 24h * 400);
  // Une construction en cours : récente, jamais touchée.
  auto live = reserve_stage(dir.path(), kStagePrefix, "engine:aji-v3-vivant.engine");
  REQUIRE(live);
  // Un dossier du même nom sans marqueur n'est pas le nôtre.
  const fs::path foreign = dir / ".aji-build-etranger";
  fs::create_directories(foreign);
  fs::last_write_time(foreign, fs::file_time_type::clock::now() - 24h * 400);

  const CachePurgeResult result = purge_engine_cache(dir.path());
  CHECK(result.removed_stages == 1);
  CHECK(result.removed_bytes >= 1000);
  CHECK_FALSE(fs::exists(*old));
  CHECK(fs::exists(*live));
  CHECK(fs::exists(foreign));
}

TEST_CASE("Cache des moteurs : le verrou d'un moteur disparu part, pas celui d'un moteur gardé",
          "[prepare][cache]") {
  const test::TempDir dir;
  const fs::path kept = engine(dir, "kept", 100, 1h);
  const fs::path lock_kept = dir / "aji-v3-kept.engine.key-lock";
  fs::last_write_time(lock_kept, fs::file_time_type::clock::now() - 24h * 90);
  // Un verrou sans moteur : ancien, il part ; récent (première construction en cours), il reste.
  const fs::path stale = dir / "aji-v3-vieux.engine.key-lock";
  const fs::path young = dir / "aji-v3-jeune.engine.key-lock";
  test::write_file(stale, "x");
  test::write_file(young, "x");
  fs::last_write_time(stale, fs::file_time_type::clock::now() - 24h * 90);

  const CachePurgeResult result = purge_engine_cache(dir.path());
  CHECK(result.removed_locks == 1);
  CHECK_FALSE(fs::exists(stale));
  CHECK(fs::exists(young));
  CHECK(fs::exists(lock_kept));
  CHECK(fs::exists(kept));
}
