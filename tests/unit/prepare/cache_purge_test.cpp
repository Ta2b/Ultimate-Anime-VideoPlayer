// Purge du cache des moteurs TensorRT (A33-04) : par âge d'usage, puis par taille, les plus anciens d'abord.
#include "prepare/cache_purge.h"

#include <chrono>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "prepare/cache.h"
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
  CHECK_FALSE(fs::exists(dir / "aji-timing-0123.cache"));
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
