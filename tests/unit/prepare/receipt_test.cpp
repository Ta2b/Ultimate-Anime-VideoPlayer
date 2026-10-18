// Reçu de qualification : sceau, relecture, falsification, plages.
#include "prepare/receipt.h"

#include <cmath>
#include <limits>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "prepare/sha256.h"
#include "support.h"

using namespace ajp::prepare;
using ajp::RefusalCode;

namespace {

Receipt sample() {
  return {.plan_hash = 0x0123456789abcdefULL,
          .created_utc = "2026-10-18T15:29:26Z",
          .driver = {.backend = "tensorrt", .device = "NVIDIA GeForce RTX 4070 Ti", .version = "580.95"},
          .engines = {{.model = "2x_AnimeJaNai_HD_V3_Compact",
                       .cache_key = "aji-v3-a.engine",
                       .sha256 = sha256_hex("a")},
                      {.model = "rife-4.26", .cache_key = "aji-v3-b.engine", .sha256 = sha256_hex("b")}},
          .goldens = {{.name = "span-1080p-0042",
                       .flip = 0.0123,
                       .flip_max = 0.05,
                       .psnr_db = 48.2,
                       .psnr_min_db = 40.0}},
          .bench = {.pairs = 600,
                    .pair_ms_mean = 12.4,
                    .pair_ms_p50 = 12.1,
                    .pair_ms_p99 = 15.0,
                    .pair_ms_max = 18.25,
                    .host_waits = 0,
                    .truehdr_waits = 3}};
}

// Remplace la première occurrence de from dans text.
std::string replaced(std::string text, std::string_view from, std::string_view to) {
  const auto at = text.find(from);
  REQUIRE(at != std::string::npos);
  return text.replace(at, from.size(), to);
}

}  // namespace

TEST_CASE("Receipt : scellé, écrit, relu à l'identique", "[prepare][receipt]") {
  const auto sealed = seal_receipt(sample());
  REQUIRE(sealed);
  CHECK(is_sha256_hex(sealed->seal));
  const std::string text = receipt_text(*sealed);
  CHECK(text.find("plan_hash = \"0123456789abcdef\"") != std::string::npos);
  CHECK(text.find("hash = \"" + driver_hash(sealed->driver) + "\"") != std::string::npos);
  const auto parsed = parse_receipt(text);
  REQUIRE(parsed);
  CHECK(receipt_text(*parsed) == text);
  CHECK(parsed->plan_hash == 0x0123456789abcdefULL);
  CHECK(parsed->bench.truehdr_waits == 3);
  CHECK(parsed->engines.size() == 2);
  CHECK(qualified(*parsed));
}

TEST_CASE("Receipt : le sceau ne dépend pas de la mise en forme", "[prepare][receipt]") {
  const std::string text = receipt_text(seal_receipt(sample()).value());
  std::string reformatted = replaced(text, "pairs = 600", "pairs   =   600   # mesurées");
  reformatted = "# commentaire ajouté\n" + reformatted;
  CHECK(parse_receipt(reformatted));
}

TEST_CASE("Receipt : toute valeur modifiée casse le sceau", "[prepare][receipt]") {
  const std::string text = receipt_text(seal_receipt(sample()).value());
  CHECK(test::refusal(parse_receipt(replaced(text, "host_waits = 0", "host_waits = 1"))) ==
        RefusalCode::ReceiptInvalid);
  CHECK(test::refusal(parse_receipt(replaced(text, "psnr_db = 48.2", "psnr_db = 49.2"))) ==
        RefusalCode::ReceiptInvalid);
  CHECK(test::refusal(parse_receipt(replaced(text, "580.95", "580.96"))) == RefusalCode::ReceiptInvalid);
  CHECK(test::refusal(parse_receipt(replaced(text, "0123456789abcdef", "0123456789abcdee"))) ==
        RefusalCode::ReceiptInvalid);
}

TEST_CASE("Receipt : refus d'un document incomplet ou d'une autre version", "[prepare][receipt]") {
  const std::string text = receipt_text(seal_receipt(sample()).value());
  CHECK(test::refusal(parse_receipt(replaced(text, "version = 1", "version = 2"))) ==
        RefusalCode::ReceiptInvalid);
  CHECK(test::refusal(parse_receipt(replaced(text, "pairs = 600\n", ""))) == RefusalCode::ReceiptInvalid);
  CHECK(test::refusal(parse_receipt(replaced(text, "pairs = 600", "pairs = -1"))) ==
        RefusalCode::ReceiptInvalid);
  CHECK(test::refusal(parse_receipt(replaced(text, "pairs = 600", "pairs = \"600\""))) ==
        RefusalCode::ReceiptInvalid);
  CHECK(test::refusal(parse_receipt(text.substr(0, text.find("[seal]")))) == RefusalCode::ReceiptInvalid);
  CHECK(test::refusal(parse_receipt("pas du [TOML")) == RefusalCode::ReceiptInvalid);
}

TEST_CASE("Receipt : une valeur hors plage est refusée au scellement", "[prepare][receipt]") {
  auto negative = sample();
  negative.bench.pair_ms_p50 = -1.0;
  CHECK(test::refusal(seal_receipt(negative)) == RefusalCode::FieldOutOfRange);
  auto unordered = sample();
  unordered.bench.pair_ms_p99 = 30.0;  // au-delà du max
  CHECK(test::refusal(seal_receipt(unordered)) == RefusalCode::FieldOutOfRange);
  auto infinite = sample();
  infinite.goldens[0].psnr_db = std::numeric_limits<double>::infinity();
  CHECK(test::refusal(seal_receipt(infinite)) == RefusalCode::FieldOutOfRange);
  auto flip = sample();
  flip.goldens[0].flip = 1.5;
  CHECK(test::refusal(seal_receipt(flip)) == RefusalCode::FieldOutOfRange);
  auto engine = sample();
  engine.engines[0].sha256 = "court";
  CHECK(test::refusal(seal_receipt(engine)) == RefusalCode::FieldOutOfRange);
}

TEST_CASE("Receipt : réels et chaînes se relisent au bit près", "[prepare][receipt]") {
  auto r = sample();
  r.driver.device = "Carte \"spéciale\" \\ tab\t日本";
  r.bench = {
      .pairs = 1, .pair_ms_mean = 0.1, .pair_ms_p50 = 1e-7, .pair_ms_p99 = 22.0, .pair_ms_max = 123456.789};
  r.goldens[0].flip = 0.0;
  const auto parsed = parse_receipt(receipt_text(seal_receipt(r).value()));
  REQUIRE(parsed);
  CHECK(parsed->driver.device == r.driver.device);
  CHECK(parsed->bench.pair_ms_mean == 0.1);
  CHECK(parsed->bench.pair_ms_p50 == 1e-7);
  CHECK(parsed->bench.pair_ms_p99 == 22.0);
  CHECK(parsed->bench.pair_ms_max == 123456.789);
}

TEST_CASE("qualified : goldens réussis, banc mesuré, aucune attente hôte sur GPU", "[prepare][receipt]") {
  CHECK(qualified(sample()));
  auto waits = sample();
  waits.bench.host_waits = 2;
  CHECK_FALSE(qualified(waits));
  waits.driver.needs_host_sync = true;  // un backend sans attente côté GPU les compte normalement
  CHECK(qualified(waits));
  auto failing = sample();
  failing.goldens[0].psnr_db = 35.0;
  CHECK_FALSE(qualified(failing));
  auto no_golden = sample();
  no_golden.goldens.clear();
  CHECK_FALSE(qualified(no_golden));
  auto no_bench = sample();
  no_bench.bench.pairs = 0;
  CHECK_FALSE(qualified(no_bench));
}

TEST_CASE("write_receipt et read_receipt sur disque", "[prepare][receipt]") {
  const test::TempDir dir;
  test::write_file(dir / "moteur.engine", "octets du moteur");
  auto r = sample();
  r.engines = {record_engine("span", "aji-v3-c.engine", dir / "moteur.engine").value()};
  CHECK(r.engines[0].sha256 == sha256_hex("octets du moteur"));
  CHECK(test::refusal(write_receipt(dir / "recu.toml", r)) == RefusalCode::ReceiptInvalid);  // pas scellé
  const auto sealed = seal_receipt(r).value();
  REQUIRE(write_receipt(dir / "recu.toml", sealed));
  CHECK(read_receipt(dir / "recu.toml").value().seal == sealed.seal);
  CHECK(test::refusal(read_receipt(dir / "absent.toml")) == RefusalCode::ReceiptInvalid);
  CHECK(test::refusal(record_engine("x", "y", dir / "absent.engine")) == RefusalCode::CacheIo);
}
