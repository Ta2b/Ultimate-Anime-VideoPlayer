// Reçu de qualification, propre à ajp : le fork n'en produisait pas.
#include "prepare/receipt.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <format>
#include <fstream>
#include <limits>
#include <sstream>
#include <string_view>
#include <utility>
#include <vector>

#include "prepare/cache_io.h"
#include "prepare/receipt_fields.h"
#include "prepare/sha256.h"
#include "prepare/utf8.h"

namespace ajp::prepare {
namespace {

std::unexpected<Refusal> invalid(std::string detail) {
  return std::unexpected(Refusal{RefusalCode::ReceiptInvalid, std::move(detail)});
}

std::unexpected<Refusal> out_of_range(std::string detail) {
  return std::unexpected(Refusal{RefusalCode::FieldOutOfRange, "reçu : " + std::move(detail)});
}

// « AAAA-MM-JJTHH:MM:SSZ » : un instant UTC à la seconde.
bool is_utc_timestamp(std::string_view text) {
  constexpr std::string_view kShape = "dddd-dd-ddTdd:dd:ddZ";
  if (text.size() != kShape.size()) return false;
  for (std::size_t i = 0; i < text.size(); ++i) {
    const bool digit = text[i] >= '0' && text[i] <= '9';
    if (kShape[i] == 'd' ? !digit : text[i] != kShape[i]) return false;
  }
  return true;
}

// TOML ne lit que de l'UTF-8 : un texte mal formé rendrait le reçu illisible.
bool texts_are_utf8(const Receipt& r) {
  std::vector<std::string_view> texts{r.driver.backend, r.driver.device, r.driver.version};
  for (const EngineRecord& e : r.engines) texts.insert(texts.end(), {e.model, e.cache_key});
  for (const GoldenRecord& g : r.goldens) texts.emplace_back(g.name);
  return std::ranges::all_of(texts, [](std::string_view text) { return is_utf8(text); });
}

// Chaque valeur écrite se relit à l'identique : sinon le sceau ne se vérifierait pas.
std::expected<void, Refusal> check(const Receipt& r) {
  constexpr auto kMaxCount = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  if (!is_utc_timestamp(r.created_utc)) return out_of_range("created_utc : AAAA-MM-JJTHH:MM:SSZ attendu");
  if (!texts_are_utf8(r)) return out_of_range("texte qui n'est pas de l'UTF-8");
  if (r.driver.backend.empty() || r.driver.device.empty() || r.driver.version.empty()) {
    return out_of_range("driver incomplet");
  }
  for (const EngineRecord& e : r.engines) {
    if (e.model.empty() || e.cache_key.empty() || !is_sha256_hex(e.sha256))
      return out_of_range("moteur " + e.model);
  }
  for (const GoldenRecord& g : r.goldens) {
    const bool flip_ok = g.flip >= 0.0 && g.flip <= 1.0 && g.flip_max >= 0.0 && g.flip_max <= 1.0;
    // Un PSNR infini (image identique) est borné par l'appelant, pas ici.
    const bool psnr_ok =
        std::isfinite(g.psnr_db) && std::isfinite(g.psnr_min_db) && g.psnr_db >= 0.0 && g.psnr_min_db >= 0.0;
    if (g.name.empty() || !flip_ok || !psnr_ok) return out_of_range("golden " + g.name);
  }
  const BenchRecord& b = r.bench;
  const bool finite = std::isfinite(b.pair_ms_mean) && std::isfinite(b.pair_ms_max);
  const bool ordered = b.pair_ms_p50 >= 0.0 && b.pair_ms_p50 <= b.pair_ms_p99 &&
                       b.pair_ms_p99 <= b.pair_ms_max && b.pair_ms_mean >= 0.0 &&
                       b.pair_ms_mean <= b.pair_ms_max;
  if (!finite || !ordered) return out_of_range("bench : 0 ≤ p50 ≤ p99 ≤ max et moyenne ≤ max attendus");
  if (b.pairs > kMaxCount || b.host_waits > kMaxCount || b.truehdr_waits > kMaxCount) {
    return out_of_range("bench : compteur au-delà de 2^63");
  }
  return {};
}

std::expected<std::uint64_t, Refusal> parse_plan_hash(const std::string& text) {
  std::uint64_t value = 0;
  const auto* end = text.data() + text.size();
  const auto [ptr, error] = std::from_chars(text.data(), end, value, 16);
  if (text.size() != 16 || error != std::errc{} || ptr != end || text != std::format("{:016x}", value)) {
    return invalid("plan_hash : 16 chiffres hexadécimaux minuscules attendus");
  }
  return value;
}

}  // namespace

std::string driver_hash(const DriverRecord& driver) {
  Sha256 hash;
  hash.text("ajp-driver-v1");
  hash.text(driver.backend);
  hash.text(driver.device);
  hash.text(driver.version);
  return hash.hex();
}

std::string canonical_text(const Receipt& r) {
  std::string out = "# Reçu de qualification ajp. Le sceau couvre tout ce qui précède [seal].\n";
  out += std::format("version = {}\n", kReceiptVersion);
  out += std::format("plan_hash = \"{:016x}\"\n", r.plan_hash);
  out += "created_utc = " + toml_string(r.created_utc) + "\n";
  out += "\n[driver]\n";
  out += "backend = " + toml_string(r.driver.backend) + "\n";
  out += "device = " + toml_string(r.driver.device) + "\n";
  out += "version = " + toml_string(r.driver.version) + "\n";
  out += std::format("needs_host_sync = {}\n", r.driver.needs_host_sync);
  out += "hash = " + toml_string(driver_hash(r.driver)) + "\n";
  out += "\n[bench]\n";
  out += std::format("pairs = {}\n", r.bench.pairs);
  out += "pair_ms_mean = " + toml_real(r.bench.pair_ms_mean) + "\n";
  out += "pair_ms_p50 = " + toml_real(r.bench.pair_ms_p50) + "\n";
  out += "pair_ms_p99 = " + toml_real(r.bench.pair_ms_p99) + "\n";
  out += "pair_ms_max = " + toml_real(r.bench.pair_ms_max) + "\n";
  out += std::format("host_waits = {}\n", r.bench.host_waits);
  out += std::format("truehdr_waits = {}\n", r.bench.truehdr_waits);
  for (const EngineRecord& e : r.engines) {
    out += "\n[[engines]]\n";
    out += "model = " + toml_string(e.model) + "\n";
    out += "cache_key = " + toml_string(e.cache_key) + "\n";
    out += "sha256 = " + toml_string(e.sha256) + "\n";
  }
  for (const GoldenRecord& g : r.goldens) {
    out += "\n[[goldens]]\n";
    out += "name = " + toml_string(g.name) + "\n";
    out += "flip = " + toml_real(g.flip) + "\n";
    out += "flip_max = " + toml_real(g.flip_max) + "\n";
    out += "psnr_db = " + toml_real(g.psnr_db) + "\n";
    out += "psnr_min_db = " + toml_real(g.psnr_min_db) + "\n";
  }
  return out;
}

std::expected<Receipt, Refusal> seal_receipt(Receipt receipt) {
  if (auto ok = check(receipt); !ok) return std::unexpected(ok.error());
  receipt.seal = sha256_hex(canonical_text(receipt));
  return receipt;
}

std::string receipt_text(const Receipt& receipt) {
  return canonical_text(receipt) + "\n[seal]\nsha256 = " + toml_string(receipt.seal) + "\n";
}

std::expected<Receipt, Refusal> parse_receipt(std::string_view text) {
  toml::parse_result parsed = toml::parse(text);
  if (!parsed) return invalid(std::format("TOML : {}", parsed.error().description()));
  const toml::table& root = parsed.table();

  FieldReader top(&root, "reçu");
  if (top.count("version") != static_cast<std::uint64_t>(kReceiptVersion) && !top.error()) {
    return invalid(std::format("version différente de {}", kReceiptVersion));
  }
  Receipt r{.created_utc = top.text("created_utc")};
  const std::string plan_hash = top.text("plan_hash");
  FieldReader driver(root["driver"].as_table(), "driver");
  r.driver = {.backend = driver.text("backend"),
              .device = driver.text("device"),
              .version = driver.text("version"),
              .needs_host_sync = driver.flag("needs_host_sync")};
  const std::string stored_driver_hash = driver.text("hash");
  FieldReader bench(root["bench"].as_table(), "bench");
  r.bench = {.pairs = bench.count("pairs"),
             .pair_ms_mean = bench.real("pair_ms_mean"),
             .pair_ms_p50 = bench.real("pair_ms_p50"),
             .pair_ms_p99 = bench.real("pair_ms_p99"),
             .pair_ms_max = bench.real("pair_ms_max"),
             .host_waits = bench.count("host_waits"),
             .truehdr_waits = bench.count("truehdr_waits")};
  FieldReader seal(root["seal"].as_table(), "seal");
  r.seal = seal.text("sha256");
  for (const FieldReader* reader : {&top, &driver, &bench, &seal}) {
    if (reader->error()) return std::unexpected(*reader->error());
  }

  auto engines = read_tables(root, "engines", [](FieldReader& f) {
    return EngineRecord{
        .model = f.text("model"), .cache_key = f.text("cache_key"), .sha256 = f.text("sha256")};
  });
  if (!engines) return std::unexpected(engines.error());
  r.engines = std::move(*engines);
  auto goldens = read_tables(root, "goldens", [](FieldReader& f) {
    return GoldenRecord{.name = f.text("name"),
                        .flip = f.real("flip"),
                        .flip_max = f.real("flip_max"),
                        .psnr_db = f.real("psnr_db"),
                        .psnr_min_db = f.real("psnr_min_db")};
  });
  if (!goldens) return std::unexpected(goldens.error());
  r.goldens = std::move(*goldens);

  auto hash = parse_plan_hash(plan_hash);
  if (!hash) return std::unexpected(hash.error());
  r.plan_hash = *hash;
  if (auto ok = check(r); !ok) return invalid(ok.error().detail);
  if (stored_driver_hash != driver_hash(r.driver)) return invalid("hachage du pilote faux");
  if (r.seal != sha256_hex(canonical_text(r))) return invalid("sceau faux : le reçu a été modifié");
  return r;
}

bool qualified(const Receipt& r) {
  const bool goldens_pass =
      !r.goldens.empty() && std::ranges::all_of(r.goldens, [](const GoldenRecord& g) { return g.passed(); });
  const bool waits_ok = r.driver.needs_host_sync || r.bench.host_waits == 0;
  return goldens_pass && r.bench.pairs > 0 && waits_ok;
}

std::expected<void, Refusal> write_receipt(const std::filesystem::path& file, const Receipt& receipt) {
  if (receipt.seal != sha256_hex(canonical_text(receipt)))
    return invalid("reçu non scellé ou modifié après sceau");
  return write_atomic(file, receipt_text(receipt));
}

std::expected<Receipt, Refusal> read_receipt(const std::filesystem::path& file) {
  std::ifstream stream(file, std::ios::binary);
  if (!stream) return invalid("reçu absent ou illisible : " + utf8_path(file));
  std::ostringstream text;
  text << stream.rdbuf();
  return parse_receipt(text.str());
}

std::expected<EngineRecord, Refusal> record_engine(std::string model, std::string cache_key,
                                                   const std::filesystem::path& engine) {
  auto sha256 = sha256_file(engine);
  if (!sha256) return std::unexpected(sha256.error());
  return EngineRecord{
      .model = std::move(model), .cache_key = std::move(cache_key), .sha256 = std::move(*sha256)};
}

}  // namespace ajp::prepare
