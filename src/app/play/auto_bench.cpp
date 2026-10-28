#include "app/play/auto_bench.h"

#include <algorithm>
#include <format>
#include <fstream>
#include <sstream>
#include <utility>

#include "core/user_overlay.h"

#ifndef TOML_EXCEPTIONS
#define TOML_EXCEPTIONS 0
#endif
#include <toml++-ajp/toml.hpp>

namespace ajp::app::play {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

constexpr std::int64_t kDefaultPeriodUs = 16'667;  // 60 Hz

}  // namespace

bool holds(const BenchRun& run) {
  if (run.presents < kMinPresents) return false;
  const double lost = static_cast<double>(run.dropped + run.missing) / static_cast<double>(run.presents);
  const std::int64_t period = run.period_us > 0 ? run.period_us : kDefaultPeriodUs;
  return lost <= kMaxLossRate && run.lag_p95_us <= kMaxLagPeriods * period;
}

std::optional<Tier> next_level(const std::vector<BenchRun>& runs, Tier estimated) {
  if (runs.empty()) return estimated;
  const auto tried = [&](Tier level) {
    return std::ranges::any_of(runs, [&](const BenchRun& run) { return run.level == level; });
  };
  const BenchRun& last = runs.back();
  const auto at = static_cast<int>(last.level);
  if (holds(last)) {
    if (at + 1 >= kTierCount || tried(static_cast<Tier>(at + 1))) return std::nullopt;
    return static_cast<Tier>(at + 1);
  }
  if (at == 0 || tried(static_cast<Tier>(at - 1))) return std::nullopt;
  // Un niveau plus bas qui tient déjà (essayé avant) : rien de plus à chercher.
  return static_cast<Tier>(at - 1);
}

Tier best_level(const std::vector<BenchRun>& runs) {
  Tier best = Tier::Economy;
  for (const BenchRun& run : runs) {
    if (holds(run) && run.level > best) best = run.level;
  }
  return best;
}

std::string format_bench(const BenchResult& result) {
  toml::table root;
  root.insert("version", std::int64_t{1});
  root.insert("gpu", result.gpu);
  root.insert("driver", result.driver);
  root.insert("level", static_cast<std::int64_t>(result.level));
  root.insert("date", result.date);
  std::ostringstream out;
  out << root << '\n';
  return out.str();
}

std::expected<BenchResult, Refusal> parse_bench(std::string_view text) {
  toml::parse_result parsed = toml::parse(text);
  if (!parsed) return refuse(RefusalCode::ProfileUnreadable, std::format("bench.toml : {}", parsed.error().description()));
  const toml::table& root = parsed.table();
  BenchResult result;
  result.gpu = root["gpu"].value_or(std::string());
  result.driver = root["driver"].value_or(std::string());
  result.date = root["date"].value_or(std::string());
  const std::int64_t level = root["level"].value_or(std::int64_t{-1});
  if (result.gpu.empty()) return refuse(RefusalCode::FieldOutOfRange, "bench.toml : gpu manque");
  if (level < 0 || level >= kTierCount) return refuse(RefusalCode::FieldOutOfRange, "bench.toml : level hors de 0 à 4");
  result.level = static_cast<Tier>(level);
  return result;
}

std::expected<BenchResult, Refusal> load_bench(const std::filesystem::path& file) {
  std::ifstream in(file, std::ios::binary);
  if (!in) return refuse(RefusalCode::FileUnreadable, file.string());
  std::ostringstream text;
  text << in.rdbuf();
  return parse_bench(text.str());
}

std::expected<void, Refusal> save_bench(const std::filesystem::path& file, const BenchResult& result) {
  return core::write_text_atomically(file, format_bench(result));
}

std::string format_reading(const BenchReading& reading) {
  toml::table root;
  root.insert("gpu", reading.gpu);
  root.insert("driver", reading.driver);
  root.insert("level", static_cast<std::int64_t>(reading.run.level));
  root.insert("presents", reading.run.presents);
  root.insert("dropped", reading.run.dropped);
  root.insert("missing", reading.run.missing);
  root.insert("lag_p95_us", reading.run.lag_p95_us);
  root.insert("period_us", reading.run.period_us);
  root.insert("interval_median_us", reading.interval_median_us);
  root.insert("interval_p95_us", reading.interval_p95_us);
  root.insert("render_median_us", reading.render_median_us);
  std::ostringstream out;
  out << root << '\n';
  return out.str();
}

std::expected<BenchReading, Refusal> parse_reading(std::string_view text) {
  toml::parse_result parsed = toml::parse(text);
  if (!parsed) return refuse(RefusalCode::ProfileUnreadable, std::format("essai : {}", parsed.error().description()));
  const toml::table& root = parsed.table();
  BenchReading reading;
  reading.gpu = root["gpu"].value_or(std::string());
  reading.driver = root["driver"].value_or(std::string());
  const std::int64_t level = root["level"].value_or(std::int64_t{-1});
  if (level < 0 || level >= kTierCount) return refuse(RefusalCode::FieldOutOfRange, "essai : level hors de 0 à 4");
  reading.run.level = static_cast<Tier>(level);
  reading.run.presents = root["presents"].value_or(std::int64_t{0});
  reading.run.dropped = root["dropped"].value_or(std::int64_t{0});
  reading.run.missing = root["missing"].value_or(std::int64_t{0});
  reading.run.lag_p95_us = root["lag_p95_us"].value_or(std::int64_t{0});
  reading.run.period_us = root["period_us"].value_or(std::int64_t{0});
  reading.interval_median_us = root["interval_median_us"].value_or(std::int64_t{0});
  reading.interval_p95_us = root["interval_p95_us"].value_or(std::int64_t{0});
  reading.render_median_us = root["render_median_us"].value_or(std::int64_t{0});
  return reading;
}

std::expected<BenchReading, Refusal> load_reading(const std::filesystem::path& file) {
  std::ifstream in(file, std::ios::binary);
  if (!in) return refuse(RefusalCode::FileUnreadable, file.string());
  std::ostringstream text;
  text << in.rdbuf();
  return parse_reading(text.str());
}

std::expected<void, Refusal> save_reading(const std::filesystem::path& file, const BenchReading& reading) {
  return core::write_text_atomically(file, format_reading(reading));
}

Tier effective_tier(const GpuFacts& gpu, const std::optional<BenchResult>& measured) {
  if (measured && measured->gpu == gpu.name && measured->driver == gpu.driver) return measured->level;
  return estimate_tier(gpu).tier;
}

}  // namespace ajp::app::play
