#include "app/play/auto_bench.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <format>
#include <fstream>
#include <optional>
#include <sstream>
#include <utility>

#include "core/text_file.h"
#include "core/utf8_path.h"

#ifndef TOML_EXCEPTIONS
#define TOML_EXCEPTIONS 0
#endif
#include <toml++-ajp/toml.hpp>

namespace ajp::app::play {
namespace {

constexpr std::int64_t kDefaultPeriodUs = 16'667;  // 60 Hz

}  // namespace

bool holds(const BenchRun& run) {
  if (run.presents < kMinPresents) return false;
  const double lost = static_cast<double>(run.dropped + run.missing) / static_cast<double>(run.presents);
  const std::int64_t period = run.period_us > 0 ? run.period_us : kDefaultPeriodUs;
  return lost <= kMaxLossRate && run.lag_p95_us <= kMaxLagPeriods * period;
}

std::optional<int> next_work_height(int tenue, int source_height) {
  constexpr std::array<int, 5> kHeights{480, 540, 720, 900, 1080};
  for (const int height : kHeights) {
    if (height <= tenue) continue;
    if (source_height > 0 && height > source_height) return std::nullopt;  // jamais plus haut que la source
    return height;
  }
  return std::nullopt;
}

std::optional<std::string> next_rife_model(const std::string& tenu) {
  if (tenu == "rife_v4.22_lite.onnx") return std::string("rife_v4.25.onnx");
  return std::nullopt;
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
  if (result.work_height > 0) root.insert("work_height", static_cast<std::int64_t>(result.work_height));
  if (!result.rife_model.empty()) root.insert("rife_model", result.rife_model);
  std::ostringstream out;
  out << root << '\n';
  return out.str();
}

std::expected<BenchResult, Refusal> parse_bench(std::string_view text) {
  toml::parse_result parsed = toml::parse(text);
  if (!parsed)
    return refuse(RefusalCode::ProfileUnreadable,
                  std::format("bench.toml : {}", parsed.error().description()));
  const toml::table& root = parsed.table();
  BenchResult result;
  result.gpu = root["gpu"].value_or(std::string());
  result.driver = root["driver"].value_or(std::string());
  result.date = root["date"].value_or(std::string());
  const std::int64_t level = root["level"].value_or(std::int64_t{-1});
  if (result.gpu.empty()) return refuse(RefusalCode::FieldOutOfRange, "bench.toml : gpu manque");
  if (level < 0 || level >= kTierCount)
    return refuse(RefusalCode::FieldOutOfRange, "bench.toml : level hors de 0 à 4");
  result.level = static_cast<Tier>(level);
  const std::int64_t height = root["work_height"].value_or(std::int64_t{0});
  if (height < 0 || height > 4320)
    return refuse(RefusalCode::FieldOutOfRange, "bench.toml : work_height hors plage");
  result.work_height = static_cast<int>(height);
  result.rife_model = root["rife_model"].value_or(std::string());
  return result;
}

std::expected<BenchResult, Refusal> load_bench(const std::filesystem::path& file) {
  std::ifstream in(file, std::ios::binary);
  if (!in) return refuse(RefusalCode::FileUnreadable, core::to_utf8(file));
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
  root.insert("interval_p99_us", reading.interval_p99_us);
  root.insert("interval_mean_us", reading.interval_mean_us);
  root.insert("render_median_us", reading.render_median_us);
  root.insert("work_height", static_cast<std::int64_t>(reading.work_height));
  root.insert("rife_model", reading.rife_model);
  root.insert("span_variant", reading.span_variant);
  root.insert("engine", reading.engine);
  root.insert("new_frames", reading.new_frames);
  root.insert("uninterpolated", reading.uninterpolated);
  if (!reading.gpu_stages.empty()) {
    toml::array stages;
    for (const gpu::GpuStageSummary& s : reading.gpu_stages) {
      stages.push_back(toml::table{{"name", s.name},
                                   {"count", static_cast<std::int64_t>(s.count)},
                                   {"total_ms", s.total_ms},
                                   {"median_ms", s.median_ms},
                                   {"p95_ms", s.p95_ms}});
    }
    root.insert("gpu_stage", std::move(stages));
  }
  std::ostringstream out;
  out << root << '\n';
  return out.str();
}

std::expected<BenchReading, Refusal> parse_reading(std::string_view text) {
  toml::parse_result parsed = toml::parse(text);
  if (!parsed)
    return refuse(RefusalCode::ProfileUnreadable, std::format("essai : {}", parsed.error().description()));
  const toml::table& root = parsed.table();
  BenchReading reading;
  reading.gpu = root["gpu"].value_or(std::string());
  reading.driver = root["driver"].value_or(std::string());
  const std::int64_t level = root["level"].value_or(std::int64_t{-1});
  if (level < 0 || level >= kTierCount)
    return refuse(RefusalCode::FieldOutOfRange, "essai : level hors de 0 à 4");
  reading.run.level = static_cast<Tier>(level);
  reading.run.presents = root["presents"].value_or(std::int64_t{0});
  reading.run.dropped = root["dropped"].value_or(std::int64_t{0});
  reading.run.missing = root["missing"].value_or(std::int64_t{0});
  reading.run.lag_p95_us = root["lag_p95_us"].value_or(std::int64_t{0});
  reading.run.period_us = root["period_us"].value_or(std::int64_t{0});
  reading.interval_median_us = root["interval_median_us"].value_or(std::int64_t{0});
  reading.interval_p95_us = root["interval_p95_us"].value_or(std::int64_t{0});
  reading.interval_p99_us = root["interval_p99_us"].value_or(std::int64_t{0});
  reading.interval_mean_us = root["interval_mean_us"].value_or(std::int64_t{0});
  reading.render_median_us = root["render_median_us"].value_or(std::int64_t{0});
  reading.work_height = static_cast<int>(root["work_height"].value_or(std::int64_t{0}));
  reading.rife_model = root["rife_model"].value_or(std::string());
  reading.span_variant = root["span_variant"].value_or(std::string());
  reading.engine = root["engine"].value_or(std::string());
  reading.new_frames = std::max<std::int64_t>(root["new_frames"].value_or(std::int64_t{0}), 0);
  reading.uninterpolated = std::max<std::int64_t>(root["uninterpolated"].value_or(std::int64_t{0}), 0);
  if (const toml::array* stages = root["gpu_stage"].as_array()) {
    for (const toml::node& node : *stages) {
      const toml::table* s = node.as_table();
      if (s == nullptr) continue;
      reading.gpu_stages.push_back(gpu::GpuStageSummary{
          .name = (*s)["name"].value_or(std::string()),
          .count =
              static_cast<std::size_t>(std::max<std::int64_t>((*s)["count"].value_or(std::int64_t{0}), 0)),
          .total_ms = (*s)["total_ms"].value_or(0.0),
          .median_ms = (*s)["median_ms"].value_or(0.0),
          .p95_ms = (*s)["p95_ms"].value_or(0.0)});
    }
  }
  return reading;
}

std::expected<BenchReading, Refusal> load_reading(const std::filesystem::path& file) {
  std::ifstream in(file, std::ios::binary);
  if (!in) return refuse(RefusalCode::FileUnreadable, core::to_utf8(file));
  std::ostringstream text;
  text << in.rdbuf();
  return parse_reading(text.str());
}

std::expected<void, Refusal> save_reading(const std::filesystem::path& file, const BenchReading& reading) {
  return core::write_text_atomically(file, format_reading(reading));
}

namespace {

// La majeure d'un driverVersion écrit en décimal ; rien s'il ne se lit pas.
std::optional<std::uint32_t> driver_major(std::string_view driver) {
  if (driver.empty() || driver.size() > 10) return std::nullopt;
  std::uint64_t raw = 0;
  for (const char c : driver) {
    if (c < '0' || c > '9') return std::nullopt;
    raw = (raw * 10) + static_cast<std::uint64_t>(c - '0');
  }
  if (raw > 0xffffffffULL) return std::nullopt;
  return static_cast<std::uint32_t>(raw >> 22U);
}

}  // namespace

bool driver_branch_changed(std::string_view measured, std::string_view now) {
  if (measured.empty() || now.empty() || measured == now) return false;
  const auto before = driver_major(measured);
  const auto after = driver_major(now);
  // Illisible d'un côté : seule l'égalité des textes dit que c'est le même.
  if (!before || !after) return true;
  return *before != *after;
}

bool measured_here(const BenchResult& measured, const GpuFacts& gpu) {
  return measured.gpu == gpu.name && !driver_branch_changed(measured.driver, gpu.driver);
}

Tier effective_tier(const GpuFacts& gpu, const std::optional<BenchResult>& measured) {
  if (measured && measured_here(*measured, gpu)) return measured->level;
  return estimate_tier(gpu).tier;
}

}  // namespace ajp::app::play
