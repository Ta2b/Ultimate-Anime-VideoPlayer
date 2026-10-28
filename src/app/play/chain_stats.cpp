#include "app/play/chain_stats.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <format>
#include <utility>

#include "app/play/build_progress.h"
#include "app/play/settings.h"
#include "app/play/span_int8.h"
#include "core/clock.h"

namespace ajp::app::play {

std::vector<EngineBadge> engine_badges(const std::filesystem::path& cache_dir,
                                       const std::filesystem::path& span, WorkSize span_size,
                                       const std::optional<std::filesystem::path>& rife,
                                       const kernels::RifeGeometry& rife_size) {
  std::vector<EngineBadge> badges;
  for (const PastBuild& past : past_builds(cache_dir)) {
    std::string file = past.onnx.filename().string();
    std::string lower_file = file;
    std::ranges::transform(lower_file, lower_file.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    const bool is_rife = lower_file.find("rife") != std::string::npos;
    SpanVariantKind variant_kind = SpanVariantKind::Int8;
    const std::optional<std::string> int8 = span_int8_source(file, &variant_kind);
    const bool loaded = is_rife ? rife && past.onnx.filename() == rife->filename() &&
                                      past.width == rife_size.padded_width &&
                                      past.height == rife_size.padded_height
                                : past.onnx.filename() == span.filename() && past.width == span_size.width &&
                                      past.height == span_size.height;
    badges.push_back(EngineBadge{
        .name = is_rife ? std::format("RIFE · {}", past.onnx.stem().string())
                : int8  ? std::format("SPAN · {} · {}", model_label(*int8), variant_text(variant_kind))
                        : std::format("SPAN · {}", model_label(file)),
        .detail = std::format("entrée {}x{} · construit en {:.0f} s", past.width, past.height, past.seconds),
        .loaded = loaded});
  }
  std::ranges::stable_partition(badges, &EngineBadge::loaded);
  return badges;
}

std::optional<std::string> ChainStats::rife() const {
  if (!interpolates_) return std::nullopt;
  const RifeCounts c = live_.counts();
  return std::format("{} {} : {} paires, {} interpolées, {} dessins tenus, {} coupes, {} en retard", engine_,
                     factor_text(live_.factor_now()), c.pairs, c.interpolated, c.held, c.cuts, c.late);
}

double ChainStats::output_fps() const {
  if (source_fps_ <= 0.0) return 0.0;
  const Factor factor = live_.factor_now();  // rien, 0 ou 1 : pas d'interpolation
  return interpolates_ && factor.den > 0 && factor.num > factor.den ? source_fps_ * factor.num / factor.den
                                                                    : source_fps_;
}

std::int64_t ChainStats::source_period_us() const {
  return source_fps_ > 0.0 ? static_cast<std::int64_t>(std::llround(1e6 / source_fps_)) : 0;
}

std::int64_t ChainStats::output_period_us() const {
  const double rate = output_fps();
  return rate > 0.0 ? static_cast<std::int64_t>(std::llround(1e6 / rate)) : 0;
}

std::optional<std::string> ChainStats::read_gpu() const {
  if (telemetry_ == nullptr) return std::nullopt;
  const std::optional<GpuSample> sample = telemetry_->sample(host_now_us());
  if (!sample) return std::nullopt;
  std::string line = telemetry_line(*sample);
  if (const std::string warning = telemetry_warning(*sample); !warning.empty()) line += " · " + warning;
  return line;
}

std::optional<std::string> ChainStats::read_memory() const {
  const auto memory = kernels_.device_memory();
  if (!memory) return std::nullopt;
  constexpr double kMiB = 1024.0 * 1024.0;
  return std::format("mémoire vidéo {:.0f} / {:.0f} Mio", static_cast<double>(memory->used) / kMiB,
                     static_cast<double>(memory->total) / kMiB);
}

void ChainStats::start_sampler() const {
  std::call_once(sampler_started_, [this] {
    sampler_ = std::jthread([this](const std::stop_token& stop) {
      std::mutex idle;
      std::condition_variable_any wake;
      while (!stop.stop_requested()) {
        auto memory = read_memory();
        auto gpu = read_gpu();
        {
          const std::scoped_lock lock(sampled_mutex_);
          memory_text_ = std::move(memory);
          gpu_text_ = std::move(gpu);
        }
        std::unique_lock lock(idle);
        wake.wait_for(lock, stop, std::chrono::milliseconds(500), [] { return false; });
      }
    });
  });
}

std::optional<std::string> ChainStats::memory() const {
  start_sampler();
  const std::scoped_lock lock(sampled_mutex_);
  return memory_text_;
}

std::optional<std::string> ChainStats::gpu() const {
  start_sampler();
  const std::scoped_lock lock(sampled_mutex_);
  return gpu_text_;
}

}  // namespace ajp::app::play
