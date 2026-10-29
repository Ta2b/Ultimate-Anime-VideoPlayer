#include "app/play/chain_stats.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <format>
#include <utility>

#include "app/play/build_progress.h"
#include "app/play/i18n.h"
#include "app/play/settings.h"
#include "app/play/span_int8.h"
#include "core/clock.h"

namespace ajp::app::play {
namespace {

// La dernière raison d'un changement de palier, dans la langue de l'interface (core/auto_rate.h :
// AutoReason).
std::string_view auto_reason_label(core::AutoReason reason) {
  switch (reason) {
    case core::AutoReason::Start:
      return i18n::tr("start");
    case core::AutoReason::Memory:
      return i18n::tr("remembered start");
    case core::AutoReason::Rise:
      return i18n::tr("lighter scene");
    case core::AutoReason::Late:
      return i18n::tr("late pairs");
    case core::AutoReason::Missed:
      return i18n::tr("missed frames");
    case core::AutoReason::Predicted:
      return i18n::tr("cost above the period");
    case core::AutoReason::Watchdog:
      return i18n::tr("no pair left");
    case core::AutoReason::Ceiling:
      return i18n::tr("display limit");
  }
  return {};
}

}  // namespace

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
        .detail = i18n::tr_format("input {}x{} · built in {:.0f} s", past.width, past.height, past.seconds),
        .loaded = loaded});
  }
  std::ranges::stable_partition(badges, &EngineBadge::loaded);
  return badges;
}

std::string auto_factor_text(int centi) {
  if (centi % 100 == 0) return std::format("{}", centi / 100);
  if (centi % 10 == 0) return std::format("{},{}", centi / 100, centi % 100 / 10);
  return std::format("{},{:02}", centi / 100, centi % 100);
}

std::string auto_status_text(const core::AutoInfo& info, double source_fps) {
  const int g = std::max(info.granularity, 1);
  const int centi = info.step * 100 / g;
  std::string text = i18n::tr_format("Auto ×{} ({:.0f} fps) · display limit ×{}", auto_factor_text(centi),
                                     source_fps * centi / 100.0, auto_factor_text(info.ceiling * 100 / g));
  if (info.predicted > 0)
    text += i18n::tr_format(" · predicted ×{}", auto_factor_text(info.predicted * 100 / g));
  if (info.cap > 0) text += i18n::tr_format(" · learned limit ×{}", auto_factor_text(info.cap * 100 / g));
  if (info.load_pct >= 0) text += i18n::tr_format(" · GPU limit {} %", info.load_pct);
  if (info.heavy) text += i18n::tr(" · heavy scene");
  text += i18n::tr_format(" · last change: {}", auto_reason_label(info.reason));
  return text;
}

std::string free_status_text(const core::FreeInfo& info) {
  std::string text = i18n::tr_format("Fast · {:.0f} fps", info.fps);
  if (info.load_pct >= 0) text += i18n::tr_format(" · GPU limit {} %", info.load_pct);
  if (info.step_us > 0) text += i18n::tr(" · service gap ") + format_ms(info.step_us);
  if (info.pressure > 1.02) {
    std::string pressure = std::format("{:.2f}", info.pressure);
    std::ranges::replace(pressure, '.', ',');
    text += i18n::tr(" · pressure ×") + pressure;
  }
  if (info.lost > 0) text += i18n::tr_format(" · {} lost", info.lost);
  return text;
}

std::optional<std::string> ChainStats::rife() const {
  if (!interpolates_) return std::nullopt;
  const RifeCounts c = live_.counts();
  std::string line =
      i18n::tr_format("{} {}: {} pairs, {} interpolated, {} held drawings, {} cuts, {} late", engine_,
                      live_.free_active() ? std::string("libre") : factor_text(live_.factor_now()), c.pairs,
                      c.interpolated, c.held, c.cuts, c.late);
  if (live_.auto_active()) line += " · " + auto_status_text(live_.auto_info(host_now_us()), source_fps_);
  if (live_.free_active()) line += " · " + free_status_text(live_.free_info(host_now_us()));
  return line;
}

double ChainStats::output_fps() const {
  if (source_fps_ <= 0.0) return 0.0;
  if (live_.free_active()) return 0.0;  // aucune cadence : l'écart des images est celui du temps de service
  const Factor factor = live_.factor_now();  // rien, 0 ou 1 : pas d'interpolation
  return interpolates_ && factor.den > 0 && factor.num > factor.den ? source_fps_ * factor.num / factor.den
                                                                    : source_fps_;
}

std::int64_t ChainStats::source_period_us() const {
  return source_fps_ > 0.0 ? static_cast<std::int64_t>(std::llround(1e6 / source_fps_)) : 0;
}

std::int64_t ChainStats::output_period_us() const {
  if (live_.free_active()) {
    const std::int64_t step = live_.free_info(host_now_us()).step_us;
    // Sur un écran fixe, un écart entre deux images est un nombre entier de vsyncs : jusqu'au vsync au-dessus
    // du pas, ce n'est pas une image manquée.
    const std::int64_t vsync = live_.vsync_period_us();
    return vsync > 0 && step > 0 ? (step + vsync - 1) / vsync * vsync : step;
  }
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
  return i18n::tr_format("video memory {:.0f} / {:.0f} MiB", static_cast<double>(memory->used) / kMiB,
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
