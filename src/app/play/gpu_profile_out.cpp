#include "app/play/gpu_profile_out.h"

#include <vector>

#include "app/play/auto_bench.h"
#include "core/log.h"
#include "core/text_file.h"

namespace ajp::app::play {

std::filesystem::path gpu_profile_path(const std::filesystem::path& stats) {
  std::filesystem::path out = stats;
  out.replace_extension(".gpu.tsv");
  return out;
}

void keep_gpu_profile(std::span<const gpu::GpuStageSummary> stages,
                      const std::optional<std::filesystem::path>& stats,
                      const std::optional<std::filesystem::path>& bench_out) {
  if (stages.empty()) return;
  const auto log = log_channel("app");
  for (const gpu::GpuStageSummary& s : stages)
    log->info("profil GPU : {} · {} mesures, total {:.0f} ms, médiane {:.2f} ms, p95 {:.2f} ms", s.name,
              s.count, s.total_ms, s.median_ms, s.p95_ms);
  if (stats) {
    if (auto written = core::write_text_atomically(gpu_profile_path(*stats), gpu::gpu_profile_tsv(stages));
        !written)
      log->warn("{}", written.error().detail);
  }
  if (bench_out) {
    auto reading = load_reading(*bench_out);
    if (!reading) return;  // pas de mesure écrite (essai refusé avant) : rien à compléter
    reading->gpu_stages.assign(stages.begin(), stages.end());
    if (auto written = save_reading(*bench_out, *reading); !written) log->warn("{}", written.error().detail);
  }
}

void write_bench_reading(const std::filesystem::path& file, const GpuFacts& gpu, const FrameStats& seen,
                         const BenchChain& chain) {
  const BenchReading reading{.gpu = gpu.name,
                             .driver = gpu.driver,
                             .run = {.level = chain.level,
                                     .presents = seen.presents,
                                     .dropped = seen.dropped,
                                     .missing = seen.missing,
                                     .lag_p95_us = seen.lag.percentile(95),
                                     .period_us = chain.period_us},
                             .interval_median_us = seen.frame_interval.median(),
                             .interval_p95_us = seen.frame_interval.percentile(95),
                             .interval_p99_us = seen.frame_interval.percentile(99),
                             .interval_mean_us = static_cast<std::int64_t>(seen.frame_interval.mean()),
                             .render_median_us = seen.render.median(),
                             .work_height = chain.work_height,
                             .rife_model = chain.rife_model,
                             .span_variant = chain.span_variant,
                             .engine = chain.engine,
                             .new_frames = seen.new_frames,
                             .uninterpolated = seen.uninterpolated};
  if (auto written = save_reading(file, reading); !written)
    log_channel("app")->warn("{}", written.error().detail);
}

}  // namespace ajp::app::play
