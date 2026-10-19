#include "app/play/chain_stats.h"

#include <algorithm>
#include <cctype>
#include <format>
#include <utility>

#include "app/play/build_progress.h"
#include "app/play/settings.h"

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
    const bool loaded = is_rife ? rife && past.onnx.filename() == rife->filename() &&
                                      past.width == rife_size.padded_width &&
                                      past.height == rife_size.padded_height
                                : past.onnx.filename() == span.filename() && past.width == span_size.width &&
                                      past.height == span_size.height;
    badges.push_back(EngineBadge{
        .name = is_rife ? std::format("RIFE · {}", past.onnx.stem().string())
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
  return std::format("{} ×{} : {} paires, {} interpolées, {} dessins tenus, {} coupes, {} en retard", engine_,
                     live_.rife_factor(), c.pairs, c.interpolated, c.held, c.cuts, c.late);
}

std::optional<std::string> ChainStats::memory() const {
  const auto memory = kernels_.device_memory();
  if (!memory) return std::nullopt;
  constexpr double kMiB = 1024.0 * 1024.0;
  return std::format("mémoire vidéo {:.0f} / {:.0f} Mio", static_cast<double>(memory->used) / kMiB,
                     static_cast<double>(memory->total) / kMiB);
}

}  // namespace ajp::app::play
