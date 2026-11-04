// Le profil GPU de fin de lecture (chantier 0.2, gpu/gpu_profile.h) : au journal, dans <stats>.gpu.tsv à côté
// de --stats, et dans la mesure du banc (--bench-out) déjà écrite.
#pragma once

#include <filesystem>
#include <optional>
#include <span>

#include "gpu/gpu_profile.h"

namespace ajp::app::play {

// « run.tsv » → « run.gpu.tsv ».
std::filesystem::path gpu_profile_path(const std::filesystem::path& stats);

void keep_gpu_profile(std::span<const gpu::GpuStageSummary> stages,
                      const std::optional<std::filesystem::path>& stats,
                      const std::optional<std::filesystem::path>& bench_out);

}  // namespace ajp::app::play
