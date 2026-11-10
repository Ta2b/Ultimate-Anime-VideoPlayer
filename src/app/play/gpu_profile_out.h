// Le profil GPU de fin de lecture (chantier 0.2, gpu/gpu_profile.h) : au journal, dans <stats>.gpu.tsv à côté
// de --stats, et dans la mesure du banc (--bench-out) déjà écrite.
#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>

#include "app/play/first_run.h"
#include "app/play/frame_stats.h"
#include "app/play/gpu_tier.h"
#include "gpu/gpu_profile.h"

namespace ajp::app::play {

// « run.tsv » → « run.gpu.tsv ».
std::filesystem::path gpu_profile_path(const std::filesystem::path& stats);

void keep_gpu_profile(std::span<const gpu::GpuStageSummary> stages,
                      const std::optional<std::filesystem::path>& stats,
                      const std::optional<std::filesystem::path>& bench_out);

// La chaîne qu'un essai du banc a jouée, pour sa mesure.
struct BenchChain {
  Tier level = Tier::Normal;
  std::int64_t period_us = 0;  // période de l'écran ; 0 : inconnue
  int work_height = 0;
  std::string rife_model;
  std::string span_variant;
  std::string engine;  // le moteur en service (« tensorrt », « vulkan »…)
};

// Écrit la mesure du banc (--bench-out, auto_bench.h) : ce que la lecture a présenté et la chaîne jouée. Un
// échec d'écriture va au journal.
void write_bench_reading(const std::filesystem::path& file, const GpuFacts& gpu, const FrameStats& seen,
                         const BenchChain& chain);

}  // namespace ajp::app::play
