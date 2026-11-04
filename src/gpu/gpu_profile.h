// Le profil GPU d'une lecture (chantier 0.2) : chaque étape mesurée par le backend (paires d'événements de
// son flux), résumée en nombre de mesures, total, médiane et 95e centile. Pur, sans CUDA : le backend le
// remplit (KernelOps::gpu_profile), le lecteur l'écrit dans <stats>.gpu.tsv et dans la mesure du banc.
#pragma once

#include <algorithm>
#include <cstddef>
#include <format>
#include <numeric>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace ajp::gpu {

struct GpuStageSummary {
  std::string name;
  std::size_t count = 0;
  double total_ms = 0.0;
  double median_ms = 0.0;
  double p95_ms = 0.0;
  friend bool operator==(const GpuStageSummary&, const GpuStageSummary&) = default;
};

// Le résumé de mesures en millisecondes (dans n'importe quel ordre) ; vide : count 0.
inline GpuStageSummary summarize(std::string name, std::vector<float> samples_ms) {
  GpuStageSummary out{.name = std::move(name), .count = samples_ms.size()};
  if (samples_ms.empty()) return out;
  std::ranges::sort(samples_ms);
  out.total_ms = std::accumulate(samples_ms.begin(), samples_ms.end(), 0.0);
  out.median_ms = samples_ms[samples_ms.size() / 2];
  out.p95_ms = samples_ms[samples_ms.size() * 95 / 100];
  return out;
}

inline constexpr std::string_view kGpuProfileHeader = "etape\tmesures\ttotal_ms\tmediane_ms\tp95_ms\n";

// <stats>.gpu.tsv : une ligne par étape, après l'en-tête.
inline std::string gpu_profile_tsv(std::span<const GpuStageSummary> stages) {
  std::string out(kGpuProfileHeader);
  for (const GpuStageSummary& s : stages)
    out +=
        std::format("{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}\n", s.name, s.count, s.total_ms, s.median_ms, s.p95_ms);
  return out;
}

}  // namespace ajp::gpu
