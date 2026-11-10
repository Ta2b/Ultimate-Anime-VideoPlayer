// Le profil GPU d'une lecture (chantier 0.2) : chaque étape mesurée par le backend (paires d'événements de
// son flux), résumée en nombre de mesures, total, médiane et 95e centile. Pur, sans CUDA : le backend le
// remplit (KernelOps::gpu_profile), le lecteur l'écrit dans <stats>.gpu.tsv et dans la mesure du banc.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
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

// Les mesures d'une étape sur toute une séance (A03-22) : le nombre, le total et le minimum sont exacts ; au
// plus `kept` mesures sont gardées pour les centiles, tirées à poids égal sur toute la séance une fois le
// plafond passé (réservoir), au lieu de ne garder que les premières sans le dire.
class StageSamples {
 public:
  static constexpr std::size_t kKept = std::size_t{1} << 18;
  explicit StageSamples(std::size_t kept = kKept) : kept_(std::max<std::size_t>(kept, 1)) {}

  void add(float ms) {
    ++count_;
    total_ms_ += ms;
    min_ms_ = count_ == 1 ? ms : std::min(min_ms_, ms);
    if (samples_.size() < kept_) {
      samples_.push_back(ms);
      return;
    }
    // Générateur congruentiel : le tirage n'a pas à être imprévisible, seulement sans biais visible.
    state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
    if (const std::uint64_t at = (state_ >> 33U) % count_; at < kept_) samples_[at] = ms;
  }
  std::uint64_t count() const { return count_; }
  double total_ms() const { return total_ms_; }
  float min_ms() const { return min_ms_; }
  bool sampled() const { return count_ > samples_.size(); }  // les centiles viennent d'un échantillon
  const std::vector<float>& samples() const { return samples_; }
  // Le résumé de la séance entière : nombre et total exacts, centiles de l'échantillon.
  GpuStageSummary summary(std::string name) const {
    GpuStageSummary out = summarize(std::move(name), samples_);
    out.count = static_cast<std::size_t>(count_);
    out.total_ms = total_ms_;
    return out;
  }

 private:
  std::size_t kept_;
  std::vector<float> samples_;
  std::uint64_t count_ = 0;
  double total_ms_ = 0.0;
  float min_ms_ = 0.0F;
  std::uint64_t state_ = 0x9E3779B97F4A7C15ULL;
};

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
