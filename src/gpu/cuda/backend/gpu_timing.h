// Temps GPU d'une étape, mesuré par paires d'événements CUDA sur le flux du backend et relu
// sans attente hôte : une paire n'est lue qu'une fois son second événement passé. Avec un
// rapporteur, toutes les kReport mesures, la médiane et le 95e centile partent au journal du
// backend ; summary() résume toutes les mesures depuis la création (profil de fin de lecture).
#pragma once

#include <array>
#include <cstddef>
#include <cuda_runtime_api.h>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace ajp::gpu::cuda {

class GpuTiming {
 public:
  static constexpr std::size_t kPairs = 16;
  static constexpr std::size_t kReport = 240;
  using Report = std::function<void(std::string_view)>;

  GpuTiming(std::string label, Report report);
  ~GpuTiming();
  GpuTiming(const GpuTiming&) = delete;
  GpuTiming& operator=(const GpuTiming&) = delete;

  void begin(cudaStream_t stream);
  void end(cudaStream_t stream);
  // « label : médiane 4,37 ms, p95 5,10 ms (1234 mesures) », ou rien sans mesure. Relit les
  // paires finies sans attendre.
  std::string summary();

 private:
  void collect();

  std::string label_;
  Report report_;
  std::array<cudaEvent_t, kPairs> start_{};
  std::array<cudaEvent_t, kPairs> stop_{};
  std::size_t begun_ = 0;  // paires commencées
  std::size_t ended_ = 0;  // paires finies (second événement enregistré)
  std::size_t read_ = 0;   // paires relues
  std::vector<float> samples_;
  std::vector<float> all_;  // toutes les mesures, pour summary() (kMaxAll au plus)
  static constexpr std::size_t kMaxAll = 1 << 18;
};

}  // namespace ajp::gpu::cuda
