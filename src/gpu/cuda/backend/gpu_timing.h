// Temps GPU d'une étape, mesuré par paires d'événements CUDA sur le flux du backend et relu
// sans attente hôte : une paire n'est lue qu'une fois son second événement passé. Avec un
// rapporteur, toutes les kReport mesures, la médiane et le 95e centile partent au journal du
// backend ; summary() résume toutes les mesures depuis la création (profil de fin de lecture).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "core/sample_reservoir.h"
#include "gpu/gpu_profile.h"

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
  // Le résumé de toutes les mesures depuis la création (0.2).
  GpuStageSummary profile();
  const std::string& label() const { return label_; }

 private:
  void collect();

  std::string label_;
  Report report_;
  std::array<cudaEvent_t, kPairs> start_{};
  std::array<cudaEvent_t, kPairs> stop_{};
  cudaStream_t open_stream_ = nullptr;  // la file du début de la paire ouverte
  std::size_t begun_ = 0;               // paires commencées
  std::size_t ended_ = 0;               // paires finies (second événement enregistré)
  std::size_t read_ = 0;                // paires relues
  std::vector<float> samples_;
  static constexpr std::size_t kMaxAll = 1 << 18;
  // Toutes les mesures, pour summary() : comptées toutes, un échantillon uniforme au-delà de kMaxAll
  // (A33-05).
  core::SampleReservoir<float> all_{kMaxAll};
};

// La latence d'une source, de son ingestion à la fin de son SPAN (0.2) : les deux bouts sont appariés par une
// clé (l'adresse du tenseur ingéré, que le SPAN lit), sur des files qui peuvent différer (voies) ; les
// événements d'une même carte se comparent d'une file à l'autre. Au plus kOpen sources en vol : la plus
// ancienne pas encore finie est oubliée.
class GpuSpan {
 public:
  static constexpr std::size_t kOpen = 16;
  explicit GpuSpan(std::string label);
  ~GpuSpan();
  GpuSpan(const GpuSpan&) = delete;
  GpuSpan& operator=(const GpuSpan&) = delete;

  void start(std::uintptr_t key, cudaStream_t stream);
  void finish(std::uintptr_t key, cudaStream_t stream);
  std::string summary();
  GpuStageSummary profile();

 private:
  struct Pair {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    std::uintptr_t key = 0;
    enum class State : unsigned char { Free, Started, Finished } state = State::Free;
  };
  void collect();

  std::string label_;
  std::array<Pair, kOpen> pairs_{};
  std::size_t next_ = 0;  // la prochaine paire à prendre (tourne)
  static constexpr std::size_t kMaxAll = 1 << 18;
  core::SampleReservoir<float> all_{kMaxAll};
};

}  // namespace ajp::gpu::cuda
