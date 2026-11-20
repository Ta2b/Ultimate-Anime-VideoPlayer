// Mesure optionnelle du retour B→A. La session aller conserve son propre historique d'indices.
#pragma once

#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <string>

#include "gpu/cuda/backend/nvof_session.h"
#include "gpu/cuda/backend/scene_roundtrip_read.h"

namespace kirari::gpu::cuda {

class SceneRoundtrip {
 public:
  static std::expected<std::unique_ptr<SceneRoundtrip>, Refusal> open(int width, int height, int divisor);
  ~SceneRoundtrip();
  SceneRoundtrip(const SceneRoundtrip&) = delete;
  SceneRoundtrip& operator=(const SceneRoundtrip&) = delete;

  // Enfile les lectures des lumas et du vecteur aller, puis le retour sur sa propre file. Vrai dès qu'il
  // faut reprotéger les lumas, même si une lecture partielle a été suivie d'un refus.
  bool start(const NvofSession& forward, int a, int b, std::uint64_t pair);
  void poll();
  std::expected<std::optional<kernels::SceneRoundtripTiles>, Refusal> take(std::uint64_t pair);
  // Seulement à l'arrêt du backend ou au changement de taille : l'appelant peut attendre la dernière paire.
  std::string finish();

 private:
  SceneRoundtrip() = default;
  std::expected<void, Refusal> allocate();
  std::expected<void, Refusal> enqueue(const NvofSession& forward, int a, int b);
  std::size_t tile_count() const;

  std::unique_ptr<NvofSession> reverse_;
  RoundtripShape shape_;
  int divisor_ = 1;
  std::int16_t* forward_ = nullptr;  // copie de l'aller avant que la session ne le réécrive
  kernels::SceneRoundtripTile* device_ = nullptr;
  kernels::SceneRoundtripTile* host_ = nullptr;
  cudaEvent_t inputs_ready_ = nullptr;
  cudaEvent_t begin_ = nullptr;
  cudaEvent_t done_ = nullptr;
  RoundtripTicket slot_;
  std::optional<Refusal> failure_;
  std::optional<kernels::SceneRoundtripTiles> completed_;
  RoundtripStats stats_;
  std::size_t photo_ = 0;
  std::uint64_t started_ = 0;
  std::uint64_t read_ = 0;
  std::uint64_t busy_ = 0;
  double measured_ms_ = 0.0;
};

}  // namespace kirari::gpu::cuda
