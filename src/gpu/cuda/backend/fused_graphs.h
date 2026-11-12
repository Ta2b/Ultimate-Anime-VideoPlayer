// Ce que partagent les moteurs fusionnés (RIFE, DistilDRBA) : leurs graphes CUDA, une chaîne de moteurs
// TensorRT et de noyaux de colle capturée en un seul graphe par jeu d'adresses, les tampons de leur essai au
// chargement et les tenseurs de leurs étapes. Le registre est GraphBook, celui de TrtEngine et du
// SPAN FP8 : mêmes captures bornées, même éviction des jeux d'adresses périmés, même guérison après une série
// de rejeux réussis (un retour de veille ne coupe pas les graphes pour toute la séance), mêmes compteurs et
// même notice.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <expected>
#include <string>
#include <string_view>
#include <vector>

#include "core/types.h"
#include "gpu/backend.h"
#include "gpu/graph_book.h"

namespace ajp::gpu::cuda {

// Un tenseur FP16 n × c × h × w à l'adresse handle + offset.
inline Tensor fused_tensor(std::uintptr_t handle, std::uint64_t offset, std::int64_t c, int h, int w,
                           std::int64_t n = 1) {
  return Tensor{.type = TensorType::F16, .shape = {.dims = {n, c, h, w}}, .handle = handle, .offset = offset};
}

inline void* fused_address(const Tensor& tensor) {
  return reinterpret_cast<void*>(tensor.handle + tensor.offset);  // NOLINT(performance-no-int-to-ptr)
}

// Les tampons et le flux de l'essai au chargement, libérés quoi qu'il arrive.
struct FusedScratch {
  std::array<void*, 3> buffers{};
  cudaStream_t stream = nullptr;
  FusedScratch() = default;
  FusedScratch(const FusedScratch&) = delete;
  FusedScratch& operator=(const FusedScratch&) = delete;
  ~FusedScratch() {
    if (stream != nullptr) cudaStreamDestroy(stream);
    for (void* buffer : buffers) {
      if (buffer != nullptr) cudaFree(buffer);
    }
  }
};

class FusedGraphs {
 public:
  // Les adresses d'entrée et de sortie (0 pour une place libre), la zone d'activation, sa génération, et
  // un drapeau de variante (DistilDRBA : temps seul).
  using Key = std::array<std::uintptr_t, 7>;
  static constexpr int kCaptureAttempts = 3;

  // who : le nom du moteur dans les refus et le journal (« RIFE fusionné »).
  FusedGraphs(std::size_t capacity, std::string who);
  ~FusedGraphs();  // détruit les graphes ; les compteurs vont au journal du module
  FusedGraphs(const FusedGraphs&) = delete;
  FusedGraphs& operator=(const FusedGraphs&) = delete;

  // Oublie tous les graphes (zone changée, fin de l'essai au chargement) ; les séries d'échecs repartent de
  // zéro (GraphBook::clear).
  void clear();
  // La chaîne enfilée sur stream par run() : rejouée par le graphe de key, capturée au deuxième passage,
  // ou lancée directement (premier passage, registre plein, graphes coupés). Une capture ou un rejeu refusés
  // ne sont pas un refus : run() fait le travail en direct. Une erreur collante : contexte perdu.
  template <typename Run>
  std::expected<void, Refusal> launch(cudaStream_t stream, const Key& key, Run&& run) {
    if (book_.skip()) return run();
    const GraphBook<Key>::Pick pick = book_.next(key);
    if (pick.step == GraphStep::Direct || pick.step == GraphStep::Overflow || pick.step == GraphStep::Warm)
      return run();  // Warm : le premier lancement fait ses allocations hors capture
    if (execs_.size() <= pick.index) execs_.resize(pick.index + 1, nullptr);
    if (pick.step == GraphStep::Capture) {
      destroy(pick.index);  // un emplacement repris par un autre jeu d'adresses garde l'ancien graphe
      cudaGraph_t captured = nullptr;
      std::expected<void, Refusal> queued;
      cudaError_t error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
      if (error == cudaSuccess) {
        queued = run();
        error = cudaStreamEndCapture(stream, &captured);  // termine aussi une capture invalidée
      }
      const std::expected<bool, Refusal> ready = instantiate(pick.index, error, captured, queued);
      if (!ready) return std::unexpected(ready.error());
      if (!*ready) return run();  // capture refusée : rien n'a tourné, le lancement direct fait le travail
    }
    const std::expected<bool, Refusal> replayed = replay(stream, pick.index);
    if (!replayed) return std::unexpected(replayed.error());
    if (*replayed) return {};
    return run();  // rejeu refusé : lancement direct, recapture au passage suivant
  }

  // Le mot pour le journal du backend (graphes coupés, registre plein), une fois ; vide sinon.
  std::string take_notice() { return book_.take_notice(); }
  const GraphCounters& counters() const { return book_.counters(); }

 private:
  void destroy(std::size_t index);
  // Le graphe capturé instancié dans execs_[index] : vrai ; faux si la capture est refusée (comptée par le
  // registre, qui coupe les graphes après kCaptureAttempts refus de suite) ; un refus si l'erreur est
  // collante. captured est détruit dans tous les cas.
  std::expected<bool, Refusal> instantiate(std::size_t index, cudaError_t error, cudaGraph_t captured,
                                           const std::expected<void, Refusal>& queued);
  // Le graphe de index rejoué : vrai ; faux sur un refus sans erreur collante (graphe détruit, compté) ; un
  // refus si l'erreur est collante.
  std::expected<bool, Refusal> replay(cudaStream_t stream, std::size_t index);

  GraphBook<Key> book_;
  std::vector<cudaGraphExec_t> execs_;  // les graphes, rangés par l'indice du registre
  std::string who_;
};

}  // namespace ajp::gpu::cuda
