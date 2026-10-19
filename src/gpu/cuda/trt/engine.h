// Moteur TensorRT désérialisé et son contexte d'exécution.
//
// Un TrtEngine appartient à un seul thread (le thread gpu) : IExecutionContext
// n'est pas réentrant. Le chemin d'une image, bind() puis enqueue(), n'alloue
// rien sur l'hôte ; seuls les refus construisent un message.
//
// Ordre d'emploi : set_input_shape() pour chaque entrée dynamique (une fois par
// changement de forme), bind() pour chaque entrée et sortie, enqueue().
#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>
#include <expected>
#include <memory>
#include <span>
#include <string_view>

#include "core/types.h"
#include "gpu/cuda/trt/types.h"

namespace ajp::gpu::trt {

// Événements facultatifs d'une exécution. input_consumed est signalé dès que
// TensorRT a fini de lire les entrées, qui peuvent alors être réécrites pendant
// que le calcul continue ; done, une fois les sorties écrites.
struct EnqueueEvents {
  cudaEvent_t input_consumed = nullptr;
  cudaEvent_t done = nullptr;
};

class TrtEngine {
 public:
  TrtEngine(TrtEngine&& other) noexcept;
  TrtEngine& operator=(TrtEngine&& other) noexcept;
  TrtEngine(const TrtEngine&) = delete;
  TrtEngine& operator=(const TrtEngine&) = delete;
  ~TrtEngine();

  // Entrées et sorties dans l'ordre du moteur.
  std::span<const TensorDesc> tensors() const;
  // nullptr si le moteur n'a pas ce tenseur.
  const TensorDesc* find(std::string_view name) const;

  int profile_count() const;
  int profile() const;
  // Mémoire d'activation que le contexte a réservée sur l'appareil.
  std::int64_t device_memory_bytes() const;

  // Passe au profil d'optimisation index (0 au chargement), sur stream. Les
  // adresses liées sont oubliées : TensorRT les attache au profil.
  std::expected<void, Refusal> select_profile(int index, cudaStream_t stream);

  // Forme d'une entrée pour les exécutions suivantes ; refusée (ShapeOutOfProfile)
  // si son rang diffère ou si une dimension sort du profil courant.
  std::expected<void, Refusal> set_input_shape(std::string_view name, std::span<const std::int64_t> dims);

  // Forme d'un tenseur pour les formes d'entrée posées, sorties comprises.
  std::expected<Dims, Refusal> shape(std::string_view name) const;

  // Adresse sur l'appareil d'une entrée ou d'une sortie, gardée pour les
  // exécutions suivantes jusqu'au prochain bind() du même tenseur.
  std::expected<void, Refusal> bind(std::string_view name, void* device);

  // enqueueV3 sur stream, qui devrait être non bloquant (CudaStream). Refuse si
  // un tenseur n'a pas d'adresse ou si une forme d'entrée manque.
  std::expected<void, Refusal> enqueue(cudaStream_t stream, const EnqueueEvents& events = {});

 private:
  friend class TrtRuntime;
  struct Impl;
  explicit TrtEngine(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace ajp::gpu::trt
