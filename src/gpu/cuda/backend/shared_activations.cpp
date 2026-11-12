#include "gpu/cuda/backend/shared_activations.h"

#include <cuda_runtime_api.h>
#include <format>

#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/cuda_warn.h"

namespace ajp::gpu::cuda {

SharedActivations::~SharedActivations() {
  if (data_ == nullptr) return;
  if (release_) {
    release_(data_);
  } else {
    cuda_warn(cudaFree(data_), "cudaFree");
  }
}

void* SharedActivations::allocate(std::size_t bytes) {
  if (alloc_) return alloc_(bytes);
  void* memory = nullptr;
  if (cudaMalloc(&memory, bytes) != cudaSuccess) {
    static_cast<void>(clear_error());
    return nullptr;
  }
  return memory;
}

void SharedActivations::release(void* memory) {
  // Les moteurs déjà posés sur la zone peuvent encore y travailler : rendue derrière toutes les voies quand
  // le backend s'en charge, sinon après synchronisation de l'appareil.
  if (release_) {
    release_(memory);
    return;
  }
  cuda_warn(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  cuda_warn(cudaFree(memory), "cudaFree");
}

std::expected<void, Refusal> SharedActivations::reserve(std::int64_t bytes) {
  if (bytes <= bytes_) return {};
  const auto refused = [bytes] {
    return refuse(RefusalCode::GpuMemoryExhausted,
                  std::format("activations partagées : {} Mo refusés par cudaMalloc", bytes >> 20));
  };
  // La nouvelle zone d'abord, l'ancienne intacte : un refus ne laisse pas aux moteurs déjà posés
  // un pointeur libéré.
  void* fresh = allocate(static_cast<std::size_t>(bytes));
  if (fresh == nullptr) {
    if (data_ == nullptr) return refused();
    // Pas de place pour les deux zones à la fois : l'ancienne part avant le nouvel essai (le pool du backend
    // la reprend dans l'ordre de la file, et se purge avant de refuser).
    const std::int64_t kept = bytes_;
    release(data_);
    data_ = nullptr;
    bytes_ = 0;
    ++generation_;
    fresh = allocate(static_cast<std::size_t>(bytes));
    if (fresh == nullptr) {
      // Refus pour le moteur qui demandait plus, lui seul : les moteurs déjà posés retrouvent une zone de
      // l'ancienne taille (une nouvelle adresse, generation a changé). Si même elle est refusée, chacun la
      // redemande à son lancement (TrtEngine::place) et seul celui qui ne l'obtient pas échoue.
      data_ = allocate(static_cast<std::size_t>(kept));
      if (data_ != nullptr) bytes_ = kept;
      return refused();
    }
  } else if (data_ != nullptr) {
    release(data_);
  }
  data_ = fresh;
  bytes_ = bytes;
  ++generation_;
  return {};
}

}  // namespace ajp::gpu::cuda
