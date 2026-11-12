// Codes d'erreur CUDA en refus nommés qui citent l'appel.
#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>
#include <expected>
#include <format>
#include <string_view>

#include "core/types.h"

namespace ajp::gpu::cuda {

// Les erreurs que la documentation de CUDA décrit comme laissant le processus dans un état incohérent :
// le contexte est perdu, toute suite d'appels CUDA échoue jusqu'à la fin du processus. A01-01 : un TDR de
// Windows ou un Xid 79 (carte tombée du bus) rendent aussi cudaErrorUnknown (999),
// cudaErrorDevicesUnavailable (46) ou cudaErrorExternalDevice (911) ; clear_error reconnaît en plus toute
// erreur qui survit à sa lecture.
constexpr bool sticky(cudaError_t error) {
  switch (error) {
    case cudaErrorUnknown:
    case cudaErrorDevicesUnavailable:
    case cudaErrorExternalDevice:
    case cudaErrorIllegalAddress:
    case cudaErrorLaunchTimeout:
    case cudaErrorIllegalInstruction:
    case cudaErrorMisalignedAddress:
    case cudaErrorInvalidAddressSpace:
    case cudaErrorInvalidPc:
    case cudaErrorLaunchFailure:
    case cudaErrorHardwareStackError:
    case cudaErrorAssert:
    case cudaErrorECCUncorrectable:
    case cudaErrorContextIsDestroyed:
      return true;
    default:
      return false;
  }
}

inline Refusal cuda_refusal(cudaError_t error, std::string_view call) {
  if (sticky(error))
    return Refusal{RefusalCode::DeviceLost, std::format("contexte CUDA perdu : relancer le lecteur ({} : {})",
                                                        call, cudaGetErrorName(error))};
  // C31-04 : un pilote trop ancien pour le runtime CUDA du paquet ne se répare pas en réinstallant le moteur
  // : SdkRejected, que l'écran dit « mettre à jour le pilote graphique ».
  if (error == cudaErrorInsufficientDriver)
    return Refusal{RefusalCode::SdkRejected,
                   std::format("{} : pilote NVIDIA trop ancien pour le CUDA du paquet ({})", call,
                               cudaGetErrorName(error))};
  if (error == cudaErrorNoDevice)
    return Refusal{RefusalCode::DeviceUnavailable,
                   std::format("{} : aucune carte CUDA ({})", call, cudaGetErrorName(error))};
  // A31-08 : la mémoire vidéo pleine (un jeu, un autre lecteur) se dit telle quelle à l'écran.
  if (error == cudaErrorMemoryAllocation)
    return Refusal{RefusalCode::GpuMemoryExhausted,
                   std::format("{} : mémoire vidéo pleine ({})", call, cudaGetErrorName(error))};
  return Refusal{RefusalCode::VendorCallFailed,
                 std::format("{} : {} ({})", call, cudaGetErrorName(error), cudaGetErrorString(error))};
}

// Lit et efface l'erreur en attente, sauf une erreur collante, qui ne s'efface pas : elle est rendue
// (cudaSuccess sinon). Une erreur que la lecture n'efface pas (rendue de nouveau aussitôt) est collante quel
// que soit son code (A01-01) : elle est rendue elle aussi.
inline cudaError_t clear_error() {
  const cudaError_t error = cudaGetLastError();
  if (error == cudaSuccess || sticky(error)) return error;
  const cudaError_t again = cudaGetLastError();
  return again == error ? error : cudaSuccess;
}

// Le refus d'une erreur lue par clear_error : un appareil perdu pour un code de la liste comme pour une
// erreur qui a survécu à sa lecture.
inline Refusal lost_refusal(cudaError_t error, std::string_view call) {
  if (sticky(error)) return cuda_refusal(error, call);
  return Refusal{RefusalCode::DeviceLost, std::format("contexte CUDA perdu : relancer le lecteur ({} : {}, "
                                                      "erreur qui ne s'efface pas)",
                                                      call, cudaGetErrorName(error))};
}

inline std::expected<void, Refusal> cuda_check(cudaError_t error, std::string_view call) {
  if (error == cudaSuccess) return {};
  return std::unexpected(cuda_refusal(error, call));
}

// Événement que l'hôte attend (cudaEventSynchronize) : le fil dort au lieu de tourner dans le pilote
// (A32-11). Les événements seulement interrogés ou attendus par un flux gardent cudaEventDisableTiming.
constexpr unsigned kHostWaitedEvent = cudaEventDisableTiming | cudaEventBlockingSync;

// Emplacement de relecture (device, host, done) : prend chaque morceau qui manque encore. Un échec
// partiel (mémoire épinglée refusée une fois) laisse les morceaux pris et reprend les autres au ticket
// suivant, au lieu de garder un emplacement à moitié prêt pour toute la séance (A01-04). Un emplacement
// que l'hôte attend passe kHostWaitedEvent.
template <typename Slot>
std::expected<void, Refusal> reserve_readback(Slot& slot, std::size_t bytes,
                                              unsigned event_flags = cudaEventDisableTiming) {
  if (slot.device == nullptr) {
    if (auto ok = cuda_check(cudaMalloc(reinterpret_cast<void**>(&slot.device), bytes), "cudaMalloc"); !ok)
      return ok;
  }
  if (slot.host == nullptr) {
    if (auto ok = cuda_check(cudaMallocHost(reinterpret_cast<void**>(&slot.host), bytes), "cudaMallocHost");
        !ok)
      return ok;
  }
  if (slot.done == nullptr)
    return cuda_check(cudaEventCreateWithFlags(&slot.done, event_flags), "cudaEventCreate");
  return {};
}

}  // namespace ajp::gpu::cuda
