// Le temps du SPAN FP8 (et de sa variante à accumulateur f16) contre celui du SPAN FP16 de TensorRT, pour la
// porte de vitesse de la dérivation (tensorrt_span_derive.cpp, C02-01) : les trois mesurés de la même façon,
// dans le même processus, sur le même flux.
//
//   - FP8 comme en lecture : le corps du réseau rejoué par un graphe CUDA, la queue lancée en direct
//     (span_fp8_engine.cpp) ; TensorRT par TrtEngine::enqueue, qui rejoue aussi son graphe dès le troisième
//     lancement ;
//   - des événements CUDA autour de chaque lot, chaque retour vérifié (C02-06) : un événement qui échoue
//     refuse la mesure, il ne compte pas 0 ms ;
//   - des tours intercalés (FP16, FP8, f16 puis f16, FP8, FP16…), pour qu'une charge qui passe les touche
//     également ; la médiane des tours ;
//   - de petits lots (kTimingWarmup + kTimingRuns passes), chacun suivi d'un rapport « mesure du temps » : la
//     préparation qui tourne pendant la lecture y rend la carte (pause_for_playback, C02-02).
#pragma once

#include <cuda_runtime_api.h>
#include <expected>

#include "core/types.h"
#include "gpu/backend.h"

namespace ajp::gpu::cuda {

class SpanFp8Net;
class TrtEngine;

// Une passe de chauffe, puis deux passes chronométrées, par lot ; onze tours.
inline constexpr int kTimingWarmup = 1;
inline constexpr int kTimingRuns = 2;
inline constexpr int kTimingRounds = 11;

// Ce qui se mesure : input en FP16 NCHW 1×3×H×W ; output_fp8 et output_trt en 1×3×2H×2W. fp8_acc16 et trt
// peuvent manquer (rien n'est alors mesuré pour eux, 0 ms).
struct SpanTimingSet {
  SpanFp8Net* fp8 = nullptr;
  SpanFp8Net* fp8_acc16 = nullptr;
  TrtEngine* trt = nullptr;
  int width = 0;
  int height = 0;
  void* input = nullptr;
  void* output_fp8 = nullptr;
  void* output_trt = nullptr;
};

struct SpanTimes {
  double fp8_ms = 0.0;
  double fp8_acc16_ms = 0.0;
  double trt_ms = 0.0;
};

// Les médianes d'une passe de chaque réseau, mesurées sur stream (la priorité la plus basse, celle de la
// dérivation). progress.report reçoit « mesure du temps » après chaque lot ; Cancelled si progress.stop le
// demande. Refus : un lancement, une capture de graphe ou un événement CUDA qui échoue.
std::expected<SpanTimes, Refusal> time_span_set(const SpanTimingSet& set, cudaStream_t stream,
                                                Progress& progress, float from, float to);

}  // namespace ajp::gpu::cuda
