// Ce que partagent les essais au chargement du GMFSS natif (gmfss_fused_trial.cpp, gmfss_pair_trial.cpp,
// docs/gmfss-natif.md) : un dessin de test, l'écart en dB, des tampons d'appareil libérés quoi qu'il arrive,
// les sommes essayées et la mesure du temps contre TensorRT.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <expected>
#include <format>
#include <limits>
#include <utility>
#include <vector>

#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/rife_native_net.h"

namespace kirari::gpu::cuda::gmfss_trial {

// Un dessin lisse, de rectangles francs sur des ondes, décalé de shift pixels vers la droite : deux images
// décalées donnent à GMFlow et à l'IFNet un mouvement à trouver.
inline float drawing(float x, float y, int c, float shift) {
  x -= shift;
  const float wave = 0.5F + 0.22F * std::sin(0.071F * x + 0.043F * y + static_cast<float>(c)) +
                     0.12F * std::sin(0.017F * x - 0.029F * y);
  const bool square = std::fmod(std::abs(x), 80.F) < 32.F && std::fmod(std::abs(y), 64.F) < 24.F;
  return std::clamp(square ? 0.9F - 0.2F * static_cast<float>(c) : wave, 0.F, 1.F);
}

// L'écart de got à want en dB, rapporté à peak ; -inf si got n'est pas fini, 200 s'ils sont égaux.
inline double psnr(const std::vector<__half>& want, const std::vector<__half>& got, double peak) {
  double sum = 0.0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    const double g = __half2float(got[i]);
    if (!std::isfinite(g)) return -std::numeric_limits<double>::infinity();
    const double d = g - static_cast<double>(__half2float(want[i]));
    sum += d * d;
  }
  const double mse = sum / static_cast<double>(want.size());
  return mse == 0.0 ? 200.0 : 10.0 * std::log10(peak * peak / mse);
}

// Des tampons d'appareil et un flux, libérés quoi qu'il arrive.
struct Scratch {
  std::vector<void*> buffers;
  cudaStream_t stream = nullptr;
  Scratch() = default;
  Scratch(const Scratch&) = delete;
  Scratch& operator=(const Scratch&) = delete;
  ~Scratch() {
    if (stream != nullptr) cudaStreamDestroy(stream);
    for (void* buffer : buffers) cudaFree(buffer);
  }
};

// Les sommes essayées, comme RIFE natif (rife_fused_native.cpp) : f16 (le débit plein des tensor cores des
// GeForce), mixtes, f32.
inline constexpr std::array<RifeNativeSums, 3> kSums{RifeNativeSums::kF16, RifeNativeSums::kMixed,
                                                     RifeNativeSums::kF32};

inline const char* sums_name(RifeNativeSums sums) {
  switch (sums) {
    case RifeNativeSums::kF16:
      return "f16";
    case RifeNativeSums::kMixed:
      return "mixtes";
    case RifeNativeSums::kF32:
      return "f32";
  }
  return "?";
}

// Le natif n'est gardé que s'il bat TensorRT d'au moins kMinGain : en deçà, l'écart tient de la mesure.
inline constexpr double kMinGain = 0.02;

// Un événement CUDA rendu quoi qu'il arrive.
struct Event {
  cudaEvent_t event = nullptr;
  Event() = default;
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  ~Event() {
    if (event != nullptr) cudaEventDestroy(event);
  }
};

// La mesure du temps, comme celle de RIFE natif : des tours intercalés (la variante 0, TensorRT, puis les
// natives, puis l'inverse), chacun d'une passe de chauffe puis de deux passes entre deux événements ; la
// médiane des tours, en ms par passe. pass(v) met en file sur stream une passe de la variante v. Au premier
// tour, quatre passes de chauffe : un premier lancement, puis la capture et le rejeu des graphes.
template <typename Pass>
std::expected<std::vector<double>, Refusal> times(cudaStream_t stream, std::size_t variants, Pass&& pass) {
  constexpr int kWarmup = 1;
  constexpr int kRuns = 2;
  constexpr int kRounds = 11;
  const auto failed = [](cudaError_t error, const char* step) {
    static_cast<void>(cudaGetLastError());
    return std::unexpected(cuda_refusal(error, std::format("GMFSS natif : mesure du temps ({})", step)));
  };
  Event start;
  Event stop;
  for (Event* event : {&start, &stop})
    if (const cudaError_t e = cudaEventCreate(&event->event); e != cudaSuccess)
      return failed(e, "cudaEventCreate");
  std::vector<std::vector<double>> rounds(variants);
  for (int round = 0; round < kRounds; ++round)
    for (std::size_t i = 0; i < variants; ++i) {
      const std::size_t v = round % 2 == 0 ? i : variants - 1 - i;
      std::expected<void, Refusal> ok;
      for (int w = 0; w < (round == 0 ? 4 : kWarmup) && ok; ++w) ok = pass(v);
      cudaError_t e = ok ? cudaEventRecord(start.event, stream) : cudaSuccess;
      for (int r = 0; r < kRuns && ok && e == cudaSuccess; ++r) ok = pass(v);
      if (ok && e == cudaSuccess) e = cudaEventRecord(stop.event, stream);
      if (ok && e == cudaSuccess) e = cudaEventSynchronize(stop.event);
      float ms = 0.F;
      if (ok && e == cudaSuccess) e = cudaEventElapsedTime(&ms, start.event, stop.event);
      if (!ok) return std::unexpected(std::move(ok.error()));
      if (e != cudaSuccess) return failed(e, "événement");
      if (!(ms > 0.F))
        return std::unexpected(Refusal{RefusalCode::VendorCallFailed, "GMFSS natif : lot mesuré à 0 ms"});
      rounds[v].push_back(static_cast<double>(ms) / kRuns);
    }
  std::vector<double> out;
  for (std::vector<double>& of : rounds) {
    std::ranges::sort(of);
    out.push_back(of[of.size() / 2]);
  }
  return out;
}

}  // namespace kirari::gpu::cuda::gmfss_trial
