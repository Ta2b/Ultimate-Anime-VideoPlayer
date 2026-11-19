// Ce que partagent les essais au chargement du GMFSS natif (gmfss_fused_trial.cpp, gmfss_pair_trial.cpp,
// docs/gmfss-natif.md) : un dessin de test, des tampons d'appareil libérés quoi qu'il arrive, la mesure du
// temps, et le choix commun (gmfss_trial.cpp) du mélange des sommes de chaque morceau (gmfss_mix.h).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <expected>
#include <format>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/gmfss_native_parts.h"

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

// Les tours de la mesure : la passe entière contre TensorRT, chaque morceau seul.
inline constexpr int kPassRounds = 11;
inline constexpr int kPieceRounds = 5;

// La mesure du temps, comme celle de RIFE natif : rounds tours intercalés (les variantes dans l'ordre, puis
// dans l'ordre inverse), chacun d'une passe de chauffe puis de deux passes entre deux événements ; la médiane
// des tours, en ms par passe. pass(v) met en file sur stream une passe de la variante v. Au premier tour,
// quatre passes de chauffe : un premier lancement, puis la capture et le rejeu des graphes.
template <typename Pass>
std::expected<std::vector<double>, Refusal> times(cudaStream_t stream, std::size_t variants, Pass&& pass,
                                                  int rounds = kPassRounds) {
  constexpr int kWarmup = 1;
  constexpr int kRuns = 2;
  const auto failed = [](cudaError_t error, const char* step) {
    static_cast<void>(cudaGetLastError());
    return std::unexpected(cuda_refusal(error, std::format("GMFSS natif : mesure du temps ({})", step)));
  };
  Event start;
  Event stop;
  for (Event* event : {&start, &stop})
    if (const cudaError_t e = cudaEventCreate(&event->event); e != cudaSuccess)
      return failed(e, "cudaEventCreate");
  std::vector<std::vector<double>> per(variants);
  for (int round = 0; round < rounds; ++round)
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
      per[v].push_back(static_cast<double>(ms) / kRuns);
    }
  std::vector<double> out;
  for (std::vector<double>& of : per) {
    std::ranges::sort(of);
    out.push_back(of[of.size() / 2]);
  }
  return out;
}

// Ce que l'essai d'une unité (la fusion, la paire) donne au choix commun. Les passes et les morceaux tournent
// sur stream ; une passe native lance le réseau aux sommes posées (parts->use).
struct Unit {
  std::string name;           // « fusion », « paire »
  int width = 0, height = 0;  // les images, pour le journal
  GmfssNativeParts* parts = nullptr;
  cudaStream_t stream = nullptr;
  std::size_t outputs = 0;  // les sorties de l'unité, chacune jugée à part
  // Une passe en file : TensorRT (native faux) ou le natif.
  std::function<std::expected<void, Refusal>(bool native)> pass;
  // Le morceau seul en file, comme une passe le lance.
  std::function<std::expected<void, Refusal>(int piece)> piece;
  // La sortie i de la dernière passe, relue sur l'hôte.
  std::function<std::expected<std::vector<__half>, Refusal>(std::size_t i)> fetch;
};

// Ce que l'essai retient : les lignes du journal, la dernière étant le verdict ; keep : le natif gardé, ses
// sommes posées et les poids qu'elles n'emploient pas libérés.
struct Outcome {
  std::vector<std::string> lines;
  bool keep = false;
};

// Une référence TensorRT, puis le natif f32 (la base) ; le temps seul de chaque option de chaque morceau ; le
// mélange le plus rapide dont chaque sortie tient min_db contre TensorRT (native_mix.h), gardé s'il bat
// TensorRT d'au moins gmfss_mix::kMinGain sur une passe entière. tuning : les secondes du réglage. Refus :
// l'essai n'a pas pu tourner.
std::expected<Outcome, Refusal> try_mix(const Unit& unit, double min_db, double tuning);

}  // namespace kirari::gpu::cuda::gmfss_trial
