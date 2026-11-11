// La dérivation native du paquet « .span8 » dans le backend TensorRT (Backend::derive_span_fp8,
// docs/span-fp8-cutlass.md) : ce que tools/span/derive_span_fp8.py fait avec ONNX Runtime, mais par les
// noyaux CUTLASS mêmes du lecteur.
//
//   1. La référence FP16 (span_fp8_ref.h) passe les images de calibration et relève le maximum de |valeur|
//      des tenseurs que le réseau FP8 quantifie ; échelle = maximum × 1,25 / 448 (gpu/span_fp8/derive.h :
//      scales_from_peaks, marge de tête B02-02), comme l'outil Python.
//   2. Le réseau FP8 (span_fp8_net.h), à ces échelles, passe les images de validation, mises de côté ; sa
//      sortie est jugée contre celle de la même référence FP16 (PSNR, SSIM de gpu/span_fp8/metrics.h).
//   3. Le temps du réseau FP8 sur cette taille est mesuré. Le lecteur le compare ensuite au SPAN FP16 de
//      TensorRT sur la même carte (app/play/span_derive.h : time_span_fp16, B02-01).
//   La variante à accumulateur f16 hors blocs (conv_cat et to_pixel, étape 3.5a) passe les mêmes images et la
//   même mesure ; l'application la garde si sa porte passe et que le PSNR moyen perd ≤ 0,3 dB.
//
// La référence est le SPAN en FP16 par les convolutions CUTLASS, pas le moteur TensorRT : construire un
// moteur TensorRT pour la dériver coûterait une minute, et ses échelles seraient relevées sur un graphe fondu
// dont les tenseurs intermédiaires ne sont plus ceux des noyaux. La comparaison au FP16 de TensorRT se fait
// par le test caché [.span_derive] (tests/unit/backend_tensorrt/span_derive_test.cpp) et l'outil
// ajp_span_fp8_run.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <format>
#include <vector>

#include "gpu/cuda/backend/span_fp8_engine.h"
#include "gpu/cuda/backend/span_fp8_net.h"
#include "gpu/cuda/backend/span_fp8_ref.h"
#include "gpu/cuda/backend/tensorrt_backend.h"
#include "gpu/span_fp8/derive.h"
#include "gpu/span_fp8/metrics.h"
#include "gpu/span_fp8/package.h"

namespace ajp::gpu::cuda {
namespace {

// Mémoire d'appareil et flux rendus quoi qu'il arrive.
struct Scratch {
  void* input = nullptr;
  void* output_ref = nullptr;
  void* output_fp8 = nullptr;
  void* workspace = nullptr;
  cudaStream_t stream = nullptr;
  Scratch() = default;
  Scratch(const Scratch&) = delete;
  Scratch& operator=(const Scratch&) = delete;
  ~Scratch() {
    if (stream != nullptr) cudaStreamDestroy(stream);
    for (void* p : {input, output_ref, output_fp8, workspace})
      if (p != nullptr) cudaFree(p);
  }
};

std::unexpected<Refusal> cuda_refusal(cudaError_t error, const char* step) {
  static_cast<void>(cudaGetLastError());
  return refuse(
      error == cudaErrorMemoryAllocation ? RefusalCode::GpuMemoryExhausted : RefusalCode::VendorCallFailed,
      std::format("dérivation du SPAN FP8 ({}) : {}", step, cudaGetErrorString(error)));
}

// Une image 8 bits planaire, en FP16 NCHW (valeurs de 0 à 1).
std::vector<__half> to_half(const SpanStill& still) {
  std::vector<__half> out(still.rgb.size());
  for (std::size_t i = 0; i < out.size(); ++i)
    out[i] = __float2half(static_cast<float>(still.rgb[i]) / 255.0F);
  return out;
}

// La sortie FP16 d'un passage, rapportée à [0, 1].
std::vector<float> to_unit(const std::vector<__half>& half) {
  std::vector<float> out(half.size());
  for (std::size_t i = 0; i < out.size(); ++i) out[i] = std::clamp(__half2float(half[i]), 0.0F, 1.0F);
  return out;
}

}  // namespace

// NOLINTNEXTLINE(readability-function-cognitive-complexity,readability-function-size)
std::expected<SpanFp8Derivation, Refusal> TensorRtBackend::derive_span_fp8(const SpanFp8Request& request,
                                                                           Progress& progress) {
  if (!SpanFp8Net::supported_device(properties_.major, properties_.minor)) {
    return refuse(
        RefusalCode::FormatUnsupported,
        std::format("SPAN FP8 (CUTLASS) : {} (sm {}.{}) n'est pas pris, ou backend construit sans CUTLASS",
                    properties_.name, properties_.major, properties_.minor));
  }
  if ((request.calibration.empty() && !request.validate_only) || request.validation.empty())
    return refuse(RefusalCode::ShapeInvalid,
                  "dérivation du SPAN FP8 : images de calibration et de validation attendues");
  const int width = request.validation.front().width;
  const int height = request.validation.front().height;
  if (const std::string why = span_fp8::size_refusal(height, width); !why.empty())
    return refuse(RefusalCode::FormatUnsupported, "dérivation du SPAN FP8 : " + why);
  const std::size_t plane = static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
  for (const auto* group : {&request.calibration, &request.validation, &request.hard})
    for (const SpanStill& still : *group)
      if (still.width != width || still.height != height || still.rgb.size() != 3 * plane)
        return refuse(RefusalCode::ShapeInvalid, "dérivation du SPAN FP8 : images de tailles différentes");
  auto parsed = span_fp8::parse(request.package);
  if (!parsed) return std::unexpected(std::move(parsed.error()));
  if (auto ok = bind(); !ok) return std::unexpected(std::move(ok.error()));
  if (request.min_free_ratio > 0.0) {
    // Étape 3.6 : une préparation de fond ne prend pas la mémoire dont la lecture a besoin. Besoin estimé :
    // zone du réseau FP8 (768 o par pixel de caractéristiques) et de la référence FP16 (1 120 o), entrée et
    // deux sorties FP16, poids.
    const auto features = static_cast<double>(plane) / 4.0;
    const double need = features * (768.0 + 1120.0) + static_cast<double>(plane) * 3.0 * 2.0 * 9.0 + 8.0e6;
    std::size_t free_bytes = 0, total_bytes = 0;
    if (const cudaError_t e = cudaMemGetInfo(&free_bytes, &total_bytes); e != cudaSuccess)
      return cuda_refusal(e, "cudaMemGetInfo");
    if (static_cast<double>(free_bytes) < request.min_free_ratio * need)
      return refuse(
          RefusalCode::GpuMemoryExhausted,
          std::format("dérivation du SPAN FP8 : {} Mo de VRAM libres, {:.0f} Mo demandés ({} × le besoin)",
                      free_bytes >> 20U, request.min_free_ratio * need / 1048576.0, request.min_free_ratio));
  }

  const auto report = [&](float fraction, const char* stage) {
    if (progress.report) progress.report(fraction, stage);
  };
  const auto cancelled = [&] { return progress.stop.stop_requested(); };
  const auto stopped = [] { return refuse(RefusalCode::Cancelled, "dérivation du SPAN FP8 interrompue"); };

  SpanFp8Weights weights = span_fp8_view(*parsed);
  SpanFp8Status status = SpanFp8Status::Ok;
  std::string why;
  report(0.0F, "référence FP16");
  const std::unique_ptr<SpanFp8Reference> reference =
      SpanFp8Reference::create(weights, height, width, status, why);
  if (!reference)
    return refuse(status == SpanFp8Status::OutOfMemory ? RefusalCode::GpuMemoryExhausted
                                                       : RefusalCode::VendorCallFailed,
                  "dérivation du SPAN FP8 : référence FP16 : " + why);

  Scratch scratch;
  const std::size_t in_bytes = 3 * plane * sizeof(__half);
  if (auto e = cudaMalloc(&scratch.input, in_bytes); e != cudaSuccess) return cuda_refusal(e, "cudaMalloc");
  if (auto e = cudaMalloc(&scratch.output_ref, in_bytes * 4); e != cudaSuccess)
    return cuda_refusal(e, "cudaMalloc");
  if (auto e = cudaMalloc(&scratch.output_fp8, in_bytes * 4); e != cudaSuccess)
    return cuda_refusal(e, "cudaMalloc");
  // Étape 3.6 : le flux à la priorité la plus basse, derrière ceux de la lecture.
  int least = 0, greatest = 0;
  if (cudaDeviceGetStreamPriorityRange(&least, &greatest) != cudaSuccess) {
    static_cast<void>(cudaGetLastError());
    least = 0;
  }
  if (auto e = cudaStreamCreateWithPriority(&scratch.stream, cudaStreamNonBlocking, least); e != cudaSuccess)
    return cuda_refusal(e, "cudaStreamCreate");
  const auto upload = [&](const SpanStill& still) -> std::expected<void, Refusal> {
    const std::vector<__half> half = to_half(still);
    if (auto e =
            cudaMemcpyAsync(scratch.input, half.data(), in_bytes, cudaMemcpyHostToDevice, scratch.stream);
        e != cudaSuccess)
      return cuda_refusal(e, "cudaMemcpy");
    // La copie lit un tampon de pile : on attend qu'elle ait fini avant de le rendre.
    if (auto e = cudaStreamSynchronize(scratch.stream); e != cudaSuccess)
      return cuda_refusal(e, "cudaMemcpy");
    return {};
  };

  // --- 1. calibration : les maxima des tenseurs, sur les images de calibration
  const auto total =
      static_cast<float>(request.calibration.size() + request.validation.size() + request.hard.size());
  float done = 0.0F;
  for (const SpanStill& still : request.calibration) {
    if (request.validate_only) break;  // validation seule : les échelles du paquet
    if (cancelled()) return stopped();
    if (auto ok = upload(still); !ok) return std::unexpected(std::move(ok.error()));
    if (!reference->run(scratch.stream, scratch.input, scratch.output_ref, true, why))
      return refuse(RefusalCode::VendorCallFailed, "dérivation du SPAN FP8 : " + why);
    done += 1.0F;
    report(0.05F + 0.55F * done / total, "calibration");
  }
  SpanFp8Derivation result;
  result.gpu_arch = std::format("sm_{}{}", properties_.major, properties_.minor);
  result.kernels_version = SpanFp8Net::kKernelsVersion;
  if (request.validate_only) {
    result.scales = parsed->scales;
  } else {
    float peaks[SpanFp8Reference::kPeaks];
    if (!reference->peaks(peaks, why))
      return refuse(RefusalCode::VendorCallFailed, "dérivation du SPAN FP8 : " + why);
    std::ranges::copy(peaks, result.peaks.begin());
    auto scales = span_fp8::scales_from_peaks(result.peaks, parsed->blocks.size());
    if (!scales) return std::unexpected(std::move(scales.error()));
    result.scales = *scales;
  }

  // --- 2. le réseau FP8 à ces échelles
  weights.scales = result.scales.data();
  weights.acc16 = request.validate_only && parsed->acc16;  // validation seule : la variante du paquet
  const std::unique_ptr<SpanFp8Net> net = SpanFp8Net::create(weights, height, width, status, why);
  if (!net)
    return refuse(status == SpanFp8Status::OutOfMemory ? RefusalCode::GpuMemoryExhausted
                                                       : RefusalCode::VendorCallFailed,
                  "dérivation du SPAN FP8 : envoi des poids : " + why);
  weights.acc16 = true;
  // Variante f16 : un refus ici ne fait pas échouer la dérivation, elle n'est simplement pas proposée.
  const std::unique_ptr<SpanFp8Net> net16 =
      request.validate_only ? nullptr : SpanFp8Net::create(weights, height, width, status, why);
  if (auto e = cudaMalloc(&scratch.workspace, static_cast<std::size_t>(net->workspace_bytes()));
      e != cudaSuccess)
    return cuda_refusal(e, "cudaMalloc");
  net->bind(scratch.workspace);
  if (net16) net16->bind(scratch.workspace);  // même zone : les deux passent l'un après l'autre

  // --- 3. validation, sur des images que la calibration n'a pas vues, puis les cas durs (étape 3.7a)
  std::vector<__half> half(3 * plane * 4);
  const auto judge_all = [&](const std::vector<SpanStill>& stills, std::vector<SpanFp8ImageScore>& scores,
                             std::vector<SpanFp8ImageScore>& scores16,
                             const char* stage) -> std::expected<void, Refusal> {
    for (const SpanStill& still : stills) {
      if (cancelled()) return stopped();
      if (auto ok = upload(still); !ok) return ok;
      if (!reference->run(scratch.stream, scratch.input, scratch.output_ref, false, why))
        return refuse(RefusalCode::VendorCallFailed, "dérivation du SPAN FP8 : " + why);
      if (!net->launch(scratch.stream, scratch.input, scratch.output_fp8, why))
        return refuse(RefusalCode::VendorCallFailed, "dérivation du SPAN FP8 : " + why);
      if (auto e = cudaStreamSynchronize(scratch.stream); e != cudaSuccess)
        return cuda_refusal(e, "exécution");
      if (auto e = cudaMemcpy(half.data(), scratch.output_ref, in_bytes * 4, cudaMemcpyDeviceToHost);
          e != cudaSuccess)
        return cuda_refusal(e, "cudaMemcpy");
      const std::vector<float> expected = to_unit(half);
      const auto score = [&](std::vector<SpanFp8ImageScore>& into) -> std::expected<void, Refusal> {
        if (auto e = cudaMemcpy(half.data(), scratch.output_fp8, in_bytes * 4, cudaMemcpyDeviceToHost);
            e != cudaSuccess)
          return cuda_refusal(e, "cudaMemcpy");
        const std::vector<float> got = to_unit(half);
        into.push_back({.psnr_db = span_fp8::psnr(expected, got),
                        .ssim = span_fp8::ssim(2 * width, 2 * height, expected, got),
                        .error_p999 = span_fp8::abs_error_quantile(expected, got, span_fp8::kErrorQuantile),
                        .dark_error_p999 = span_fp8::dark_abs_error_quantile(2 * width, 2 * height, expected,
                                                                             got, span_fp8::kErrorQuantile)});
        return {};
      };
      if (auto ok = score(scores); !ok) return ok;
      if (net16) {
        if (!net16->launch(scratch.stream, scratch.input, scratch.output_fp8, why))
          return refuse(RefusalCode::VendorCallFailed, "dérivation du SPAN FP8 (f16) : " + why);
        if (auto e = cudaStreamSynchronize(scratch.stream); e != cudaSuccess)
          return cuda_refusal(e, "exécution");
        if (auto ok = score(scores16); !ok) return ok;
      }
      done += 1.0F;
      report(0.05F + 0.55F * done / total, stage);
    }
    return {};
  };
  if (auto ok = judge_all(request.validation, result.scores, result.scores_acc16, "validation"); !ok)
    return std::unexpected(std::move(ok.error()));
  if (auto ok = judge_all(request.hard, result.hard_scores, result.hard_scores_acc16, "validation"); !ok)
    return std::unexpected(std::move(ok.error()));

  // --- 4. le temps du réseau FP8 (et de sa variante f16)
  report(0.9F, "mesure du temps");
  const auto time = [&](SpanFp8Net& timed_net) -> std::expected<double, Refusal> {
    constexpr int kWarmup = 5;
    constexpr int kRuns = 21;
    std::vector<cudaEvent_t> events(kRuns + 1, nullptr);
    for (cudaEvent_t& event : events) {
      if (auto e = cudaEventCreate(&event); e != cudaSuccess) {
        for (cudaEvent_t made : events)
          if (made != nullptr) cudaEventDestroy(made);
        return cuda_refusal(e, "cudaEventCreate");
      }
    }
    std::expected<void, Refusal> timed;
    for (int i = 0; i < kWarmup && timed; ++i)
      if (!timed_net.launch(scratch.stream, scratch.input, scratch.output_fp8, why))
        timed = refuse(RefusalCode::VendorCallFailed, "dérivation du SPAN FP8 : " + why);
    for (int i = 0; i < kRuns && timed; ++i) {
      cudaEventRecord(events[static_cast<std::size_t>(i)], scratch.stream);
      if (!timed_net.launch(scratch.stream, scratch.input, scratch.output_fp8, why))
        timed = refuse(RefusalCode::VendorCallFailed, "dérivation du SPAN FP8 : " + why);
    }
    cudaEventRecord(events[kRuns], scratch.stream);
    if (const cudaError_t e = cudaStreamSynchronize(scratch.stream); e != cudaSuccess && timed)
      timed = cuda_refusal(e, "mesure");
    double median = 0.0;
    if (timed) {
      std::vector<float> ms;
      for (std::size_t i = 0; i < static_cast<std::size_t>(kRuns); ++i) {
        float one = 0.0F;
        cudaEventElapsedTime(&one, events[i], events[i + 1]);
        ms.push_back(one);
      }
      std::ranges::sort(ms);
      median = static_cast<double>(ms[ms.size() / 2]);
    }
    for (cudaEvent_t event : events) cudaEventDestroy(event);
    if (!timed) return std::unexpected(std::move(timed.error()));
    return median;
  };
  auto median = time(*net);
  if (!median) return std::unexpected(std::move(median.error()));
  result.kernels_ms = *median;
  if (net16) {
    auto median16 = time(*net16);
    if (!median16) return std::unexpected(std::move(median16.error()));
    result.kernels_ms_acc16 = *median16;
  }
  result.reference = "cutlass-fp16";
  report(1.0F, "terminé");
  return result;
}

}  // namespace ajp::gpu::cuda
