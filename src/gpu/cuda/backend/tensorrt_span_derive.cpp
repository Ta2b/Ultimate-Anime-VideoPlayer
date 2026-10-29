// La dérivation native du paquet « .span8 » dans le backend TensorRT (Backend::derive_span_fp8,
// docs/span-fp8-cutlass.md) : ce que tools/span/derive_span_fp8.py fait avec ONNX Runtime, mais par les
// noyaux CUTLASS mêmes du lecteur.
//
//   1. La référence FP16 (span_fp8_ref.h) passe les images de calibration et relève le maximum de |valeur|
//   des
//      25 tenseurs que le réseau FP8 quantifie ; échelle = maximum / 448, comme l'outil Python.
//   2. Le réseau FP8 (span_fp8_net.h), à ces échelles, passe les images de validation, mises de côté ; sa
//      sortie est jugée contre celle de la même référence FP16 (PSNR, SSIM de gpu/span_fp8/metrics.h).
//   3. Le temps du réseau FP8 sur cette taille est mesuré.
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

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

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

std::expected<SpanFp8Derivation, Refusal> TensorRtBackend::derive_span_fp8(const SpanFp8Request& request,
                                                                           Progress& progress) {
  if (!SpanFp8Net::supported_device(properties_.major, properties_.minor)) {
    return refuse(
        RefusalCode::FormatUnsupported,
        std::format("SPAN FP8 (CUTLASS) : {} (sm {}.{}) n'est pas pris, ou backend construit sans CUTLASS",
                    properties_.name, properties_.major, properties_.minor));
  }
  if (request.calibration.empty() || request.validation.empty())
    return refuse(RefusalCode::ShapeInvalid,
                  "dérivation du SPAN FP8 : images de calibration et de validation attendues");
  const int width = request.calibration.front().width;
  const int height = request.calibration.front().height;
  if (const std::string why = span_fp8::size_refusal(height, width); !why.empty())
    return refuse(RefusalCode::FormatUnsupported, "dérivation du SPAN FP8 : " + why);
  const std::size_t plane = static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
  for (const auto* group : {&request.calibration, &request.validation})
    for (const SpanStill& still : *group)
      if (still.width != width || still.height != height || still.rgb.size() != 3 * plane)
        return refuse(RefusalCode::ShapeInvalid, "dérivation du SPAN FP8 : images de tailles différentes");
  auto parsed = span_fp8::parse(request.package);
  if (!parsed) return std::unexpected(std::move(parsed.error()));
  if (auto ok = bind(); !ok) return std::unexpected(std::move(ok.error()));

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
  if (auto e = cudaStreamCreateWithFlags(&scratch.stream, cudaStreamNonBlocking); e != cudaSuccess)
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
  const auto total = static_cast<float>(request.calibration.size() + request.validation.size());
  float done = 0.0F;
  for (const SpanStill& still : request.calibration) {
    if (cancelled()) return stopped();
    if (auto ok = upload(still); !ok) return std::unexpected(std::move(ok.error()));
    if (!reference->run(scratch.stream, scratch.input, scratch.output_ref, true, why))
      return refuse(RefusalCode::VendorCallFailed, "dérivation du SPAN FP8 : " + why);
    done += 1.0F;
    report(0.05F + 0.55F * done / total, "calibration");
  }
  SpanFp8Derivation result;
  float peaks[SpanFp8Reference::kPeaks];
  if (!reference->peaks(peaks, why))
    return refuse(RefusalCode::VendorCallFailed, "dérivation du SPAN FP8 : " + why);
  std::ranges::copy(peaks, result.peaks.begin());
  auto scales = span_fp8::scales_from_peaks(result.peaks);
  if (!scales) return std::unexpected(std::move(scales.error()));
  result.scales = *scales;

  // --- 2. le réseau FP8 à ces échelles
  weights.scales = result.scales.data();
  const std::unique_ptr<SpanFp8Net> net = SpanFp8Net::create(weights, height, width, status, why);
  if (!net)
    return refuse(status == SpanFp8Status::OutOfMemory ? RefusalCode::GpuMemoryExhausted
                                                       : RefusalCode::VendorCallFailed,
                  "dérivation du SPAN FP8 : envoi des poids : " + why);
  if (auto e = cudaMalloc(&scratch.workspace, static_cast<std::size_t>(net->workspace_bytes()));
      e != cudaSuccess)
    return cuda_refusal(e, "cudaMalloc");
  net->bind(scratch.workspace);

  // --- 3. validation, sur des images que la calibration n'a pas vues
  std::vector<__half> half(3 * plane * 4);
  for (const SpanStill& still : request.validation) {
    if (cancelled()) return stopped();
    if (auto ok = upload(still); !ok) return std::unexpected(std::move(ok.error()));
    if (!reference->run(scratch.stream, scratch.input, scratch.output_ref, false, why))
      return refuse(RefusalCode::VendorCallFailed, "dérivation du SPAN FP8 : " + why);
    if (!net->launch(scratch.stream, scratch.input, scratch.output_fp8, why))
      return refuse(RefusalCode::VendorCallFailed, "dérivation du SPAN FP8 : " + why);
    if (auto e = cudaStreamSynchronize(scratch.stream); e != cudaSuccess) return cuda_refusal(e, "exécution");
    if (auto e = cudaMemcpy(half.data(), scratch.output_ref, in_bytes * 4, cudaMemcpyDeviceToHost);
        e != cudaSuccess)
      return cuda_refusal(e, "cudaMemcpy");
    const std::vector<float> expected = to_unit(half);
    if (auto e = cudaMemcpy(half.data(), scratch.output_fp8, in_bytes * 4, cudaMemcpyDeviceToHost);
        e != cudaSuccess)
      return cuda_refusal(e, "cudaMemcpy");
    const std::vector<float> got = to_unit(half);
    result.scores.push_back({.psnr_db = span_fp8::psnr(expected, got),
                             .ssim = span_fp8::ssim(2 * width, 2 * height, expected, got)});
    done += 1.0F;
    report(0.05F + 0.55F * done / total, "validation");
  }

  // --- 4. le temps du réseau FP8
  report(0.9F, "mesure du temps");
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
    if (!net->launch(scratch.stream, scratch.input, scratch.output_fp8, why))
      timed = refuse(RefusalCode::VendorCallFailed, "dérivation du SPAN FP8 : " + why);
  for (int i = 0; i < kRuns && timed; ++i) {
    cudaEventRecord(events[static_cast<std::size_t>(i)], scratch.stream);
    if (!net->launch(scratch.stream, scratch.input, scratch.output_fp8, why))
      timed = refuse(RefusalCode::VendorCallFailed, "dérivation du SPAN FP8 : " + why);
  }
  cudaEventRecord(events[kRuns], scratch.stream);
  if (const cudaError_t e = cudaStreamSynchronize(scratch.stream); e != cudaSuccess && timed)
    timed = cuda_refusal(e, "mesure");
  if (timed) {
    std::vector<float> ms;
    for (std::size_t i = 0; i < static_cast<std::size_t>(kRuns); ++i) {
      float one = 0.0F;
      cudaEventElapsedTime(&one, events[i], events[i + 1]);
      ms.push_back(one);
    }
    std::ranges::sort(ms);
    result.kernels_ms = static_cast<double>(ms[ms.size() / 2]);
  }
  for (cudaEvent_t event : events) cudaEventDestroy(event);
  if (!timed) return std::unexpected(std::move(timed.error()));
  result.reference = "cutlass-fp16";
  report(1.0F, "terminé");
  return result;
}

}  // namespace ajp::gpu::cuda
