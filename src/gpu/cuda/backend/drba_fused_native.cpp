#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <format>
#include <vector>

#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/drba_fused_engine.h"
#include "gpu/cuda/backend/gmfss_trial.h"

namespace kirari::gpu::cuda {
namespace {

constexpr int kInputPitch = 112;
constexpr int kTrtPitch = 56;
constexpr int kUsedChannels = 20;
constexpr int kOutputPitch = DrbaTailNative::kOutputPitch;

struct TailQuality {
  double snr = 0;
  double flow = 0;
  double mask = 0;
};

TailQuality tail_quality(std::span<const __half> reference, std::span<const __half> result) {
  double square = 0, power = 0, flow = 0, mask = 0;
  const std::size_t cells = result.size() / kOutputPitch;
  for (std::size_t px = 0; px < cells; ++px) {
    for (std::size_t c = 0; c < kUsedChannels; ++c) {
      const double a = __half2float(reference[px * kTrtPitch + c]);
      const double b = __half2float(result[px * kOutputPitch + c]);
      if (!std::isfinite(a) || !std::isfinite(b)) return {.snr = -1.0, .flow = 1.0, .mask = 1.0};
      const double error = a - b;
      power += a * a;
      square += error * error;
      auto& worst = c < 16 ? flow : mask;
      worst = std::max(worst, std::abs(error));
    }
  }
  return {.snr = square == 0.0 ? 200.0 : 10.0 * std::log10(power / square), .flow = flow, .mask = mask};
}

// Les mêmes 48 canaux fixes, puis un autre instant pour les entrées mobiles.
std::vector<__half> tail_mire(int height, int width, int instant) {
  std::vector<__half> data(static_cast<std::size_t>(height) * static_cast<std::size_t>(width) * kInputPitch);
  std::size_t index = 0;
  const float shift = static_cast<float>(instant) * 3.0F;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < kInputPitch; ++c) {
        float v = 0.0F;
        if (c < 109) {
          const float dx = static_cast<float>(x) + (c >= 57 ? shift : 0.0F);
          v = gmfss_trial::drawing(dx, static_cast<float>(y), c % 3, 0.0F);
          if (c >= 9 && c != 95) v = (v - 0.5F) * 0.5F;
          if (c == 95) v = 0.625F + static_cast<float>(instant) * 0.75F;
        }
        data[index++] = __float2half_rn(v);
      }
    }
  }
  return data;
}

struct TrialDrain {
  cudaStream_t stream;
  ~TrialDrain() { static_cast<void>(cudaStreamSynchronize(stream)); }
};

}  // namespace

std::expected<std::string, Refusal> DrbaFusedEngine::try_native_tail(
    cudaStream_t stream, const std::shared_ptr<SharedActivations>& arena,
    std::span<const RifeNativeConv> recipe) {
  constexpr int kLast = kernels::kDrbaFusedBlocks - 1;
  const kernels::DrbaFusedDims x = kernels::drba_fused_x(plan_, kLast, height_, width_);
  const kernels::DrbaFusedDims y = kernels::drba_fused_y(plan_, kLast, height_, width_);
  const auto elements = [](kernels::DrbaFusedDims dims, int pitch) {
    return static_cast<std::size_t>(dims.width) * static_cast<std::size_t>(dims.height) *
           static_cast<std::size_t>(pitch);
  };
  cudaError_t result = cudaSuccess;
  auto native = DrbaTailNative::create(recipe, stream, result);
  if (!native) return std::unexpected(cuda_refusal(result, "DistilDRBA : préparation du bloc natif"));
  FusedScratch scratch;
  FusedGraphs graphs(3, "DistilDRBA : essai du bloc natif");
  const TrialDrain drain{stream};
  const std::array sizes{elements(x, kInputPitch), elements(y, kTrtPitch), elements(y, kOutputPitch)};
  for (std::size_t i = 0; i < sizes.size(); ++i) {
    if (auto ok = cuda_check(cudaMalloc(&scratch.buffers[i], sizes[i] * sizeof(__half)),
                             "DistilDRBA : tampons de l'essai du bloc 2");
        !ok)
      return std::unexpected(std::move(ok.error()));
  }

  const auto input =
      fused_tensor(reinterpret_cast<std::uintptr_t>(scratch.buffers[0]), 0, x.channels, x.height, x.width);
  const auto output =
      fused_tensor(reinterpret_cast<std::uintptr_t>(scratch.buffers[1]), 0, y.channels, y.height, y.width);
  const auto trt_pass = [&] {
    return blocks_[kLast]->enqueue_plain(stream, arena, std::span(&input, 1), std::span(&output, 1));
  };
  const auto native_pass = [&](bool reuse) {
    return cuda_check(native->enqueue(stream, scratch.buffers[0], scratch.buffers[2], reuse),
                      "DistilDRBA : essai du bloc 2 natif");
  };
  const auto output_at = [&](std::size_t slot, std::vector<__half>& host) -> std::expected<void, Refusal> {
    auto error = cudaMemcpyAsync(host.data(), scratch.buffers[slot], sizes[slot] * sizeof(__half),
                                 cudaMemcpyDeviceToHost, stream);
    if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
    return cuda_check(error, "DistilDRBA : récupération du bloc 2");
  };
  std::vector<__half> reference(sizes[1]), candidate(sizes[2]), rebuilt(sizes[2]);
  TailQuality score{.snr = 200.0};
  for (const int instant : {0, 1}) {
    const auto mire = tail_mire(x.height, x.width, instant);
    if (auto ok = cuda_check(cudaMemcpyAsync(scratch.buffers[0], mire.data(), sizes[0] * sizeof(__half),
                                             cudaMemcpyHostToDevice, stream),
                             "DistilDRBA : mire du bloc 2");
        !ok)
      return std::unexpected(std::move(ok.error()));
    if (auto ok = cuda_check(cudaStreamSynchronize(stream), "DistilDRBA : copie de la mire"); !ok)
      return std::unexpected(std::move(ok.error()));
    if (auto ok = trt_pass(); !ok) return std::unexpected(std::move(ok.error()));
    if (auto ok = native_pass(instant != 0); !ok) return std::unexpected(std::move(ok.error()));
    if (auto ok = output_at(1, reference); !ok) return std::unexpected(std::move(ok.error()));
    if (auto ok = output_at(2, candidate); !ok) return std::unexpected(std::move(ok.error()));
    if (instant == 1) {
      if (auto ok = native_pass(false); !ok) return std::unexpected(std::move(ok.error()));
      if (auto ok = output_at(2, rebuilt); !ok) return std::unexpected(std::move(ok.error()));
      const auto used = [&](std::size_t i) { return i % kOutputPitch < kUsedChannels; };
      for (std::size_t i = 0; i < candidate.size(); ++i)
        if (used(i) && __half_as_ushort(candidate[i]) != __half_as_ushort(rebuilt[i]))
          return "DistilDRBA : bloc natif écarté, réemploi différent du recalcul";
    }
    const auto quality = tail_quality(reference, candidate);
    score.snr = std::min(score.snr, quality.snr);
    score.flow = std::max(score.flow, quality.flow);
    score.mask = std::max(score.mask, quality.mask);
  }
  if (!(score.snr >= 50.0) || std::max(score.flow, score.mask) > 0.0625)
    return std::format("DistilDRBA : bloc natif écarté, SNR {:.2f} dB, écarts flot {:.5f} et masque {:.5f}",
                       score.snr, score.flow, score.mask);

  const auto timed = gmfss_trial::times(
      stream, 3,
      [&](std::size_t mode) {
        return graphs.launch(stream, FusedGraphs::Key{mode},
                             [&] { return mode == 0 ? trt_pass() : native_pass(mode == 2); });
      },
      gmfss_trial::kPieceRounds);
  if (!timed) return std::unexpected(timed.error());
  const double trt = (*timed)[0], cold = (*timed)[1], reused = (*timed)[2];
  const bool keep = graphs.counters().captures == 3 && cold <= 1.03 * trt && reused <= 0.95 * trt;
  if (keep) tail_ = std::move(native);
  return std::format(
      "DistilDRBA : bloc natif {}, {:.2f} dB, écarts flot {:.5f} et masque {:.5f}, "
      "TensorRT {:.3f} ms, natif {:.3f} ms, avec réemploi {:.3f} ms",
      keep ? "retenu" : "écarté", score.snr, score.flow, score.mask, trt, cold, reused);
}

}  // namespace kirari::gpu::cuda
