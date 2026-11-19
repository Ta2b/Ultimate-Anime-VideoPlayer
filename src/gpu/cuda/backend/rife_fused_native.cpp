// L'essai au chargement du RIFE natif (RifeFusedEngine::try_native, docs/rife-natif.md) : une paire de test
// passe par la chaîne avec les moteurs TensorRT, puis avec les noyaux natifs (sommes f16, puis f32), et le
// natif n'est gardé que si sa sortie tient le seuil contre celle de TensorRT. La paire : un dessin lisse
// (ondes, un carré aux bords francs) et le même décalé de quelques pixels, instant 0,5 ; le flot de RIFE y a
// du travail sans sortir de ce que le réseau voit en lecture.
#include <algorithm>
#include <cmath>
#include <cuda_fp16.h>
#include <format>
#include <limits>
#include <utility>

#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/rife_fused_engine.h"

namespace kirari::gpu::cuda {
namespace {

// L'entrée v1 de RIFE (11 plans) pour la paire de test : A, B (A décalé de (5, 3) pixels), l'instant, la
// grille normalisée et ses pas.
std::vector<__half> test_pair(int h, int w) {
  const auto plane = static_cast<std::size_t>(h) * static_cast<std::size_t>(w);
  std::vector<__half> input(plane * kernels::kFusedInputPlanes);
  const auto drawing = [](float x, float y, int channel) {
    const auto c = static_cast<float>(channel);
    const float wave =
        0.5F + 0.22F * std::sin(0.071F * x + 0.043F * y + c) + 0.12F * std::sin(0.017F * x - 0.029F * y);
    const bool square = std::fmod(std::abs(x), 160.F) < 64.F && std::fmod(std::abs(y), 128.F) < 48.F;
    return std::clamp(square ? 0.9F - 0.2F * c : wave, 0.F, 1.F);
  };
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const std::size_t at =
          static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x);
      const auto fx = static_cast<float>(x);
      const auto fy = static_cast<float>(y);
      for (int c = 0; c < 3; ++c) {
        input[static_cast<std::size_t>(c) * plane + at] = __float2half(drawing(fx, fy, c));
        input[static_cast<std::size_t>(3 + c) * plane + at] = __float2half(drawing(fx - 5.F, fy - 3.F, c));
      }
      input[6 * plane + at] = __float2half(0.5F);
      input[7 * plane + at] = __float2half(2.F * fx / static_cast<float>(w - 1) - 1.F);
      input[8 * plane + at] = __float2half(2.F * fy / static_cast<float>(h - 1) - 1.F);
      input[9 * plane + at] = __float2half(2.F / static_cast<float>(w - 1));
      input[10 * plane + at] = __float2half(2.F / static_cast<float>(h - 1));
    }
  return input;
}

// Le PSNR de got contre want, plan par plan de taille plane, au pic max(1, |want| le plus grand du plan
// groupé) : l'image va de 0 à 1, le flot en pixels peut dépasser 1. Le plus bas des groupes (images : 3
// plans ; mouvement : flot 4 plans, masque 1 plan). Une valeur non finie, des deux côtés : -infini (une
// référence fautive ne laisse rien passer).
double worst_psnr(const std::vector<float>& want, const std::vector<float>& got, std::size_t plane,
                  bool motion) {
  const std::vector<std::pair<std::size_t, std::size_t>> groups =
      motion ? std::vector<std::pair<std::size_t, std::size_t>>{{0, 4}, {4, 5}}
             : std::vector<std::pair<std::size_t, std::size_t>>{{0, 3}};
  double worst = std::numeric_limits<double>::infinity();
  for (const auto& [first, end] : groups) {
    double peak = 1.0;
    double sum = 0.0;
    for (std::size_t i = first * plane; i < end * plane; ++i) {
      if (!std::isfinite(got[i]) || !std::isfinite(want[i])) return -std::numeric_limits<double>::infinity();
      peak = std::max(peak, static_cast<double>(std::abs(want[i])));
      const double d = static_cast<double>(got[i]) - static_cast<double>(want[i]);
      sum += d * d;
    }
    const double mse = sum / static_cast<double>((end - first) * plane);
    worst = std::min(worst, mse == 0.0 ? 200.0 : 10.0 * std::log10(peak * peak / mse));
  }
  return worst;
}

}  // namespace

void RifeFusedEngine::forget_passes() {
  graphs_.clear();
  static_cast<void>(graphs_.take_notice());
  kept_.reset();
  hint_ = Hint::None;
  warm_ready_ = false;
  warm_next_ = false;
}

std::expected<std::vector<float>, Refusal> RifeFusedEngine::trial_output(
    cudaStream_t stream, const std::shared_ptr<SharedActivations>& shared, const Tensor& input,
    std::span<const Tensor> outputs, std::size_t elements) {
  forget_passes();
  // Trois passes : directe, capture du graphe, rejeu ; la dernière est comparée.
  for (int pass = 0; pass < 3; ++pass) {
    const Tensor inputs[]{input};
    if (auto ok = enqueue(stream, shared, inputs, outputs); !ok)
      return std::unexpected(std::move(ok.error()));
    if (auto ok = cuda_check(cudaStreamSynchronize(stream), "RIFE natif : essai"); !ok)
      return std::unexpected(std::move(ok.error()));
  }
  std::vector<__half> half(elements);
  if (auto ok =
          cuda_check(cudaMemcpy(half.data(), fused_address(outputs[0]), elements * 2, cudaMemcpyDeviceToHost),
                     "RIFE natif : copie de l'essai");
      !ok)
    return std::unexpected(std::move(ok.error()));
  std::vector<float> out(elements);
  std::ranges::transform(half, out.begin(), [](__half v) { return __half2float(v); });
  return out;
}

std::expected<std::string, Refusal> RifeFusedEngine::try_native(
    const std::vector<RifeNativePiece>& pieces, const std::shared_ptr<SharedActivations>& shared,
    double min_db) {
  if (!hwc8_)
    return refuse(RefusalCode::FormatUnsupported,
                  "RIFE natif : blocs TensorRT en plans, le natif lit et écrit X et Y en HWC8");
  const int h = height_;
  const int w = width_;
  const auto plane = static_cast<std::size_t>(h) * static_cast<std::size_t>(w);
  // Entrée et sorties d'un seul tenant : flot (4 plans) puis masque, ou l'image (3 plans).
  const std::size_t out_planes = motion_ ? 5 : 3;
  FusedScratch scratch;
  // Une erreur collante (contexte perdu) le dit comme telle : l'appelant ne garde pas la chaîne pour autant.
  const auto failed = [](cudaError_t error, const char* step) {
    static_cast<void>(cudaGetLastError());
    return std::unexpected(cuda_refusal(error, std::format("RIFE natif : essai ({})", step)));
  };
  const std::vector<__half> host = test_pair(h, w);
  if (cudaError_t e = cudaMalloc(&scratch.buffers[0], host.size() * 2); e != cudaSuccess)
    return failed(e, "cudaMalloc");
  if (cudaError_t e = cudaMalloc(&scratch.buffers[1], out_planes * plane * 2); e != cudaSuccess)
    return failed(e, "cudaMalloc");
  if (cudaError_t e = cudaMemcpy(scratch.buffers[0], host.data(), host.size() * 2, cudaMemcpyHostToDevice);
      e != cudaSuccess)
    return failed(e, "cudaMemcpy");
  if (cudaError_t e = cudaStreamCreateWithFlags(&scratch.stream, cudaStreamNonBlocking); e != cudaSuccess)
    return failed(e, "cudaStreamCreate");
  // Le travail en vol sur les voies finit avant : l'essai écrit dans la zone partagée (comme load).
  if (cudaError_t e = cudaDeviceSynchronize(); e != cudaSuccess) return failed(e, "cudaDeviceSynchronize");
  const auto in = reinterpret_cast<std::uintptr_t>(scratch.buffers[0]);
  const auto out = reinterpret_cast<std::uintptr_t>(scratch.buffers[1]);
  const Tensor input = fused_tensor(in, 0, kernels::kFusedInputPlanes, h, w);
  std::vector<Tensor> outputs;
  if (motion_)
    outputs = {fused_tensor(out, 0, 4, h, w), fused_tensor(out, 4U * plane * 2U, 1, h, w)};
  else
    outputs = {fused_tensor(out, 0, 3, h, w)};

  native_on_ = false;
  auto reference = trial_output(scratch.stream, shared, input, outputs, out_planes * plane);
  if (!reference) {
    forget_passes();  // les graphes capturés visent les tampons de l'essai, libérés au retour
    return std::unexpected(std::move(reference.error()));
  }
  std::vector<int> reduce(plan_.reduce.begin(), plan_.reduce.begin() + plan_.blocks);
  const std::int64_t trt_activation = own_activation_;
  std::string tried;
  for (const bool acc32 : {false, true}) {
    std::string error;
    native_on_ = false;
    own_activation_ = trt_activation;
    native_ = RifeNativeNet::create(pieces, reduce, h, w, acc32, error);
    if (!native_) {
      forget_passes();
      return refuse(RefusalCode::FormatUnsupported, "RIFE natif : " + error);
    }
    // La zone de travail des noyaux s'ajoute aux activations : les tampons de la chaîne montent au-dessus.
    own_activation_ = std::max(trt_activation, native_->workspace_bytes());
    native_on_ = true;
    auto got = trial_output(scratch.stream, shared, input, outputs, out_planes * plane);
    if (!got) {
      native_on_ = false;
      native_.reset();
      own_activation_ = trt_activation;
      forget_passes();
      return std::unexpected(std::move(got.error()));
    }
    const double db = worst_psnr(*reference, *got, plane, motion_);
    tried += std::format("{}sommes {} : {:.1f} dB", tried.empty() ? "" : ", ", acc32 ? "f32" : "f16", db);
    if (db >= min_db) {
      forget_passes();
      return std::format("RIFE natif retenu ({}×{}, {} contre TensorRT, seuil {:.0f} dB)", w, h, tried,
                         min_db);
    }
  }
  native_on_ = false;
  native_.reset();
  own_activation_ = trt_activation;
  forget_passes();
  return std::format(
      "RIFE natif écarté, TensorRT garde la chaîne ({}×{}, {} contre TensorRT, seuil {:.0f} dB)", w, h, tried,
      min_db);
}

}  // namespace kirari::gpu::cuda
