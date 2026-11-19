// L'essai au chargement du RIFE natif (RifeFusedEngine::try_native, docs/rife-natif.md) : une paire de test
// passe par la chaîne avec les moteurs TensorRT, puis avec les noyaux natifs (sommes f16, mixtes, f32) ;
// les variantes dont la sortie tient le seuil contre celle de TensorRT sont chronométrées avec lui, et la
// plus rapide n'est gardée que si elle bat TensorRT. La paire : un dessin lisse (ondes, un carré aux bords
// francs) et le même décalé de quelques pixels, instant 0,5 ; le flot de RIFE y a du travail sans sortir de
// ce que le réseau voit en lecture.
#include <algorithm>
#include <array>
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

// La mesure du temps, comme celle du SPAN (span_timing.h) : des tours intercalés, chacun d'une passe de
// chauffe puis de deux passes entre deux événements ; la médiane des tours. Le natif n'est gardé que s'il
// bat TensorRT d'au moins kMinGain : en deçà, l'écart tient de la mesure.
constexpr int kTimeWarmup = 1;
constexpr int kTimeRuns = 2;
constexpr int kTimeRounds = 11;
constexpr double kMinGain = 0.02;

constexpr std::array<RifeNativeSums, 3> kSums{RifeNativeSums::kF16, RifeNativeSums::kMixed,
                                              RifeNativeSums::kF32};

const char* sums_name(RifeNativeSums sums) {
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

double median(std::vector<double> of) {
  std::ranges::sort(of);
  return of.empty() ? 0.0 : of[of.size() / 2];
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

std::expected<std::vector<double>, Refusal> RifeFusedEngine::trial_times(
    cudaStream_t stream, const std::shared_ptr<SharedActivations>& shared, const Tensor& input,
    std::span<const Tensor> outputs, std::vector<std::unique_ptr<RifeNativeNet>>& nets) {
  const auto failed = [](cudaError_t error, const char* step) {
    static_cast<void>(cudaGetLastError());
    return std::unexpected(cuda_refusal(error, std::format("RIFE natif : mesure du temps ({})", step)));
  };
  Event start;
  Event stop;
  for (Event* event : {&start, &stop})
    if (const cudaError_t e = cudaEventCreate(&event->event); e != cudaSuccess)
      return failed(e, "cudaEventCreate");
  const Tensor inputs[]{input};
  // Une passe de lecture ×2 : la B d'avant devient A, seule la nouvelle B s'encode. Chaque variante a ses
  // graphes (la variante est dans leur clé), rejoués comme en lecture une fois capturés.
  const auto pass = [&]() -> std::expected<void, Refusal> {
    next_pair_follows();
    return enqueue(stream, shared, inputs, outputs);
  };
  // Le lot de la variante v : son réseau prend la place de native_ le temps du lot, puis y retourne. warm :
  // passes de chauffe (au premier tour, un premier lancement puis une capture par emplacement de A).
  const auto batch = [&](std::size_t v, int warm) -> std::expected<double, Refusal> {
    std::swap(native_, nets[v]);
    native_on_ = native_ != nullptr;
    std::expected<void, Refusal> ok;
    for (int i = 0; i < warm && ok; ++i) ok = pass();
    cudaError_t e = ok ? cudaEventRecord(start.event, stream) : cudaSuccess;
    for (int i = 0; i < kTimeRuns && ok && e == cudaSuccess; ++i) ok = pass();
    if (ok && e == cudaSuccess) e = cudaEventRecord(stop.event, stream);
    if (ok && e == cudaSuccess) e = cudaEventSynchronize(stop.event);
    float ms = 0.F;
    if (ok && e == cudaSuccess) e = cudaEventElapsedTime(&ms, start.event, stop.event);
    std::swap(native_, nets[v]);
    native_on_ = false;
    if (!ok) return std::unexpected(std::move(ok.error()));
    if (e != cudaSuccess) return failed(e, "événement");
    if (!(ms > 0.F)) return refuse(RefusalCode::VendorCallFailed, "RIFE natif : lot mesuré à 0 ms");
    return static_cast<double>(ms) / kTimeRuns;
  };
  std::vector<std::vector<double>> rounds(nets.size());
  for (int round = 0; round < kTimeRounds; ++round)
    for (std::size_t i = 0; i < nets.size(); ++i) {
      const std::size_t v = round % 2 == 0 ? i : nets.size() - 1 - i;  // TensorRT, natifs, puis l'inverse
      auto ms = batch(v, round == 0 ? 4 : kTimeWarmup);
      if (!ms) return std::unexpected(std::move(ms.error()));
      rounds[v].push_back(*ms);
    }
  std::vector<double> times(rounds.size());
  std::ranges::transform(rounds, times.begin(), median);
  return times;
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

  // Quoi qu'il arrive, l'essai rend la chaîne à TensorRT ou à la variante retenue, sans graphe de l'essai
  // (ils visent les tampons de l'essai, libérés au retour), et laisse les comptes d'encodage de la lecture.
  const std::int64_t trt_activation = own_activation_;
  const auto saved_encodes = encodes_;
  const auto settle = [&](std::unique_ptr<RifeNativeNet> keep) {
    native_ = std::move(keep);
    native_on_ = native_ != nullptr;
    own_activation_ = native_on_ ? std::max(trt_activation, native_->workspace_bytes()) : trt_activation;
    forget_passes();
    encodes_ = saved_encodes;
  };
  native_on_ = false;
  auto reference = trial_output(scratch.stream, shared, input, outputs, out_planes * plane);
  if (!reference) {
    settle(nullptr);
    return std::unexpected(std::move(reference.error()));
  }
  // La justesse de chaque variante contre TensorRT ; celles qui tiennent le seuil vont à la mesure du temps.
  const std::vector<int> reduce(plan_.reduce.begin(), plan_.reduce.begin() + plan_.blocks);
  std::vector<std::unique_ptr<RifeNativeNet>> nets(1);  // [0] : TensorRT
  std::int64_t region_floor = trt_activation;
  std::string quality;
  for (const RifeNativeSums sums : kSums) {
    std::string error;
    auto net = RifeNativeNet::create(pieces, reduce, h, w, sums, error);
    if (!net) {
      settle(nullptr);
      return refuse(RefusalCode::FormatUnsupported, "RIFE natif : " + error);
    }
    // La zone de travail des noyaux s'ajoute aux activations : les tampons de la chaîne montent au-dessus.
    own_activation_ = std::max(trt_activation, net->workspace_bytes());
    native_ = std::move(net);
    native_on_ = true;
    auto got = trial_output(scratch.stream, shared, input, outputs, out_planes * plane);
    native_on_ = false;
    if (!got) {
      settle(nullptr);
      return std::unexpected(std::move(got.error()));
    }
    const double db = worst_psnr(*reference, *got, plane, motion_);
    quality += std::format("{}{} {:.1f} dB", quality.empty() ? "" : ", ", sums_name(sums), db);
    if (db >= min_db) {
      region_floor = std::max(region_floor, native_->workspace_bytes());
      nets.push_back(std::move(native_));
    }
    native_.reset();
  }
  const std::string where =
      std::format("{}×{}, contre TensorRT : {}, seuil {:.0f} dB", w, h, quality, min_db);
  if (nets.size() == 1) {
    settle(nullptr);
    return std::format("RIFE natif écarté, TensorRT garde la chaîne ({})", where);
  }
  // Toutes les variantes sous le même plancher : la place des tampons ne change pas d'un lot à l'autre.
  own_activation_ = region_floor;
  forget_passes();
  auto times = trial_times(scratch.stream, shared, input, outputs, nets);
  if (!times) {
    settle(nullptr);
    return std::unexpected(std::move(times.error()));
  }
  std::string speed = std::format("TensorRT {:.2f} ms", (*times)[0]);
  std::size_t best = 0;
  for (std::size_t v = 1; v < nets.size(); ++v) {
    speed += std::format(", {} {:.2f} ms", sums_name(nets[v]->sums()), (*times)[v]);
    if (best == 0 || (*times)[v] < (*times)[best]) best = v;
  }
  const bool faster = (*times)[best] <= (*times)[0] * (1.0 - kMinGain);
  const std::string verdict = faster
                                  ? std::format("RIFE natif retenu, sommes {}", sums_name(nets[best]->sums()))
                                  : std::string("RIFE natif écarté, TensorRT plus rapide");
  settle(faster ? std::move(nets[best]) : nullptr);
  return std::format("{} ({} ; passe de lecture ×2 : {})", verdict, where, speed);
}

}  // namespace kirari::gpu::cuda
