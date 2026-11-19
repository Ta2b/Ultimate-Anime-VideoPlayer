// L'essai au chargement de la paire native de GMFSS (GmfssPairEngine::try_native, docs/gmfss-natif.md) :
// deux images de test, la seconde décalée de quelques pixels, passent par le moteur TensorRT de la paire
// entière, puis par le flot et les réseaux natifs en sommes f16, mixtes et f32. Chaque sortie est comparée à
// celle de TensorRT, l'écart rapporté à son étendue (au moins 1 : les caractéristiques et les flots ne sont
// pas des images) ; les variantes dont la pire tient le seuil sont chronométrées contre TensorRT, et la plus
// rapide n'est gardée que si elle le bat d'au moins 2 % (gmfss_trial.h).
#include <algorithm>
#include <cmath>
#include <cuda_fp16.h>
#include <format>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/fused_graphs.h"
#include "gpu/cuda/backend/gmfss_pair_engine.h"
#include "gpu/cuda/backend/gmfss_trial.h"

namespace kirari::gpu::cuda {
namespace {

std::size_t elements(const Shape& shape) {
  std::size_t count = 1;
  for (const std::int64_t d : shape.dims) count *= static_cast<std::size_t>(d);
  return count;
}

// frames : l'image A aux trois premiers plans, B (A décalée de 4 pixels) aux trois suivants.
std::vector<__half> test_frames(const Shape& shape) {
  const auto h = static_cast<int>(shape.dims[2]);
  const auto w = static_cast<int>(shape.dims[3]);
  std::vector<__half> out(elements(shape));
  std::size_t at = 0;
  for (int plane = 0; plane < static_cast<int>(shape.dims[1]); ++plane)
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x)
        out[at++] = __float2half(gmfss_trial::drawing(static_cast<float>(x), static_cast<float>(y), plane % 3,
                                                      plane < 3 ? 0.F : 4.F));
  return out;
}

double peak_of(const std::vector<__half>& values) {
  float peak = 1.F;
  for (const __half v : values) peak = std::max(peak, std::abs(__half2float(v)));
  return peak;
}

using Planes = std::vector<std::vector<__half>>;

// Les sorties de la passe, relues sur l'hôte.
std::expected<Planes, Refusal> fetch_all(std::span<const Tensor> outputs) {
  Planes out;
  for (const Tensor& t : outputs) {
    std::vector<__half> host(elements(t.shape));
    if (auto ok =
            cuda_check(cudaMemcpy(host.data(), fused_address(t), host.size() * 2, cudaMemcpyDeviceToHost),
                       "GMFSS natif : copie de l'essai");
        !ok)
      return std::unexpected(std::move(ok.error()));
    out.push_back(std::move(host));
  }
  return out;
}

// La pire sortie contre la référence, l'écart rapporté à l'étendue de la référence ; at reçoit sa place.
double worst_of(const Planes& reference, const Planes& got, std::size_t& at) {
  double worst = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < reference.size(); ++i) {
    const double db = gmfss_trial::psnr(reference[i], got[i], peak_of(reference[i]));
    if (db < worst) {
      worst = db;
      at = i;
    }
  }
  return worst;
}

}  // namespace

std::expected<std::string, Refusal> GmfssPairEngine::try_native(
    const prepare::GmfssPairNets& nets_in, const std::shared_ptr<SharedActivations>& shared, double min_db) {
  gmfss_trial::Scratch scratch;
  const auto device = [&](std::size_t count) -> std::expected<void*, Refusal> {
    void* at = nullptr;
    if (auto ok = cuda_check(cudaMalloc(&at, count * 2), "GMFSS natif : essai de la paire"); !ok)
      return std::unexpected(std::move(ok.error()));
    scratch.buffers.push_back(at);
    return at;
  };
  const Shape frames_shape = full_->input_shape(0);
  const std::vector<__half> frames_host = test_frames(frames_shape);
  auto frames_at = device(frames_host.size());
  if (!frames_at) return std::unexpected(std::move(frames_at.error()));
  if (auto ok = cuda_check(
          cudaMemcpy(*frames_at, frames_host.data(), frames_host.size() * 2, cudaMemcpyHostToDevice),
          "GMFSS natif : essai de la paire");
      !ok)
    return std::unexpected(std::move(ok.error()));
  const Tensor frames{
      .type = TensorType::F16, .shape = frames_shape, .handle = reinterpret_cast<std::uintptr_t>(*frames_at)};
  std::vector<Tensor> outputs(kOutputs);
  for (std::size_t i = 0; i < kOutputs; ++i) {
    auto at = device(elements(full_->output_shape(i)));
    if (!at) return std::unexpected(std::move(at.error()));
    outputs[i] = Tensor{.type = TensorType::F16,
                        .shape = full_->output_shape(i),
                        .handle = reinterpret_cast<std::uintptr_t>(*at)};
  }
  if (auto ok =
          cuda_check(cudaStreamCreateWithFlags(&scratch.stream, cudaStreamNonBlocking), "cudaStreamCreate");
      !ok)
    return std::unexpected(std::move(ok.error()));
  // Le travail en vol sur les voies finit avant : l'essai écrit dans la zone partagée (comme load).
  if (auto ok = cuda_check(cudaDeviceSynchronize(), "GMFSS natif : essai de la paire"); !ok)
    return std::unexpected(std::move(ok.error()));
  const Tensor inputs[]{frames};
  // Trois passes (directe, capture du graphe de TensorRT, rejeu) ; la dernière est comparée.
  const auto run = [&]() -> std::expected<void, Refusal> {
    for (int pass = 0; pass < 3; ++pass) {
      if (auto ok = enqueue(scratch.stream, shared, inputs, outputs); !ok) return ok;
      if (auto ok = cuda_check(cudaStreamSynchronize(scratch.stream), "GMFSS natif : essai de la paire"); !ok)
        return ok;
    }
    return {};
  };
  native_on_ = false;
  if (auto ok = run(); !ok) return std::unexpected(std::move(ok.error()));
  auto reference = fetch_all(outputs);
  if (!reference) return std::unexpected(std::move(reference.error()));
  // Quoi qu'il arrive, l'essai laisse la paire à TensorRT ou à la variante retenue.
  const auto settle = [&](std::unique_ptr<GmfssPairNativeNet> keep) {
    native_ = std::move(keep);
    native_on_ = native_ != nullptr;
  };
  const auto give_up = [&](Refusal refusal) -> std::expected<std::string, Refusal> {
    settle(nullptr);
    return std::unexpected(std::move(refusal));
  };
  // La justesse de chaque variante contre TensorRT ; celles dont la pire sortie tient le seuil vont à la
  // mesure du temps.
  std::vector<std::unique_ptr<GmfssPairNativeNet>> nets(1);  // [0] : TensorRT
  std::string quality;
  for (const RifeNativeSums sums : gmfss_trial::kSums) {
    std::string error;
    native_ = GmfssPairNativeNet::create(nets_in, height_, width_, sums, error);
    if (!native_) return give_up(Refusal{RefusalCode::FormatUnsupported, "GMFSS natif : paire : " + error});
    native_on_ = true;
    auto ran = run();
    native_on_ = false;
    if (!ran) return give_up(std::move(ran.error()));
    auto got = fetch_all(outputs);
    if (!got) return give_up(std::move(got.error()));
    std::size_t at = 0;
    const double worst = worst_of(*reference, *got, at);
    quality += std::format("{}{} {:.1f} dB au pire ({})", quality.empty() ? "" : ", ",
                           gmfss_trial::sums_name(sums), worst, output_names()[at]);
    if (worst >= min_db) nets.push_back(std::move(native_));
    native_.reset();
  }
  const std::string where =
      std::format("{}×{}, contre TensorRT : {}, seuil {:.0f} dB", 2 * width_, 2 * height_, quality, min_db);
  if (nets.size() == 1) {
    settle(nullptr);
    return std::format("GMFSS natif écarté, TensorRT garde la paire ({})", where);
  }
  // Une passe de la variante v : son réseau prend la place de native_ le temps de la passe.
  const auto pass = [&](std::size_t v) -> std::expected<void, Refusal> {
    std::swap(native_, nets[v]);
    native_on_ = native_ != nullptr;
    auto ok = enqueue(scratch.stream, shared, inputs, outputs);
    std::swap(native_, nets[v]);
    native_on_ = false;
    return ok;
  };
  auto choice = gmfss_trial::choose(scratch.stream, nets, pass);
  if (!choice) return give_up(std::move(choice.error()));
  const std::size_t best = choice->best;
  const std::string verdict =
      best != 0 ? std::format("GMFSS natif retenu pour la paire, sommes {} (zone {} Mio)",
                              gmfss_trial::sums_name(nets[best]->sums()), nets[best]->workspace_bytes() >> 20)
                : std::string("GMFSS natif écarté, TensorRT plus rapide pour la paire");
  settle(best != 0 ? std::move(nets[best]) : nullptr);
  return std::format("{} ({} ; une passe : {})", verdict, where, choice->speed);
}

}  // namespace kirari::gpu::cuda
