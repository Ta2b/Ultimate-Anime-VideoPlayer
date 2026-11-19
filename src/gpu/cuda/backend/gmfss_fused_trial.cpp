// L'essai au chargement du GMFSS natif (GmfssFusedEngine::try_native, docs/gmfss-natif.md) : des entrées de
// test passent par le moteur TensorRT de la fusion entière, puis par l'IFNet et le GridNet natif en sommes
// f16, mixtes et f32. Les variantes dont la sortie tient le seuil contre celle de TensorRT sont chronométrées
// contre lui ; la plus rapide n'est gardée que si elle le bat d'au moins 2 % (gmfss_trial.h). Les entrées :
// deux images lisses décalées de quelques pixels (l'IFNet y a du travail), leurs plans recopiés en tête des
// warped*_1, et des caractéristiques en ondes de l'ordre de celles de FeatureNet.
#include <algorithm>
#include <cmath>
#include <cuda_fp16.h>
#include <format>
#include <limits>
#include <utility>
#include <vector>

#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/fused_graphs.h"
#include "gpu/cuda/backend/gmfss_fused_engine.h"
#include "gpu/cuda/backend/gmfss_trial.h"

namespace kirari::gpu::cuda {
namespace {

// Les plans d'une entrée de test : images aux trois premiers plans des entrées half* et warped*_1, ondes
// au-delà (de pente et de phase propres au plan), l'instant à 0,5.
std::vector<__half> test_input(const Shape& shape, int input, float shift) {
  const auto c_count = static_cast<int>(shape.dims[1]);
  const auto h = static_cast<int>(shape.dims[2]);
  const auto w = static_cast<int>(shape.dims[3]);
  std::vector<__half> out(static_cast<std::size_t>(c_count) * static_cast<std::size_t>(h) *
                          static_cast<std::size_t>(w));
  std::size_t at = 0;
  for (int c = 0; c < c_count; ++c)
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        float v = 0.5F;
        if (h > 1 && c < 3) {
          v = gmfss_trial::drawing(static_cast<float>(x), static_cast<float>(y), c, shift);
        } else if (h > 1) {
          const float k = 0.05F + 0.003F * static_cast<float>((c * 7 + input) % 17);
          v = 0.8F * std::sin(k * static_cast<float>(x) + 0.7F * static_cast<float>(c)) *
              std::cos(0.6F * k * static_cast<float>(y) - static_cast<float>(input));
        }
        out[at++] = __float2half(v);
      }
  return out;
}

}  // namespace

std::expected<std::string, Refusal> GmfssFusedEngine::try_native(
    const prepare::GmfssGrid& grid, const std::shared_ptr<SharedActivations>& shared, double min_db) {
  gmfss_trial::Scratch scratch;
  const auto upload = [&](const std::vector<__half>& host) -> std::expected<Tensor, Refusal> {
    void* device = nullptr;
    if (auto ok = cuda_check(cudaMalloc(&device, host.size() * 2), "GMFSS natif : essai"); !ok)
      return std::unexpected(std::move(ok.error()));
    scratch.buffers.push_back(device);
    if (auto ok = cuda_check(cudaMemcpy(device, host.data(), host.size() * 2, cudaMemcpyHostToDevice),
                             "GMFSS natif : essai");
        !ok)
      return std::unexpected(std::move(ok.error()));
    return Tensor{.type = TensorType::F16, .shape = {}, .handle = reinterpret_cast<std::uintptr_t>(device)};
  };
  std::vector<Tensor> inputs(input_names().size());
  for (std::size_t i = 0; i < kInputs; ++i) {
    const Shape shape = full_->input_shape(at_[i]);
    // B (half1, warped1_*) est A décalée de 4 pixels.
    const float shift = i == kHalf1 || i >= kW11 ? 4.F : 0.F;
    auto made = upload(test_input(shape, static_cast<int>(i), shift));
    if (!made) return std::unexpected(std::move(made.error()));
    made->shape = shape;
    inputs[at_[i]] = *made;
  }
  const Shape out_shape = full_->output_shape(0);
  const auto elements = static_cast<std::size_t>(3 * out_shape.dims[2] * out_shape.dims[3]);
  auto out = upload(std::vector<__half>(elements));
  if (!out) return std::unexpected(std::move(out.error()));
  out->shape = out_shape;
  if (auto ok =
          cuda_check(cudaStreamCreateWithFlags(&scratch.stream, cudaStreamNonBlocking), "cudaStreamCreate");
      !ok)
    return std::unexpected(std::move(ok.error()));
  // Le travail en vol sur les voies finit avant : l'essai écrit dans la zone partagée (comme load).
  if (auto ok = cuda_check(cudaDeviceSynchronize(), "GMFSS natif : essai"); !ok)
    return std::unexpected(std::move(ok.error()));
  const Tensor outputs[]{*out};
  const auto run = [&]() -> std::expected<std::vector<__half>, Refusal> {
    // Trois passes (directe, capture du graphe de TensorRT, rejeu) ; la dernière est comparée.
    for (int pass = 0; pass < 3; ++pass) {
      if (auto ok = enqueue(scratch.stream, shared, inputs, outputs); !ok)
        return std::unexpected(std::move(ok.error()));
      if (auto ok = cuda_check(cudaStreamSynchronize(scratch.stream), "GMFSS natif : essai"); !ok)
        return std::unexpected(std::move(ok.error()));
    }
    std::vector<__half> host(elements);
    if (auto ok =
            cuda_check(cudaMemcpy(host.data(), fused_address(*out), elements * 2, cudaMemcpyDeviceToHost),
                       "GMFSS natif : copie de l'essai");
        !ok)
      return std::unexpected(std::move(ok.error()));
    return host;
  };
  native_on_ = false;
  auto reference = run();
  if (!reference) return std::unexpected(std::move(reference.error()));
  // Quoi qu'il arrive, l'essai laisse la fusion à TensorRT ou à la variante retenue.
  const auto settle = [&](std::unique_ptr<GmfssNativeNet> keep) {
    native_ = std::move(keep);
    native_on_ = native_ != nullptr;
  };
  // La justesse de chaque variante contre TensorRT ; celles qui tiennent le seuil vont à la mesure du temps.
  std::vector<std::unique_ptr<GmfssNativeNet>> nets(1);  // [0] : TensorRT
  std::string quality;
  for (const RifeNativeSums sums : gmfss_trial::kSums) {
    std::string error;
    auto net = GmfssNativeNet::create(grid, height_, width_, sums, error);
    if (!net) {
      settle(nullptr);
      return refuse(RefusalCode::FormatUnsupported, "GMFSS natif : " + error);
    }
    native_ = std::move(net);
    native_on_ = true;
    auto got = run();
    native_on_ = false;
    if (!got) {
      settle(nullptr);
      return std::unexpected(std::move(got.error()));
    }
    const double db = gmfss_trial::psnr(*reference, *got, 1.0);
    quality += std::format("{}{} {:.1f} dB", quality.empty() ? "" : ", ", gmfss_trial::sums_name(sums), db);
    if (db >= min_db) nets.push_back(std::move(native_));
    native_.reset();
  }
  const std::string where =
      std::format("{}×{}, contre TensorRT : {}, seuil {:.0f} dB", 2 * width_, 2 * height_, quality, min_db);
  if (nets.size() == 1) {
    settle(nullptr);
    return std::format("GMFSS natif écarté, TensorRT garde la fusion ({})", where);
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
  if (!choice) {
    settle(nullptr);
    return std::unexpected(std::move(choice.error()));
  }
  const std::size_t best = choice->best;
  const std::string verdict =
      best != 0 ? std::format("GMFSS natif retenu pour la fusion, sommes {} (zone {} Mio)",
                              gmfss_trial::sums_name(nets[best]->sums()), nets[best]->workspace_bytes() >> 20)
                : std::string("GMFSS natif écarté, TensorRT plus rapide pour la fusion");
  settle(best != 0 ? std::move(nets[best]) : nullptr);
  return std::format("{} ({} ; une passe : {})", verdict, where, choice->speed);
}

}  // namespace kirari::gpu::cuda
