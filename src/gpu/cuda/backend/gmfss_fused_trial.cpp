// L'essai au chargement du GMFSS natif pour la fusion (GmfssFusedEngine::try_native, docs/gmfss-natif.md) :
// le GridNet natif réglé au banc, puis des entrées de test passent par le moteur TensorRT de la fusion
// entière et par l'IFNet et le GridNet natif ; le choix des sommes de chaque morceau (les trois lignes, la
// queue) est celui de gmfss_trial.h. Les entrées : deux images lisses décalées de quelques pixels (l'IFNet y
// a du travail), leurs plans recopiés en tête des warped*_1, et des caractéristiques en ondes de l'ordre de
// celles de FeatureNet.
#include <chrono>
#include <cmath>
#include <cuda_fp16.h>
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

std::expected<std::vector<std::string>, Refusal> GmfssFusedEngine::try_native(
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
  // Quoi qu'il arrive, l'essai laisse la fusion à TensorRT ou au natif retenu.
  native_.reset();
  native_on_ = false;
  const auto started = std::chrono::steady_clock::now();
  std::string error;
  std::unique_ptr<GmfssNativeNet> net = GmfssNativeNet::create(grid, height_, width_, scratch.stream, error);
  if (!net) {
    if (const cudaError_t lost = clear_error(); lost != cudaSuccess)
      return std::unexpected(lost_refusal(lost, "GMFSS natif : réglage de la fusion"));
    return refuse(RefusalCode::FormatUnsupported, "GMFSS natif : " + error);
  }
  const double tuning = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  const Tensor outputs[]{*out};
  native_ = std::move(net);
  gmfss_trial::Unit unit{
      .name = "fusion",
      .width = 2 * width_,
      .height = 2 * height_,
      .parts = native_.get(),
      .stream = scratch.stream,
      .outputs = 1,
      .pass =
          [&](bool native) {
            native_on_ = native;
            auto ok = enqueue(scratch.stream, shared, inputs, outputs);
            native_on_ = false;
            return ok;
          },
      .piece =
          [&](int piece) { return cuda_check(native_->run_piece(scratch.stream, piece), "GMFSS natif"); },
      .fetch = [&](std::size_t) -> std::expected<std::vector<__half>, Refusal> {
        std::vector<__half> host(elements);
        if (auto ok =
                cuda_check(cudaMemcpy(host.data(), fused_address(*out), elements * 2, cudaMemcpyDeviceToHost),
                           "GMFSS natif : copie de l'essai");
            !ok)
          return std::unexpected(std::move(ok.error()));
        return host;
      }};
  auto outcome = gmfss_trial::try_mix(unit, min_db, tuning);
  if (!outcome || !outcome->keep) native_.reset();
  native_on_ = native_ != nullptr;
  if (!outcome) return std::unexpected(std::move(outcome.error()));
  return std::move(outcome->lines);
}

}  // namespace kirari::gpu::cuda
