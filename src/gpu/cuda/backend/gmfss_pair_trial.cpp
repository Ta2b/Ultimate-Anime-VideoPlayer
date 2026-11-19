// L'essai au chargement de la paire native de GMFSS (GmfssPairEngine::try_native, docs/gmfss-natif.md) : le
// FeatureNet et le MetricNet natifs réglés au banc, puis deux images de test, la seconde décalée de quelques
// pixels, passent par le moteur TensorRT de la paire entière et par le flot et les réseaux natifs ; le choix
// des sommes de chaque réseau est celui de gmfss_trial.h, chacune des quatorze sorties jugée à part, l'écart
// rapporté à son étendue (au moins 1 : les caractéristiques et les flots ne sont pas des images).
#include <chrono>
#include <cuda_fp16.h>
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

}  // namespace

std::expected<std::vector<std::string>, Refusal> GmfssPairEngine::try_native(
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
  // Quoi qu'il arrive, l'essai laisse la paire à TensorRT ou au natif retenu.
  native_.reset();
  native_on_ = false;
  const auto started = std::chrono::steady_clock::now();
  std::string error;
  std::unique_ptr<GmfssPairNativeNet> net =
      GmfssPairNativeNet::create(nets_in, height_, width_, scratch.stream, error);
  if (!net) {
    if (const cudaError_t lost = clear_error(); lost != cudaSuccess)
      return std::unexpected(lost_refusal(lost, "GMFSS natif : réglage de la paire"));
    return refuse(RefusalCode::FormatUnsupported, "GMFSS natif : paire : " + error);
  }
  const double tuning = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  native_ = std::move(net);
  const Tensor inputs[]{frames};
  gmfss_trial::Unit unit{
      .name = "paire",
      .width = 2 * width_,
      .height = 2 * height_,
      .parts = native_.get(),
      .stream = scratch.stream,
      .outputs = kOutputs,
      .pass =
          [&](bool native) {
            native_on_ = native;
            auto ok = enqueue(scratch.stream, shared, inputs, outputs);
            native_on_ = false;
            return ok;
          },
      .piece = [&](int piece) { return enqueue_piece(scratch.stream, piece, frames, outputs); },
      .fetch = [&](std::size_t i) -> std::expected<std::vector<__half>, Refusal> {
        std::vector<__half> host(elements(outputs[i].shape));
        if (auto ok = cuda_check(
                cudaMemcpy(host.data(), fused_address(outputs[i]), host.size() * 2, cudaMemcpyDeviceToHost),
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
