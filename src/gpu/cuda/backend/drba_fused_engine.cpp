#include "gpu/cuda/backend/drba_fused_engine.h"

#include <algorithm>
#include <format>
#include <utility>

#include "core/utf8_path.h"
#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/drba_fused_kernels.h"

namespace ajp::gpu::cuda {

DrbaFusedEngine::~DrbaFusedEngine() = default;

std::expected<std::unique_ptr<DrbaFusedEngine>, Refusal> DrbaFusedEngine::load(
    nvinfer1::IRuntime& runtime, const std::filesystem::path& cache_dir, const DrbaFusedKey& key,
    const std::shared_ptr<SharedActivations>& shared) {
  if (!shared) return refuse(RefusalCode::GraphInvalid, "DistilDRBA fusionné : zone d'activation absente");
  if (!kernels::drba_fused_size_valid(key.plan, key.height, key.width) ||
      key.engines.size() != static_cast<std::size_t>(kernels::kDrbaFusedBlocks) + 1)
    return refuse(RefusalCode::GraphInvalid,
                  "DistilDRBA fusionné : découpage ou taille que les noyaux ne prennent pas");
  std::unique_ptr<DrbaFusedEngine> engine(new DrbaFusedEngine());
  engine->plan_ = key.plan;
  engine->height_ = key.height;
  engine->width_ = key.width;
  const int h = key.height;
  const int w = key.width;
  const int t = key.plan.tail;
  std::int64_t activation = 0;

  auto encode = TrtEngine::load(runtime, cache_dir / core::path_from_utf8(key.engines[0]), shared);
  if (!encode) return std::unexpected(std::move(encode.error()));
  const bool encode_ok =
      (*encode)->input_shape(0) == Shape{.dims = {3, 3, h / t, w / t}} &&
      (*encode)->output_shape(0) == Shape{.dims = {3, kernels::kDrbaFusedEncodeChannels, h / t, w / t}};
  if (!encode_ok)
    return refuse(RefusalCode::ShapeInvalid, "DistilDRBA fusionné : moteur d'encodage de forme inattendue");
  activation = std::max(activation, (*encode)->activation_bytes());
  engine->encode_ = std::move(*encode);
  for (int b = 0; b < kernels::kDrbaFusedBlocks; ++b) {
    auto block = TrtEngine::load(
        runtime, cache_dir / core::path_from_utf8(key.engines[static_cast<std::size_t>(b) + 1]), shared);
    if (!block) return std::unexpected(std::move(block.error()));
    const kernels::DrbaFusedDims x = kernels::drba_fused_x(key.plan, b, h, w);
    const kernels::DrbaFusedDims y = kernels::drba_fused_y(key.plan, b, h, w);
    const bool ok = (*block)->input_shape(0) == Shape{.dims = {1, x.channels, x.height, x.width}} &&
                    (*block)->output_shape(0) == Shape{.dims = {1, y.channels, y.height, y.width}};
    if (!ok)
      return refuse(RefusalCode::ShapeInvalid,
                    std::format("DistilDRBA fusionné : moteur du bloc {} de forme inattendue", b));
    activation = std::max(activation, (*block)->activation_bytes());
    engine->blocks_[static_cast<std::size_t>(b)] = std::move(*block);
  }
  engine->layout_ = kernels::drba_fused_layout(key.plan, h, w);
  engine->scratch_offset_ = kernels::drba_fused_align(static_cast<std::size_t>(activation));
  if (auto ok = engine->use_zone(shared); !ok) return std::unexpected(std::move(ok.error()));

  // Une paire de zéros trois fois par toute la chaîne, sur un flux à part : exécution à froid, capture,
  // rejeu.
  const std::size_t in_bytes = std::size_t{kernels::kDrbaFusedInputPlanes} * static_cast<std::size_t>(h) *
                               static_cast<std::size_t>(w) * 2U;
  const std::size_t out_bytes =
      std::size_t{3} * static_cast<std::size_t>(h) * static_cast<std::size_t>(w) * 2U;
  FusedScratch scratch;
  void*& scratch_input01 = scratch.buffers[0];
  void*& scratch_input12 = scratch.buffers[1];
  void*& scratch_output = scratch.buffers[2];
  const auto failed = [](cudaError_t error, const char* step) {
    static_cast<void>(cudaGetLastError());
    return refuse(
        RefusalCode::VendorCallFailed,
        std::format("DistilDRBA fusionné : essai au chargement ({}) : {}", step, cudaGetErrorString(error)));
  };
  for (void** buffer : {&scratch_input01, &scratch_input12}) {
    if (cudaError_t e = cudaMalloc(buffer, in_bytes); e != cudaSuccess) return failed(e, "cudaMalloc");
    if (cudaError_t e = cudaMemset(*buffer, 0, in_bytes); e != cudaSuccess) return failed(e, "cudaMemset");
  }
  if (cudaError_t e = cudaMalloc(&scratch_output, out_bytes); e != cudaSuccess)
    return failed(e, "cudaMalloc");
  if (cudaError_t e = cudaMemset(scratch_output, 0, out_bytes); e != cudaSuccess)
    return failed(e, "cudaMemset");
  if (cudaError_t e = cudaStreamCreateWithFlags(&scratch.stream, cudaStreamNonBlocking); e != cudaSuccess)
    return failed(e, "cudaStreamCreate");
  // L'essai écrit dans la zone d'activation partagée de la voie, sur un flux à lui : le travail encore
  // en vol sur les voies (SPAN, un autre moteur) finit avant (A03-05). enqueue nomme ce moteur
  // dernier utilisateur de la zone : celui qui y gardait ses tampons les refait.
  if (cudaError_t e = cudaDeviceSynchronize(); e != cudaSuccess) return failed(e, "cudaDeviceSynchronize");
  const Tensor inputs[]{fused_tensor(reinterpret_cast<std::uintptr_t>(scratch_input01), 0,
                                     kernels::kDrbaFusedInputPlanes, h, w),
                        fused_tensor(reinterpret_cast<std::uintptr_t>(scratch_input12), 0,
                                     kernels::kDrbaFusedInputPlanes, h, w)};
  const Tensor outputs[]{fused_tensor(reinterpret_cast<std::uintptr_t>(scratch_output), 0, 3, h, w)};
  for (int pass = 0; pass < 3; ++pass) {
    if (auto ok = engine->enqueue(scratch.stream, shared, inputs, outputs); !ok)
      return std::unexpected(std::move(ok.error()));
    if (cudaError_t e = cudaStreamSynchronize(scratch.stream); e != cudaSuccess)
      return failed(e, "exécution");
  }
  // Une capture refusée pendant l'essai laisse la chaîne tourner sans graphe : ce n'est pas un refus. Les
  // graphes de l'essai, sur des adresses qui disparaissent, ne servent plus ; les séries d'échecs repartent
  // de zéro, et sa notice part avec eux. Ce que l'essai a gardé dans la zone (D6) désignait ses tampons,
  // libérés au retour : une paire de la chaîne allouée à la même adresse ne doit pas le reprendre.
  engine->graphs_.clear();
  static_cast<void>(engine->graphs_.take_notice());
  engine->kept_.reset();
  engine->times_only_next_ = false;
  return engine;
}

std::expected<void, Refusal> DrbaFusedEngine::use_zone(const std::shared_ptr<SharedActivations>& zone) {
  const auto needed = static_cast<std::int64_t>(scratch_offset_ + layout_.bytes);
  if (auto ok = zone->reserve(needed); !ok) return ok;
  zone_ = zone;
  zone_generation_ = zone->generation();
  graphs_.clear();  // les graphes capturés l'étaient sur une autre adresse
  return {};
}

std::expected<void, Refusal> DrbaFusedEngine::run(cudaStream_t stream,
                                                  const std::shared_ptr<SharedActivations>& arena,
                                                  std::span<const Tensor> inputs,
                                                  std::span<const Tensor> outputs, bool times_only) {
  const int h = height_;
  const int w = width_;
  const int t = plan_.tail;
  auto* base = static_cast<std::byte*>(arena->data()) + scratch_offset_;
  const auto at = [base](std::size_t offset) { return reinterpret_cast<std::uintptr_t>(base + offset); };
  const auto at_ptr = [base](std::size_t offset) { return static_cast<void*>(base + offset); };
  const void* const pair01 = fused_address(inputs[0]);
  const void* const pair12 = fused_address(inputs[1]);

  if (auto ok = cuda_check(launch_drba_pre({.height = h,
                                            .width = w,
                                            .reduce0 = kernels::drba_fused_reduce(plan_, 0),
                                            .reduce1 = kernels::drba_fused_reduce(plan_, 1),
                                            .reduce2 = kernels::drba_fused_reduce(plan_, 2),
                                            .pair01 = pair01,
                                            .pair12 = pair12,
                                            .x0 = at_ptr(layout_.x[0]),
                                            .x1 = at_ptr(layout_.x[1]),
                                            .x2 = at_ptr(layout_.x[2]),
                                            .times_only = times_only},
                                           stream),
                           "drba_pre<<<>>>");
      !ok)
    return ok;
  // Les images réduites et leurs encodages de l'appel précédent sont toujours là (D6) : seul le temps change.
  if (!times_only) {
    // L'encodeur lit les images réduites au début de X_2 et y écrit les encodages juste après.
    const std::size_t enc_plane = static_cast<std::size_t>(h / t) * static_cast<std::size_t>(w / t);
    const Tensor images = fused_tensor(at(layout_.x[2]), 0, 3, h / t, w / t, 3);
    const Tensor encoded = fused_tensor(at(layout_.x[2]), kernels::kDrbaFusedImagePlanes * enc_plane * 2U,
                                        kernels::kDrbaFusedEncodeChannels, h / t, w / t, 3);
    if (auto ok = encode_->enqueue_plain(stream, arena, std::span(&images, 1), std::span(&encoded, 1)); !ok)
      return ok;
    if (auto ok = cuda_check(launch_drba_post({.enc_height = h / t,
                                               .enc_width = w / t,
                                               .x2 = at_ptr(layout_.x[2]),
                                               .x1 = at_ptr(layout_.x[1]),
                                               .x0 = at_ptr(layout_.x[0])},
                                              stream),
                             "drba_post<<<>>>");
        !ok)
      return ok;
  }
  for (int b = 0; b < kernels::kDrbaFusedBlocks; ++b) {
    const auto i = static_cast<std::size_t>(b);
    const kernels::DrbaFusedDims x = kernels::drba_fused_x(plan_, b, h, w);
    const kernels::DrbaFusedDims y = kernels::drba_fused_y(plan_, b, h, w);
    const Tensor block_in = fused_tensor(at(layout_.x[i]), 0, x.channels, x.height, x.width);
    const Tensor block_out = fused_tensor(at(layout_.y[i]), 0, y.channels, y.height, y.width);
    if (auto ok = blocks_[i]->enqueue_plain(stream, arena, std::span(&block_in, 1), std::span(&block_out, 1));
        !ok)
      return ok;
    if (b + 1 < kernels::kDrbaFusedBlocks) {
      const kernels::DrbaFusedDims next = kernels::drba_fused_x(plan_, b + 1, h, w);
      const DrbaGlueParams glue{.height = next.height,
                                .width = next.width,
                                .y = at_ptr(layout_.y[i]),
                                .pair01 = pair01,
                                .full_height = h,
                                .full_width = w,
                                .x = at_ptr(layout_.x[i + 1])};
      if (auto ok = cuda_check(launch_drba_glue(glue, stream), "drba_glue<<<>>>"); !ok) return ok;
    }
  }
  const auto last = static_cast<std::size_t>(kernels::kDrbaFusedBlocks - 1);
  return cuda_check(launch_drba_finish(
                        {
                            .height = h,
                            .width = w,
                            .upscale = t,
                            .y = at_ptr(layout_.y[last]),
                            .pair01 = pair01,
                            .pair12 = pair12,
                            .out = outputs.size() == 1 ? fused_address(outputs[0]) : nullptr,
                            .flow = outputs.size() == 2 ? fused_address(outputs[0]) : nullptr,
                            .mask = outputs.size() == 2 ? fused_address(outputs[1]) : nullptr,
                        },
                        stream),
                    "drba_finish<<<>>>");
}

std::expected<void, Refusal> DrbaFusedEngine::enqueue(cudaStream_t stream,
                                                      const std::shared_ptr<SharedActivations>& arena,
                                                      std::span<const Tensor> inputs,
                                                      std::span<const Tensor> outputs) {
  if (inputs.size() != 2 || (outputs.size() != 1 && outputs.size() != 2))
    return refuse(RefusalCode::ShapeInvalid,
                  "DistilDRBA fusionné : nombre d'entrées ou de sorties différent du moteur");
  const int h = height_;
  const int w = width_;
  const auto matches = [](const Tensor& t, const Shape& shape) {
    return t.type == TensorType::F16 && t.shape == shape && t.handle != 0;
  };
  if (!matches(inputs[0], input_shape()) || !matches(inputs[1], input_shape()))
    return refuse(RefusalCode::ShapeInvalid,
                  "DistilDRBA fusionné : entrée de forme ou de type différents du moteur");
  const bool motion = outputs.size() == 2;
  if (!motion && !matches(outputs[0], Shape{.dims = {1, 3, h, w}}))
    return refuse(RefusalCode::ShapeInvalid,
                  "DistilDRBA fusionné : sortie de forme ou de type différents du moteur");
  if (motion && (!matches(outputs[0], Shape{.dims = {1, 4, h, w}}) ||
                 !matches(outputs[1], Shape{.dims = {1, 1, h, w}})))
    return refuse(RefusalCode::ShapeInvalid,
                  "DistilDRBA fusionné : flot 1×4 et masque 1×1 à la taille du moteur attendus");
  // D6 : les mêmes paires, la même zone, que personne n'a touchée depuis l'appel précédent mené au bout.
  const std::uintptr_t pair01 = inputs[0].handle + inputs[0].offset;
  const std::uintptr_t pair12 = inputs[1].handle + inputs[1].offset;
  const bool times_only = std::exchange(times_only_next_, false) && kept_ && arena->last_user() == this &&
                          kept_->zone == arena->data() && kept_->generation == arena->generation() &&
                          kept_->pair01 == pair01 && kept_->pair12 == pair12;
  kept_.reset();
  auto queued = enqueue_chain(stream, arena, inputs, outputs, times_only);
  arena->set_last_user(this);  // après les morceaux, qui se sont nommés
  if (queued)
    kept_ =
        Kept{.pair01 = pair01, .pair12 = pair12, .zone = arena->data(), .generation = arena->generation()};
  return queued;
}

std::expected<void, Refusal> DrbaFusedEngine::enqueue_chain(cudaStream_t stream,
                                                            const std::shared_ptr<SharedActivations>& arena,
                                                            std::span<const Tensor> inputs,
                                                            std::span<const Tensor> outputs,
                                                            bool times_only) {
  const bool motion = outputs.size() == 2;
  // Lancé sur une autre voie que celle du chargement, ou la zone a grandi : tampons repris à leur place,
  // graphes jetés.
  const auto needed = static_cast<std::int64_t>(scratch_offset_ + layout_.bytes);
  if (arena != zone_ || arena->bytes() < needed || zone_generation_ != arena->generation()) {
    if (auto ok = use_zone(arena); !ok) return ok;
  }
  const FusedGraphs::Key key{inputs[0].handle + inputs[0].offset,
                             inputs[1].handle + inputs[1].offset,
                             outputs[0].handle + outputs[0].offset,
                             motion ? outputs[1].handle + outputs[1].offset : 0,
                             reinterpret_cast<std::uintptr_t>(arena->data()),
                             static_cast<std::uintptr_t>(arena->generation()),
                             times_only ? 1U : 0U};
  return graphs_.launch(stream, key, [&] { return run(stream, arena, inputs, outputs, times_only); });
}

}  // namespace ajp::gpu::cuda
