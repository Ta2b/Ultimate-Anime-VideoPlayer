#include "gpu/cuda/backend/rife_fused_engine.h"

#include <algorithm>
#include <format>
#include <utility>

#include "core/utf8_path.h"
#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/rife_fused_kernels.h"

namespace ajp::gpu::cuda {

RifeFusedEngine::~RifeFusedEngine() = default;

std::expected<std::unique_ptr<RifeFusedEngine>, Refusal> RifeFusedEngine::load(
    nvinfer1::IRuntime& runtime, const std::filesystem::path& cache_dir, const RifeFusedKey& key,
    const std::shared_ptr<SharedActivations>& shared) {
  if (!shared) return refuse(RefusalCode::GraphInvalid, "RIFE fusionné : zone d'activation absente");
  if (!kernels::rife_fused_size_valid(key.plan, key.height, key.width) ||
      key.engines.size() != static_cast<std::size_t>(key.plan.blocks) + 1)
    return refuse(RefusalCode::GraphInvalid,
                  "RIFE fusionné : découpage ou taille que les noyaux ne prennent pas");
  std::unique_ptr<RifeFusedEngine> engine(new RifeFusedEngine());
  engine->plan_ = key.plan;
  engine->height_ = key.height;
  engine->width_ = key.width;
  engine->motion_ = key.motion;
  engine->outputs_ =
      key.motion ? std::vector<std::string>{"flow", "mask"} : std::vector<std::string>{"output"};
  const int h = key.height;
  const int w = key.width;
  std::int64_t activation = 0;

  auto encode = TrtEngine::load(runtime, cache_dir / core::path_from_utf8(key.engines[0]), shared);
  if (!encode) return std::unexpected(std::move(encode.error()));
  // Une image à la fois (fusion3) : un moteur d'encodage de l'ancienne découpe, en lot de 2, est refusé.
  const bool encode_ok = (*encode)->input_shape(0) == Shape{.dims = {1, 3, h, w}} &&
                         (*encode)->output_shape(0) == Shape{.dims = {1, 4, h, w}};
  if (!encode_ok)
    return refuse(RefusalCode::ShapeInvalid, "RIFE fusionné : moteur d'encodage de forme inattendue");
  activation = std::max(activation, (*encode)->activation_bytes());
  engine->encode_ = std::move(*encode);
  for (int b = 0; b < key.plan.blocks; ++b) {
    auto block = TrtEngine::load(
        runtime, cache_dir / core::path_from_utf8(key.engines[static_cast<std::size_t>(b) + 1]), shared);
    if (!block) return std::unexpected(std::move(block.error()));
    const kernels::FusedDims x = kernels::rife_fused_x(key.plan, b, h, w);
    const kernels::FusedDims y = kernels::rife_fused_y(key.plan, b, h, w);
    const bool ok = (*block)->input_shape(0) == Shape{.dims = {1, x.channels, x.height, x.width}} &&
                    (*block)->output_shape(0) == Shape{.dims = {1, y.channels, y.height, y.width}};
    if (!ok)
      return refuse(RefusalCode::ShapeInvalid,
                    std::format("RIFE fusionné : moteur du bloc {} de forme inattendue", b));
    activation = std::max(activation, (*block)->activation_bytes());
    engine->blocks_.push_back(std::move(*block));
  }
  engine->layout_ = kernels::rife_fused_layout(key.plan, h, w);
  engine->scratch_offset_ = kernels::fused_align(static_cast<std::size_t>(activation));
  if (auto ok = engine->use_zone(shared); !ok) return std::unexpected(std::move(ok.error()));

  // Une image de zéros trois fois par toute la chaîne, sur un flux à part : exécution à froid, capture,
  // rejeu.
  const std::size_t in_bytes = std::size_t{kernels::kFusedInputPlanes} * static_cast<std::size_t>(h) *
                               static_cast<std::size_t>(w) * 2U;
  const std::size_t out_bytes =
      std::size_t{5} * static_cast<std::size_t>(h) * static_cast<std::size_t>(w) * 2U;
  FusedScratch scratch;
  void*& scratch_input = scratch.buffers[0];
  void*& scratch_output = scratch.buffers[1];
  const auto failed = [](cudaError_t error, const char* step) {
    static_cast<void>(cudaGetLastError());
    return refuse(RefusalCode::VendorCallFailed, std::format("RIFE fusionné : essai au chargement ({}) : {}",
                                                             step, cudaGetErrorString(error)));
  };
  if (cudaError_t e = cudaMalloc(&scratch_input, in_bytes); e != cudaSuccess) return failed(e, "cudaMalloc");
  if (cudaError_t e = cudaMalloc(&scratch_output, out_bytes); e != cudaSuccess)
    return failed(e, "cudaMalloc");
  if (cudaError_t e = cudaMemset(scratch_input, 0, in_bytes); e != cudaSuccess)
    return failed(e, "cudaMemset");
  if (cudaError_t e = cudaMemset(scratch_output, 0, out_bytes); e != cudaSuccess)
    return failed(e, "cudaMemset");
  if (cudaError_t e = cudaStreamCreateWithFlags(&scratch.stream, cudaStreamNonBlocking); e != cudaSuccess)
    return failed(e, "cudaStreamCreate");
  // L'essai écrit dans la zone d'activation partagée de la voie, sur un flux à lui : le travail encore
  // en vol sur les voies (SPAN, un autre moteur) finit avant (A03-05). enqueue nomme ce moteur
  // dernier utilisateur de la zone : celui qui y gardait ses tampons les refait.
  if (cudaError_t e = cudaDeviceSynchronize(); e != cudaSuccess) return failed(e, "cudaDeviceSynchronize");
  const auto in = reinterpret_cast<std::uintptr_t>(scratch_input);
  const auto out = reinterpret_cast<std::uintptr_t>(scratch_output);
  const Tensor input = fused_tensor(in, 0, kernels::kFusedInputPlanes, h, w);
  std::vector<Tensor> outputs;
  if (key.motion) {
    outputs = {
        fused_tensor(out, 0, 4, h, w),
        fused_tensor(out + 4U * static_cast<std::size_t>(h) * static_cast<std::size_t>(w) * 2U, 0, 1, h, w)};
  } else {
    outputs = {fused_tensor(out, 0, 3, h, w)};
  }
  for (int pass = 0; pass < 3; ++pass) {
    const Tensor inputs[]{input};
    if (auto ok = engine->enqueue(scratch.stream, shared, inputs, outputs); !ok)
      return std::unexpected(std::move(ok.error()));
    if (cudaError_t e = cudaStreamSynchronize(scratch.stream); e != cudaSuccess)
      return failed(e, "exécution");
  }
  // Une capture refusée pendant l'essai laisse la chaîne tourner sans graphe : ce n'est pas un refus. Les
  // graphes de l'essai, sur des adresses qui disparaissent, ne servent plus ; les séries d'échecs repartent
  // de zéro, et sa notice part avec eux. L'encodage que l'essai a gardé venait de ses tampons, libérés au
  // retour : une entrée de la chaîne allouée à la même adresse ne doit pas le reprendre.
  engine->graphs_.clear();
  static_cast<void>(engine->graphs_.take_notice());
  engine->kept_.reset();
  engine->hint_ = Hint::None;
  return engine;
}

std::expected<void, Refusal> RifeFusedEngine::use_zone(const std::shared_ptr<SharedActivations>& zone) {
  const auto needed = static_cast<std::int64_t>(scratch_offset_ + layout_.bytes);
  if (auto ok = zone->reserve(needed); !ok) return ok;
  zone_ = zone;
  zone_generation_ = zone->generation();
  graphs_.clear();  // les graphes capturés l'étaient sur une autre adresse
  return {};
}

std::expected<void, Refusal> RifeFusedEngine::run(cudaStream_t stream,
                                                  const std::shared_ptr<SharedActivations>& arena,
                                                  const Tensor& input, std::span<const Tensor> outputs,
                                                  Encode encode) {
  const int h = height_;
  const int w = width_;
  auto* base = static_cast<std::byte*>(arena->data()) + scratch_offset_;
  const auto at = [base](std::size_t offset) { return reinterpret_cast<std::uintptr_t>(base + offset); };
  const auto at_ptr = [base](std::size_t offset) { return static_cast<void*>(base + offset); };
  const void* const in = fused_address(input);

  // L'encodage de A (4 plans) puis celui de B, une image par passe de l'encodeur ; l'image B est aux plans
  // 3 à 5 de l'entrée. Second : l'encodage de B de l'appel précédent, intact, devient celui de A.
  const std::size_t image_bytes =
      std::size_t{3} * static_cast<std::size_t>(h) * static_cast<std::size_t>(w) * 2U;
  const std::size_t encode_bytes =
      std::size_t{4} * static_cast<std::size_t>(h) * static_cast<std::size_t>(w) * 2U;
  const auto encode_image = [&](std::uint64_t image_offset,
                                std::size_t encode_offset) -> std::expected<void, Refusal> {
    const Tensor image = fused_tensor(input.handle, input.offset + image_offset, 3, h, w);
    const Tensor encoded = fused_tensor(at(layout_.encode + encode_offset), 0, 4, h, w);
    return encode_->enqueue_plain(stream, arena, std::span(&image, 1), std::span(&encoded, 1));
  };
  if (encode == Encode::Both) {
    if (auto ok = encode_image(0, 0); !ok) return ok;
  } else if (encode == Encode::Second) {
    if (auto ok = cuda_check(cudaMemcpyAsync(at_ptr(layout_.encode), at_ptr(layout_.encode + encode_bytes),
                                             encode_bytes, cudaMemcpyDeviceToDevice, stream),
                             "cudaMemcpyAsync (encodage de B vers A)");
        !ok)
      return ok;
  }
  if (encode != Encode::None) {
    if (auto ok = encode_image(image_bytes, encode_bytes); !ok) return ok;
  }

  const kernels::FusedDims x0 = kernels::rife_fused_x(plan_, 0, h, w);
  if (auto ok = cuda_check(launch_fused_entry({.height = h,
                                               .width = w,
                                               .reduce = plan_.reduce[0],
                                               .out_height = x0.height,
                                               .out_width = x0.width,
                                               .input = in,
                                               .encode = at_ptr(layout_.encode),
                                               .x = at_ptr(layout_.x[0])},
                                              stream),
                           "fused_entry<<<>>>");
      !ok)
    return ok;
  for (int b = 0; b < plan_.blocks; ++b) {
    const auto i = static_cast<std::size_t>(b);
    const kernels::FusedDims x = kernels::rife_fused_x(plan_, b, h, w);
    const kernels::FusedDims y = kernels::rife_fused_y(plan_, b, h, w);
    const Tensor block_in = fused_tensor(at(layout_.x[i]), 0, x.channels, x.height, x.width);
    const Tensor block_out = fused_tensor(at(layout_.y[i]), 0, y.channels, y.height, y.width);
    if (auto ok = blocks_[i]->enqueue_plain(stream, arena, std::span(&block_in, 1), std::span(&block_out, 1));
        !ok)
      return ok;
    if (b + 1 < plan_.blocks) {
      const kernels::FusedDims next = kernels::rife_fused_x(plan_, b + 1, h, w);
      FusedGlueParams glue{.height = h,
                           .width = w,
                           .reduce = plan_.reduce[i + 1],
                           .out_height = next.height,
                           .out_width = next.width,
                           .flows = b + 1,
                           .input = in,
                           .encode = at_ptr(layout_.encode),
                           .x = at_ptr(layout_.x[i + 1])};
      for (int k = 0; k <= b; ++k) {
        const auto j = static_cast<std::size_t>(k);
        const kernels::FusedDims yk = kernels::rife_fused_y(plan_, k, h, w);
        glue.source[j] = {
            .y = at_ptr(layout_.y[j]), .height = yk.height, .width = yk.width, .scale = plan_.reduce[j]};
      }
      if (auto ok = cuda_check(launch_fused_glue(glue, stream), "fused_glue<<<>>>"); !ok) return ok;
    }
  }
  const auto last = static_cast<std::size_t>(plan_.blocks - 1);
  const kernels::FusedDims y_last = kernels::rife_fused_y(plan_, plan_.blocks - 1, h, w);
  FusedFinishParams finish{.height = h,
                           .width = w,
                           .input = in,
                           .x_last = at_ptr(layout_.x[last]),
                           .y_last = at_ptr(layout_.y[last]),
                           .y_height = y_last.height,
                           .y_width = y_last.width};
  if (motion_) {
    finish.flow = fused_address(outputs[0]);
    finish.mask = fused_address(outputs[1]);
  } else {
    finish.image = fused_address(outputs[0]);
  }
  return cuda_check(launch_fused_finish(finish, stream), "fused_finish<<<>>>");
}

std::expected<void, Refusal> RifeFusedEngine::enqueue(cudaStream_t stream,
                                                      const std::shared_ptr<SharedActivations>& arena,
                                                      std::span<const Tensor> inputs,
                                                      std::span<const Tensor> outputs) {
  const Hint hint = std::exchange(hint_, Hint::None);
  if (!arena) return refuse(RefusalCode::GraphInvalid, "RIFE fusionné : zone d'activation absente");
  if (inputs.size() != 1 || outputs.size() != outputs_.size())
    return refuse(RefusalCode::ShapeInvalid,
                  "RIFE fusionné : nombre d'entrées ou de sorties différent du moteur");
  const int h = height_;
  const int w = width_;
  const auto matches = [](const Tensor& t, const Shape& shape) {
    return t.type == TensorType::F16 && t.shape == shape && t.handle != 0;
  };
  if (!matches(inputs[0], Shape{.dims = {1, kernels::kFusedInputPlanes, h, w}}))
    return refuse(RefusalCode::ShapeInvalid,
                  "RIFE fusionné : entrée de forme ou de type différents du moteur");
  if (motion_ ? !(matches(outputs[0], Shape{.dims = {1, 4, h, w}}) &&
                  matches(outputs[1], Shape{.dims = {1, 1, h, w}}))
              : !matches(outputs[0], Shape{.dims = {1, 3, h, w}}))
    return refuse(RefusalCode::ShapeInvalid,
                  "RIFE fusionné : sortie de forme ou de type différents du moteur");
  // Lancé sur une autre voie que celle du chargement, ou la zone a grandi : tampons repris à leur place,
  // graphes jetés.
  const auto needed = static_cast<std::int64_t>(scratch_offset_ + layout_.bytes);
  if (arena != zone_ || arena->bytes() < needed || zone_generation_ != arena->generation()) {
    if (auto ok = use_zone(arena); !ok) {
      kept_.reset();
      return ok;
    }
  }
  // L'encodage gardé sert si la zone, une fois prise, est celle du dernier appel mené au bout et que personne
  // n'y a écrit depuis ; tout le reste (autre voie, zone agrandie, autre moteur, appel en échec) encode les
  // deux images.
  const std::uintptr_t input = inputs[0].handle + inputs[0].offset;
  const bool intact = kept_ && arena->last_user() == this && kept_->zone == arena->data() &&
                      kept_->generation == arena->generation();
  Encode encode = Encode::Both;
  if (intact && hint == Hint::TimesOnly && kept_->input == input) encode = Encode::None;
  if (intact && hint == Hint::Follows) encode = Encode::Second;
  kept_.reset();
  auto queued = enqueue_chain(stream, arena, inputs, outputs, encode);
  arena->set_last_user(this);  // après les morceaux, qui se sont nommés
  if (queued) kept_ = Kept{.input = input, .zone = arena->data(), .generation = arena->generation()};
  return queued;
}

std::expected<void, Refusal> RifeFusedEngine::enqueue_chain(cudaStream_t stream,
                                                            const std::shared_ptr<SharedActivations>& arena,
                                                            std::span<const Tensor> inputs,
                                                            std::span<const Tensor> outputs, Encode encode) {
  const FusedGraphs::Key key{inputs[0].handle + inputs[0].offset,
                             outputs[0].handle + outputs[0].offset,
                             outputs.size() > 1 ? outputs[1].handle + outputs[1].offset : 0,
                             0,
                             reinterpret_cast<std::uintptr_t>(arena->data()),
                             static_cast<std::uintptr_t>(arena->generation()),
                             static_cast<std::uintptr_t>(encode)};
  return graphs_.launch(stream, key, [&] { return run(stream, arena, inputs[0], outputs, encode); });
}

}  // namespace ajp::gpu::cuda
