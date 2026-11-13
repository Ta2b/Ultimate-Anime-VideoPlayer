#include "gpu/cuda/backend/rife_fused_engine.h"

#include <algorithm>
#include <format>
#include <utility>

#include "core/utf8_path.h"
#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/rife_fused_kernels.h"

namespace ajp::gpu::cuda {
namespace {

// Pose la fenêtre d'accès persistante du flux (bytes 0 : la retire) ; les noyaux enfilés ou capturés ensuite
// la portent. hitRatio : la part de la fenêtre qui tient dans la réserve persistante.
cudaError_t set_l2_window(cudaStream_t stream, void* base, std::size_t bytes, std::size_t persisting) {
  cudaStreamAttrValue value{};
  value.accessPolicyWindow.base_ptr = base;
  value.accessPolicyWindow.num_bytes = bytes;
  value.accessPolicyWindow.hitRatio =
      bytes == 0 ? 0.0F : std::min(1.0F, static_cast<float>(persisting) / static_cast<float>(bytes));
  value.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
  value.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
  return cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &value);
}

}  // namespace

RifeFusedEngine::~RifeFusedEngine() {
  if (zone_) zone_->drop_region(this);  // sa région ne repousse plus celles des autres moteurs fusionnés
}

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
  engine->warp_ = key.warp;
  engine->outputs_ =
      key.motion ? std::vector<std::string>{"flow", "mask"} : std::vector<std::string>{"output"};
  engine->window_ = kernels::Window{.x = 0, .y = 0, .width = key.width, .height = key.height};
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
  // Les blocs grossiers, repris à chaud (next_warm) : ceux à la réduction du premier, sauf le dernier.
  while (engine->warm_blocks_ + 1 < key.plan.blocks &&
         key.plan.reduce[static_cast<std::size_t>(engine->warm_blocks_)] == key.plan.reduce[0]) {
    ++engine->warm_blocks_;
  }
  engine->layout_ = kernels::rife_fused_layout(key.plan, h, w);
  engine->own_activation_ = activation;
  engine->scratch_offset_ = engine->scratch_offset_for(*shared);
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
  // en vol sur les voies (SPAN, un autre moteur) finit avant (A03-05). enqueue déclare ce qu'il y écrit :
  // un autre moteur qui y gardait ses tampons les refait.
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
  engine->warm_ready_ = false;
  return engine;
}

void RifeFusedEngine::set_window(const kernels::Window& window) {
  const bool inside = window.x >= 0 && window.y >= 0 && window.width >= 1 && window.height >= 1 &&
                      window.x + window.width <= width_ && window.y + window.height <= height_;
  window_ = inside ? window : kernels::Window{.x = 0, .y = 0, .width = width_, .height = height_};
}

std::size_t RifeFusedEngine::scratch_offset_for(SharedActivations& zone) {
  // Sa propre région, au-dessus des activations et des régions des autres moteurs fusionnés de la zone :
  // RIFE plein et RIFE demi-taille (échelle par paire) ne s'effacent plus l'un l'autre leur encodage gardé.
  return static_cast<std::size_t>(
      zone.region_offset(this, own_activation_, static_cast<std::int64_t>(layout_.bytes)));
}

std::expected<void, Refusal> RifeFusedEngine::use_zone(const std::shared_ptr<SharedActivations>& zone) {
  const auto needed = static_cast<std::int64_t>(scratch_offset_ + layout_.bytes);
  if (auto ok = zone->reserve(needed); !ok) return ok;
  if (zone_ && zone_ != zone) zone_->drop_region(this);  // la région suit le moteur dans sa nouvelle zone
  zone_ = zone;
  zone_generation_ = zone->generation();
  graphs_.clear();  // les graphes capturés l'étaient sur une autre adresse
  return {};
}

// L'entrée du bloc block (1 et après) : les flots des blocs d'avant, l'entrée et les deux encodages.
std::expected<void, Refusal> RifeFusedEngine::launch_glue(cudaStream_t stream, const void* in, int block,
                                                          std::byte* base, std::size_t encode_a,
                                                          std::size_t encode_b) {
  const int h = height_;
  const int w = width_;
  const auto at_ptr = [base](std::size_t offset) { return static_cast<void*>(base + offset); };
  const auto i = static_cast<std::size_t>(block);
  const kernels::FusedDims next = kernels::rife_fused_x(plan_, block, h, w);
  FusedGlueParams glue{.height = h,
                       .width = w,
                       .reduce = plan_.reduce[i],
                       .out_height = next.height,
                       .out_width = next.width,
                       .flows = block,
                       .input = in,
                       .encode = at_ptr(encode_a),
                       .encode_b = at_ptr(encode_b),
                       .x = at_ptr(layout_.x[i])};
  for (int k = 0; k < block; ++k) {
    const auto j = static_cast<std::size_t>(k);
    const kernels::FusedDims yk = kernels::rife_fused_y(plan_, k, h, w);
    glue.source[j] = {
        .y = at_ptr(layout_.y[j]), .height = yk.height, .width = yk.width, .scale = plan_.reduce[j]};
  }
  return cuda_check(launch_fused_glue(glue, stream), "fused_glue<<<>>>");
}

std::expected<void, Refusal> RifeFusedEngine::run(cudaStream_t stream,
                                                  const std::shared_ptr<SharedActivations>& arena,
                                                  const Tensor& input, std::span<const Tensor> outputs,
                                                  const Pass& pass) {
  const Encode encode = pass.encode;
  const unsigned a_slot = pass.a_slot;
  const int h = height_;
  const int w = width_;
  auto* base = static_cast<std::byte*>(arena->data()) + scratch_offset_;
  const auto at = [base](std::size_t offset) { return reinterpret_cast<std::uintptr_t>(base + offset); };
  const auto at_ptr = [base](std::size_t offset) { return static_cast<void*>(base + offset); };
  const void* const in = fused_address(input);

  // Deux emplacements d'encodage (4 plans chacun) : A dans a_slot, B dans l'autre, une image par passe de
  // l'encodeur ; l'image B est aux plans 3 à 5 de l'entrée. Second : l'encodage de B de l'appel précédent,
  // intact, devient celui de A sans copie (a_slot a basculé), et B s'encode à la place de l'ancien A.
  const std::size_t image_bytes =
      std::size_t{3} * static_cast<std::size_t>(h) * static_cast<std::size_t>(w) * 2U;
  const std::size_t encode_bytes =
      std::size_t{4} * static_cast<std::size_t>(h) * static_cast<std::size_t>(w) * 2U;
  const auto encode_image = [&](std::uint64_t image_offset,
                                std::size_t encode_offset) -> std::expected<void, Refusal> {
    const Tensor image = fused_tensor(input.handle, input.offset + image_offset, 3, h, w);
    const Tensor encoded = fused_tensor(at(encode_offset), 0, 4, h, w);
    return encode_->enqueue_plain(stream, arena, std::span(&image, 1), std::span(&encoded, 1));
  };
  const std::size_t encode_a = layout_.encode + a_slot * encode_bytes;
  const std::size_t encode_b = layout_.encode + (a_slot ^ 1U) * encode_bytes;
  if (encode == Encode::Both) {
    if (auto ok = encode_image(0, encode_a); !ok) return ok;
  }
  if (encode != Encode::None) {
    if (auto ok = encode_image(image_bytes, encode_b); !ok) return ok;
  }

  // À chaud (P6.15), les Y des blocs grossiers restent ceux de l'appel précédent : la chaîne reprend à
  // l'entrée du premier bloc fin, que la colle bâtit sur eux.
  const int first = pass.first_block;
  if (first > 0) {
    if (auto ok = launch_glue(stream, in, first, base, encode_a, encode_b); !ok) return ok;
  } else {
    const kernels::FusedDims x0 = kernels::rife_fused_x(plan_, 0, h, w);
    if (auto ok = cuda_check(launch_fused_entry({.height = h,
                                                 .width = w,
                                                 .reduce = plan_.reduce[0],
                                                 .out_height = x0.height,
                                                 .out_width = x0.width,
                                                 .input = in,
                                                 .encode = at_ptr(encode_a),
                                                 .encode_b = at_ptr(encode_b),
                                                 .x = at_ptr(layout_.x[0])},
                                                stream),
                             "fused_entry<<<>>>");
        !ok)
      return ok;
  }
  for (int b = first; b < plan_.blocks; ++b) {
    const auto i = static_cast<std::size_t>(b);
    const kernels::FusedDims x = kernels::rife_fused_x(plan_, b, h, w);
    const kernels::FusedDims y = kernels::rife_fused_y(plan_, b, h, w);
    const Tensor block_in = fused_tensor(at(layout_.x[i]), 0, x.channels, x.height, x.width);
    const Tensor block_out = fused_tensor(at(layout_.y[i]), 0, y.channels, y.height, y.width);
    if (auto ok = blocks_[i]->enqueue_plain(stream, arena, std::span(&block_in, 1), std::span(&block_out, 1));
        !ok)
      return ok;
    if (b + 1 < plan_.blocks) {
      if (auto ok = launch_glue(stream, in, b + 1, base, encode_a, encode_b); !ok) return ok;
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
    finish.cubic = warp_ == kernels::RifeWarp::CatmullRom ? 1 : 0;
    finish.window_left = window_.x;
    finish.window_top = window_.y;
    finish.window_width = window_.width;
    finish.window_height = window_.height;
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
  // Lancé sur une autre voie que celle du chargement, la zone a grandi, ou un moteur entier de la zone a des
  // activations plus hautes que les tampons (un SPAN chargé après) : tampons repris au-dessus, graphes jetés.
  const std::size_t offset = scratch_offset_for(*arena);
  const auto needed = static_cast<std::int64_t>(offset + layout_.bytes);
  if (arena != zone_ || offset != scratch_offset_ || arena->bytes() < needed ||
      zone_generation_ != arena->generation()) {
    scratch_offset_ = offset;
    if (auto ok = use_zone(arena); !ok) {
      kept_.reset();
      return ok;
    }
  }
  const std::uintptr_t input = inputs[0].handle + inputs[0].offset;
  Pass pass{.encode = choose_encode(*arena, hint, input)};
  // L'emplacement de A : remis à zéro pour deux images neuves, basculé quand B devient A, gardé sinon.
  pass.a_slot = pass.encode == Encode::Both ? 0U : pass.encode == Encode::Second ? a_slot_ ^ 1U : a_slot_;
  // À chaud : les Y gardés servent aux mêmes conditions que l'encodage gardé.
  pass.first_block =
      std::exchange(warm_next_, false) && warm_ready_ && kept_intact(*arena) ? warm_blocks_ : 0;
  warm_ready_ = false;  // repris si la chaîne est enfilée jusqu'au bout
  kept_.reset();
  ++encodes_[static_cast<std::size_t>(pass.encode)];
  auto queued = enqueue_chain(stream, arena, inputs, outputs, pass);
  // Ses tampons, au-dessus des activations de ses morceaux (qui se sont déclarés) : le gardien d'avant, s'il
  // était un autre moteur fusionné, les refera.
  arena->wrote(this, static_cast<std::int64_t>(scratch_offset_),
               static_cast<std::int64_t>(scratch_offset_ + layout_.bytes));
  if (queued) {
    arena->keep(this, static_cast<std::int64_t>(scratch_offset_),
                static_cast<std::int64_t>(scratch_offset_ + layout_.bytes));
    kept_ = Kept{
        .input = input, .zone = arena->data(), .generation = arena->generation(), .offset = scratch_offset_};
    a_slot_ = pass.a_slot;
    warm_ready_ = true;
  }
  return queued;
}

bool RifeFusedEngine::kept_intact(const SharedActivations& arena) const {
  return kept_ && arena.kept_by(this) && kept_->zone == arena.data() &&
         kept_->generation == arena.generation() && kept_->offset == scratch_offset_;
}

RifeFusedEngine::Encode RifeFusedEngine::choose_encode(const SharedActivations& arena, Hint hint,
                                                       std::uintptr_t input) const {
  // L'encodage gardé sert si la zone, une fois prise, est celle du dernier appel mené au bout, les tampons à
  // la même place, et qu'aucun autre moteur n'a écrit sur eux depuis (C03-01 : un SPAN qui passe entre deux
  // appels n'écrit que sous eux) ; tout le reste (autre voie, zone agrandie, autre moteur fusionné, appel en
  // échec) encode les deux images.
  const bool intact = kept_intact(arena);
  if (intact && kept_ && hint == Hint::TimesOnly && kept_->input == input) return Encode::None;
  if (intact && hint == Hint::Follows) return Encode::Second;
  return Encode::Both;
}

std::expected<void, Refusal> RifeFusedEngine::enqueue_chain(cudaStream_t stream,
                                                            const std::shared_ptr<SharedActivations>& arena,
                                                            std::span<const Tensor> inputs,
                                                            std::span<const Tensor> outputs,
                                                            const Pass& pass) {
  // La fenêtre de l'image, paramètre du warp final : seize bits par champ (tailles bornées par
  // rife_fused_size_valid).
  const auto field = [](int v) { return static_cast<std::uintptr_t>(static_cast<std::uint16_t>(v)); };
  const FusedGraphs::Key key{
      inputs[0].handle + inputs[0].offset,
      outputs[0].handle + outputs[0].offset,
      outputs.size() > 1 ? outputs[1].handle + outputs[1].offset : 0,
      field(window_.x) | field(window_.y) << 16U | field(window_.width) << 32U | field(window_.height) << 48U,
      reinterpret_cast<std::uintptr_t>(arena->data()),
      static_cast<std::uintptr_t>(arena->generation()),
      static_cast<std::uintptr_t>(pass.encode) | static_cast<std::uintptr_t>(pass.a_slot) << 8U |
          static_cast<std::uintptr_t>(pass.first_block) << 16U |
          static_cast<std::uintptr_t>(l2_persisting_ != 0 && l2_max_window_ != 0) << 24U};
  const auto chain = [&] { return run(stream, arena, inputs[0], outputs, pass); };
  if (l2_persisting_ == 0 || l2_max_window_ == 0) return graphs_.launch(stream, key, chain);
  // Les deux emplacements de l'encodage (4 plans FP16 chacun), relus par la colle de chaque bloc.
  auto* encode = static_cast<std::byte*>(arena->data()) + scratch_offset_ + layout_.encode;
  const std::size_t bytes = std::min(
      std::size_t{16} * static_cast<std::size_t>(height_) * static_cast<std::size_t>(width_), l2_max_window_);
  if (auto ok = cuda_check(set_l2_window(stream, encode, bytes, l2_persisting_), "cudaStreamSetAttribute");
      !ok)
    return ok;
  auto launched = graphs_.launch(stream, key, chain);
  // Retirée aussitôt : les autres moteurs de la voie (SPAN, moteurs entiers) n'en héritent pas.
  const cudaError_t reset = set_l2_window(stream, nullptr, 0, 0);
  if (!launched) return launched;
  return cuda_check(reset, "cudaStreamSetAttribute");
}

}  // namespace ajp::gpu::cuda
