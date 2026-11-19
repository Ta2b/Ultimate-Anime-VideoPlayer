#include "gpu/cuda/backend/tfix_fused_engine.h"

#include <algorithm>
#include <format>
#include <utility>

#include "core/utf8_path.h"
#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/tfix_fused_kernels.h"
#include "prepare/onnx_cache.h"
#include "prepare/onnx_frames.h"

namespace kirari::gpu::cuda {
namespace {

// Les noms des entrées de la fenêtre, ceux du graphe du cache de l'encodeur (app/play/temporal_fix_model.h).
constexpr std::string_view kInput = "input";
constexpr std::string_view kFeatures = "features";

int work(int net, bool half) {
  return half ? 2 * net : net;
}

bool matches(const Tensor& t, const Shape& shape) {
  return t.type == TensorType::F16 && t.shape == shape && t.handle != 0;
}

std::expected<std::unique_ptr<TrtEngine>, Refusal> load_piece(
    nvinfer1::IRuntime& runtime, const std::filesystem::path& cache_dir, const std::string& file,
    const Shape& in, const Shape& out, const std::shared_ptr<SharedActivations>& shared) {
  auto engine = TrtEngine::load(runtime, cache_dir / core::path_from_utf8(file), shared);
  if (!engine) return std::unexpected(std::move(engine.error()));
  if (!((*engine)->input_shape(0) == in) || !((*engine)->output_shape(0) == out) ||
      (*engine)->input_pitch(0) != 0 || (*engine)->output_pitch(0) != 0)
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("TemporalFix fusionné : moteur {} de forme inattendue", file));
  return engine;
}

Shape dims_shape(std::int64_t n, const kernels::TfixDims& d) {
  return {.dims = {n, d.channels, d.height, d.width}};
}

}  // namespace

// --- L'encodeur ---

std::expected<std::unique_ptr<TfixEncoderEngine>, Refusal> TfixEncoderEngine::load(
    nvinfer1::IRuntime& runtime, const std::filesystem::path& cache_dir, const TfixFusedKey& key,
    const std::shared_ptr<SharedActivations>& shared) {
  if (key.window || key.engines.size() != 1 || !kernels::tfix_size_valid(key.height, key.width))
    return refuse(RefusalCode::GraphInvalid, "TemporalFix fusionné : clé d'encodeur invalide");
  std::unique_ptr<TfixEncoderEngine> engine(new TfixEncoderEngine());
  engine->height_ = key.height;
  engine->width_ = key.width;
  engine->half_ = key.half;
  auto encode = load_piece(runtime, cache_dir, key.engines[0],
                           {.dims = {1, kernels::kTfixImage, key.height, key.width}},
                           {.dims = {1, kernels::kTfixTraits, key.height, key.width}}, shared);
  if (!encode) return std::unexpected(std::move(encode.error()));
  engine->encode_ = std::move(*encode);
  return engine;
}

Shape TfixEncoderEngine::input_shape() const {
  return {.dims = {1, kernels::kTfixImage, work(height_, half_), work(width_, half_)}};
}

Shape TfixEncoderEngine::output_shape() const {
  return {.dims = {1, kernels::tfix_packed_planes(height_, width_, half_), work(height_, half_),
                   work(width_, half_)}};
}

std::expected<void, Refusal> TfixEncoderEngine::enqueue(cudaStream_t stream,
                                                        const std::shared_ptr<SharedActivations>& arena,
                                                        std::span<const Tensor> inputs,
                                                        std::span<const Tensor> outputs) {
  if (inputs.size() != 1 || outputs.size() != 1 || !matches(inputs[0], input_shape()) ||
      !matches(outputs[0], output_shape()))
    return refuse(RefusalCode::ShapeInvalid,
                  "TemporalFix fusionné : encodeur, entrée ou sortie différente du moteur");
  const kernels::TfixPacked packed = kernels::tfix_packed(height_, width_, half_);
  const Tensor& rgb = inputs[0];
  const Tensor& out = outputs[0];
  void* const packed_ptr = fused_address(out);
  Tensor image = rgb;
  if (half_) {
    if (auto ok = cuda_check(launch_tfix_pool(fused_address(rgb), packed_ptr, height_, width_, stream),
                             "tfix_pool<<<>>>");
        !ok)
      return ok;
    image = fused_tensor(out.handle, out.offset + packed.image * 2U, kernels::kTfixImage, height_, width_);
  }
  const Tensor traits =
      fused_tensor(out.handle, out.offset + packed.traits * 2U, kernels::kTfixTraits, height_, width_);
  if (auto ok = encode_->enqueue_plain(stream, arena, std::span(&image, 1), std::span(&traits, 1)); !ok)
    return ok;
  return cuda_check(launch_tfix_pyramid(fused_address(rgb), packed_ptr, height_, width_, half_, stream),
                    "tfix_pyramid<<<>>>");
}

// --- La fenêtre ---

TfixWindowEngine::~TfixWindowEngine() {
  if (zone_) zone_->drop_region(this);
}

Shape TfixWindowEngine::image_shape() const {
  return {.dims = {1, kernels::kTfixImage, work(height_, half_), work(width_, half_)}};
}

Shape TfixWindowEngine::packed_shape() const {
  return {.dims = {1, kernels::tfix_packed_planes(height_, width_, half_), work(height_, half_),
                   work(width_, half_)}};
}

std::expected<std::unique_ptr<TfixWindowEngine>, Refusal> TfixWindowEngine::load(
    nvinfer1::IRuntime& runtime, const std::filesystem::path& cache_dir, const TfixFusedKey& key,
    const std::shared_ptr<SharedActivations>& shared) {
  if (!shared) return refuse(RefusalCode::GraphInvalid, "TemporalFix fusionné : zone d'activation absente");
  if (!key.window || key.engines.size() != 4 || !kernels::tfix_size_valid(key.height, key.width))
    return refuse(RefusalCode::GraphInvalid, "TemporalFix fusionné : clé de fenêtre invalide");
  std::unique_ptr<TfixWindowEngine> engine(new TfixWindowEngine());
  const int h = key.height;
  const int w = key.width;
  engine->height_ = h;
  engine->width_ = w;
  engine->half_ = key.half;
  for (int r = 0; r < kernels::kTfixFrames; ++r)
    engine->inputs_.push_back(prepare::frame_input_name(kInput, r));
  for (int r = 0; r < kernels::kTfixFrames; ++r)
    engine->inputs_.push_back(prepare::feature_input_name(kFeatures, r));
  std::int64_t activation = 0;
  for (int b = 0; b < kernels::kTfixLevels; ++b) {
    const auto i = static_cast<std::size_t>(b);
    auto block = load_piece(runtime, cache_dir, key.engines[i],
                            dims_shape(kernels::kTfixSupports, kernels::tfix_block_x(b, h, w)),
                            dims_shape(kernels::kTfixSupports, kernels::tfix_block_y(b, h, w)), shared);
    if (!block) return std::unexpected(std::move(block.error()));
    activation = std::max(activation, (*block)->activation_bytes());
    engine->blocks_[i] = std::move(*block);
  }
  const kernels::TfixDims half_level = kernels::tfix_level(2, h, w, kernels::kTfixConf);
  auto conf = load_piece(runtime, cache_dir, key.engines[3], dims_shape(kernels::kTfixSupports, half_level),
                         {.dims = {kernels::kTfixSupports, 1, half_level.height, half_level.width}}, shared);
  if (!conf) return std::unexpected(std::move(conf.error()));
  activation = std::max(activation, (*conf)->activation_bytes());
  engine->conf_ = std::move(*conf);
  engine->layout_ = kernels::tfix_layout(h, w, key.half);
  engine->own_activation_ = activation;
  engine->scratch_offset_ = engine->scratch_offset_for(*shared);
  if (auto ok = engine->use_zone(shared); !ok) return std::unexpected(std::move(ok.error()));

  // Une fenêtre de zéros trois fois par toute la chaîne, sur un flux à part : les quatorze entrées lisent les
  // deux mêmes tampons.
  const auto plane =
      static_cast<std::size_t>(work(h, key.half)) * static_cast<std::size_t>(work(w, key.half)) * 2U;
  const std::size_t image_bytes = kernels::kTfixImage * plane;
  const std::size_t packed_bytes =
      static_cast<std::size_t>(kernels::tfix_packed_planes(h, w, key.half)) * plane;
  FusedScratch scratch;
  const auto failed = [](cudaError_t error, const char* step) {
    static_cast<void>(cudaGetLastError());
    return refuse(
        RefusalCode::VendorCallFailed,
        std::format("TemporalFix fusionné : essai au chargement ({}) : {}", step, cudaGetErrorString(error)));
  };
  const std::array<std::size_t, 3> sizes{image_bytes, packed_bytes, image_bytes};
  for (std::size_t i = 0; i < sizes.size(); ++i) {
    if (cudaError_t e = cudaMalloc(&scratch.buffers[i], sizes[i]); e != cudaSuccess)
      return failed(e, "cudaMalloc");
    if (cudaError_t e = cudaMemset(scratch.buffers[i], 0, sizes[i]); e != cudaSuccess)
      return failed(e, "cudaMemset");
  }
  if (cudaError_t e = cudaStreamCreateWithFlags(&scratch.stream, cudaStreamNonBlocking); e != cudaSuccess)
    return failed(e, "cudaStreamCreate");
  if (cudaError_t e = cudaDeviceSynchronize(); e != cudaSuccess) return failed(e, "cudaDeviceSynchronize");
  const Shape image = engine->image_shape();
  const Shape packed = engine->packed_shape();
  std::vector<Tensor> inputs;
  for (int r = 0; r < kernels::kTfixFrames; ++r)
    inputs.push_back({.type = TensorType::F16,
                      .shape = image,
                      .handle = reinterpret_cast<std::uintptr_t>(scratch.buffers[0]),
                      .offset = 0});
  for (int r = 0; r < kernels::kTfixFrames; ++r)
    inputs.push_back({.type = TensorType::F16,
                      .shape = packed,
                      .handle = reinterpret_cast<std::uintptr_t>(scratch.buffers[1]),
                      .offset = 0});
  const Tensor out{.type = TensorType::F16,
                   .shape = image,
                   .handle = reinterpret_cast<std::uintptr_t>(scratch.buffers[2]),
                   .offset = 0};
  for (int pass = 0; pass < 3; ++pass) {
    if (auto ok = engine->enqueue(scratch.stream, shared, inputs, std::span(&out, 1)); !ok)
      return std::unexpected(std::move(ok.error()));
    if (cudaError_t e = cudaStreamSynchronize(scratch.stream); e != cudaSuccess)
      return failed(e, "exécution");
  }
  engine->graphs_.clear();
  static_cast<void>(engine->graphs_.take_notice());
  return engine;
}

std::size_t TfixWindowEngine::scratch_offset_for(SharedActivations& zone) {
  return static_cast<std::size_t>(
      zone.region_offset(this, own_activation_, static_cast<std::int64_t>(layout_.bytes())));
}

std::expected<void, Refusal> TfixWindowEngine::follow_zone(const std::shared_ptr<SharedActivations>& arena) {
  const std::size_t offset = scratch_offset_for(*arena);
  const auto needed = static_cast<std::int64_t>(offset + layout_.bytes());
  if (arena == zone_ && offset == scratch_offset_ && arena->bytes() >= needed &&
      zone_generation_ == arena->generation())
    return {};
  scratch_offset_ = offset;
  return use_zone(arena);
}

std::expected<void, Refusal> TfixWindowEngine::use_zone(const std::shared_ptr<SharedActivations>& zone) {
  const auto needed = static_cast<std::int64_t>(scratch_offset_ + layout_.bytes());
  if (auto ok = zone->reserve(needed); !ok) return ok;
  if (zone_ && zone_ != zone) zone_->drop_region(this);
  zone_ = zone;
  zone_generation_ = zone->generation();
  graphs_.clear();  // capturés sur une autre adresse
  return {};
}

std::expected<void, Refusal> TfixWindowEngine::run(cudaStream_t stream,
                                                   const std::shared_ptr<SharedActivations>& arena) {
  const int h = height_;
  const int w = width_;
  void* const base = static_cast<std::byte*>(arena->data()) + scratch_offset_;
  const auto at = [base](std::size_t elements) {
    return reinterpret_cast<std::uintptr_t>(base) + elements * 2U;
  };
  const TfixShape shape = tfix_shape(h, w, half_, base);
  const auto block = [&](int b) -> std::expected<void, Refusal> {
    const auto i = static_cast<std::size_t>(b);
    const kernels::TfixDims x = kernels::tfix_block_x(b, h, w);
    const kernels::TfixDims y = kernels::tfix_block_y(b, h, w);
    const Tensor in =
        fused_tensor(at(layout_.x[i]), 0, x.channels, x.height, x.width, kernels::kTfixSupports);
    const Tensor out =
        fused_tensor(at(layout_.y[i]), 0, y.channels, y.height, y.width, kernels::kTfixSupports);
    return blocks_[i]->enqueue_plain(stream, arena, std::span(&in, 1), std::span(&out, 1));
  };
  if (auto ok = cuda_check(launch_tfix_first(shape, stream), "tfix_first<<<>>>"); !ok) return ok;
  if (auto ok = block(0); !ok) return ok;
  for (int level = 1; level < kernels::kTfixLevels; ++level) {
    if (auto ok = cuda_check(launch_tfix_glue(shape, level, stream), "tfix_glue<<<>>>"); !ok) return ok;
    if (auto ok = block(level); !ok) return ok;
  }
  if (auto ok = cuda_check(launch_tfix_conf(shape, stream), "tfix_conf<<<>>>"); !ok) return ok;
  const kernels::TfixDims half_level = kernels::tfix_level(2, h, w, 1);
  const Tensor conf_in = fused_tensor(at(layout_.conf_in), 0, kernels::kTfixConf, half_level.height,
                                      half_level.width, kernels::kTfixSupports);
  const Tensor conf_out =
      fused_tensor(at(layout_.conf_out), 0, 1, half_level.height, half_level.width, kernels::kTfixSupports);
  if (auto ok = conf_->enqueue_plain(stream, arena, std::span(&conf_in, 1), std::span(&conf_out, 1)); !ok)
    return ok;
  return cuda_check(launch_tfix_blend(shape, stream), "tfix_blend<<<>>>");
}

std::expected<void, Refusal> TfixWindowEngine::enqueue(cudaStream_t stream,
                                                       const std::shared_ptr<SharedActivations>& arena,
                                                       std::span<const Tensor> inputs,
                                                       std::span<const Tensor> outputs) {
  const auto frames = static_cast<std::size_t>(kernels::kTfixFrames);
  if (inputs.size() != 2 * frames || outputs.size() != 1)
    return refuse(RefusalCode::ShapeInvalid,
                  "TemporalFix fusionné : nombre d'entrées ou de sorties différent du moteur");
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    if (!matches(inputs[i], i < frames ? image_shape() : packed_shape()))
      return refuse(
          RefusalCode::ShapeInvalid,
          std::format("TemporalFix fusionné : entrée {} de forme ou de type différents du moteur", i));
  }
  if (!matches(outputs[0], image_shape()))
    return refuse(RefusalCode::ShapeInvalid,
                  "TemporalFix fusionné : sortie de forme ou de type différents du moteur");
  if (auto ok = follow_zone(arena); !ok) return ok;
  // La table des adresses, copiée sur le flux avant la chaîne : hors du graphe, qui la lit.
  TfixTable table;
  for (std::size_t r = 0; r < frames; ++r) {
    table.rgb[r] = fused_address(inputs[r]);
    table.packed[r] = fused_address(inputs[frames + r]);
  }
  table.out = fused_address(outputs[0]);
  auto* device_table = static_cast<std::byte*>(arena->data()) + scratch_offset_ + layout_.table * 2U;
  if (auto ok =
          cuda_check(cudaMemcpyAsync(device_table, &table, sizeof(table), cudaMemcpyHostToDevice, stream),
                     "cudaMemcpyAsync (table TemporalFix)");
      !ok)
    return ok;
  const FusedGraphs::Key key{reinterpret_cast<std::uintptr_t>(arena->data()),
                             static_cast<std::uintptr_t>(arena->generation()),
                             scratch_offset_,
                             0,
                             0,
                             0,
                             0};
  auto queued = graphs_.launch(stream, key, [&] { return run(stream, arena); });
  arena->wrote(this, static_cast<std::int64_t>(scratch_offset_),
               static_cast<std::int64_t>(scratch_offset_ + layout_.bytes()));
  return queued;
}

}  // namespace kirari::gpu::cuda
