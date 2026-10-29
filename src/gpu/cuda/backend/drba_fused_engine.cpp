#include "gpu/cuda/backend/drba_fused_engine.h"

#include <algorithm>
#include <charconv>
#include <format>
#include <utility>

#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/drba_fused_kernels.h"

namespace ajp::gpu::cuda {
namespace {

constexpr std::string_view kKeyPrefix = "drba-fused:";

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

// Libère les tampons d'essai quoi qu'il arrive.
struct Scratch {
  void* input01 = nullptr;
  void* input12 = nullptr;
  void* output = nullptr;
  cudaStream_t stream = nullptr;
  ~Scratch() {
    if (stream != nullptr) cudaStreamDestroy(stream);
    if (input01 != nullptr) cudaFree(input01);
    if (input12 != nullptr) cudaFree(input12);
    if (output != nullptr) cudaFree(output);
  }
};

std::optional<int> parse_int(std::string_view text) {
  int value = 0;
  const auto [stop, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || stop != text.data() + text.size()) return std::nullopt;
  return value;
}

std::vector<std::string_view> split(std::string_view text, char separator) {
  std::vector<std::string_view> parts;
  while (true) {
    const std::size_t at = text.find(separator);
    parts.push_back(text.substr(0, at));
    if (at == std::string_view::npos) return parts;
    text.remove_prefix(at + 1);
  }
}

Tensor tensor_of(std::uintptr_t handle, std::uint64_t offset, std::int64_t c, int h, int w,
                 std::int64_t n = 1) {
  return Tensor{.type = TensorType::F16, .shape = {.dims = {n, c, h, w}}, .handle = handle, .offset = offset};
}

const void* address_of(const Tensor& tensor) {
  return reinterpret_cast<const void*>(tensor.handle + tensor.offset);  // NOLINT(performance-no-int-to-ptr)
}

}  // namespace

std::string drba_fused_artifact_key(const DrbaFusedKey& key) {
  std::string engines;
  for (const std::string& engine : key.engines)
    engines += std::format("{}{}", engines.empty() ? "" : ",", engine);
  return std::format("{}{}x{}:{}:{}", kKeyPrefix, key.height, key.width, key.plan.tail, engines);
}

std::optional<DrbaFusedKey> drba_fused_artifact_key_parse(std::string_view text) {
  if (!text.starts_with(kKeyPrefix)) return std::nullopt;
  text.remove_prefix(kKeyPrefix.size());
  const std::vector<std::string_view> fields = split(text, ':');
  if (fields.size() != 3) return std::nullopt;
  const std::vector<std::string_view> size = split(fields[0], 'x');
  if (size.size() != 2) return std::nullopt;
  const auto height = parse_int(size[0]);
  const auto width = parse_int(size[1]);
  const auto tail = parse_int(fields[1]);
  if (!height || !width || !tail) return std::nullopt;
  DrbaFusedKey key{.height = *height, .width = *width, .plan = {.tail = *tail}};
  for (const std::string_view part : split(fields[2], ',')) {
    // Un nom de fichier seul : la clé ne sort jamais du dossier du cache.
    if (part.empty() || part.find_first_of("/\\") != std::string_view::npos || part == "." || part == "..")
      return std::nullopt;
    key.engines.emplace_back(part);
  }
  if (!kernels::drba_fused_size_valid(key.plan, key.height, key.width) ||
      key.engines.size() != static_cast<std::size_t>(kernels::kDrbaFusedBlocks) + 1)
    return std::nullopt;
  return key;
}

DrbaFusedEngine::~DrbaFusedEngine() {
  drop_graphs();
}

void DrbaFusedEngine::drop_graphs() {
  for (const Graph& graph : graphs_) {
    if (graph.exec != nullptr) cudaGraphExecDestroy(graph.exec);
  }
  graphs_.clear();
}

std::string DrbaFusedEngine::take_notice() {
  return std::exchange(notice_, std::string());
}

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

  auto encode = TrtEngine::load(runtime, cache_dir / key.engines[0], shared);
  if (!encode) return std::unexpected(std::move(encode.error()));
  const bool encode_ok =
      (*encode)->input_shape(0) == Shape{.dims = {3, 3, h / t, w / t}} &&
      (*encode)->output_shape(0) == Shape{.dims = {3, kernels::kDrbaFusedEncodeChannels, h / t, w / t}};
  if (!encode_ok)
    return refuse(RefusalCode::ShapeInvalid, "DistilDRBA fusionné : moteur d'encodage de forme inattendue");
  activation = std::max(activation, (*encode)->activation_bytes());
  engine->encode_ = std::move(*encode);
  for (int b = 0; b < kernels::kDrbaFusedBlocks; ++b) {
    auto block = TrtEngine::load(runtime, cache_dir / key.engines[static_cast<std::size_t>(b) + 1], shared);
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
  Scratch scratch;
  const auto failed = [](cudaError_t error, const char* step) {
    static_cast<void>(cudaGetLastError());
    return refuse(
        RefusalCode::VendorCallFailed,
        std::format("DistilDRBA fusionné : essai au chargement ({}) : {}", step, cudaGetErrorString(error)));
  };
  for (void** buffer : {&scratch.input01, &scratch.input12}) {
    if (cudaError_t e = cudaMalloc(buffer, in_bytes); e != cudaSuccess) return failed(e, "cudaMalloc");
    if (cudaError_t e = cudaMemset(*buffer, 0, in_bytes); e != cudaSuccess) return failed(e, "cudaMemset");
  }
  if (cudaError_t e = cudaMalloc(&scratch.output, out_bytes); e != cudaSuccess)
    return failed(e, "cudaMalloc");
  if (cudaError_t e = cudaMemset(scratch.output, 0, out_bytes); e != cudaSuccess)
    return failed(e, "cudaMemset");
  if (cudaError_t e = cudaStreamCreateWithFlags(&scratch.stream, cudaStreamNonBlocking); e != cudaSuccess)
    return failed(e, "cudaStreamCreate");
  const Tensor inputs[]{
      tensor_of(reinterpret_cast<std::uintptr_t>(scratch.input01), 0, kernels::kDrbaFusedInputPlanes, h, w),
      tensor_of(reinterpret_cast<std::uintptr_t>(scratch.input12), 0, kernels::kDrbaFusedInputPlanes, h, w)};
  const Tensor outputs[]{tensor_of(reinterpret_cast<std::uintptr_t>(scratch.output), 0, 3, h, w)};
  for (int pass = 0; pass < 3; ++pass) {
    if (auto ok = engine->enqueue(scratch.stream, shared, inputs, outputs); !ok)
      return std::unexpected(std::move(ok.error()));
    if (cudaError_t e = cudaStreamSynchronize(scratch.stream); e != cudaSuccess)
      return failed(e, "exécution");
  }
  // Une capture refusée pendant l'essai laisse la chaîne tourner sans graphe : ce n'est pas un refus. Les
  // graphes de l'essai, sur des adresses qui disparaissent, ne servent plus.
  engine->drop_graphs();
  engine->capture_failures_ = 0;
  engine->notice_.clear();
  return engine;
}

std::expected<void, Refusal> DrbaFusedEngine::use_zone(const std::shared_ptr<SharedActivations>& zone) {
  const auto needed = static_cast<std::int64_t>(scratch_offset_ + layout_.bytes);
  if (auto ok = zone->reserve(needed); !ok) return ok;
  zone_ = zone;
  drop_graphs();  // les graphes capturés l'étaient sur une autre adresse
  return {};
}

std::expected<void, Refusal> DrbaFusedEngine::run(cudaStream_t stream,
                                                  const std::shared_ptr<SharedActivations>& arena,
                                                  std::span<const Tensor> inputs, const Tensor& output) {
  const int h = height_;
  const int w = width_;
  const int t = plan_.tail;
  auto* base = static_cast<std::byte*>(arena->data()) + scratch_offset_;
  const auto at = [base](std::size_t offset) { return reinterpret_cast<std::uintptr_t>(base + offset); };
  const auto at_ptr = [base](std::size_t offset) { return static_cast<void*>(base + offset); };
  const void* const pair01 = address_of(inputs[0]);
  const void* const pair12 = address_of(inputs[1]);

  if (auto ok = cuda_check(launch_drba_pre({.height = h,
                                            .width = w,
                                            .reduce0 = kernels::drba_fused_reduce(plan_, 0),
                                            .reduce1 = kernels::drba_fused_reduce(plan_, 1),
                                            .reduce2 = kernels::drba_fused_reduce(plan_, 2),
                                            .pair01 = pair01,
                                            .pair12 = pair12,
                                            .x0 = at_ptr(layout_.x[0]),
                                            .x1 = at_ptr(layout_.x[1]),
                                            .x2 = at_ptr(layout_.x[2])},
                                           stream),
                           "drba_pre<<<>>>");
      !ok)
    return ok;
  // L'encodeur lit les images réduites au début de X_2 et y écrit les encodages juste après.
  const std::size_t enc_plane = static_cast<std::size_t>(h / t) * static_cast<std::size_t>(w / t);
  const Tensor images = tensor_of(at(layout_.x[2]), 0, 3, h / t, w / t, 3);
  const Tensor encoded = tensor_of(at(layout_.x[2]), kernels::kDrbaFusedImagePlanes * enc_plane * 2U,
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
  for (int b = 0; b < kernels::kDrbaFusedBlocks; ++b) {
    const auto i = static_cast<std::size_t>(b);
    const kernels::DrbaFusedDims x = kernels::drba_fused_x(plan_, b, h, w);
    const kernels::DrbaFusedDims y = kernels::drba_fused_y(plan_, b, h, w);
    const Tensor block_in = tensor_of(at(layout_.x[i]), 0, x.channels, x.height, x.width);
    const Tensor block_out = tensor_of(at(layout_.y[i]), 0, y.channels, y.height, y.width);
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
                                .x = at_ptr(layout_.x[i + 1]),
                                .grid_x = drba_grid_step(next.width),
                                .grid_y = drba_grid_step(next.height)};
      if (auto ok = cuda_check(launch_drba_glue(glue, stream), "drba_glue<<<>>>"); !ok) return ok;
    }
  }
  const auto last = static_cast<std::size_t>(kernels::kDrbaFusedBlocks - 1);
  return cuda_check(
      launch_drba_finish(
          {.height = h,
           .width = w,
           .upscale = t,
           .y = at_ptr(layout_.y[last]),
           .pair01 = pair01,
           .pair12 = pair12,
           .out = const_cast<void*>(address_of(output)),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
           .grid_x = drba_grid_step(w),
           .grid_y = drba_grid_step(h)},
          stream),
      "drba_finish<<<>>>");
}

std::expected<void, Refusal> DrbaFusedEngine::enqueue(cudaStream_t stream,
                                                      const std::shared_ptr<SharedActivations>& arena,
                                                      std::span<const Tensor> inputs,
                                                      std::span<const Tensor> outputs) {
  if (inputs.size() != 2 || outputs.size() != 1)
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
  if (!matches(outputs[0], Shape{.dims = {1, 3, h, w}}))
    return refuse(RefusalCode::ShapeInvalid,
                  "DistilDRBA fusionné : sortie de forme ou de type différents du moteur");
  // Lancé sur une autre voie que celle du chargement, ou la zone a grandi : tampons repris à leur place,
  // graphes jetés.
  const auto needed = static_cast<std::int64_t>(scratch_offset_ + layout_.bytes);
  if (arena != zone_ || arena->bytes() < needed ||
      (!graphs_.empty() && graphs_.front().generation != arena->generation())) {
    if (auto ok = use_zone(arena); !ok) return ok;
  }
  if (graphs_off_) return run(stream, arena, inputs, outputs[0]);

  const std::array<std::uintptr_t, 3> addresses{inputs[0].handle + inputs[0].offset,
                                                inputs[1].handle + inputs[1].offset,
                                                outputs[0].handle + outputs[0].offset};
  auto found = std::ranges::find_if(graphs_, [&](const Graph& g) {
    return g.addresses == addresses && g.zone == reinterpret_cast<std::uintptr_t>(arena->data()) &&
           g.generation == arena->generation();
  });
  if (found == graphs_.end()) {
    if (graphs_.size() >= kMaxGraphs) return run(stream, arena, inputs, outputs[0]);
    graphs_.push_back(Graph{.addresses = addresses,
                            .zone = reinterpret_cast<std::uintptr_t>(arena->data()),
                            .generation = arena->generation()});
    found = graphs_.end() - 1;
  }
  Graph& graph = *found;
  if (graph.exec != nullptr) {
    if (cudaGraphLaunch(graph.exec, stream) != cudaSuccess)
      return refuse(RefusalCode::VendorCallFailed, "DistilDRBA fusionné : cudaGraphLaunch a échoué");
    return {};
  }
  if (!graph.warmed) {  // le premier lancement fait ses allocations hors capture
    graph.warmed = true;
    return run(stream, arena, inputs, outputs[0]);
  }
  cudaGraph_t captured = nullptr;
  std::expected<void, Refusal> queued;
  cudaError_t error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
  if (error == cudaSuccess) {
    queued = run(stream, arena, inputs, outputs[0]);
    error = cudaStreamEndCapture(stream, &captured);  // termine aussi une capture invalidée
  }
  if (error == cudaSuccess && queued && captured != nullptr)
    error = cudaGraphInstantiate(&graph.exec, captured, 0);
  if (captured != nullptr) cudaGraphDestroy(captured);
  if (error != cudaSuccess || !queued || captured == nullptr) {
    // Capture refusée : rien n'a tourné (la capture enregistre sans exécuter), le lancement direct fait le
    // travail.
    graph.exec = nullptr;
    static_cast<void>(cudaGetLastError());
    ++capture_failures_;
    if (capture_failures_ >= kCaptureAttempts) {
      graphs_off_ = true;
      notice_ = std::format("graphes CUDA coupés : capture refusée {} fois, dernière erreur : {}",
                            capture_failures_,
                            !queued                ? queued.error().detail
                            : error != cudaSuccess ? cudaGetErrorString(error)
                                                   : "graphe vide");
    }
    return run(stream, arena, inputs, outputs[0]);
  }
  if (cudaGraphLaunch(graph.exec, stream) != cudaSuccess)
    return refuse(RefusalCode::VendorCallFailed, "DistilDRBA fusionné : cudaGraphLaunch a échoué");
  return {};
}

}  // namespace ajp::gpu::cuda
