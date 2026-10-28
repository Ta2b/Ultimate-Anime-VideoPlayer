#include "gpu/cuda/backend/rife_fused_engine.h"

#include <algorithm>
#include <charconv>
#include <format>
#include <optional>
#include <utility>

#include "gpu/cuda/backend/cuda_status.h"
#include "gpu/cuda/backend/rife_fused_kernels.h"

namespace ajp::gpu::cuda {
namespace {

constexpr std::string_view kKeyPrefix = "rife-fused:";

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

// Libère les tampons d'essai quoi qu'il arrive.
struct Scratch {
  void* input = nullptr;
  void* output = nullptr;
  cudaStream_t stream = nullptr;
  ~Scratch() {
    if (stream != nullptr) cudaStreamDestroy(stream);
    if (input != nullptr) cudaFree(input);
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

void* address_of(const Tensor& tensor) {
  return reinterpret_cast<void*>(tensor.handle + tensor.offset);  // NOLINT(performance-no-int-to-ptr)
}

}  // namespace

std::string rife_fused_artifact_key(const RifeFusedKey& key) {
  std::string reduce;
  for (int b = 0; b < key.plan.blocks; ++b)
    reduce += std::format("{}{}", b == 0 ? "" : ",", key.plan.reduce[static_cast<std::size_t>(b)]);
  std::string engines;
  for (const std::string& engine : key.engines)
    engines += std::format("{}{}", engines.empty() ? "" : ",", engine);
  return std::format("{}{}x{}:{}:{}:{}", kKeyPrefix, key.height, key.width, key.motion ? "motion" : "image",
                     reduce, engines);
}

std::optional<RifeFusedKey> rife_fused_artifact_key_parse(std::string_view text) {
  if (!text.starts_with(kKeyPrefix)) return std::nullopt;
  text.remove_prefix(kKeyPrefix.size());
  const std::vector<std::string_view> fields = split(text, ':');
  if (fields.size() != 4) return std::nullopt;
  const std::vector<std::string_view> size = split(fields[0], 'x');
  if (size.size() != 2) return std::nullopt;
  const auto height = parse_int(size[0]);
  const auto width = parse_int(size[1]);
  if (!height || !width || (fields[1] != "image" && fields[1] != "motion")) return std::nullopt;
  RifeFusedKey key{.height = *height, .width = *width, .motion = fields[1] == "motion"};
  for (const std::string_view part : split(fields[2], ',')) {
    const auto d = parse_int(part);
    if (!d || key.plan.blocks >= kernels::kFusedMaxBlocks) return std::nullopt;
    key.plan.reduce[static_cast<std::size_t>(key.plan.blocks++)] = *d;
  }
  for (const std::string_view part : split(fields[3], ',')) {
    // Un nom de fichier seul : la clé ne sort jamais du dossier du cache.
    if (part.empty() || part.find_first_of("/\\") != std::string_view::npos || part == "." || part == "..")
      return std::nullopt;
    key.engines.emplace_back(part);
  }
  if (!kernels::rife_fused_size_valid(key.plan, key.height, key.width) ||
      key.engines.size() != static_cast<std::size_t>(key.plan.blocks) + 1)
    return std::nullopt;
  return key;
}

RifeFusedEngine::~RifeFusedEngine() {
  drop_graphs();
}

void RifeFusedEngine::drop_graphs() {
  for (const Graph& graph : graphs_) {
    if (graph.exec != nullptr) cudaGraphExecDestroy(graph.exec);
  }
  graphs_.clear();
}

std::string RifeFusedEngine::take_notice() {
  return std::exchange(notice_, std::string());
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
  engine->outputs_ =
      key.motion ? std::vector<std::string>{"flow", "mask"} : std::vector<std::string>{"output"};
  const int h = key.height;
  const int w = key.width;
  std::int64_t activation = 0;

  auto encode = TrtEngine::load(runtime, cache_dir / key.engines[0], shared);
  if (!encode) return std::unexpected(std::move(encode.error()));
  const bool encode_ok = (*encode)->input_shape(0) == Shape{.dims = {2, 3, h, w}} &&
                         (*encode)->output_shape(0) == Shape{.dims = {2, 4, h, w}};
  if (!encode_ok)
    return refuse(RefusalCode::ShapeInvalid, "RIFE fusionné : moteur d'encodage de forme inattendue");
  activation = std::max(activation, (*encode)->activation_bytes());
  engine->encode_ = std::move(*encode);
  for (int b = 0; b < key.plan.blocks; ++b) {
    auto block = TrtEngine::load(runtime, cache_dir / key.engines[static_cast<std::size_t>(b) + 1], shared);
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
  Scratch scratch;
  const auto failed = [](cudaError_t error, const char* step) {
    static_cast<void>(cudaGetLastError());
    return refuse(RefusalCode::VendorCallFailed, std::format("RIFE fusionné : essai au chargement ({}) : {}",
                                                             step, cudaGetErrorString(error)));
  };
  if (cudaError_t e = cudaMalloc(&scratch.input, in_bytes); e != cudaSuccess) return failed(e, "cudaMalloc");
  if (cudaError_t e = cudaMalloc(&scratch.output, out_bytes); e != cudaSuccess)
    return failed(e, "cudaMalloc");
  if (cudaError_t e = cudaMemset(scratch.input, 0, in_bytes); e != cudaSuccess)
    return failed(e, "cudaMemset");
  if (cudaError_t e = cudaMemset(scratch.output, 0, out_bytes); e != cudaSuccess)
    return failed(e, "cudaMemset");
  if (cudaError_t e = cudaStreamCreateWithFlags(&scratch.stream, cudaStreamNonBlocking); e != cudaSuccess)
    return failed(e, "cudaStreamCreate");
  const auto in = reinterpret_cast<std::uintptr_t>(scratch.input);
  const auto out = reinterpret_cast<std::uintptr_t>(scratch.output);
  const Tensor input = tensor_of(in, 0, kernels::kFusedInputPlanes, h, w);
  std::vector<Tensor> outputs;
  if (key.motion) {
    outputs = {
        tensor_of(out, 0, 4, h, w),
        tensor_of(out + 4U * static_cast<std::size_t>(h) * static_cast<std::size_t>(w) * 2U, 0, 1, h, w)};
  } else {
    outputs = {tensor_of(out, 0, 3, h, w)};
  }
  for (int pass = 0; pass < 3; ++pass) {
    const Tensor inputs[]{input};
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

std::expected<void, Refusal> RifeFusedEngine::use_zone(const std::shared_ptr<SharedActivations>& zone) {
  const auto needed = static_cast<std::int64_t>(scratch_offset_ + layout_.bytes);
  if (auto ok = zone->reserve(needed); !ok) return ok;
  zone_ = zone;
  drop_graphs();  // les graphes capturés l'étaient sur une autre adresse
  return {};
}

std::expected<void, Refusal> RifeFusedEngine::run(cudaStream_t stream,
                                                  const std::shared_ptr<SharedActivations>& arena,
                                                  const Tensor& input, std::span<const Tensor> outputs) {
  const int h = height_;
  const int w = width_;
  auto* base = static_cast<std::byte*>(arena->data()) + scratch_offset_;
  const auto at = [base](std::size_t offset) { return reinterpret_cast<std::uintptr_t>(base + offset); };
  const auto at_ptr = [base](std::size_t offset) { return static_cast<void*>(base + offset); };
  const void* const in = address_of(input);

  const Tensor images = tensor_of(input.handle, input.offset, 3, h, w, 2);
  const Tensor encoded = tensor_of(at(layout_.encode), 0, 4, h, w, 2);
  if (auto ok = encode_->enqueue_plain(stream, arena, std::span(&images, 1), std::span(&encoded, 1)); !ok)
    return ok;

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
    const Tensor block_in = tensor_of(at(layout_.x[i]), 0, x.channels, x.height, x.width);
    const Tensor block_out = tensor_of(at(layout_.y[i]), 0, y.channels, y.height, y.width);
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
    finish.flow = address_of(outputs[0]);
    finish.mask = address_of(outputs[1]);
  } else {
    finish.image = address_of(outputs[0]);
  }
  return cuda_check(launch_fused_finish(finish, stream), "fused_finish<<<>>>");
}

std::expected<void, Refusal> RifeFusedEngine::enqueue(cudaStream_t stream,
                                                      const std::shared_ptr<SharedActivations>& arena,
                                                      std::span<const Tensor> inputs,
                                                      std::span<const Tensor> outputs) {
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
  if (arena != zone_ || arena->bytes() < needed ||
      (!graphs_.empty() && graphs_.front().generation != arena->generation())) {
    if (auto ok = use_zone(arena); !ok) return ok;
  }
  if (graphs_off_) return run(stream, arena, inputs[0], outputs);

  const std::array<std::uintptr_t, 3> addresses{
      inputs[0].handle + inputs[0].offset, outputs[0].handle + outputs[0].offset,
      outputs.size() > 1 ? outputs[1].handle + outputs[1].offset : 0};
  auto found = std::ranges::find_if(graphs_, [&](const Graph& g) {
    return g.addresses == addresses && g.zone == reinterpret_cast<std::uintptr_t>(arena->data()) &&
           g.generation == arena->generation();
  });
  if (found == graphs_.end()) {
    if (graphs_.size() >= kMaxGraphs) return run(stream, arena, inputs[0], outputs);
    graphs_.push_back(Graph{.addresses = addresses,
                            .zone = reinterpret_cast<std::uintptr_t>(arena->data()),
                            .generation = arena->generation()});
    found = graphs_.end() - 1;
  }
  Graph& graph = *found;
  if (graph.exec != nullptr) {
    if (cudaGraphLaunch(graph.exec, stream) != cudaSuccess)
      return refuse(RefusalCode::VendorCallFailed, "RIFE fusionné : cudaGraphLaunch a échoué");
    return {};
  }
  if (!graph.warmed) {  // le premier lancement fait ses allocations hors capture
    graph.warmed = true;
    return run(stream, arena, inputs[0], outputs);
  }
  cudaGraph_t captured = nullptr;
  std::expected<void, Refusal> queued;
  cudaError_t error = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
  if (error == cudaSuccess) {
    queued = run(stream, arena, inputs[0], outputs);
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
    return run(stream, arena, inputs[0], outputs);
  }
  if (cudaGraphLaunch(graph.exec, stream) != cudaSuccess)
    return refuse(RefusalCode::VendorCallFailed, "RIFE fusionné : cudaGraphLaunch a échoué");
  return {};
}

}  // namespace ajp::gpu::cuda
