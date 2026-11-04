// Le SPAN entier en FP8 dans le backend TensorRT : « construire » un paquet « .span8 » ne lance pas trtexec,
// cela vérifie le paquet et la taille ; « charger » envoie les poids et essaie les noyaux. Tout refus ici
// fait reprendre la chaîne sur TensorRT FP16 (chain_setup.cpp) : le SPAN d'origine n'est jamais touché.
#include <format>

#include "gpu/cuda/backend/ring_slot.h"
#include "gpu/cuda/backend/span_fp8_engine.h"
#include "gpu/cuda/backend/tensorrt_backend.h"
#include "gpu/span_fp8/package.h"

namespace ajp::gpu::cuda {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

}  // namespace

std::expected<Artifact, Refusal> TensorRtBackend::build_span_fp8(const ModelSpec& model,
                                                                 const ShapeSet& shapes) {
  if (!SpanFp8Net::supported_device(properties_.major, properties_.minor)) {
    return refuse(
        RefusalCode::FormatUnsupported,
        std::format("SPAN FP8 (CUTLASS) : {} (sm {}.{}) n'est pas pris, ou backend construit sans CUTLASS",
                    properties_.name, properties_.major, properties_.minor));
  }
  const Shape& shape = shapes.opt;
  if (!(shapes.min == shape) || !(shapes.max == shape) || shape.dims[0] != 1 || shape.dims[1] != 3) {
    return refuse(RefusalCode::ShapeInvalid, "SPAN FP8 : une forme fixe 1×3×H×W est attendue");
  }
  const std::int64_t height = shape.dims[2], width = shape.dims[3];
  constexpr std::int64_t kMaxSide =
      32768;  // size_refusal juge la taille ; ce plafond évite le rétrécissement
  if (height <= 0 || width <= 0 || height > kMaxSide || width > kMaxSide)
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("SPAN FP8 : taille {}x{} hors de portée", width, height));
  if (const std::string why = span_fp8::size_refusal(static_cast<int>(height), static_cast<int>(width));
      !why.empty())
    return refuse(RefusalCode::FormatUnsupported, "SPAN FP8 : " + why);
  auto bytes = read_span_fp8_package(model.onnx_path);
  if (!bytes) return std::unexpected(std::move(bytes.error()));
  auto parsed = span_fp8::parse(*bytes);  // un paquet abîmé se refuse ici, avant tout envoi sur l'appareil
  if (!parsed) return std::unexpected(std::move(parsed.error()));
  return Artifact{.cache_key = span_fp8_artifact_key(span_fp8::crc32(*bytes), static_cast<int>(height),
                                                     static_cast<int>(width)),
                  .path = model.onnx_path};
}

std::expected<std::unique_ptr<Engine>, Refusal> TensorRtBackend::load_span_fp8(const Artifact& artifact) {
  const std::optional<Shape> shape = span_fp8_artifact_shape(artifact.cache_key);
  if (!shape) return refuse(RefusalCode::GraphInvalid, "SPAN FP8 : clé d'artefact illisible");
  auto engine = SpanFp8Engine::load(artifact.path, static_cast<int>(shape->dims[2]),
                                    static_cast<int>(shape->dims[3]), arena());
  if (!engine) return std::unexpected(std::move(engine.error()));
  log(BackendLogLevel::Info,
      std::format("SPAN FP8 (noyaux CUTLASS) chargé : {}×{}", shape->dims[3], shape->dims[2]));
  return std::unique_ptr<Engine>(std::move(*engine));
}

std::expected<void, Refusal> TensorRtBackend::enqueue_ring(Engine& engine, std::span<const Tensor> inputs,
                                                           std::span<Tensor> outputs,
                                                           const RingTarget& target) {
  auto* fused = dynamic_cast<SpanFp8Engine*>(&engine);
  // La sortie doit avoir la taille de l'emplacement (write_ring le refuserait sinon) : le chemin commun dit
  // alors pourquoi.
  const bool direct = fused != nullptr && target.slot != nullptr && outputs.size() == 1 &&
                      outputs[0].type == TensorType::F16 && outputs[0].shape.dims[2] == target.slot->height &&
                      outputs[0].shape.dims[3] == target.slot->width;
  if (!direct) return Backend::enqueue_ring(engine, inputs, outputs, target);
  return enqueue_span_fp8(*fused, inputs, outputs, &target);
}

std::expected<void, Refusal> TensorRtBackend::enqueue_span_fp8(SpanFp8Engine& engine,
                                                               std::span<const Tensor> inputs,
                                                               std::span<Tensor> outputs,
                                                               const RingTarget* ring) {
  if (auto ok = bind(); !ok) return ok;
  // Les noyaux lisent l'entrée, écrivent la sortie et leur zone d'activation.
  const std::shared_ptr<SharedActivations>& zone = arena();
  std::vector<std::uintptr_t> reads;
  std::vector<std::uintptr_t> writes{reinterpret_cast<std::uintptr_t>(zone.get()),
                                     reinterpret_cast<std::uintptr_t>(&engine)};
  for (const Tensor& tensor : inputs) reads.push_back(tensor.handle);
  for (const Tensor& tensor : outputs) writes.push_back(tensor.handle);
  if (auto ok = order(reads, writes); !ok) return ok;
  const Shape shape = engine.input_shape();
  GpuTiming& timing = stage(std::format("moteur FP8 {}×{}×{}", shape.dims[1], shape.dims[2], shape.dims[3]));
  role(Role::Upscale).begin(stream_);
  timing.begin(stream_);
  ImportedSlot* imported = nullptr;
  auto queued =
      engine.enqueue_body(stream_, zone, inputs, outputs).and_then([&]() -> std::expected<void, Refusal> {
        if (ring != nullptr) {
          // L'attente de l'emplacement vient après le corps : SPAN n'est pas retenu derrière un emplacement
          // que l'affichage tient encore, seule la queue l'est.
          auto begun = begin_slot(*ring->slot, ring->sync);
          if (!begun) return std::unexpected(std::move(begun.error()));
          imported = *begun;
        }
        const SpanFp8Ring surface = imported == nullptr ? SpanFp8Ring{}
                                                        : SpanFp8Ring{.surface = imported->surface(),
                                                                      .width = ring->slot->width,
                                                                      .height = ring->slot->height};
        return engine.enqueue_tail(stream_, outputs, surface);
      });
  timing.end(stream_);
  role(Role::Upscale).end(stream_);
  if (!inputs.empty()) chain_span_->finish(inputs[0].handle, stream_);
  if (std::string notice = engine.take_notice(); !notice.empty())
    log(BackendLogLevel::Warn, std::format("SPAN FP8 {}×{} : {}", shape.dims[3], shape.dims[2], notice));
  if (!queued || imported == nullptr) return queued;
  return finish_slot(*imported, ring->sync);
}

}  // namespace ajp::gpu::cuda
