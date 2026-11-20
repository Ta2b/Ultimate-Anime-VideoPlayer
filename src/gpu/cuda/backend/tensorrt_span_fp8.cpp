// Le SPAN entier en FP8 dans le backend TensorRT : « construire » un paquet « .span8 » ne lance pas trtexec,
// cela vérifie le paquet et la taille ; « charger » envoie les poids et essaie les noyaux. Tout refus ici
// fait reprendre la chaîne sur TensorRT FP16 (chain_setup.cpp) : le SPAN d'origine n'est jamais touché.
#include <array>
#include <format>
#include <optional>
#include <span>

#include "gpu/cuda/backend/ring_slot.h"
#include "gpu/cuda/backend/span_fp8_engine.h"
#include "gpu/cuda/backend/tensorrt_backend.h"
#include "gpu/span_fp8/package.h"

namespace kirari::gpu::cuda {
namespace {

// Le FP8 sur Ada seulement ; le FP16 natif aussi sur RTX 50 (SpanFp8Net::supported_fp16_device).
std::expected<void, Refusal> device_refusal(const cudaDeviceProp& properties, bool fp16) {
  const bool supported = fp16 ? SpanFp8Net::supported_fp16_device(properties.major, properties.minor)
                              : SpanFp8Net::supported_device(properties.major, properties.minor);
  if (supported) return {};
  return refuse(
      RefusalCode::FormatUnsupported,
      std::format("SPAN {} (CUTLASS) : {} (sm {}.{}) n'est pas pris, ou backend construit sans CUTLASS",
                  fp16 ? "FP16 natif" : "FP8", properties.name, properties.major, properties.minor));
}

}  // namespace

std::expected<Artifact, Refusal> TensorRtBackend::build_span_fp8(const ModelSpec& model,
                                                                 const ShapeSet& shapes) {
  if (!SpanFp8Net::supported_device(properties_.major, properties_.minor) &&
      !SpanFp8Net::supported_fp16_device(properties_.major, properties_.minor))
    return std::unexpected(device_refusal(properties_, false).error());
  const Shape& shape = shapes.opt;
  if (!(shapes.min == shape) || !(shapes.max == shape) || shape.dims[0] != 1 || shape.dims[1] != 3) {
    return refuse(RefusalCode::ShapeInvalid, "SPAN FP8 : une forme fixe 1×3×H×W est attendue");
  }
  const std::int64_t height = shape.dims[2], width = shape.dims[3];
  // Un seul plafond, celui de size_refusal (B02-04) ; borné avant le rétrécissement en int.
  if (height <= 0 || width <= 0 || height > span_fp8::kMaxSide || width > span_fp8::kMaxSide)
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("SPAN FP8 : taille {}x{} hors de 1 à {}", width, height, span_fp8::kMaxSide));
  if (const std::string why = span_fp8::size_refusal(static_cast<int>(height), static_cast<int>(width));
      !why.empty())
    return refuse(RefusalCode::FormatUnsupported, "SPAN FP8 : " + why);
  auto bytes = read_span_fp8_package(model.onnx_path);
  if (!bytes) return std::unexpected(std::move(bytes.error()));
  auto parsed = span_fp8::parse(*bytes);  // un paquet abîmé se refuse ici, avant tout envoi sur l'appareil
  if (!parsed) return std::unexpected(std::move(parsed.error()));
  if (auto ok = device_refusal(properties_, parsed->fp16); !ok) return std::unexpected(std::move(ok.error()));
  return Artifact{.cache_key = span_fp8_artifact_key(span_fp8::crc32(*bytes), static_cast<int>(height),
                                                     static_cast<int>(width)),
                  .path = model.onnx_path};
}

std::expected<std::unique_ptr<Engine>, Refusal> TensorRtBackend::load_span_fp8(const Artifact& artifact) {
  const std::optional<Shape> shape = span_fp8_artifact_shape(artifact.cache_key);
  const std::optional<std::uint32_t> crc = span_fp8_artifact_crc(artifact.cache_key);
  if (!shape || !crc) return refuse(RefusalCode::GraphInvalid, "SPAN FP8 : clé d'artefact illisible");
  const bool native_blocks =
      span_fp8_native_blocks_ && SpanFp8Net::supported_device(properties_.major, properties_.minor);
  const bool vertical_winograd = properties_.major == 8 && properties_.minor == 9;
  auto engine = SpanFp8Engine::load(artifact.path, *crc, static_cast<int>(shape->dims[2]),
                                    static_cast<int>(shape->dims[3]), arena(), span_native_, native_blocks,
                                    vertical_winograd);
  if (!engine) return std::unexpected(std::move(engine.error()));
  if ((*engine)->fp16())
    log(BackendLogLevel::Info,
        std::format("SPAN FP16 natif chargé : {}×{}, sommes {}", shape->dims[3], shape->dims[2],
                    (*engine)->sums() == SpanNativeSums::F32 ? "f32" : "f16"));
  else
    log(BackendLogLevel::Info,
        std::format("SPAN FP8 ({}) chargé : {}×{}{}",
                    (*engine)->native_blocks() ? "ReLU natives" : "noyaux CUTLASS", shape->dims[3],
                    shape->dims[2], (*engine)->native_edges() ? ", bords natifs" : ""));
  if (const auto notice = (*engine)->native_blocks_notice(); !notice.empty())
    log(BackendLogLevel::Info, notice);
  if (const auto notice = (*engine)->winograd_notice(); !notice.empty()) log(BackendLogLevel::Info, notice);
  if (span_fp8_native_blocks_ && !native_blocks)
    log(BackendLogLevel::Info, "SPAN FP8 : option des ReLU natives disponible sur Ada");
  if (const std::optional<double> db = (*engine)->native_check_db())
    log(BackendLogLevel::Info,
        std::format(
            "SPAN FP16 natif ({}) : sortie de la carte à {:.1f} dB du calcul du processeur, au chargement",
            properties_.name, *db));
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
  if (auto ok = bind_joined(ring); !ok) return ok;
  // Les noyaux lisent l'entrée, écrivent la sortie et leur zone d'activation ; par tuiles, ils lisent aussi
  // la sortie d'avant pour la décaler ou la copier (span_fp8_tiles.cpp).
  const std::shared_ptr<SharedActivations>& zone = arena();
  std::array<Tensor, 2> reads{};
  std::span<const Tensor> ordered = inputs;
  if (const Tensor* previous = engine.tile_read(); previous != nullptr && inputs.size() == 1) {
    reads = {inputs[0], *previous};
    ordered = reads;
  }
  if (auto ok = order_engine(&engine, ordered, outputs, false); !ok) return ok;
  const Shape shape = engine.input_shape();
  const char* precision = engine.fp16()            ? "SPAN FP16 natif"
                          : engine.native_blocks() ? "FP8 ReLU natives"
                                                   : "FP8";
  GpuTiming& timing =
      stage_named("moteur {} {}×{}×{}", precision, shape.dims[1], shape.dims[2], shape.dims[3]);
  role(Role::Upscale).begin(stream_);
  timing.begin(stream_);
  // Les trois parts au profil de fin (span_parts_) : le temps du rôle en lecture, plus long qu'au banc, dit
  // ainsi s'il vient des noyaux eux-mêmes (corps) ou de l'anneau que l'affichage tient (attente).
  span_part(SpanPart::Body).begin(stream_);
  ImportedSlot* imported = nullptr;
  auto queued = engine.enqueue_body(stream_, zone, inputs, outputs, ring != nullptr)
                    .and_then([&]() -> std::expected<void, Refusal> {
                      span_part(SpanPart::Body).end(stream_);
                      if (ring != nullptr) {
                        // L'attente de l'emplacement vient après le corps : SPAN n'est pas retenu derrière un
                        // emplacement que l'affichage tient encore, seule la queue l'est.
                        span_part(SpanPart::SlotWait).begin(stream_);
                        auto begun = begin_slot(*ring->slot, ring->sync);
                        span_part(SpanPart::SlotWait).end(stream_);
                        if (!begun) return std::unexpected(std::move(begun.error()));
                        imported = *begun;
                      }
                      const SpanFp8Ring surface = imported == nullptr
                                                      ? SpanFp8Ring{}
                                                      : SpanFp8Ring{.surface = imported->surface(),
                                                                    .width = ring->slot->width,
                                                                    .height = ring->slot->height};
                      span_part(SpanPart::Tail).begin(stream_);
                      auto tail = engine.enqueue_tail(stream_, outputs, surface);
                      span_part(SpanPart::Tail).end(stream_);
                      return tail;
                    });
  timing.end(stream_);
  role(Role::Upscale).end(stream_);
  if (!inputs.empty()) chain_span_->finish(inputs[0].handle, stream_);
  if (std::string notice = engine.take_notice(); !notice.empty())
    log(BackendLogLevel::Warn, std::format("{} {}×{} : {}", precision, shape.dims[3], shape.dims[2], notice));
  if (!queued || imported == nullptr) return queued;
  return finish_slot(*imported, ring->sync);
}

}  // namespace kirari::gpu::cuda
