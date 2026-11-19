// TemporalFix fusionné dans le backend TensorRT (kernels/spec/tfix_fused.h) : « construire » un manifeste
// « .tfixfused » lance trtexec sur chacun des morceaux qu'il nomme (l'encodeur, ou les trois blocs et la tête
// de confiance), à forme fixe ; « charger » les charge ensemble. Tout refus ici fait reprendre l'étage sur le
// graphe entier du cache de l'encodeur (chain_temporal.cpp).
#include <cstdint>
#include <format>
#include <fstream>
#include <iterator>

#include "core/utf8_path.h"
#include "gpu/cuda/backend/tensorrt_backend.h"
#include "gpu/cuda/backend/tfix_fused_engine.h"
#include "prepare/model_source.h"
#include "prepare/onnx_split.h"
#include "prepare/rife_native.h"
#include "prepare/sha256.h"
#include "prepare/tfix_split.h"

namespace kirari::gpu::cuda {
namespace {

// Le seuil de l'essai de TemporalFix natif contre TensorRT FP16 (docs/tfix-natif.md) ; la séance vise 50 dB.
constexpr double kNativeMinDb = 45.0;

// Les couches des morceaux que nomme le manifeste (le même découpage que les moteurs TensorRT), vérifiés par
// leur empreinte, rangées pour TfixNativeNet : l'encodeur seul (coupé avant son Resize), ou les blocs et la
// tête, les autres places vides.
std::expected<std::vector<RifeNativePiece>, Refusal> native_pieces(
    const std::filesystem::path& path, const prepare::TfixFusedManifest& manifest) {
  const bool window = manifest.role == prepare::TfixRole::Window;
  std::vector<RifeNativePiece> pieces(2 + kernels::kTfixLevels);
  for (std::size_t i = 0; i < manifest.pieces.size(); ++i) {
    const std::filesystem::path onnx = path.parent_path() / core::path_from_utf8(manifest.pieces[i]);
    auto sha = prepare::sha256_file_known(onnx);
    if (!sha) return std::unexpected(std::move(sha.error()));
    if (i < manifest.digests.size() && *sha != manifest.digests[i])
      return refuse(RefusalCode::ModelUnreadable,
                    std::format("TemporalFix natif : {} a changé", manifest.pieces[i]));
    auto bytes = prepare::read_onnx_bytes(onnx);
    if (!bytes) return std::unexpected(std::move(bytes.error()));
    int in = kernels::kTfixImage;
    int out = kernels::kTfixTraits;
    if (window) {
      const auto b = static_cast<int>(i);
      in = b < kernels::kTfixLevels ? kernels::tfix_block_x(b, 16, 16).channels : kernels::kTfixConf;
      out = b < kernels::kTfixLevels ? kernels::kTfixY : 1;
    } else {
      // L'encodeur sans son Resize : les noyaux natifs l'agrandissent eux-mêmes.
      const prepare::TfixSplit split;
      const prepare::SplitTensor from{.name = split.encode_input,
                                      .dims = {1, kernels::kTfixImage, manifest.height, manifest.width}};
      const prepare::SplitTensor to{.name = split.encode_last};
      auto cut = prepare::extract_subgraph(*bytes, std::span(&from, 1), std::span(&to, 1), "encode_native");
      if (!cut) return std::unexpected(std::move(cut.error()));
      bytes = std::move(*cut);
    }
    auto layers = prepare::rife_native_layers(*bytes);
    if (!layers) return std::unexpected(std::move(layers.error()));
    if (auto ok = prepare::conv_chain_check(*layers, in, out); !ok)
      return std::unexpected(std::move(ok.error()));
    RifeNativePiece& piece = pieces[window ? i + 1 : 0];
    for (prepare::RifeNativeLayer& layer : *layers) {
      piece.push_back(RifeNativeConv{.transposed = layer.transposed,
                                     .cin = layer.cin,
                                     .cout = layer.cout,
                                     .stride = layer.stride,
                                     .weights = std::move(layer.weights),
                                     .bias = std::move(layer.bias),
                                     .slope = std::move(layer.slope),
                                     .scale = std::move(layer.scale)});
    }
  }
  return pieces;
}

}  // namespace

std::expected<Artifact, Refusal> TensorRtBackend::build_tfix_fused(const ModelSpec& model,
                                                                   const ShapeSet& shapes,
                                                                   Progress& progress) {
  std::ifstream file(model.onnx_path, std::ios::binary);
  const std::string text{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
  if (!file && !file.eof())
    return refuse(RefusalCode::FileUnreadable,
                  std::format("TemporalFix fusionné : {} illisible", core::to_utf8(model.onnx_path)));
  auto manifest = prepare::parse_tfix_manifest(text);
  if (!manifest) return std::unexpected(std::move(manifest.error()));
  const int h = manifest->height;
  const int w = manifest->width;
  const std::int64_t scale = manifest->half ? 2 : 1;
  // L'étage demande la forme d'une image de travail.
  const Shape expected{.dims = {1, kernels::kTfixImage, h * scale, w * scale}};
  if (!(shapes.min == expected) || !(shapes.opt == expected) || !(shapes.max == expected)) {
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("TemporalFix fusionné : forme fixe 1×3×{}×{} attendue, demandée {}", h * scale,
                              w * scale, shape_text(shapes)));
  }
  const prepare::EngineCheck check =
      [this](const std::filesystem::path& engine) -> std::expected<void, Refusal> {
    if (auto ok = bind(); !ok) return ok;
    auto loaded = TrtEngine::load(*runtime_, engine);
    if (!loaded) return std::unexpected(std::move(loaded.error()));
    return {};
  };
  const prepare::TrtexecBuilder builder(trtexec_config(), check);
  const std::filesystem::path folder = model.onnx_path.parent_path();
  const bool window = manifest->role == prepare::TfixRole::Window;
  TfixFusedKey key{.window = window, .height = h, .width = w, .half = manifest->half};
  const std::size_t count = manifest->pieces.size();
  for (std::size_t i = 0; i < count; ++i) {
    const std::filesystem::path onnx = folder / core::path_from_utf8(manifest->pieces[i]);
    auto sha = prepare::sha256_file_known(onnx);
    if (!sha) return std::unexpected(std::move(sha.error()));
    const ModelSpec piece{
        .id = core::to_utf8(onnx.stem()), .onnx_path = onnx, .sha256 = *sha, .precision = TensorType::F16};
    Shape input{.dims = {1, kernels::kTfixImage, h, w}};
    if (window) {
      const auto b = static_cast<int>(i);
      const kernels::TfixDims x = b < kernels::kTfixLevels ? kernels::tfix_block_x(b, h, w)
                                                           : kernels::tfix_level(2, h, w, kernels::kTfixConf);
      input = {.dims = {kernels::kTfixSupports, x.channels, x.height, x.width}};
    }
    Progress part{.report =
                      [&progress, i, count](float fraction, const std::string& stage) {
                        if (!progress.report) return;
                        progress.report((static_cast<float>(i) + fraction) / static_cast<float>(count),
                                        std::format("TemporalFix fusionné {}/{} : {}", i + 1, count, stage));
                      },
                  .stop = progress.stop};
    auto built = builder.build(piece, ShapeSet{.min = input, .opt = input, .max = input}, part);
    if (!built) return std::unexpected(std::move(built.error()));
    key.engines.push_back(core::to_utf8(built->path.filename()));
  }
  return Artifact{.cache_key = tfix_fused_artifact_key(key), .path = model.onnx_path};
}

std::expected<std::unique_ptr<Engine>, Refusal> TensorRtBackend::load_tfix_fused(const Artifact& artifact) {
  const std::optional<TfixFusedKey> key = tfix_fused_artifact_key_parse(artifact.cache_key);
  if (!key) return refuse(RefusalCode::GraphInvalid, "TemporalFix fusionné : clé d'artefact illisible");
  // --tfix-native : les morceaux du manifeste, lus seulement si l'essai est demandé.
  const auto pieces = [&]() -> std::expected<std::vector<RifeNativePiece>, Refusal> {
    std::ifstream file(artifact.path, std::ios::binary);
    const std::string text{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    auto manifest = prepare::parse_tfix_manifest(text);
    if (!manifest) return std::unexpected(std::move(manifest.error()));
    return native_pieces(artifact.path, *manifest);
  };
  // L'essai et sa ligne du journal : retenu (Info), écarté ou refusé (Warn) ; TensorRT reste chargé.
  const auto try_native = [&](auto& engine) {
    auto parts = pieces();
    auto kept = parts ? engine.try_native(*parts, arena(), kNativeMinDb)
                      : std::expected<std::string, Refusal>(std::unexpected(std::move(parts.error())));
    if (kept)
      log(engine.native() ? BackendLogLevel::Info : BackendLogLevel::Warn, *kept);
    else
      log(BackendLogLevel::Warn, "TemporalFix natif refusé, TensorRT garde l'étage : " + kept.error().detail);
  };
  if (!key->window) {
    auto engine = TfixEncoderEngine::load(*runtime_, cache_dir_, *key, arena());
    if (!engine) return std::unexpected(std::move(engine.error()));
    if (tfix_native_) try_native(**engine);
    return std::unique_ptr<Engine>(std::move(*engine));
  }
  auto engine = TfixWindowEngine::load(*runtime_, cache_dir_, *key, arena());
  if (!engine) return std::unexpected(std::move(engine.error()));
  if (tfix_native_) try_native(**engine);
  log(BackendLogLevel::Info, std::format("TemporalFix fusionné chargé : réseau {}×{}{}", key->width,
                                         key->height, key->half ? ", demi-taille" : ""));
  return std::unique_ptr<Engine>(std::move(*engine));
}

std::expected<void, Refusal> TensorRtBackend::enqueue_tfix_fused(Engine& engine,
                                                                 std::span<const Tensor> inputs,
                                                                 std::span<Tensor> outputs) {
  if (auto ok = bind(); !ok) return ok;
  const std::shared_ptr<SharedActivations>& zone = arena();
  if (auto ok = order_engine(&engine, inputs, outputs, false); !ok) return ok;
  auto* window = dynamic_cast<TfixWindowEngine*>(&engine);
  const Shape shape = inputs.empty() ? Shape{} : inputs[0].shape;
  GpuTiming& timing = stage_named("TemporalFix fusionné {} {}×{}", window != nullptr ? "fenêtre" : "encodeur",
                                  shape.dims[3], shape.dims[2]);
  role(Role::Upscale).begin(stream_);
  timing.begin(stream_);
  std::expected<void, Refusal> queued;
  if (window != nullptr) {
    queued = window->enqueue(stream_, zone, inputs, outputs);
  } else {
    queued = dynamic_cast<TfixEncoderEngine&>(engine).enqueue(stream_, zone, inputs, outputs);
  }
  timing.end(stream_);
  role(Role::Upscale).end(stream_);
  if (window != nullptr) {
    if (std::string notice = window->take_notice(); !notice.empty())
      log(BackendLogLevel::Warn, std::format("TemporalFix fusionné : {}", notice));
  }
  return queued;
}

}  // namespace kirari::gpu::cuda
