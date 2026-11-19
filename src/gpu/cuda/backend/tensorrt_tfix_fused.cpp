// TemporalFix fusionné dans le backend TensorRT (kernels/spec/tfix_fused.h) : « construire » un manifeste
// « .tfixfused » lance trtexec sur chacun des morceaux qu'il nomme (l'encodeur, ou les trois blocs et la tête
// de confiance), à forme fixe ; « charger » les charge ensemble. Tout refus ici fait reprendre l'étage sur le
// graphe entier du cache de l'encodeur (chain_temporal.cpp).
#include <format>
#include <fstream>
#include <iterator>

#include "core/utf8_path.h"
#include "gpu/cuda/backend/tensorrt_backend.h"
#include "gpu/cuda/backend/tfix_fused_engine.h"
#include "prepare/sha256.h"
#include "prepare/tfix_split.h"

namespace kirari::gpu::cuda {

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
  const int scale = manifest->half ? 2 : 1;
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
  if (!key->window) {
    auto engine = TfixEncoderEngine::load(*runtime_, cache_dir_, *key, arena());
    if (!engine) return std::unexpected(std::move(engine.error()));
    return std::unique_ptr<Engine>(std::move(*engine));
  }
  auto engine = TfixWindowEngine::load(*runtime_, cache_dir_, *key, arena());
  if (!engine) return std::unexpected(std::move(engine.error()));
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
