// RIFE fusionné dans le backend TensorRT (kernels/spec/rife_fused.h) : « construire » un manifeste «
// .rifefused » lance trtexec sur chacun des six morceaux qu'il nomme ; « charger » les charge ensemble et
// fait un essai de toute la chaîne. Tout refus ici fait reprendre la chaîne sur le moteur RIFE entier
// (chain_setup.cpp) : l'ONNX d'origine n'est jamais touché.
#include <format>
#include <fstream>
#include <iterator>
#include <system_error>

#include "core/utf8_path.h"
#include "gpu/cuda/backend/rife_fused_engine.h"
#include "gpu/cuda/backend/tensorrt_backend.h"
#include "prepare/rife_split.h"
#include "prepare/sha256.h"

namespace ajp::gpu::cuda {
std::expected<Artifact, Refusal> TensorRtBackend::build_rife_fused(const ModelSpec& model,
                                                                   const ShapeSet& shapes,
                                                                   Progress& progress) {
  std::ifstream file(model.onnx_path, std::ios::binary);
  const std::string text{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
  if (!file && !file.eof())
    return refuse(RefusalCode::FileUnreadable,
                  std::format("RIFE fusionné : {} illisible", core::to_utf8(model.onnx_path)));
  auto manifest = prepare::parse_manifest(text);
  if (!manifest) return std::unexpected(std::move(manifest.error()));
  const int h = manifest->height;
  const int w = manifest->width;
  const Shape expected{.dims = {1, kernels::kFusedInputPlanes, h, w}};
  if (!(shapes.min == expected) || !(shapes.opt == expected) || !(shapes.max == expected)) {
    return refuse(
        RefusalCode::ShapeInvalid,
        std::format("RIFE fusionné : forme fixe 1×11×{}×{} attendue, le manifeste dit {}×{}", h, w, h, w));
  }
  // Un moteur publié doit se désérialiser ici : sinon le constructeur le refait.
  const prepare::EngineCheck check =
      [this](const std::filesystem::path& engine) -> std::expected<void, Refusal> {
    if (auto ok = bind(); !ok) return ok;
    auto loaded = TrtEngine::load(*runtime_, engine);
    if (!loaded) return std::unexpected(std::move(loaded.error()));
    return {};
  };
  const prepare::TrtexecBuilder builder(trtexec_config(), check);
  const std::filesystem::path folder = model.onnx_path.parent_path();
  const std::size_t count = manifest->pieces.size();
  RifeFusedKey key{.height = h,
                   .width = w,
                   .motion = manifest->output == prepare::RifeFusedOutput::Motion,
                   .warp = manifest->output == prepare::RifeFusedOutput::ImageCubic ? kernels::RifeWarp::CatmullRom
                                                                                   : kernels::RifeWarp::Bilinear,
                   .plan = manifest->plan};
  for (std::size_t i = 0; i < count; ++i) {
    const std::filesystem::path onnx =
        folder / core::path_from_utf8(manifest->pieces[i]);  // noms en UTF-8 (A04-04)
    auto sha = prepare::sha256_file_known(onnx);
    if (!sha) return std::unexpected(std::move(sha.error()));
    const ModelSpec piece{
        .id = core::to_utf8(onnx.stem()), .onnx_path = onnx, .sha256 = *sha, .precision = TensorType::F16};
    Shape input{.dims = {2, 3, h, w}};
    if (i > 0) {
      const kernels::FusedDims x = kernels::rife_fused_x(manifest->plan, static_cast<int>(i) - 1, h, w);
      input = {.dims = {1, x.channels, x.height, x.width}};
    }
    // La progression de la chaîne : le morceau i occupe le i-ième sixième.
    Progress part{.report =
                      [&progress, i, count](float fraction, const std::string& stage) {
                        if (!progress.report) return;
                        progress.report((static_cast<float>(i) + fraction) / static_cast<float>(count),
                                        std::format("RIFE fusionné {}/{} : {}", i + 1, count, stage));
                      },
                  .stop = progress.stop};
    auto built = builder.build(piece, ShapeSet{.min = input, .opt = input, .max = input}, part);
    if (!built) return std::unexpected(std::move(built.error()));
    key.engines.push_back(core::to_utf8(built->path.filename()));
  }
  return Artifact{.cache_key = rife_fused_artifact_key(key), .path = model.onnx_path};
}

std::expected<std::unique_ptr<Engine>, Refusal> TensorRtBackend::load_rife_fused(const Artifact& artifact) {
  const std::optional<RifeFusedKey> key = rife_fused_artifact_key_parse(artifact.cache_key);
  if (!key) return refuse(RefusalCode::GraphInvalid, "RIFE fusionné : clé d'artefact illisible");
  auto engine = RifeFusedEngine::load(*runtime_, cache_dir_, *key, arena());
  if (!engine) return std::unexpected(std::move(engine.error()));
  log(BackendLogLevel::Info,
      std::format("RIFE fusionné chargé : {}×{}, {} blocs, sortie {}", key->width, key->height,
                  key->plan.blocks,
                  key->motion ? "flot et masque"
                              : (key->warp == kernels::RifeWarp::CatmullRom ? "image (warp Catmull-Rom)" : "image")));
  return std::unique_ptr<Engine>(std::move(*engine));
}

std::expected<void, Refusal> TensorRtBackend::enqueue_rife_fused(RifeFusedEngine& engine,
                                                                 std::span<const Tensor> inputs,
                                                                 std::span<Tensor> outputs) {
  if (auto ok = bind(); !ok) return ok;
  // La chaîne lit l'entrée, écrit les sorties, la zone d'activation et son propre contexte.
  const std::shared_ptr<SharedActivations>& zone = arena();
  std::vector<std::uintptr_t> reads;
  std::vector<std::uintptr_t> writes{reinterpret_cast<std::uintptr_t>(zone.get()),
                                     reinterpret_cast<std::uintptr_t>(&engine)};
  for (const Tensor& tensor : inputs) reads.push_back(tensor.handle);
  for (const Tensor& tensor : outputs) writes.push_back(tensor.handle);
  if (auto ok = order(reads, writes); !ok) return ok;
  const Shape shape = engine.input_shape();
  GpuTiming& timing = stage(std::format("moteur RIFE fusionné {}×{}", shape.dims[3], shape.dims[2]));
  role(Role::Interpolation).begin(stream_);
  timing.begin(stream_);
  auto queued = engine.enqueue(stream_, zone, inputs, outputs);
  timing.end(stream_);
  role(Role::Interpolation).end(stream_);
  if (std::string notice = engine.take_notice(); !notice.empty())
    log(BackendLogLevel::Warn, std::format("RIFE fusionné {}×{} : {}", shape.dims[3], shape.dims[2], notice));
  return queued;
}

}  // namespace ajp::gpu::cuda
