// La fusion de GMFSS au GridNet natif dans le backend TensorRT (prepare/gmfss_fused.h, docs/gmfss-natif.md) :
// « construire » un manifeste « .gmfssfused » lance trtexec sur la fusion entière (le même moteur que sans
// le natif, repris du cache) et sur son IFNet ; « charger » les charge ensemble et, sous --gmfss-native,
// essaie le GridNet natif contre la fusion entière. Tout refus du natif laisse la fusion à TensorRT.
#include <format>
#include <fstream>
#include <iterator>
#include <utility>

#include "core/utf8_path.h"
#include "gpu/cuda/backend/gmfss_fused_engine.h"
#include "gpu/cuda/backend/tensorrt_backend.h"
#include "prepare/gmfss_fused.h"
#include "prepare/gmfss_native.h"
#include "prepare/model_source.h"
#include "prepare/sha256.h"

namespace kirari::gpu::cuda {
namespace {

// Le seuil de l'essai contre TensorRT FP16 (docs/gmfss-natif.md) ; la séance finale vise 50 dB.
constexpr double kNativeMinDb = 45.0;

std::expected<prepare::GmfssFusedManifest, Refusal> read_manifest(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  const std::string text{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
  if (!file && !file.eof())
    return refuse(RefusalCode::FileUnreadable,
                  std::format("GMFSS natif : {} illisible", core::to_utf8(path)));
  return prepare::parse_gmfss_manifest(text);
}

// Le GridNet lu dans la fusion entière que nomme le manifeste, vérifiée par son empreinte.
std::expected<prepare::GmfssGrid, Refusal> native_grid(const std::filesystem::path& manifest_path) {
  auto manifest = read_manifest(manifest_path);
  if (!manifest) return std::unexpected(std::move(manifest.error()));
  const std::filesystem::path fusion =
      manifest_path.parent_path() / core::path_from_utf8(manifest->pieces[0]);
  auto sha = prepare::sha256_file_known(fusion);
  if (!sha) return std::unexpected(std::move(sha.error()));
  if (*sha != manifest->digests[0])
    return refuse(RefusalCode::ModelUnreadable,
                  std::format("GMFSS natif : {} a changé", manifest->pieces[0]));
  auto bytes = prepare::read_onnx_bytes(fusion);
  if (!bytes) return std::unexpected(std::move(bytes.error()));
  return prepare::gmfss_grid(*bytes);
}

}  // namespace

std::expected<Artifact, Refusal> TensorRtBackend::build_gmfss_fused(const ModelSpec& model,
                                                                    const ShapeSet& shapes,
                                                                    Progress& progress) {
  auto manifest = read_manifest(model.onnx_path);
  if (!manifest) return std::unexpected(std::move(manifest.error()));
  const int h = manifest->height;
  const int w = manifest->width;
  // L'étage demande la forme de half0, celle de la fusion entière.
  const Shape expected{.dims = {1, 3, h, w}};
  if (!(shapes.min == expected) || !(shapes.opt == expected) || !(shapes.max == expected))
    return refuse(
        RefusalCode::ShapeInvalid,
        std::format("GMFSS natif : forme fixe 1×3×{}×{} attendue, demandée {}", h, w, shape_text(shapes)));
  const prepare::EngineCheck check =
      [this](const std::filesystem::path& engine) -> std::expected<void, Refusal> {
    if (auto ok = bind(); !ok) return ok;
    auto loaded = TrtEngine::load(*runtime_, engine);
    if (!loaded) return std::unexpected(std::move(loaded.error()));
    return {};
  };
  const prepare::TrtexecBuilder builder(trtexec_config(), check);
  GmfssFusedKey key{.height = h, .width = w};
  const std::size_t count = manifest->pieces.size();
  for (std::size_t i = 0; i < count; ++i) {
    const std::filesystem::path onnx =
        model.onnx_path.parent_path() / core::path_from_utf8(manifest->pieces[i]);
    auto sha = prepare::sha256_file_known(onnx);
    if (!sha) return std::unexpected(std::move(sha.error()));
    if (*sha != manifest->digests[i])
      return refuse(RefusalCode::ModelUnreadable,
                    std::format("GMFSS natif : {} a changé", manifest->pieces[i]));
    // La fusion entière comme le lecteur la construit sans le natif (sans pavage L2 ni niveau 5,
    // gpu::ModelSpec::tuned) : la même clé, le moteur déjà construit est repris. L'IFNet, un RIFE 4.26
    // à GridSample, prend les réglages de RIFE.
    const ModelSpec piece{.id = core::to_utf8(onnx.stem()),
                          .onnx_path = onnx,
                          .sha256 = *sha,
                          .precision = TensorType::F16,
                          .tuned = i != 0};
    Progress part{.report =
                      [&progress, i, count](float fraction, const std::string& stage) {
                        if (!progress.report) return;
                        progress.report((static_cast<float>(i) + fraction) / static_cast<float>(count),
                                        std::format("GMFSS natif {}/{} : {}", i + 1, count, stage));
                      },
                  .stop = progress.stop};
    auto built = builder.build(piece, ShapeSet{.min = expected, .opt = expected, .max = expected}, part);
    if (!built) return std::unexpected(std::move(built.error()));
    key.engines.push_back(core::to_utf8(built->path.filename()));
  }
  return Artifact{.cache_key = gmfss_fused_artifact_key(key), .path = model.onnx_path};
}

std::expected<std::unique_ptr<Engine>, Refusal> TensorRtBackend::load_gmfss_fused(const Artifact& artifact) {
  const std::optional<GmfssFusedKey> key = gmfss_fused_artifact_key_parse(artifact.cache_key);
  if (!key) return refuse(RefusalCode::GraphInvalid, "GMFSS natif : clé d'artefact illisible");
  auto engine = GmfssFusedEngine::load(*runtime_, cache_dir_, *key, arena());
  if (!engine) return std::unexpected(std::move(engine.error()));
  if (gmfss_native_) {
    auto grid = native_grid(artifact.path);
    auto kept = grid ? (*engine)->try_native(*grid, arena(), kNativeMinDb)
                     : std::expected<std::string, Refusal>(std::unexpected(std::move(grid.error())));
    if (kept)
      log((*engine)->native() ? BackendLogLevel::Info : BackendLogLevel::Warn, *kept);
    else
      log(BackendLogLevel::Warn, "GMFSS natif refusé, TensorRT garde la fusion : " + kept.error().detail);
  }
  return std::unique_ptr<Engine>(std::move(*engine));
}

std::expected<void, Refusal> TensorRtBackend::enqueue_gmfss_fused(GmfssFusedEngine& engine,
                                                                  std::span<const Tensor> inputs,
                                                                  std::span<Tensor> outputs) {
  if (auto ok = bind(); !ok) return ok;
  const std::shared_ptr<SharedActivations>& zone = arena();
  if (auto ok = order_engine(&engine, inputs, outputs, false); !ok) return ok;
  const Shape shape = outputs.empty() ? Shape{} : outputs[0].shape;
  GpuTiming& timing = stage_named("GMFSS fusion {} {}×{}", engine.native() ? "native" : "TensorRT",
                                  shape.dims[3], shape.dims[2]);
  role(Role::Interpolation).begin(stream_);
  timing.begin(stream_);
  auto queued = engine.enqueue(stream_, zone, inputs, outputs);
  timing.end(stream_);
  role(Role::Interpolation).end(stream_);
  return queued;
}

}  // namespace kirari::gpu::cuda
