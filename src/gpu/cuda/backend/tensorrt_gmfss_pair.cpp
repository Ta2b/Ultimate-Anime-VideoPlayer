// La paire de GMFSS au FeatureNet et au MetricNet natifs dans le backend TensorRT
// (prepare/gmfss_pair_split.h, docs/gmfss-natif.md) : « construire » un manifeste « .gmfsspair » lance
// trtexec sur la paire entière (le même moteur que sans le natif, repris du cache) et sur son flot ;
// « charger » les charge ensemble et, sous --gmfss-native, essaie les réseaux natifs contre la paire entière.
// Tout refus du natif laisse la paire à TensorRT.
#include <cstdint>
#include <format>
#include <fstream>
#include <iterator>
#include <utility>

#include "core/utf8_path.h"
#include "gpu/cuda/backend/gmfss_pair_engine.h"
#include "gpu/cuda/backend/tensorrt_backend.h"
#include "prepare/gmfss_fused.h"
#include "prepare/gmfss_pair_native.h"
#include "prepare/gmfss_pair_split.h"

namespace kirari::gpu::cuda {
namespace {

std::expected<prepare::GmfssPairManifest, Refusal> read_manifest(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  const std::string text{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
  if (!file && !file.eof())
    return refuse(RefusalCode::FileUnreadable,
                  std::format("GMFSS natif : {} illisible", core::to_utf8(path)));
  return prepare::parse_gmfss_pair_manifest(text);
}

}  // namespace

std::expected<Artifact, Refusal> TensorRtBackend::build_gmfss_pair(const ModelSpec& model,
                                                                   const ShapeSet& shapes,
                                                                   Progress& progress) {
  auto manifest = read_manifest(model.onnx_path);
  if (!manifest) return std::unexpected(std::move(manifest.error()));
  const int h = manifest->height;
  const int w = manifest->width;
  // L'étage demande la forme de frames : les deux images au double de la fusion.
  const Shape expected{.dims = {1, 6, 2 * std::int64_t{h}, 2 * std::int64_t{w}}};
  if (!(shapes.min == expected) || !(shapes.opt == expected) || !(shapes.max == expected))
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("GMFSS natif : forme fixe 1×6×{}×{} attendue, demandée {}", 2 * h, 2 * w,
                              shape_text(shapes)));
  // La paire entière comme le lecteur la construit sans le natif (sans pavage L2 ni niveau 5) : la même clé,
  // le moteur déjà construit est repris ; le flot de même.
  auto engines = build_gmfss_pieces(model.onnx_path.parent_path(), manifest->pieces, manifest->digests,
                                    expected, false, progress);
  if (!engines) return std::unexpected(std::move(engines.error()));
  const GmfssFusedKey key{.height = h, .width = w, .engines = std::move(*engines)};
  return Artifact{.cache_key = gmfss_pair_artifact_key(key), .path = model.onnx_path};
}

std::expected<std::unique_ptr<Engine>, Refusal> TensorRtBackend::load_gmfss_pair(const Artifact& artifact) {
  const std::optional<GmfssFusedKey> key = gmfss_pair_artifact_key_parse(artifact.cache_key);
  if (!key) return refuse(RefusalCode::GraphInvalid, "GMFSS natif : clé d'artefact illisible");
  auto manifest = read_manifest(artifact.path);
  if (!manifest) return std::unexpected(std::move(manifest.error()));
  auto engine = GmfssPairEngine::load(*runtime_, cache_dir_, *key, manifest->cut, arena());
  if (!engine) return std::unexpected(std::move(engine.error()));
  if (gmfss_native_) {
    // Les poids du FeatureNet et du MetricNet, lus dans la paire FP16 que nomme le manifeste.
    auto bytes =
        prepare::read_gmfss_piece(artifact.path.parent_path(), manifest->pieces[0], manifest->digests[0]);
    auto nets =
        bytes ? prepare::gmfss_pair_nets(*bytes)
              : std::expected<prepare::GmfssPairNets, Refusal>(std::unexpected(std::move(bytes.error())));
    auto kept =
        nets ? (*engine)->try_native(*nets, arena(), kGmfssNativeMinDb)
             : std::expected<std::vector<std::string>, Refusal>(std::unexpected(std::move(nets.error())));
    // Comme pour la fusion : un contexte perdu fait échouer le chargement, tout autre refus garde TensorRT.
    if (!kept && kept.error().code == RefusalCode::DeviceLost)
      return std::unexpected(std::move(kept.error()));
    if (kept) {
      for (std::size_t i = 0; i < kept->size(); ++i)
        log(i + 1 == kept->size() && !(*engine)->native() ? BackendLogLevel::Warn : BackendLogLevel::Info,
            (*kept)[i]);
    } else
      log(BackendLogLevel::Warn, "GMFSS natif refusé, TensorRT garde la paire : " + kept.error().detail);
  }
  return std::unique_ptr<Engine>(std::move(*engine));
}

std::expected<void, Refusal> TensorRtBackend::enqueue_gmfss_pair(GmfssPairEngine& engine,
                                                                 std::span<const Tensor> inputs,
                                                                 std::span<Tensor> outputs) {
  if (auto ok = bind(); !ok) return ok;
  const std::shared_ptr<SharedActivations>& zone = arena();
  if (auto ok = order_engine(&engine, inputs, outputs, false); !ok) return ok;
  const Shape shape = inputs.empty() ? Shape{} : inputs[0].shape;
  GpuTiming& timing = stage_named("GMFSS paire {} {}×{}", engine.native() ? "native" : "TensorRT",
                                  shape.dims[3], shape.dims[2]);
  role(Role::Interpolation).begin(stream_);
  timing.begin(stream_);
  auto queued = engine.enqueue(stream_, zone, inputs, outputs);
  timing.end(stream_);
  role(Role::Interpolation).end(stream_);
  return queued;
}

}  // namespace kirari::gpu::cuda
