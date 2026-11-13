// RIFE fusionné dans le backend TensorRT (kernels/spec/rife_fused.h) : « construire » un manifeste «
// .rifefused » lance trtexec sur chacun des six morceaux qu'il nomme ; « charger » les charge ensemble et
// fait un essai de toute la chaîne. Tout refus ici fait reprendre la chaîne sur le moteur RIFE entier
// (chain_setup.cpp) : l'ONNX d'origine n'est jamais touché.
#include <algorithm>
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
        std::format("RIFE fusionné : forme fixe 1×11×{}×{} (celle du manifeste) attendue, demandée {}", h, w,
                    shape_text(shapes)));
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
                   .warp = manifest->output == prepare::RifeFusedOutput::ImageCubic
                               ? kernels::RifeWarp::CatmullRom
                               : kernels::RifeWarp::Bilinear,
                   .plan = manifest->plan};
  for (std::size_t i = 0; i < count; ++i) {
    const std::filesystem::path onnx =
        folder / core::path_from_utf8(manifest->pieces[i]);  // noms en UTF-8 (A04-04)
    auto sha = prepare::sha256_file_known(onnx);
    if (!sha) return std::unexpected(std::move(sha.error()));
    const ModelSpec piece{
        .id = core::to_utf8(onnx.stem()), .onnx_path = onnx, .sha256 = *sha, .precision = TensorType::F16};
    Shape input{.dims = {1, 3, h, w}};  // l'encodeur, une image à la fois (fusion3)
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
  if (l2_keep_)
    (*engine)->keep_in_l2(persisting_l2(), static_cast<std::size_t>(properties_.accessPolicyMaxWindowSize));
  log(BackendLogLevel::Info,
      std::format(
          "RIFE fusionné chargé : {}×{}, {} blocs, sortie {}", key->width, key->height, key->plan.blocks,
          key->motion ? "flot et masque"
                      : (key->warp == kernels::RifeWarp::CatmullRom ? "image (warp Catmull-Rom)" : "image")));
  return std::unique_ptr<Engine>(std::move(*engine));
}

std::size_t TensorRtBackend::persisting_l2() {
  if (l2_persisting_) return *l2_persisting_;
  // Au plus 16 Mo : de quoi tenir les deux encodages d'une paire en 720p sans prendre le L2 de SPAN.
  constexpr std::size_t kCeiling = std::size_t{16} << 20U;
  const std::size_t bytes =
      std::min(static_cast<std::size_t>(properties_.persistingL2CacheMaxSize), kCeiling);
  l2_persisting_ = 0;
  if (bytes == 0) {
    log(BackendLogLevel::Warn, "--l2-keep : la carte n'a pas de réserve persistante de L2, rien n'est posé");
    return 0;
  }
  if (const cudaError_t error = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, bytes);
      error != cudaSuccess) {
    log(BackendLogLevel::Warn,
        std::format("--l2-keep : réserve de L2 refusée ({}), rien n'est posé", cudaGetErrorName(error)));
    return 0;
  }
  l2_persisting_ = bytes;
  log(BackendLogLevel::Info,
      std::format("--l2-keep : {} Mo de L2 persistant pour l'encodage gardé de RIFE (fenêtre au plus {} Mo)",
                  bytes >> 20U, properties_.accessPolicyMaxWindowSize >> 20));
  return bytes;
}

std::expected<void, Refusal> TensorRtBackend::enqueue_rife_fused(RifeFusedEngine& engine,
                                                                 std::span<const Tensor> inputs,
                                                                 std::span<Tensor> outputs) {
  // Un refus avant la chaîne ne laisse pas l'annonce de cet appel au suivant.
  if (auto ok = bind(); !ok) {
    engine.drop_hint();
    return ok;
  }
  // La chaîne lit l'entrée, écrit les sorties, la zone d'activation et son propre contexte.
  const std::shared_ptr<SharedActivations>& zone = arena();
  if (auto ok = order_engine(&engine, inputs, outputs, false); !ok) {
    engine.drop_hint();
    return ok;
  }
  const Shape shape = engine.input_shape();
  // Le warp final ne sort pas de l'image (C03-03) : sa fenêtre, connue de rife_input ; sinon le tenseur.
  const std::uintptr_t key = inputs.empty() ? 0 : inputs[0].handle + inputs[0].offset;
  const auto known = std::ranges::find(rife_windows_, key, &RifeWindow::key);
  engine.set_window(known != rife_windows_.end() ? known->window : kernels::Window{});
  GpuTiming& timing = stage_named("moteur RIFE fusionné {}×{}", shape.dims[3], shape.dims[2]);
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
