// DistilDRBA fusionné dans le backend TensorRT (kernels/spec/drba_fused.h) : « construire » un manifeste
// « .drbafused » lance trtexec sur chacun des quatre morceaux qu'il nomme ; « charger » les charge ensemble
// et fait un essai de toute la chaîne. Tout refus ici fait reprendre la chaîne sur le moteur DistilDRBA
// entier (chain_setup.cpp) : l'ONNX d'origine n'est jamais touché.
#include <algorithm>
#include <cstdlib>
#include <format>
#include <fstream>
#include <iterator>
#include <string_view>
#include <system_error>

#include "core/utf8_path.h"
#include "gpu/cuda/backend/drba_fused_engine.h"
#include "gpu/cuda/backend/tensorrt_backend.h"
#include "prepare/drba_split.h"
#include "prepare/model_source.h"
#include "prepare/rife_native.h"
#include "prepare/sha256.h"

namespace kirari::gpu::cuda {
namespace {

RifeNativePiece native_tail_recipe(const std::filesystem::path& artifact, const DrbaFusedKey& key) {
  if (key.height != 1088 || key.width != 1920 || key.plan.tail != 4) return {};
  const char* setting = std::getenv("KIRARI_AB_DRBA_NATIVE_TAIL");  // env: banc
  if (setting != nullptr && std::string_view(setting) == "0") return {};
  std::ifstream file(artifact, std::ios::binary);
  const std::string text{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
  auto manifest = prepare::parse_drba_manifest(text);
  if (!manifest || manifest->pieces.empty()) return {};
  const auto path = artifact.parent_path() / core::path_from_utf8(manifest->pieces.back());
  const auto bytes = prepare::read_onnx_bytes(path);
  if (!bytes) return {};
  // Dernier bloc de distil_drba_v2_lite du paquet models-interp-1 en 1920×1088.
  const auto data = std::string_view(reinterpret_cast<const char*>(bytes->data()), bytes->size());
  if (prepare::sha256_hex(data) != "32fe601966d60e38af254b366a8b44808e82a02a8d1532cd2a07984a53a0432f")
    return {};
  auto layers = prepare::rife_native_layers(*bytes);
  if (!layers || !prepare::rife_native_check(*layers, 109, 52)) return {};
  RifeNativePiece native;
  native.reserve(layers->size());
  for (auto& layer : *layers) {
    native.push_back({.transposed = layer.transposed,
                      .cin = layer.cin,
                      .cout = layer.cout,
                      .stride = layer.stride,
                      .weights = std::move(layer.weights),
                      .bias = std::move(layer.bias),
                      .slope = std::move(layer.slope),
                      .scale = std::move(layer.scale)});
  }
  return native;
}

}  // namespace

std::expected<Artifact, Refusal> TensorRtBackend::build_drba_fused(const ModelSpec& model,
                                                                   const ShapeSet& shapes,
                                                                   Progress& progress) {
  std::ifstream file(model.onnx_path, std::ios::binary);
  const std::string text{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
  if (!file && !file.eof())
    return refuse(RefusalCode::FileUnreadable,
                  std::format("DistilDRBA fusionné : {} illisible", core::to_utf8(model.onnx_path)));
  auto manifest = prepare::parse_drba_manifest(text);
  if (!manifest) return std::unexpected(std::move(manifest.error()));
  const int h = manifest->height;
  const int w = manifest->width;
  const Shape expected{.dims = {1, kernels::kDrbaFusedInputPlanes, h, w}};
  if (!(shapes.min == expected) || !(shapes.opt == expected) || !(shapes.max == expected)) {
    return refuse(RefusalCode::ShapeInvalid,
                  std::format("DistilDRBA fusionné : forme fixe 1×7×{}×{} (celle du manifeste) attendue, "
                              "demandée {}",
                              h, w, shape_text(shapes)));
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
  const bool measured_hwc8 =
      h == 1088 && w == 1920 && manifest->plan.tail == 4 && properties_.major == 8 && properties_.minor == 9;
  prepare::TrtexecConfig tail_config = trtexec_config();
  tail_config.hwc8_io = true;
  const prepare::TrtexecBuilder tail_builder(std::move(tail_config), check);
  const std::filesystem::path folder = model.onnx_path.parent_path();
  const std::size_t count = manifest->pieces.size();
  const int t = manifest->plan.tail;
  DrbaFusedKey key{.height = h, .width = w, .plan = manifest->plan};
  for (std::size_t i = 0; i < count; ++i) {
    const std::filesystem::path onnx =
        folder / core::path_from_utf8(manifest->pieces[i]);  // noms en UTF-8 (A04-04)
    auto sha = prepare::sha256_file_known(onnx);
    if (!sha) return std::unexpected(std::move(sha.error()));
    const ModelSpec piece{
        .id = core::to_utf8(onnx.stem()), .onnx_path = onnx, .sha256 = *sha, .precision = TensorType::F16};
    Shape input{.dims = {3, 3, h / t, w / t}};
    if (i > 0) {
      const kernels::DrbaFusedDims x = kernels::drba_fused_x(manifest->plan, static_cast<int>(i) - 1, h, w);
      input = {.dims = {1, x.channels, x.height, x.width}};
    }
    // La progression de la chaîne : le morceau i occupe le i-ième quart.
    Progress part{.report =
                      [&progress, i, count](float fraction, const std::string& stage) {
                        if (!progress.report) return;
                        progress.report((static_cast<float>(i) + fraction) / static_cast<float>(count),
                                        std::format("DistilDRBA fusionné {}/{} : {}", i + 1, count, stage));
                      },
                  .stop = progress.stop};
    const ShapeSet fixed{.min = input, .opt = input, .max = input};
    const bool packed_tail = measured_hwc8 && i + 1 == count;
    auto built = (packed_tail ? tail_builder : builder).build(piece, fixed, part);
    if (!built && packed_tail && !part.stop.stop_requested()) {
      log(BackendLogLevel::Warn, "DistilDRBA : bloc final HWC8 refusé, essai en plans");
      built = builder.build(piece, fixed, part);
    }
    if (!built) return std::unexpected(std::move(built.error()));
    key.engines.push_back(core::to_utf8(built->path.filename()));
  }
  return Artifact{.cache_key = drba_fused_artifact_key(key), .path = model.onnx_path};
}

std::expected<std::unique_ptr<Engine>, Refusal> TensorRtBackend::load_drba_fused(const Artifact& artifact) {
  const std::optional<DrbaFusedKey> key = drba_fused_artifact_key_parse(artifact.cache_key);
  if (!key) return refuse(RefusalCode::GraphInvalid, "DistilDRBA fusionné : clé d'artefact illisible");
  const auto native_tail = drba_native_tail_ && properties_.major == 8 && properties_.minor == 9
                               ? native_tail_recipe(artifact.path, *key)
                               : RifeNativePiece{};
  auto engine = DrbaFusedEngine::load(*runtime_, cache_dir_, *key, arena(), native_tail);
  if (!engine) return std::unexpected(std::move(engine.error()));
  if (!(*engine)->tail_notice().empty()) log(BackendLogLevel::Info, (*engine)->tail_notice());
  log(BackendLogLevel::Info, std::format("DistilDRBA fusionné chargé : {}×{}, réduction {}", key->width,
                                         key->height, key->plan.tail));
  return std::unique_ptr<Engine>(std::move(*engine));
}

std::expected<void, Refusal> TensorRtBackend::enqueue_drba_fused(DrbaFusedEngine& engine,
                                                                 std::span<const Tensor> inputs,
                                                                 std::span<Tensor> outputs) {
  // Un refus avant la chaîne ne laisse pas l'annonce de cet appel au suivant (C03-05) : la paire suivante
  // réécrit les mêmes tampons pair01 et pair12.
  if (auto ok = bind(); !ok) {
    engine.drop_hint();
    return ok;
  }
  // La chaîne lit les entrées, écrit la sortie, la zone d'activation et son propre contexte.
  const std::shared_ptr<SharedActivations>& zone = arena();
  if (auto ok = order_engine(&engine, inputs, outputs, false); !ok) {
    engine.drop_hint();
    return ok;
  }
  const Shape shape = engine.input_shape();
  GpuTiming& timing = stage_named("moteur DistilDRBA fusionné {}×{}", shape.dims[3], shape.dims[2]);
  role(Role::Interpolation).begin(stream_);
  timing.begin(stream_);
  // Sur pair01, tous les consommateurs du temps peuvent lire le scalaire posé par rife_time.
  // Une entrée ordinaire garde son plan dense. Une autre chaîne qui lirait l'entrée ferait reposer le
  // plan entier (restore_rife_times), ce qui retire également la marque.
  const bool scalar_time =
      !inputs.empty() && std::ranges::find(stale_times_, inputs[0].handle + inputs[0].offset,
                                           &StaleTime::key) != stale_times_.end();
  auto queued = engine.enqueue(stream_, zone, inputs, outputs, scalar_time);
  if (queued) {
    note_rife_reader(inputs[0], true);
    note_rife_reader(inputs[1], true);  // l'entrée 12 ne fournit que des plans RGB au moteur
  }
  timing.end(stream_);
  role(Role::Interpolation).end(stream_);
  if (std::string notice = engine.take_notice(); !notice.empty())
    log(BackendLogLevel::Warn,
        std::format("DistilDRBA fusionné {}×{} : {}", shape.dims[3], shape.dims[2], notice));
  return queued;
}

}  // namespace kirari::gpu::cuda
