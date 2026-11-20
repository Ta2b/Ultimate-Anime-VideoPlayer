// RIFE fusionné dans le backend TensorRT (kernels/spec/rife_fused.h) : « construire » un manifeste «
// .rifefused » lance trtexec sur chacun des six morceaux qu'il nomme ; « charger » les charge ensemble et
// fait un essai de toute la chaîne. Tout refus ici fait reprendre la chaîne sur le moteur RIFE entier
// (chain_setup.cpp) : l'ONNX d'origine n'est jamais touché.
#include <algorithm>
#include <cstdint>
#include <format>
#include <fstream>
#include <iterator>
#include <system_error>
#include <utility>

#include "core/file_sync.h"
#include "core/utf8_path.h"
#include "gpu/cuda/backend/rife_fused_engine.h"
#include "gpu/cuda/backend/tensorrt_backend.h"
#include "prepare/derived_files.h"
#include "prepare/model_source.h"
#include "prepare/onnx_split.h"
#include "prepare/rife_native.h"
#include "prepare/rife_split.h"
#include "prepare/sha256.h"

namespace kirari::gpu::cuda {
namespace {

// Le seuil de l'essai du RIFE natif contre TensorRT FP16 (docs/rife-natif.md) ; la séance finale vise 50 dB.
constexpr double kNativeMinDb = 45.0;

// Les couches de l'encodeur et des blocs, lues dans les morceaux ONNX que nomme le manifeste (le même
// découpage que les moteurs TensorRT, poids de 4.25 lite déjà permutés), vérifiées par leur empreinte.
std::expected<std::vector<RifeNativePiece>, Refusal> native_pieces(
    const std::filesystem::path& manifest_path) {
  std::ifstream file(manifest_path, std::ios::binary);
  const std::string text{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
  if (!file && !file.eof())
    return refuse(RefusalCode::FileUnreadable,
                  std::format("RIFE natif : {} illisible", core::to_utf8(manifest_path)));
  auto manifest = prepare::parse_manifest(text);
  if (!manifest) return std::unexpected(std::move(manifest.error()));
  std::vector<RifeNativePiece> pieces;
  for (std::size_t i = 0; i < manifest->pieces.size(); ++i) {
    const std::filesystem::path onnx =
        manifest_path.parent_path() / core::path_from_utf8(manifest->pieces[i]);
    if (i < manifest->digests.size()) {
      auto sha = prepare::sha256_file_known(onnx);
      if (!sha) return std::unexpected(std::move(sha.error()));
      if (*sha != manifest->digests[i])
        return refuse(RefusalCode::ModelUnreadable,
                      std::format("RIFE natif : {} a changé", manifest->pieces[i]));
    }
    auto bytes = prepare::read_onnx_bytes(onnx);
    if (!bytes) return std::unexpected(std::move(bytes.error()));
    auto layers = prepare::rife_native_layers(*bytes);
    if (!layers) return std::unexpected(std::move(layers.error()));
    const int in = i == 0 ? 3 : kernels::rife_fused_block_channels(static_cast<int>(i) - 1);
    const int out = i == 0 ? kernels::kFusedEncodePlanes / 2 : kernels::kFusedOutChannels;
    if (auto ok = prepare::rife_native_check(*layers, in, out); !ok)
      return std::unexpected(std::move(ok.error()));
    RifeNativePiece& piece = pieces.emplace_back();
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

// Le préfixe de l'encodeur Lite livré, coupé juste avant sa transposée.
std::expected<ModelSpec, Refusal> lite_encoder_body(const std::filesystem::path& source, int h, int w) {
  auto bytes = prepare::read_onnx_bytes(source);
  if (!bytes) return std::unexpected(std::move(bytes.error()));
  const prepare::SplitTensor in{.name = "/Reshape_output_0", .dims = {1, 3, h, w}};
  const prepare::SplitTensor out{.name = "/encode/relu_2/LeakyRelu_output_0", .dims = {1, 16, h / 2, w / 2}};
  auto cut = prepare::extract_subgraph(*bytes, std::span(&in, 1), std::span(&out, 1), "encode-body");
  if (!cut) return std::unexpected(std::move(cut.error()));
  const auto digest = prepare::sha256_hex(prepare::byte_text(*cut));
  const auto onnx =
      source.parent_path() / core::path_from_utf8(std::format("{}-body-{}.onnx", core::to_utf8(source.stem()),
                                                              digest.substr(0, 16)));
  if (!prepare::file_matches(onnx, cut->size(), digest)) {
    if (auto stored = core::write_atomic(onnx, *cut); !stored)
      return std::unexpected(std::move(stored.error()));
  }
  return ModelSpec{.id = core::to_utf8(onnx.stem()),
                   .onnx_path = onnx,
                   .sha256 = digest,
                   .precision = TensorType::F16,
                   .tuned = false};
}

bool supports_lite_encoder_body(const cudaDeviceProp& device, const RifeFusedKey& key,
                                const std::filesystem::path& manifest) {
  // L'empreinte du Lite livré fait partie du nom dérivé (prepare/rife_split.cpp).
  return device.major == 8 && device.minor == 9 && key.motion && key.height == 1088 && key.width == 1920 &&
         core::to_utf8(manifest.stem()).ends_with("-8b30b93ef8ab0b3c-motion");
}

std::expected<std::string, Refusal> build_lite_encoder_body(const std::filesystem::path& source,
                                                            const RifeFusedKey& key,
                                                            prepare::TrtexecConfig config,
                                                            const prepare::EngineCheck& check,
                                                            Progress& progress) {
  auto body = lite_encoder_body(source, key.height, key.width);
  if (!body) return std::unexpected(std::move(body.error()));
  config.hwc8_output = true;
  const prepare::TrtexecBuilder builder(std::move(config), check);
  const Shape in{.dims = {1, 3, key.height, key.width}};
  const auto count = static_cast<float>(key.engines.size());
  Progress part{.report =
                    [&progress, count](float fraction, const std::string& stage) {
                      if (progress.report)
                        progress.report((count + fraction) / (count + 1.0F),
                                        "RIFE Lite : encodeur hybride : " + stage);
                    },
                .stop = progress.stop};
  auto built = builder.build(*body, ShapeSet{.min = in, .opt = in, .max = in}, part);
  if (!built) return std::unexpected(std::move(built.error()));
  return core::to_utf8(built->path.filename());
}

// L'identité d'un verdict gardé (rife_native_verdict.h) : la carte (nom, architecture, UUID), le pilote, les
// moteurs de la chaîne (clé d'artefact : taille, variante, warp, plan, noms des moteurs), les sources des
// noyaux natifs et le seuil. Un changement de l'un refait l'essai.
std::string verdict_identity(const cudaDeviceProp& device, int driver, const std::string& artifact_key) {
  std::string uuid;
  for (const char byte : device.uuid.bytes) uuid += std::format("{:02x}", static_cast<std::uint8_t>(byte));
  return std::format("{}|sm{}.{}|{}|CUDA {}|{}|{}|{:.1f}", device.name, device.major, device.minor, uuid,
                     driver, artifact_key, KIRARI_RIFE_NATIVE_SOURCES, kNativeMinDb);
}

}  // namespace

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
  // Les blocs, sous --rife-hwc8, en E/S HWC8 (kTrialRifeHwc8) ; l'encodeur reste en plans : sa sortie est
  // relue par plans par les noyaux de colle, et ses copies ne pèsent que 2 % de la passe.
  prepare::TrtexecConfig block_config = trtexec_config();
  block_config.hwc8_io = rife_hwc8_ || rife_native_;  // le natif lit et écrit X et Y en HWC8
  const prepare::TrtexecBuilder block_builder(std::move(block_config), check);
  const std::filesystem::path folder = model.onnx_path.parent_path();
  const std::size_t count = manifest->pieces.size();
  RifeFusedKey key{.height = h,
                   .width = w,
                   .motion = manifest->output == prepare::RifeFusedOutput::Motion,
                   .warp = manifest->output == prepare::RifeFusedOutput::ImageCubic
                               ? kernels::RifeWarp::CatmullRom
                               : kernels::RifeWarp::Bilinear,
                   .plan = manifest->plan};
  const bool try_body = rife_native_ && supports_lite_encoder_body(properties_, key, model.onnx_path);
  const auto total = count + (try_body ? 1U : 0U);
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
                      [&progress, i, total](float fraction, const std::string& stage) {
                        if (!progress.report) return;
                        progress.report((static_cast<float>(i) + fraction) / static_cast<float>(total),
                                        std::format("RIFE fusionné {}/{} : {}", i + 1, total, stage));
                      },
                  .stop = progress.stop};
    auto built = (i == 0 ? builder : block_builder)
                     .build(piece, ShapeSet{.min = input, .opt = input, .max = input}, part);
    if (!built) return std::unexpected(std::move(built.error()));
    key.engines.push_back(core::to_utf8(built->path.filename()));
  }
  if (try_body) {
    auto body = build_lite_encoder_body(folder / core::path_from_utf8(manifest->pieces.front()), key,
                                        trtexec_config(), check, progress);
    if (body)
      key.encode_body = std::move(*body);
    else if (body.error().code == RefusalCode::Cancelled)
      return std::unexpected(std::move(body.error()));
    else
      log(BackendLogLevel::Warn, "RIFE Lite : préfixe hybride écarté : " + body.error().detail);
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
  if (rife_native_) {
    // Un refus laisse la chaîne à TensorRT, chargée et intacte, sauf un contexte perdu : un noyau fautif
    // pendant l'essai (accès hors limites) le perd, TensorRT n'y peut plus rien non plus, le chargement
    // échoue au lieu de rendre une chaîne morte.
    auto kept = settle_rife_native(**engine, artifact);
    if (!kept && kept.error().code == RefusalCode::DeviceLost)
      return std::unexpected(std::move(kept.error()));
    if (!kept)
      log(BackendLogLevel::Warn, "RIFE natif refusé, TensorRT garde la chaîne : " + kept.error().detail);
  }
  log(BackendLogLevel::Info,
      std::format(
          "RIFE fusionné chargé : {}×{}, {} blocs, sortie {}", key->width, key->height, key->plan.blocks,
          key->motion ? "flot et masque"
                      : (key->warp == kernels::RifeWarp::CatmullRom ? "image (warp Catmull-Rom)" : "image")));
  return std::unique_ptr<Engine>(std::move(*engine));
}

std::expected<void, Refusal> TensorRtBackend::settle_rife_native(RifeFusedEngine& engine,
                                                                 const Artifact& artifact) {
  auto pieces = native_pieces(artifact.path);
  if (!pieces) return std::unexpected(std::move(pieces.error()));
  const auto finish = [&]() -> std::expected<void, Refusal> {
    if (!engine.can_try_hybrid_encoder()) return {};
    auto hybrid = engine.try_hybrid_encoder(pieces->front().back(), arena());
    if (!hybrid && hybrid.error().code == RefusalCode::DeviceLost)
      return std::unexpected(std::move(hybrid.error()));
    if (hybrid)
      log(BackendLogLevel::Info, *hybrid);
    else
      log(BackendLogLevel::Info, "RIFE Lite : encodeur complet conservé (" + hybrid.error().detail + ")");
    return {};
  };
  const std::filesystem::path file = cache_dir_ / rife_verdict::file_name(verdict_identity(
                                                      properties_, driver_version_, artifact.cache_key));
  std::ifstream in(file, std::ios::binary);
  const std::string text{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
  if (const auto verdict = rife_verdict::parse(text, pieces->size(), rife_verdict::kOptionsMax)) {
    auto cached = engine.try_cached(*pieces, arena(), *verdict);
    if (!cached) return std::unexpected(std::move(cached.error()));
    for (const std::string& line : cached->lines)
      log(cached->held ? BackendLogLevel::Info : BackendLogLevel::Warn, line);
    if (cached->held) return finish();
  }
  auto trial = engine.try_native(*pieces, arena(), kNativeMinDb);
  if (!trial) return std::unexpected(std::move(trial.error()));
  // Les temps et la justesse de chaque morceau, le profil des couches, puis le verdict.
  for (std::size_t i = 0; i < trial->lines.size(); ++i)
    log(i + 1 == trial->lines.size() && !engine.native() ? BackendLogLevel::Warn : BackendLogLevel::Info,
        trial->lines[i]);
  if (auto saved = core::write_text_atomically(file, rife_verdict::text_of(trial->verdict)); !saved)
    log(BackendLogLevel::Warn, "RIFE natif : verdict non gardé, l'essai se refera : " + saved.error().detail);
  return finish();
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

}  // namespace kirari::gpu::cuda
