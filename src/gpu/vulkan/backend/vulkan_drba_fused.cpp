// DistilDRBA fusionné dans le backend Vulkan (kernels/spec/drba_fused.h) : « construire » un manifeste
// « .drbafused » construit chacun des quatre morceaux qu'il nomme en artefact .ajpvk, à sa forme fixe ;
// « charger » les charge et les relie par les noyaux de colle (drba_fused_engine.h). Tout refus ici fait
// reprendre la chaîne sur le DistilDRBA entier (chain_setup.cpp) : l'ONNX d'origine n'est jamais touché.
#include <format>
#include <fstream>
#include <iterator>

#include "core/utf8_path.h"
#include "gpu/vulkan/backend/drba_fused_engine.h"
#include "gpu/vulkan/backend/vulkan_backend.h"
#include "gpu/vulkan/compute/refuse.h"
#include "prepare/drba_split.h"
#include "prepare/sha256.h"

namespace ajp::gpu::vk {

std::expected<Artifact, Refusal> VulkanBackend::build_drba_fused(const ModelSpec& model,
                                                                 const ShapeSet& shapes, Progress& progress) {
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
                  std::format("DistilDRBA fusionné : forme fixe 1×7×{}×{} attendue", h, w));
  }
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
    Progress part{.report =
                      [&progress, i, count](float fraction, const std::string& stage) {
                        if (!progress.report) return;
                        progress.report((static_cast<float>(i) + fraction) / static_cast<float>(count),
                                        std::format("DistilDRBA fusionné {}/{} : {}", i + 1, count, stage));
                      },
                  .stop = progress.stop};
    auto built = build(piece, ShapeSet{.min = input, .opt = input, .max = input}, part);
    if (!built) return std::unexpected(std::move(built.error()));
    key.artifacts.push_back(core::to_utf8(built->path));
  }
  return Artifact{.cache_key = drba_fused_artifact_key(key), .path = model.onnx_path};
}

std::expected<std::unique_ptr<Engine>, Refusal> VulkanBackend::load_drba_fused(const Artifact& artifact) {
  const std::optional<DrbaFusedKey> key = drba_fused_artifact_key_parse(artifact.cache_key);
  if (!key) return refuse(RefusalCode::GraphInvalid, "DistilDRBA fusionné : clé d'artefact illisible");
  std::vector<std::unique_ptr<VulkanEngine>> pieces;
  for (const std::string& path : key->artifacts) {
    auto loaded = load(Artifact{.cache_key = {}, .path = core::path_from_utf8(path)});
    if (!loaded) return std::unexpected(std::move(loaded.error()));
    auto* piece = dynamic_cast<VulkanEngine*>(loaded->get());
    if (piece == nullptr)
      return refuse(RefusalCode::GraphInvalid, "DistilDRBA fusionné : morceau d'un autre moteur");
    pieces.emplace_back(piece);
    [[maybe_unused]] auto* released = loaded->release();  // le morceau appartient désormais à pieces
  }
  auto engine = DrbaFusedEngine::create(*context_, *memory_, *pipelines_, *key, std::move(pieces));
  if (!engine) return std::unexpected(std::move(engine.error()));
  log(BackendLogLevel::Info, std::format("DistilDRBA fusionné chargé : {}×{}, réduction {}", key->width,
                                         key->height, key->plan.tail));
  return std::unique_ptr<Engine>(std::move(*engine));
}

}  // namespace ajp::gpu::vk
