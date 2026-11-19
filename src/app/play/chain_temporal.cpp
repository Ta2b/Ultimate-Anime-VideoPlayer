// La stabilisation temporelle dans la chaîne (temporal_fix.h, upscale_temporal.cpp) : le modèle de la force
// choisie, fixé à la taille de travail et son entrée coupée en sept images, construit puis chargé. TensorRT
// seulement pour l'instant : l'import du moteur Vulkan ne connaît pas encore tous ses opérateurs (CumSum,
// ReduceSum, ReduceL2…). Un refus laisse la lecture sans elle, la raison au journal, retenu pour la séance
// (C08-12).
#include <format>
#include <utility>
#include <vector>

#include "app/play/chain_setup.h"
#include "app/play/i18n.h"
#include "app/play/paths.h"
#include "app/play/temporal_fix_model.h"
#include "core/log.h"
#include "core/utf8_path.h"

namespace kirari::app::play {
namespace {

// Première construction annoncée tant qu'aucune passée ne la donne : 79 s pour le graphe à pleine taille en
// 1080p sur une RTX 4070 Ti (niveau 3, sans pavage).
constexpr double kTemporalFixFirstBuildS = 80.0;
// L'encodeur seul : quatre convolutions et un Resize, construit en quelques secondes.
constexpr double kTemporalFixEncoderFirstBuildS = 10.0;
// La largeur que le profil laisse libre sous celle du travail : comme vs_temporalfix (width - 8).
constexpr int kTemporalFixLooseWidth = 8;

}  // namespace

std::vector<std::filesystem::path> temporal_fix_dirs(const PlayerPaths& paths) {
  return {paths.models / kTemporalFixDir};
}

std::optional<bool> load_temporal_fix(Player& p, const gpu::BackendInit& init, ChainEngines& c,
                                      const PlaySettings& settings, InnerFallbacks& inner,
                                      const std::string& library) {
  const TemporalFix fix = settings.temporal_fix;
  if (fix == TemporalFix::Off) return true;
  const auto log = log_channel("app");
  if (c.backend_name != "tensorrt") {
    log->info(
        "stabilisation temporelle : le moteur {} ne la fait pas tourner (TensorRT seulement), lecture "
        "sans elle",
        c.backend_name);
    return true;
  }
  const std::vector<std::filesystem::path> dirs = temporal_fix_dirs(p.paths);
  const std::optional<std::filesystem::path> found = find_temporal_fix(fix, dirs);
  if (!found) {
    log->warn("stabilisation temporelle : {} introuvable dans les modèles ({}), lecture sans elle",
              level_of(fix).model, kTemporalFixDir);
    return true;
  }
  const std::string key = inner_key(settings.temporal_fix_full ? "temporal-fix-full" : "temporal-fix",
                                    library, *found, c.work.width, c.work.height);
  if (inner.refused(ChainStage::Upscale, key) != nullptr) {
    log->info("stabilisation temporelle refusée plus tôt dans la séance : lecture sans elle");
    return true;
  }
  const auto refused = [&](const Refusal& refusal) {
    log->warn("stabilisation temporelle refusée ({}) : lecture sans elle", refusal.detail);
    inner.failed(ChainStage::Upscale, key, refusal);
    return true;
  };
  const bool half = !settings.temporal_fix_full;
  if (!temporal_fix_fits(c.work.width, c.work.height, half)) {
    log->info("stabilisation temporelle : taille de travail {}x{} impaire{}, lecture sans elle", c.work.width,
              c.work.height, half ? " à demi-taille" : "");
    return true;
  }
  auto graph = temporal_fix_graph(*found, half, paths::derived_models(p.paths));
  if (!graph) return refused(graph.error());
  auto encoder_graph = temporal_fix_encoder(*found, half, paths::derived_models(p.paths));
  if (!encoder_graph) return refused(encoder_graph.error());
  // La forme d'une image ; les traits gardés suivent le même profil, avec leurs canaux (ModelSource::twins).
  // Le profil laisse la largeur libre de quelques pixels (loose_width), le niveau d'optimisation 3 sans
  // pavage (tuned = false) : 28 ms au lieu de 42 en 1080p, construit en 79 s au lieu de 8 min (RTX 4070 Ti,
  // docs/temporalfix.md).
  const gpu::Shape shape{.dims = {1, 3, c.work.height, c.work.width}};
  const auto build = [&](const std::filesystem::path& onnx, double first_build_s) {
    return prepare_engine(p, init, *c.backend, onnx, shape,
                          EngineLabel{.name = i18n::tr("Temporal stabilization"),
                                      .model = core::to_utf8(found->stem()),
                                      .width = c.work.width,
                                      .height = c.work.height,
                                      .first_build_s = first_build_s,
                                      .tuned = false,
                                      .loose_width = kTemporalFixLooseWidth},
                          gpu::Lane::Upscale);
  };
  auto engine = build(*graph, kTemporalFixFirstBuildS);
  if (!engine) return refused(engine.error());
  if (!*engine) return std::nullopt;  // fenêtre fermée pendant la construction
  if ((**engine)->input_names().size() != 2 * static_cast<std::size_t>(kTemporalFixFrames)) {
    return refused(Refusal{
        RefusalCode::ModelMismatch,
        std::format("{} entrées au lieu de {}", (**engine)->input_names().size(), 2 * kTemporalFixFrames)});
  }
  auto encoder = build(*encoder_graph, kTemporalFixEncoderFirstBuildS);
  if (!encoder) return refused(encoder.error());
  if (!*encoder) return std::nullopt;
  if ((**encoder)->input_names().size() != 1) {
    return refused(Refusal{RefusalCode::ModelMismatch, std::format("encodeur à {} entrées au lieu d'une",
                                                                   (**encoder)->input_names().size())});
  }
  c.temporal_fix = std::move(**engine);
  c.temporal_encoder = std::move(**encoder);
  c.temporal_features = prepare::feature_channels(temporal_fix_cache(half));
  log->info("stabilisation temporelle {} : {} en {}x{}{}", name(fix), core::to_utf8(found->filename()),
            c.work.width, c.work.height, half ? ", réseau à demi-taille" : "");
  return true;
}

}  // namespace kirari::app::play
