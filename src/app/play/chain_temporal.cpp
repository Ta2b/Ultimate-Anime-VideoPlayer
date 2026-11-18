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

namespace ajp::app::play {
namespace {

// Première construction annoncée tant qu'aucune passée ne la donne : un petit graphe (un demi-million de
// poids), à GridSample ; à mesurer sur une RTX.
constexpr double kTemporalFixFirstBuildS = 120.0;

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
  const std::string key = inner_key("temporal-fix", library, *found, c.work.width, c.work.height);
  if (inner.refused(ChainStage::Upscale, key) != nullptr) {
    log->info("stabilisation temporelle refusée plus tôt dans la séance : lecture sans elle");
    return true;
  }
  const auto refused = [&](const Refusal& refusal) {
    log->warn("stabilisation temporelle refusée ({}) : lecture sans elle", refusal.detail);
    inner.failed(ChainStage::Upscale, key, refusal);
    return true;
  };
  auto fixed = static_temporal_fix(*found, c.work.width, c.work.height, paths::derived_models(p.paths));
  if (!fixed) return refused(fixed.error());
  // La forme d'une des sept entrées : prepare_engine ne s'en sert que pour l'étiquette d'un graphe statique.
  const gpu::Shape shape{.dims = {1, 3, c.work.height, c.work.width}};
  auto engine = prepare_engine(p, init, *c.backend, *fixed, shape,
                               EngineLabel{.name = i18n::tr("Temporal stabilization"),
                                           .model = core::to_utf8(found->stem()),
                                           .width = c.work.width,
                                           .height = c.work.height,
                                           .first_build_s = kTemporalFixFirstBuildS},
                               gpu::Lane::Upscale);
  if (!engine) return refused(engine.error());
  if (!*engine) return std::nullopt;  // fenêtre fermée pendant la construction
  if ((**engine)->input_names().size() != static_cast<std::size_t>(kTemporalFixFrames)) {
    return refused(Refusal{
        RefusalCode::ModelMismatch,
        std::format("{} entrées au lieu de {}", (**engine)->input_names().size(), kTemporalFixFrames)});
  }
  c.temporal_fix = std::move(**engine);
  log->info("stabilisation temporelle {} : {} en {}x{}", name(fix), core::to_utf8(found->filename()),
            c.work.width, c.work.height);
  return true;
}

}  // namespace ajp::app::play
