// GMFSS Fortuna Union dans la chaîne (upscale_gmfss.cpp) : ses deux graphes trouvés à côté de RIFE, fixés aux
// plans remplis de la lecture, construits puis chargés. Le mélange (niveau Élevé : RIFE 4.26 et GMFSS au
// milieu des paires fortes) à la taille où RIFE mesure ; GMFSS seul à la taille de travail. TensorRT
// seulement (BackendCaps::gmfss : GMFlow et ses transformeurs). Un refus laisse RIFE seul, la raison au
// journal, retenu pour la séance (C08-12).
#include <format>
#include <optional>
#include <string>
#include <utility>

#include "app/play/chain_setup.h"
#include "app/play/gmfss_model.h"
#include "app/play/i18n.h"
#include "app/play/paths.h"
#include "core/log.h"
#include "core/utf8_path.h"
#include "kernels/spec/gmfss.h"
#include "kernels/spec/rife.h"

namespace ajp::app::play {
namespace {

// Premières constructions mesurées dans le lecteur sur une RTX 4070 Ti (travail 720p) :
// paire 1280x768 en 179 s, fusion 640x384 en 218 s (trtexec, mêmes réglages).
constexpr double kGmfssPairFirstBuildS = 170.0;    // 179 s à 1280x768, ramenés à 1280x720
constexpr double kGmfssFusionFirstBuildS = 220.0;  // à 640x384, sous kReferencePixels : telle quelle

}  // namespace

GmfssUse gmfss_wanted(const PlaySettings& settings) {
  if (settings.rife_factor < 2) return GmfssUse::Off;
  if (settings.motion_engine == MotionEngine::Gmfss) return GmfssUse::Only;
  return settings.motion_engine == MotionEngine::Rife && settings.gmfss_middle ? GmfssUse::Middle
                                                                               : GmfssUse::Off;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
std::optional<bool> load_gmfss(Player& p, const gpu::BackendInit& init, ChainEngines& c,
                               const PlaySettings& settings, InnerFallbacks& inner,
                               const std::string& library) {
  const GmfssUse use = gmfss_wanted(settings);
  if (use == GmfssUse::Off) return true;
  const auto log = log_channel("app");
  const char* const without = use == GmfssUse::Only ? "RIFE à la place" : "RIFE seul";
  if (use == GmfssUse::Middle && !c.rife_motion) {
    log->warn("GMFSS au milieu des paires fortes : RIFE mouvement absent, {}", without);
    return true;
  }
  if (!c.backend->capabilities().gmfss) {
    log->warn("GMFSS union : le moteur {} ne le fait pas tourner (TensorRT seulement), {}", c.backend_name,
              without);
    return true;
  }
  const std::optional<GmfssModels> found = find_gmfss(c.model_places);
  if (!found) {
    log->warn("GMFSS union : {} et {} introuvables à côté de RIFE, {}", kGmfssPairFile, kGmfssFusionFile,
              without);
    return true;
  }
  const WorkSize size = use == GmfssUse::Only ? c.work : c.rife_size;
  const kernels::RifeGeometry g = kernels::rife_geometry(size.width, size.height, kernels::kGmfssAlign);
  const std::string key = inner_key("gmfss", library, found->pair, g.padded_width, g.padded_height);
  if (inner.refused(ChainStage::Motion, key) != nullptr) {
    log->info("GMFSS union refusé plus tôt dans la séance : {}", without);
    return true;
  }
  const auto refused = [&](const Refusal& refusal) {
    log->warn("GMFSS union refusé ({}) : {}", refusal.detail, without);
    inner.failed(ChainStage::Motion, key, refusal);
    return true;
  };
  auto fixed = static_gmfss(*found, g.padded_width, g.padded_height, paths::derived_models(p.paths));
  if (!fixed) return refused(fixed.error());
  // Les deux graphes sans pavage L2 ni niveau 5 (gpu::ModelSpec::tuned). first_s : la durée annoncée tant
  // qu'aucune construction passée ne la donne, mesurée sur une RTX 4070 Ti en travail 720p.
  const auto build = [&](const std::filesystem::path& model, const gpu::Shape& shape, std::string name,
                         gpu::Lane lane, double first_s) {
    return prepare_engine(p, init, *c.backend, model, shape,
                          EngineLabel{.name = std::move(name),
                                      .model = core::to_utf8(model.stem()),
                                      .width = static_cast<int>(shape.dims[3]),
                                      .height = static_cast<int>(shape.dims[2]),
                                      .first_build_s = first_s,
                                      .tuned = false},
                          lane);
  };
  auto pair =
      build(fixed->pair, gpu::Shape{.dims = {1, kernels::kGmfssFramePlanes, g.padded_height, g.padded_width}},
            i18n::tr("GMFSS (pair)"), gpu::Lane::Pair, kGmfssPairFirstBuildS);
  if (!pair) return refused(pair.error());
  if (!*pair) return std::nullopt;
  auto fusion = build(fixed->fusion, gpu::Shape{.dims = {1, 3, g.padded_height / 2, g.padded_width / 2}},
                      i18n::tr("GMFSS (fusion)"), gpu::Lane::Interpolate, kGmfssFusionFirstBuildS);
  if (!fusion) return refused(fusion.error());
  if (!*fusion) return std::nullopt;
  c.gmfss_pair = std::move(**pair);
  c.gmfss_fusion = std::move(**fusion);
  c.gmfss_use = use;
  c.gmfss_px = static_cast<float>(settings.gmfss_motion);
  log->info("GMFSS union : {} en {}x{}",
            use == GmfssUse::Only ? "à chaque instant" : "au milieu des paires fortes", size.width,
            size.height);
  return true;
}

std::expected<bool, StageFailure> load_motion(Player& p, const gpu::BackendInit& init, ChainEngines& c,
                                              const PlaySettings& settings, InnerFallbacks& inner,
                                              const std::string& library) {
  const GmfssUse use = gmfss_wanted(settings);
  if (use == GmfssUse::Only && !load_gmfss(p, init, c, settings, inner, library)) return false;
  auto rife = load_rife(p, init, c, settings, inner, library);
  if (!rife || !*rife) return rife;
  if (use == GmfssUse::Middle && !load_gmfss(p, init, c, settings, inner, library)) return false;
  return true;
}

}  // namespace ajp::app::play
