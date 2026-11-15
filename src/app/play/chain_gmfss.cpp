// GMFSS Fortuna Union dans la chaîne (upscale_gmfss.cpp) : ses deux graphes trouvés à côté de RIFE, fixés aux
// plans remplis de la lecture, construits puis chargés. Le mélange (niveau Élevé : RIFE 4.26 et GMFSS au
// milieu des paires fortes) à la taille où RIFE mesure ; GMFSS seul à la taille de travail. TensorRT
// seulement (BackendCaps::gmfss : GMFlow et ses transformeurs). Un refus laisse RIFE seul, la raison au
// journal, retenu pour la séance (C08-12).
#include <algorithm>
#include <cmath>
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

GmfssCost gmfss_cost(WorkSize size, bool half) {
  // Mesuré sur une RTX 4070 Ti à 1280x720 : paire 35 ms (12 au demi-flot), fusion 12 ms, splats 3,3 ms ;
  // à l'échelle des pixels.
  constexpr double kReference = 1280.0 * 720.0;
  const double scale = static_cast<double>(size.width) * static_cast<double>(size.height) / kReference;
  const auto us = [scale](double ms) { return static_cast<std::int64_t>(std::llround(ms * 1000.0 * scale)); };
  const double pair_ms = half ? 12.0 : 35.0;
  return {.total_us = us(pair_ms + 12.0 + 3.3), .pair_us = us(pair_ms)};
}

int gmfss_auto_cap(const ChainEngines& c, int cap, double source_fps) {
  if (c.gmfss_use == GmfssUse::Off || source_fps <= 0.0) return cap;
  // Le demi-flot ou les 540 lignes du mélange léger : environ deux fois moins cher.
  const bool half = c.played.gmfss_half_flow || c.played.gmfss_light;
  // GMFSS à chaque instant : ×2 (×2,5). Très élevé au milieu de chaque paire : ×2,5, ×3,5 avec l'un des
  // deux, ×4 avec les deux (mesuré sur une RTX 4070 Ti, 720p au demi-flot comme 540p : 54 i/s sans image
  // manquée, environ 25 ms par image source pour 41,7 ; plafonné à ×2,5, l'Auto restait au palier 55 i/s de
  // 165 Hz, le suivant étant 82,5) ; RIFE reprend les paires que GMFSS ne tient pas.
  // Le mélange d'Élevé : ×3 (×4).
  const bool both = c.played.gmfss_half_flow && c.played.gmfss_light;
  const bool every = c.gmfss_use == GmfssUse::Only;
  const bool very_high = c.played.motion_engine == MotionEngine::Gmfss;
  const double mixed = both ? 4.0 : half ? 3.5 : 2.5;
  const double factor = every ? (half ? 2.5 : 2.0) : very_high ? mixed : (half ? 4.0 : 3.0);
  const int most = static_cast<int>(std::ceil(source_fps * factor));
  return cap > 0 ? std::min(cap, most) : most;
}

GmfssUse gmfss_wanted(const PlaySettings& settings) {
  if (settings.rife_factor < 2) return GmfssUse::Off;
  // Très élevé : le milieu de chaque paire par la chaîne du mélange (RIFE en secours), ou la fusion à chaque
  // instant sans RIFE (gmfss_every_instant).
  if (settings.motion_engine == MotionEngine::Gmfss)
    return settings.gmfss_every_instant ? GmfssUse::Only : GmfssUse::Middle;
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
  if (settings.gmfss_half_flow && found->half_pair.empty())
    log->warn("GMFSS union : {} introuvable, la paire au flot entier", kGmfssHalfPairFile);
  const GmfssModels chosen = gmfss_chosen(*found, settings.gmfss_half_flow);
  const WorkSize size = use == GmfssUse::Only ? c.work : c.rife_size;
  const kernels::RifeGeometry g = kernels::rife_geometry(size.width, size.height, kernels::kGmfssAlign);
  const std::string key = inner_key("gmfss", library, chosen.pair, g.padded_width, g.padded_height);
  if (inner.refused(ChainStage::Motion, key) != nullptr) {
    log->info("GMFSS union refusé plus tôt dans la séance : {}", without);
    return true;
  }
  const auto refused = [&](const Refusal& refusal) {
    log->warn("GMFSS union refusé ({}) : {}", refusal.detail, without);
    inner.failed(ChainStage::Motion, key, refusal);
    return true;
  };
  auto fixed = static_gmfss(chosen, g.padded_width, g.padded_height, paths::derived_models(p.paths));
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
  // Très élevé par le mélange : chaque paire qui bouge (seuil 0).
  c.gmfss_px =
      settings.motion_engine == MotionEngine::Gmfss ? 0.0F : static_cast<float>(settings.gmfss_motion);
  const bool half = !chosen.half_pair.empty() && chosen.pair == chosen.half_pair;
  const GmfssCost cost = gmfss_cost(size, half);
  c.gmfss_cost_us = cost.total_us;
  c.gmfss_pair_us = cost.pair_us;
  log->info("GMFSS union : {} en {}x{}",
            use == GmfssUse::Only ? "à chaque instant"
            : c.gmfss_px == 0.0F  ? "au milieu de chaque paire"
                                  : "au milieu des paires fortes",
            size.width, size.height);
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
