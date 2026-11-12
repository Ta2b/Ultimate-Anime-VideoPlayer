// L'échelle par paire (PlaySettings::pair_scale) : le second RIFE mouvement, à demi-taille, que l'étage gpu
// prend pour les paires à mouvement fort (upscale_half.cpp).
#include <utility>

#include "app/play/chain_setup.h"
#include "app/play/motion_model.h"
#include "app/play/upscale_stage.h"
#include "core/log.h"
#include "core/utf8_path.h"

namespace ajp::app::play {

// Facultatif, comme le RIFE des paires calmes : un refus (découpe, construction, mémoire) laisse la lecture à
// pleine taille, la raison au journal. Le fusionné d'abord quand le RIFE principal l'est, sinon la variante
// mouvement entière.
std::optional<bool> load_half_rife(Player& p, const gpu::BackendInit& init, ChainEngines& c,
                                   const PlaySettings& settings, const std::filesystem::path& models_dir) {
  if (!settings.pair_scale || !c.rife_motion || !c.rife_model) return true;
  const auto log = log_channel("app");
  const WorkSize half{.width = pair_scale_side(c.rife_size.width),
                      .height = pair_scale_side(c.rife_size.height)};
  const kernels::RifeGeometry g = kernels::rife_geometry(half.width, half.height, c.rife_align);
  std::optional<std::unique_ptr<gpu::Engine>> engine;
  if (c.rife_fused) {
    auto fused = prepare_rife_fused(p, init, *c.backend, *c.rife_model, models_dir, g,
                                    prepare::RifeFusedOutput::Motion);
    if (fused && !*fused) return std::nullopt;
    if (fused) engine = std::move(**fused);
    if (!fused)
      log->warn("RIFE à demi-taille : fusionné indisponible ({}), RIFE entier", fused.error().detail);
  }
  if (!engine) {
    auto derived = motion_model(*c.rife_model, models_dir);
    if (!derived) {
      log->warn("échelle par paire ignorée : {}", derived.error().detail);
      return true;
    }
    auto loaded = prepare_engine(p, init, *c.backend, *derived,
                                 gpu::Shape{.dims = {1, c.rife_planes, g.padded_height, g.padded_width}},
                                 EngineLabel{.name = "RIFE demi-taille",
                                             .model = core::to_utf8(derived->stem()),
                                             .width = g.padded_width,
                                             .height = g.padded_height},
                                 gpu::Lane::Interpolate);
    if (loaded && !*loaded) return std::nullopt;
    if (!loaded) {
      log->warn("échelle par paire ignorée : {}", loaded.error().detail);
      return true;
    }
    engine = std::move(**loaded);
  }
  c.rife_motion_half = std::move(*engine);
  c.rife_half_size = half;
  log->info("échelle par paire : mouvement fort estimé en {}x{} (au lieu de {}x{})", half.width, half.height,
            c.rife_size.width, c.rife_size.height);
  return true;
}

}  // namespace ajp::app::play
