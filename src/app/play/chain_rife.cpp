// RIFE dans la chaîne : le modèle au cran de précision, le fusionné, la variante mouvement et ses passes à
// l'essai (paires calmes, demi-taille), sinon RIFE image. Sorti de chain_setup.cpp (C08-14).
#include <cstdint>
#include <filesystem>
#include <format>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "app/play/chain_setup.h"
#include "app/play/i18n.h"
#include "app/play/motion_model.h"
#include "app/play/paths.h"
#include "app/play/rife_fused_choice.h"
#include "core/log.h"
#include "core/utf8_path.h"

namespace ajp::app::play {
namespace {

std::unexpected<StageFailure> stage_refused(Refusal refusal, std::optional<ChainStage> stage) {
  return std::unexpected(StageFailure{.refusal = std::move(refusal), .stage = stage});
}

// P6.17, essai (--rife-calm-model) : la variante mouvement d'un RIFE plus léger, à la forme de rife_motion,
// pour la passe du milieu des paires calmes. Facultative : un modèle aux autres plans ou à un autre
// remplissage, une dérivation ou une construction refusée la laissent de côté, la raison au journal (essai :
// un refus n'arrête pas la lecture). C08-12 : un refus de construction est retenu pour la séance (inner_key),
// la réouverture suivante ne la refait pas. Faux : la fenêtre fermée pendant la construction.
bool load_calm_rife(Player& p, const gpu::BackendInit& init, ChainEngines& c, InnerFallbacks& inner,
                    const std::string& library, const std::filesystem::path& models_dir) {
  if (!c.rife_motion || !p.options.rife_calm_model) return true;
  const auto log = log_channel("app");
  const std::filesystem::path& calm = *p.options.rife_calm_model;
  auto planes = rife_input_planes(calm);
  if (!planes || *planes != c.rife_planes || rife_padding_multiple(calm) != c.rife_align) {
    log->warn("RIFE des paires calmes ignoré : {} n'a pas les plans ni le remplissage du RIFE principal",
              core::to_utf8(calm.filename()));
    return true;
  }
  auto derived = motion_model(calm, models_dir);
  if (!derived) {
    log->warn("RIFE des paires calmes ignoré : {}", derived.error().detail);
    return true;
  }
  const kernels::RifeGeometry& g = c.geometry;
  const std::string key = inner_key("rife-calm", library, *derived, g.padded_width, g.padded_height);
  if (inner.refused(ChainStage::Motion, key) != nullptr) return true;
  auto loaded = prepare_engine(p, init, *c.backend, *derived,
                               gpu::Shape{.dims = {1, c.rife_planes, g.padded_height, g.padded_width}},
                               EngineLabel{.name = "RIFE calme",
                                           .model = core::to_utf8(derived->stem()),
                                           .width = g.padded_width,
                                           .height = g.padded_height},
                               gpu::Lane::Interpolate);
  if (!loaded) {
    log->warn("RIFE des paires calmes ignoré : {}", loaded.error().detail);
    inner.failed(ChainStage::Motion, key, loaded.error());
    return true;
  }
  if (!*loaded) return false;
  c.rife_motion_calm = std::move(**loaded);
  log->info("RIFE des paires calmes : {}", core::to_utf8(derived->stem()));
  return true;
}

// Rien (fenêtre fermée pendant la construction) : « arrêt » ; un refus : journal et RIFE entier ; une
// annulation (« Annuler et regarder ») : l'interpolation part sans autre construction (A08-01, A31-15).
enum class Fused : std::uint8_t { Loaded, Declined, Closed, Cancelled };

// Un chargement de RIFE : ce que partagent le fusionné, la variante mouvement et RIFE image.
struct RifeLoad {
  Player& p;
  const gpu::BackendInit& init;
  ChainEngines& c;
  const PlaySettings& settings;
  InnerFallbacks& inner;
  const std::string& library;
  std::filesystem::path models_dir;
  bool fused_wanted = false;
  std::optional<Refusal> cancelled;

  // RIFE fusionné (kernels/spec/rife_fused.h) à la place du moteur entier, à la géométrie g.
  Fused try_fused(const kernels::RifeGeometry& g, bool motion, std::unique_ptr<gpu::Engine>& into);
  // Un moteur RIFE entier de `model` à la géométrie g.
  std::expected<std::optional<std::unique_ptr<gpu::Engine>>, Refusal> engine(
      const std::filesystem::path& model, const kernels::RifeGeometry& g);
  // RIFE par son mouvement : une passe au milieu de la paire la sert toute quand le mouvement est faible ;
  // sans variante dérivable, RIFE image comme avant.
  std::expected<bool, StageFailure> motion_variant();
  // RIFE image, fusionné sinon entier, quand la variante mouvement ne tourne pas.
  std::expected<bool, StageFailure> image();
};

Fused RifeLoad::try_fused(const kernels::RifeGeometry& g, bool motion, std::unique_ptr<gpu::Engine>& into) {
  if (!fused_wanted) return Fused::Declined;
  const auto log = log_channel("app");
  const prepare::RifeFusedOutput output = rife_fused_output(motion, settings.cubic_warp);
  const std::string key = inner_key(std::format("rife-fused {}", prepare::output_name(output)), library,
                                    *c.rife_model, g.padded_width, g.padded_height);
  if (const Refusal* before = inner.refused(ChainStage::Motion, key)) {
    c.rife_fused_refused = before->code;  // le panneau dit encore pourquoi l'entier tourne
    return Fused::Declined;
  }
  auto fused = prepare_rife_fused(p, init, *c.backend, *c.rife_model, models_dir, g, output);
  if (!fused && watch_now(fused.error())) {
    cancelled = std::move(fused.error());
    return Fused::Cancelled;
  }
  if (!fused) {
    log->warn("RIFE fusionné indisponible ({}) : RIFE entier", fused.error().detail);
    c.rife_fused_refused = fused.error().code;
    inner.failed(ChainStage::Motion, key, fused.error());
    return Fused::Declined;
  }
  if (!*fused) return Fused::Closed;
  into = std::move(**fused);
  c.rife_fused = true;
  log->info("RIFE fusionné : six moteurs et noyaux de colle, {}", motion ? "flot et masque" : "image");
  return Fused::Loaded;
}

std::expected<std::optional<std::unique_ptr<gpu::Engine>>, Refusal> RifeLoad::engine(
    const std::filesystem::path& model, const kernels::RifeGeometry& g) {
  return prepare_engine(p, init, *c.backend, model,
                        gpu::Shape{.dims = {1, c.rife_planes, g.padded_height, g.padded_width}},
                        EngineLabel{.name = "RIFE",
                                    .model = core::to_utf8(model.stem()),
                                    .width = g.padded_width,
                                    .height = g.padded_height},
                        gpu::Lane::Interpolate);
}

std::expected<bool, StageFailure> RifeLoad::motion_variant() {
  const auto log = log_channel("app");
  auto derived = motion_model(*c.rife_model, models_dir);
  if (!derived) {
    log->warn("RIFE mouvement indisponible ({}) : RIFE image", derived.error().detail);
    return true;
  }
  const Fused tried = try_fused(c.geometry, true, c.rife_motion);
  if (tried == Fused::Closed) return false;
  if (tried == Fused::Cancelled) return stage_refused(std::move(*cancelled), ChainStage::Motion);
  if (tried == Fused::Loaded) return true;
  const std::string key =
      inner_key("rife-motion", library, *derived, c.geometry.padded_width, c.geometry.padded_height);
  if (inner.refused(ChainStage::Motion, key) != nullptr) {
    log->info("RIFE mouvement refusé plus tôt dans la séance : RIFE image");
    return true;
  }
  auto loaded = engine(*derived, c.geometry);
  if (!loaded && watch_now(loaded.error()))
    return stage_refused(std::move(loaded.error()), ChainStage::Motion);
  if (!loaded) {
    // La variante mouvement refusée (construction, mémoire) : RIFE image reste à essayer (A29-01).
    log->warn("RIFE mouvement refusé ({}) : RIFE image", loaded.error().detail);
    inner.failed(ChainStage::Motion, key, loaded.error());
    return true;
  }
  if (!*loaded) return false;
  c.rife_motion = std::move(**loaded);
  return true;
}

std::expected<bool, StageFailure> RifeLoad::image() {
  const Fused tried = try_fused(c.image_geometry, false, c.rife);
  if (tried == Fused::Closed) return false;
  if (tried == Fused::Cancelled) return stage_refused(std::move(*cancelled), ChainStage::Motion);
  if (c.rife) return true;
  auto loaded = engine(*c.rife_model, c.image_geometry);
  if (!loaded) return stage_refused(std::move(loaded.error()), ChainStage::Motion);
  if (!*loaded) return false;
  c.rife = std::move(**loaded);
  return true;
}

}  // namespace

std::expected<bool, StageFailure> load_rife(Player& p, const gpu::BackendInit& init, ChainEngines& c,
                                            const PlaySettings& settings, InnerFallbacks& inner,
                                            const std::string& library) {
  const auto log = log_channel("app");
  const bool use_rife = settings.rife_factor >= 2 && c.rife_model && !c.use_drba() && !c.use_apple();
  c.rife_align = use_rife ? rife_padding_multiple(*c.rife_model) : kernels::kRifeAlign;
  c.geometry = kernels::rife_geometry(c.rife_size.width, c.rife_size.height, c.rife_align);
  c.image_geometry = kernels::rife_geometry(c.work.width, c.work.height, c.rife_align);
  if (use_rife) {
    auto planes = rife_input_planes(*c.rife_model);
    if (!planes) return stage_refused(std::move(planes.error()), ChainStage::Motion);
    c.rife_planes = *planes;
  }
  RifeLoad load{.p = p,
                .init = init,
                .c = c,
                .settings = settings,
                .inner = inner,
                .library = library,
                .models_dir = paths::derived_models(p.paths)};
  // Le fusionné quand rife_fused_choice.h le veut et que le modèle se coupe ; tout échec garde le RIFE
  // entier.
  const std::string fused_declined = use_rife
                                         ? rife_fused_declined(settings, c.backend->capabilities().rife_fused,
                                                               c.rife_planes, model_tier(c.rife_model))
                                         : i18n::tr("no RIFE");
  load.fused_wanted = fused_declined.empty();
  if (use_rife && settings.rife_fused && !load.fused_wanted)
    log->info("RIFE fusionné non utilisé : {}", fused_declined);
  if (use_rife && !p.options.rife_image) {
    auto motion = load.motion_variant();
    if (!motion || !*motion) return motion;
  }
  if (!load_calm_rife(p, init, c, inner, library, load.models_dir)) return false;
  if (!load_half_rife(p, init, c, settings, inner, library, load.models_dir)) return false;
  if (use_rife && !c.rife_motion) {
    auto image = load.image();
    if (!image || !*image) return image;
  }
  if (c.rife_motion) {
    log->info("RIFE mouvement : estimé en {}x{}, warp en {}x{}", c.rife_size.width, c.rife_size.height,
              c.work.width, c.work.height);
  }
  return true;
}

}  // namespace ajp::app::play
