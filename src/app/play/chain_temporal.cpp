// La stabilisation temporelle dans la chaîne (temporal_fix.h, upscale_temporal.cpp) : le modèle de la force
// choisie, fixé à la taille de travail et son entrée coupée en sept images, construit puis chargé. TensorRT
// seulement pour l'instant : l'import du moteur Vulkan ne connaît pas encore tous ses opérateurs (CumSum,
// ReduceSum, ReduceL2…). Un refus laisse la lecture sans elle, la raison au journal, retenu pour la séance
// (C08-12).
//
// Sous TensorRT, la version fusionnée passe d'abord (kernels/spec/tfix_fused.h, docs/temporalfix.md) : le
// modèle coupé en cinq petits moteurs reliés par des noyaux CUDA, les six appuis en un lot, la fenêtre en un
// graphe CUDA. Si elle ne s'applique pas (structure du modèle inconnue, taille, construction refusée), la
// raison va au journal (retenue pour la séance) et le graphe à cache de l'encodeur prend la place ; une
// préparation arrêtée par l'utilisateur ne lance pas ce graphe. --no-tfix-fused (essai) prend le graphe à
// cache de l'encodeur, pour la mesure ABAB.
#include <cstdint>
#include <format>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "app/play/chain_setup.h"
#include "app/play/i18n.h"
#include "app/play/paths.h"
#include "app/play/temporal_fix_model.h"
#include "core/log.h"
#include "core/utf8_path.h"
#include "kernels/spec/tfix_fused.h"
#include "prepare/tfix_split.h"

namespace kirari::app::play {
namespace {

// Première construction annoncée tant qu'aucune passée ne la donne : 79 s pour le graphe à pleine taille en
// 1080p sur une RTX 4070 Ti (niveau 3, sans pavage).
constexpr double kTemporalFixFirstBuildS = 80.0;
// L'encodeur seul : quatre convolutions et un Resize, construit en quelques secondes.
constexpr double kTemporalFixEncoderFirstBuildS = 10.0;
// La largeur que le profil laisse libre sous celle du travail : comme vs_temporalfix (width - 8).
constexpr int kTemporalFixLooseWidth = 8;
// La fenêtre fusionnée : quatre petits moteurs à forme fixe (blocs et tête, lots de six).
constexpr double kTemporalFixFusedFirstBuildS = 40.0;

enum class FusedLoad : std::uint8_t { Loaded, Declined, Stopped, Closed };

// Les deux moteurs fusionnés, l'encodeur puis la fenêtre. Declined : la raison au journal (retenue pour la
// séance sous key quand c'est un refus du moteur), le graphe à cache de l'encodeur suit ; Stopped : la
// préparation arrêtée (« Annuler et regarder », vidéo glissée, réglage changé, appareil perdu), dans stopped,
// et rien d'autre ne se construit ; Closed : la fenêtre a été fermée pendant une construction.
FusedLoad load_fused(Player& p, const gpu::BackendInit& init, ChainEngines& c,
                     const std::filesystem::path& onnx, bool half, InnerFallbacks& inner,
                     const std::string& key, std::optional<Refusal>& stopped) {
  const auto log = log_channel("app");
  if (!p.options.temporal_fix_fused) {
    log->info("stabilisation temporelle : --no-tfix-fused, graphe à cache de l'encodeur");
    return FusedLoad::Declined;
  }
  const int net_height = half ? c.work.height / 2 : c.work.height;
  const int net_width = half ? c.work.width / 2 : c.work.width;
  if (!kernels::tfix_size_valid(net_height, net_width)) {
    log->info(
        "stabilisation temporelle : réseau {}x{} hors des noyaux fusionnés, graphe à cache de l'encodeur",
        net_width, net_height);
    return FusedLoad::Declined;
  }
  if (inner.refused(ChainStage::Upscale, key) != nullptr) {
    log->info(
        "stabilisation temporelle fusionnée refusée plus tôt dans la séance : graphe à cache de l'encodeur");
    return FusedLoad::Declined;
  }
  const gpu::Shape shape{.dims = {1, kernels::kTfixImage, c.work.height, c.work.width}};
  const auto build =
      [&](prepare::TfixRole role,
          double first_build_s) -> std::expected<std::optional<std::unique_ptr<gpu::Engine>>, Refusal> {
    auto manifest =
        prepare::derive_tfix_fused(onnx, paths::derived_models(p.paths), role, net_height, net_width, half);
    if (!manifest) return std::unexpected(std::move(manifest.error()));
    return prepare_engine(p, init, *c.backend, *manifest, shape,
                          EngineLabel{.name = i18n::tr("Temporal stabilization"),
                                      .model = core::to_utf8(onnx.stem()),
                                      .width = c.work.width,
                                      .height = c.work.height,
                                      .first_build_s = first_build_s},
                          gpu::Lane::Upscale);
  };
  const auto declined = [&](Refusal refusal) {
    // Une préparation arrêtée n'est pas un refus du moteur : le graphe ne se construit pas à sa place.
    if (watch_now(refusal) || preparation_stopped(refusal) || refusal.code == RefusalCode::DeviceLost) {
      stopped = std::move(refusal);
      return FusedLoad::Stopped;
    }
    log->warn("stabilisation temporelle fusionnée refusée ({}) : graphe à cache de l'encodeur",
              refusal.detail);
    inner.failed(ChainStage::Upscale, key, refusal);
    return FusedLoad::Declined;
  };
  auto encoder = build(prepare::TfixRole::Encoder, kTemporalFixEncoderFirstBuildS);
  if (!encoder) return declined(std::move(encoder.error()));
  if (!*encoder) return FusedLoad::Closed;
  auto window = build(prepare::TfixRole::Window, kTemporalFixFusedFirstBuildS);
  if (!window) return declined(std::move(window.error()));
  if (!*window) return FusedLoad::Closed;
  const std::size_t frames = 2 * static_cast<std::size_t>(kTemporalFixFrames);
  if ((**encoder)->input_names().size() != 1 || (**window)->input_names().size() != frames) {
    return declined(
        Refusal{RefusalCode::ModelMismatch,
                std::format("{} et {} entrées au lieu de 1 et {}", (**encoder)->input_names().size(),
                            (**window)->input_names().size(), frames)});
  }
  c.temporal_fix = std::move(**window);
  c.temporal_encoder = std::move(**encoder);
  c.temporal_features = kernels::tfix_packed_planes(net_height, net_width, half);
  return FusedLoad::Loaded;
}

// Le graphe à cache de l'encodeur (ou, whole, le graphe en un morceau), construit puis chargé ; refused dit
// un refus et rend la lecture sans la stabilisation.
std::optional<bool> load_graph(Player& p, const gpu::BackendInit& init, ChainEngines& c,
                               const std::filesystem::path& found, TemporalFix fix, bool half, bool whole,
                               const std::function<bool(const Refusal&)>& refused) {
  const auto log = log_channel("app");
  auto graph = temporal_fix_graph(found, half, paths::derived_models(p.paths), whole);
  if (!graph) return refused(graph.error());
  // La forme d'une image ; les traits gardés suivent le même profil, avec leurs canaux (ModelSource::twins).
  // Le profil laisse la largeur libre de quelques pixels (loose_width), le niveau d'optimisation 3 sans
  // pavage (tuned = false) : 28 ms au lieu de 42 en 1080p, construit en 79 s au lieu de 8 min (RTX 4070 Ti,
  // docs/temporalfix.md).
  const gpu::Shape shape{.dims = {1, 3, c.work.height, c.work.width}};
  const auto build = [&](const std::filesystem::path& onnx, double first_build_s) {
    return prepare_engine(p, init, *c.backend, onnx, shape,
                          EngineLabel{.name = i18n::tr("Temporal stabilization"),
                                      .model = core::to_utf8(found.stem()),
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
  const int inputs = whole ? kTemporalFixFrames : 2 * kTemporalFixFrames;
  if ((**engine)->input_names().size() != static_cast<std::size_t>(inputs)) {
    return refused(
        Refusal{RefusalCode::ModelMismatch,
                std::format("{} entrées au lieu de {}", (**engine)->input_names().size(), inputs)});
  }
  if (whole) {
    c.temporal_fix = std::move(**engine);
    log->info("stabilisation temporelle {} : {} en {}x{}{}, en un seul moteur (--no-tfix-cache)", name(fix),
              core::to_utf8(found.filename()), c.work.width, c.work.height,
              half ? ", réseau à demi-taille" : "");
    return true;
  }
  auto encoder_graph = temporal_fix_encoder(found, half, paths::derived_models(p.paths));
  if (!encoder_graph) return refused(encoder_graph.error());
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
  log->info("stabilisation temporelle {} : {} en {}x{}{}", name(fix), core::to_utf8(found.filename()),
            c.work.width, c.work.height, half ? ", réseau à demi-taille" : "");
  return true;
}

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
  // --no-tfix-cache : le graphe en un morceau, sans cache de l'encodeur (le retour en arrière).
  const bool whole = c.temporal_fix_whole;
  // La version fusionnée a sa propre clé : son refus retenu laisse le graphe à cache essayer.
  const std::string fused_key =
      inner_key(settings.temporal_fix_full ? "temporal-fix-fused-full" : "temporal-fix-fused", library,
                *found, c.work.width, c.work.height);
  std::optional<Refusal> stopped;
  switch (whole ? FusedLoad::Declined : load_fused(p, init, c, *found, half, inner, fused_key, stopped)) {
    case FusedLoad::Loaded:
      log->info("stabilisation temporelle {} fusionnée : {} en {}x{}{}", name(fix),
                core::to_utf8(found->filename()), c.work.width, c.work.height,
                half ? ", réseau à demi-taille" : "");
      return true;
    case FusedLoad::Stopped:
      return refused(*stopped);
    case FusedLoad::Closed:
      return std::nullopt;
    case FusedLoad::Declined:
      break;
  }
  return load_graph(p, init, c, *found, fix, half, whole, refused);
}

}  // namespace kirari::app::play
