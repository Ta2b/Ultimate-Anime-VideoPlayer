#include <optional>
#include <utility>
#include <vector>

#include "app/play/anime4k_choice.h"
#include "app/play/chain_setup.h"
#include "app/play/model_choice.h"
#include "core/log.h"

// Anime4K à la place de SPAN, au départ de la chaîne (prepare_chain_once, chain_setup.cpp) : son choix, ses
// shaders sur l'écran et la liste des modèles du menu.
namespace ajp::app::play {
namespace {

// La liste des modèles ONNX du menu quand Anime4K remplace SPAN : rien n'est chargé, et un --model qui nomme
// Anime4K n'est pas un fichier à chercher. Aucun modèle ONNX : la liste ne propose qu'Anime4K.
std::optional<ModelChoice> span_menu_list(const Player& p, const PlaySettings& settings) {
  PlayOptions listing = p.options;
  listing.model.reset();
  std::erase(listing.given, "--model");
  auto chosen = choose_model(listing, p.paths.home, settings, p.paths.models);
  if (!chosen) return std::nullopt;
  return std::move(*chosen);
}

}  // namespace

// SPAN ou Anime4K, quand l'agrandissement tourne (span_runs ; c.work déjà posé) : Anime4K choisi, la liste du
// menu et ses shaders lus sur l'écran, pour la lecture comme pour l'export ; refusés, la chaîne reste sans
// agrandissement (l'image de travail, agrandie par le filtre de la fenêtre). Vrai : SPAN est à charger.
bool take_upscaler(Player& p, ChainEngines& c, const PlaySettings& settings) {
  const int height = p.video.height;
  const bool upscales = span_runs(settings, height);
  if (settings.upscale && !upscales)
    log_channel("app")->info("restaurer les traits coupé : source {}p gardée telle quelle, sans SPAN",
                             height);
  c.anime4k = anime4k_choice(settings.model, upscales, height, c.work.height);
  if (c.anime4k) c.span_list = span_menu_list(p, settings);
  if (auto shaders = p.screen.set_anime4k(c.anime4k); !shaders) {
    log_channel("app")->warn("Anime4K refusé ({}) : image de travail sans agrandissement",
                             shaders.error().detail);
    c.anime4k.reset();
  }
  return upscales && !anime4k_size(settings.model);
}

}  // namespace ajp::app::play
