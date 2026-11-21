#include "app/play/launch_data.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <format>
#include <memory>
#include <string>
#include <utility>

#include "app/play/data_dirs.h"
#include "app/play/legacy_dirs.h"
#include "app/play/pumped_task.h"
#include "core/log.h"
#include "core/thread.h"
#include "core/utf8_path.h"

namespace kirari::app::play {

DataPlaces locate_launch_data(const std::filesystem::path& data, const std::filesystem::path& home,
                              const std::filesystem::path& xdg_cache, bool normal_launch) {
  const auto said = [](const std::string& line) { log_channel("app")->info("données : {}", line); };
  // Moteurs, modèles et caches en local (P3.15) ; ceux d'une version d'avant y sont déplacés une fois. Sous
  // Linux, les caches qui se refont seuls vont ensuite dans ~/.cache/kirari (A36-08).
  DataPlaces places{.heavy = normal_launch ? local_data(data) : data};
  places.cache = cache_data(home, places.heavy, xdg_cache);
  // --data-dir ou --bench-out peuvent viser un cache jetable sur un autre disque. Déplacer les plans
  // personnels dans ce cache les ferait disparaître à la fin de l'essai.
  if (!normal_launch) return places;
  move_heavy_data(data, places.heavy, said);
  // Le dossier des caches d'ajp (<caches>/ajp) devient celui de Kirari (<caches>/kirari), moteurs construits
  // compris (legacy_dirs.h).
  if (places.cache != places.heavy)
    adopt_legacy_dir(places.cache.parent_path() / kLegacyName, places.cache, said);
  move_cache_data(places.heavy, places.cache, said);
  // Les caches qu'une version d'avant a laissés dans ~/.cache/ajp suivent quand $XDG_CACHE_HOME les met
  // ailleurs (sans quoi les moteurs se reconstruisaient).
  if (!home.empty()) move_cache_data(home / ".cache" / kLegacyName, places.cache, said);
  return places;
}

DataPlaces settle_data(const std::filesystem::path& data, const PlayerPlace& place, bool normal_launch) {
  std::filesystem::path xdg_cache;
#if defined(__linux__)
  // C34-13 : le dossier des caches que la session a choisi, un fait du système et non une option.
  if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg != nullptr)  // env: système
    xdg_cache = core::path_from_utf8(xdg);
#endif
  const DataPlaces places = locate_launch_data(data, place.home, xdg_cache, normal_launch);
  // La mise à jour n'est plus qu'une notification : les archives préparées et la copie de l'ancienne
  // installation des versions qui s'appliquaient elles-mêmes se retirent (A26-06) ; le cache des moteurs
  // TensorRT est borné (A33-04).
  if (!normal_launch) return places;
  const auto said = [](const std::string& line) { log_channel("app")->info("données : {}", line); };
  purge_misplaced(user_data(data), said);
  // C18-05 : la purge précède toute construction de moteur, elle reste avant la fenêtre ; sa durée se dit
  // quand elle se voit, pour mesurer ce qu'elle coûte au lancement sur un disque lent.
  const auto tidy_from = std::chrono::steady_clock::now();
  tidy_at_launch(data, places.cache / "engines",
                 [](const std::string& line) { log_channel("app")->info("{}", line); });
  if (const auto took = std::chrono::steady_clock::now() - tidy_from; took > std::chrono::milliseconds(100))
    said(std::format("rangement au lancement en {} ms",
                     std::chrono::duration_cast<std::chrono::milliseconds>(took).count()));
  // Un lecteur inscrit dont le dossier a bougé suit sa nouvelle place (A19-06) ; hors du chemin de la
  // première image (C18-05) : registre ou fiche .desktop, et update-desktop-database, sur un fil de fond.
  const auto registered = std::make_shared<std::atomic<bool>>(false);
  auto registration = start_thread(
      "inscription", ThreadReport([registered](const Refusal&) { registered->store(true); }),
      [place, registered](const std::stop_token&, const ThreadReport&) {
        if (refresh_registration(place))
          log_channel("app")->info("données : inscription refaite à la nouvelle place du lecteur");
        registered->store(true);
      });
  if (registration) abandon_thread(std::move(*registration), [registered] { return registered->load(); });
  return places;
}

}  // namespace kirari::app::play
