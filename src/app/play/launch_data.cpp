#include "app/play/launch_data.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <format>
#include <memory>
#include <string>
#include <utility>

#include "app/play/data_dirs.h"
#include "app/play/pumped_task.h"
#include "core/log.h"
#include "core/thread.h"
#include "core/utf8_path.h"

namespace ajp::app::play {

DataPlaces settle_data(const std::filesystem::path& data, const PlayerPlace& place, bool purge_updates) {
  const auto said = [](const std::string& line) { log_channel("app")->info("données : {}", line); };
  // Moteurs, modèles et caches en local (P3.15) ; ceux d'une version d'avant y sont déplacés une fois. Sous
  // Linux, les caches qui se refont seuls vont ensuite dans ~/.cache/ajp (A36-08).
  DataPlaces places{.heavy = local_data(data)};
  move_heavy_data(data, places.heavy, said);
  std::filesystem::path xdg_cache;
#if defined(__linux__)
  // C34-13 : le dossier des caches que la session a choisi, un fait du système et non une option.
  if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg != nullptr)  // env: système
    xdg_cache = core::path_from_utf8(xdg);
#endif
  places.cache = cache_data(place.home, places.heavy, xdg_cache);
  move_cache_data(places.heavy, places.cache, said);
  // Les caches qu'une version d'avant a laissés dans ~/.cache/ajp suivent (sans quoi les moteurs se
  // reconstruisaient).
  if (!place.home.empty()) move_cache_data(place.home / ".cache" / "ajp", places.cache, said);
  // La mise à jour n'est plus qu'une notification : les archives préparées et la copie de l'ancienne
  // installation des versions qui s'appliquaient elles-mêmes se retirent (A26-06) ; le cache des moteurs
  // TensorRT est borné (A33-04).
  if (!purge_updates) return places;  // --data-dir, banc : ni le dossier de l'utilisateur ni son bureau
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

}  // namespace ajp::app::play
