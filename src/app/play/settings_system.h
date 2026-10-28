// L'onglet Système du menu des paramètres : le lecteur inscrit auprès du système, pour paraître
// dans « Ouvrir avec » et devenir le lecteur des vidéos (default_player.h).
#pragma once

#include <chrono>
#include <expected>
#include <filesystem>
#include <future>
#include <memory>
#include <string>

#include "app/play/default_player.h"
#include "app/play/settings.h"

namespace ajp::app::play {

// Ce que la dernière action de l'onglet a donné, montré sous ses boutons jusqu'à la suivante.
struct SystemNote {
  std::string text;
  bool refused = false;
  // L'action en cours sur un autre fil : xdg-mime et consorts peuvent durer des secondes, et le
  // thread de présentation ne les attend pas.
  std::shared_ptr<std::future<std::expected<std::string, Refusal>>> pending;
};

// Le bouton « Évaluer ma machine » : bench_note porte son essai (auto_bench_run.h) ; video est la vidéo lue
// (vide : l'écran d'accueil, rien à jouer) ; pause demande la mise en pause de la lecture avant l'essai.
struct BenchNote {
  std::string text;
  bool refused = false;
  std::shared_ptr<std::future<std::expected<std::string, Refusal>>> pending;
};

// Rend vrai quand l'évaluation vient d'être lancée : l'appelant met alors la lecture en pause.
bool bench_page(const PlayerPlace& place, const std::filesystem::path& video, BenchNote& bench);

// Général > Mises à jour : le mode, le canal, la fréquence, « Vérifier maintenant », le retour arrière.
struct UpdateNote {
  std::string text;
  bool refused = false;
  std::shared_ptr<std::future<std::expected<std::string, Refusal>>> pending;
};
void updates_page(PlaySettings& d, const PlayerPlace& place, UpdateNote& note);

// Le lecteur par défaut et « Ouvrir avec » (Général > Lecteur par défaut).
// Avec « Ne plus me le demander » (dismissed) et, quand le lecteur est déjà celui des vidéos, un mot qui le
// dit.
void default_player_page(const PlayerPlace& place, SystemNote& last, bool& dismissed, bool already_default);

// La fenêtre de proposition au démarrage (une fois par lancement, tant que AJP n'est pas le lecteur par
// défaut et que « Ne plus demander » n'est pas coché) : « Définir par défaut », « Plus tard », « Ne plus
// demander ». Dessinée au centre du cadre ImGui en cours ; rend vrai quand elle se ferme, dismissed dit si «
// Ne plus demander » est coché.
bool default_player_prompt(const PlayerPlace& place, SystemNote& last, bool& dismissed, float scale);

}  // namespace ajp::app::play
