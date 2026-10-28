// L'onglet Système du menu des paramètres : le lecteur inscrit auprès du système, pour paraître
// dans « Ouvrir avec » et devenir le lecteur des vidéos (default_player.h).
#pragma once

#include <chrono>
#include <expected>
#include <filesystem>
#include <future>
#include <memory>
#include <stop_token>
#include <string>

#include "app/play/default_player.h"
#include "app/play/settings.h"
#include "app/play/span_derive.h"

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
  bool finished = false;  // l'évaluation vient de finir avec un niveau gardé : la page le dit, une fois
};

// Image IA > Agrandissement (réglages avancés) : « Préparer le mode Performance (SPAN FP8) » (span_derive.h),
// une tâche de fond avec sa progression et son résultat, sur le modèle de « Évaluer ma machine ».
struct PrepareNote {
  std::string text;
  bool refused = false;
  std::shared_ptr<std::future<std::expected<std::string, Refusal>>> pending;
  std::shared_ptr<SpanDeriveProgress> progress;
  std::shared_ptr<std::stop_source> stop;  // le panneau l'arrête en disparaissant (settings_panel.cpp)
};

// Lance la préparation ; faux si une autre est en cours ou si elle n'est pas offerte.
bool start_span_prepare(const SpanPrepare& prepare, PrepareNote& note);
// La section du bouton, de sa progression, de son résultat et de la case « Préparer en tâche de fond ».
void span_prepare_section(PlaySettings& d, const SpanPrepare& prepare, PrepareNote& note);
// Après « Évaluer ma machine » (bench.finished) : lance la préparation quand la case est cochée, qu'une vidéo
// est ouverte, que rien n'est déjà prêt et que le niveau gardé dans data/bench.toml est Rapide ou plus.
bool prepare_after_bench(const PlaySettings& d, const SpanPrepare& prepare, const std::filesystem::path& data,
                         PrepareNote& note);

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
