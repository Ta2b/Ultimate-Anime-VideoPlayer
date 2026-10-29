// L'onglet Système du menu des paramètres : le lecteur inscrit auprès du système, pour paraître
// dans « Ouvrir avec » et devenir le lecteur des vidéos (default_player.h).
#pragma once

#include <chrono>
#include <cstdint>
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

struct BenchState;  // l'essai en cours et de quoi le défaire (settings_system.cpp ; auto_bench.h le décrit)

// Le bouton « Optimiser pour ma machine » : l'essai en cours (étapes, progression, annulation), son résultat
// en une phrase, et de quoi défaire les réglages posés.
struct BenchNote {
  std::string text;  // le résultat en une phrase, ou ce qui a manqué
  bool refused = false;
  std::shared_ptr<BenchState> state;
  ~BenchNote();
  BenchNote();
  BenchNote(const BenchNote&) = delete;
  BenchNote& operator=(const BenchNote&) = delete;
  // Arrête l'essai en cours (le panneau qui disparaît).
  void stop();
};

// Ce que la page sait de la machine et de la vidéo ouverte.
struct BenchContext {
  int display_height = 1080;
  double display_hz = 60.0;
  int source_height = 1080;
  double source_fps = 23.976;
  SpanPrepare span;        // la préparation du mode Performance (étape 4) : offerte avec une vidéo ouverte
  std::string fp8_reason;  // pourquoi le mode Performance est impossible ici (vide : possible)
  // Le dossier des réglages (celui de play.toml : bench.toml, settings.toml et la séquence de test y sont).
  // PlayerPlace::data est la racine des données de l'utilisateur, un ou deux niveaux plus haut : y écrire le
  // résultat le rendait introuvable (« jamais évaluée ») et les essais perdaient les moteurs choisis.
  std::filesystem::path data_dir;
};

enum class BenchEvent : std::uint8_t {
  None,
  Started,  // l'optimisation vient de partir : l'appelant met la lecture en pause
  Applied,  // elle a fini : le brouillon porte les réglages recommandés, à appliquer
  Undone,   // « Annuler ces réglages » : le brouillon est revenu à l'état d'avant
};

// Image IA > Agrandissement (réglages avancés) : « Refaire la préparation sur cette vidéo » (span_derive.h),
// la tâche partagée avec la boucle du lecteur (span_auto.h) avec sa progression et son résultat.
struct PrepareNote {
  std::string text;
  bool refused = false;
  bool completed = false;  // la tâche vient de finir : la boucle du lecteur le lit une fois (span_auto.h)
  std::shared_ptr<std::future<std::expected<std::string, Refusal>>> pending;
  std::shared_ptr<SpanDeriveProgress> progress;
  std::shared_ptr<std::stop_source> stop;  // le lecteur l'arrête en se fermant
};

// Lance la préparation ; faux si une autre est en cours ou si elle n'est pas offerte.
bool start_span_prepare(const SpanPrepare& prepare, PrepareNote& note);
// La section de la préparation : son état, sa progression, son résultat et « Refaire la préparation sur cette
// vidéo ». Discrète : réglages avancés seulement.
void span_prepare_section(const SpanPrepare& prepare, PrepareNote& note);

// La page « Optimiser pour ma machine » : un bouton, l'étape en cours, une barre, Annuler, le résultat. draft
// : les réglages du panneau, que l'optimisation règle (préréglage Auto sur la mesure) ou que « Annuler ces
// réglages » rétablit (avec le bench.toml d'avant). Une optimisation tourne : le panneau demande des images
// même en pause, pour que l'étape et la barre avancent.
bool bench_running(const BenchNote& bench);

BenchEvent bench_page(const PlayerPlace& place, const std::filesystem::path& video,
                      const BenchContext& context, BenchNote& bench, PlaySettings& draft);

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
