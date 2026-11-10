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
#include <vector>

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

// Vidéo > Post-traitement (réglages avancés) : « Refaire la préparation sur cette vidéo » (span_derive.h),
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

// Général > Application : le mode, le canal, la fréquence, « Vérifier maintenant », le retour arrière.
// Le panneau qui se ferme arrête la vérification ou le téléchargement en cours (P2.6) : le futur attendu par
// le destructeur rend la main au lieu de tenir la fermeture jusqu'à 30 minutes.
struct UpdateNote {
  std::string text;
  bool refused = false;
  std::shared_ptr<std::stop_source> stop = std::make_shared<std::stop_source>();
  std::shared_ptr<std::future<std::expected<std::string, Refusal>>> pending;  // détruit avant stop
  UpdateNote() = default;
  ~UpdateNote() { stop->request_stop(); }
  UpdateNote(const UpdateNote&) = delete;
  UpdateNote& operator=(const UpdateNote&) = delete;
  UpdateNote(UpdateNote&&) = delete;
  UpdateNote& operator=(UpdateNote&&) = delete;
};
// La langue de l'interface : anglais ou français, prise en direct.
void language_section(PlaySettings& d);
void updates_page(PlaySettings& d, const PlayerPlace& place, UpdateNote& note);

// Le lecteur par défaut et « Ouvrir avec » (Général > Application).
// Quand le lecteur est déjà celui des vidéos, un mot qui le dit. Rien ne le propose au démarrage.
void default_player_page(const PlayerPlace& place, SystemNote& last, bool already_default);

// « Signaler un problème » (Général > À propos, P1.33) : la version, la carte et la fin de ajp.log,
// déjà nettoyé à l'écriture (core/log_redact.h), copiées dans le presse-papiers ; puis la page de
// nouvelle issue s'ouvre sur le modèle .github/ISSUE_TEMPLATE/bug.yml.
// La carte graphique de la lecture (P3.16) : Automatique ou l'une des cartes, au prochain lancement ; rien
// avec une seule carte.
void gpu_section(PlaySettings& d, const std::vector<std::string>& gpus);
void report_problem_section(const PlayerPlace& place, const std::string& gpu_name);

}  // namespace ajp::app::play
