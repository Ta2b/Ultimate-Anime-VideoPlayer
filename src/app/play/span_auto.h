// La préparation automatique du mode Performance de SPAN (FP8, noyaux CUTLASS ; span_derive.h). Le cran
// « Performance » de la précision de SPAN se choisit comme les autres : au premier démarrage avec ce cran
// (ou dès qu'on le choisit), le lecteur fabrique lui-même le paquet « .span8 » adapté à la vidéo (une minute
// environ), pendant que la lecture reste en FP16 ; à la fin, la chaîne se rouvre et l'utilise. Si c'est
// impossible, une phrase dit pourquoi et ce qu'il faut faire.
//
// Ici : la logique (quand démarrer, que dire, quand rouvrir) et la tâche partagée entre la boucle du lecteur
// et le panneau des réglages (qui la montre et sait la refaire). Rien ne dépend du panneau ouvert.
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "app/play/precision.h"
#include "app/play/settings_system.h"
#include "app/play/span_derive.h"

namespace ajp::app::play {

// Ce que la machine sait du mode Performance de SPAN (pour dire pourquoi il manque).
struct SpanFp8Facts {
  std::string backend_name;       // le moteur en service (« tensorrt », « vulkan »…)
  std::string gpu_name;           // « NVIDIA GeForce RTX 4050 Laptop GPU »
  bool tensorrt_present = false;  // le module TensorRT est installé (il peut ne pas être en service)
  bool capability = false;        // le moteur en service annonce BackendCaps::span_fp8_cutlass
};

// Vide si le mode Performance est possible ; sinon une phrase qui dit pourquoi et ce qu'il faut faire. Une
// RTX 40 de portable (Ada, calcul 8.9) est une carte prise : seul le moteur peut manquer.
std::string span_fp8_unavailable_reason(const SpanFp8Facts& facts);

// La tâche partagée : l'état d'une préparation (le fil, sa progression, son résultat) et la clé de la
// dernière tentative (modèle + vidéo), pour ne pas la relancer en boucle après un refus. Détruite avec le
// lecteur : le destructeur arrête la tâche et l'attend.
struct SpanJob {
  std::shared_ptr<PrepareNote> note = std::make_shared<PrepareNote>();
  std::string attempted;  // clé de la dernière préparation lancée automatiquement
  SpanJob() = default;
  SpanJob(const SpanJob&) = delete;
  SpanJob& operator=(const SpanJob&) = delete;
  ~SpanJob();
};

// Range le résultat d'une tâche finie dans note (text, refused, completed) ; sans effet si elle tourne
// encore.
void poll_prepare(PrepareNote& note);
// Arrête la tâche en cours et l'attend.
void stop_and_wait(PrepareNote& note);

// « Préparation du mode Performance… 37 % ».
std::string span_progress_text(float fraction);
// La phrase d'un échec : « Mode Performance impossible, lecture en FP16 : <détail> ».
std::string span_failure_text(std::string_view detail);

// Faut-il lancer la préparation automatique ? Le cran Performance est choisi, la préparation est offerte
// (moteur capable, vidéo ouverte, modèle pris), rien n'est prêt, aucun reçu ne l'a déjà refusée, aucune tâche
// ne tourne et cette clé n'a pas déjà été tentée.
bool should_start_span_auto(PrecisionTier precision, const SpanPrepare& prepare, bool job_running,
                            std::string_view key, std::string_view attempted);

// Ce que la boucle du lecteur doit faire de la tâche à ce tour.
struct SpanAutoTick {
  std::optional<std::string> notify;  // à afficher à l'écran
  bool reopen = false;                // le paquet est prêt : rouvrir la chaîne pour l'utiliser
};
SpanAutoTick tick_span_auto(PrecisionTier precision, const SpanPrepare& prepare, SpanJob& job,
                            std::string_view key);

}  // namespace ajp::app::play
