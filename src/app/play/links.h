// Ce que la session de lecture (thread main) échange avec le reste du lecteur.
//
// La session ne connaît ni SDL, ni libplacebo, ni les threads : elle parle à des
// interfaces, que player.cpp relie aux vrais modules et que les tests remplacent par des
// doublures.
//   - MainInbox : ce que les autres threads envoient au thread main (images du ring,
//     sous-titres, fin de flux, refus). main ne l'attend jamais : il la vide entre deux
//     vsyncs (drain).
//   - Stages : les boîtes des threads source, gpu, audio et subs, vues de main.
//   - Screen : la fenêtre, la composition et la présentation (render et present).
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <utility>

#include "core/inbox.h"
#include "core/messages.h"
#include "core/ring.h"
#include "core/types.h"
#include "subs/subtitle_frame.h"
#include "subs/subtitle_renderer.h"

struct ImDrawData;

namespace ajp::app::play {

// La source a livré sa dernière image et son dernier paquet pour cette époque.
struct StreamEnded {
  std::uint32_t epoch = 0;
};

// Un thread s'est arrêté ou a refusé quelque chose : affiché tel quel. fatal arrête la
// lecture (thread arrêté, fichier illisible).
struct StageRefusal {
  std::string stage;  // « source », « gpu », « audio », « subs »
  Refusal refusal;
  bool fatal = false;
};

// Slots du ring au plus : la borne des RingFrame dans la boîte de main.
inline constexpr std::size_t kMaxRingSlots = 16;

using MainInbox = Inbox<RingFrame, subs::SubtitleFrame, StreamEnded, StageRefusal>;
inline constexpr MainInbox::Bounds kMainInboxBounds{kMaxRingSlots, subs::kMaxPendingRequests, 4, 64};

// Message suivant sans attendre, ou rien. main est le seul consommateur de sa boîte : un
// message compté par pending<>() y est encore quand wait() le prend.
inline std::optional<MainInbox::Message> try_take(MainInbox& inbox) {
  const std::size_t pending = inbox.pending<RingFrame>() + inbox.pending<subs::SubtitleFrame>() +
                              inbox.pending<StreamEnded>() + inbox.pending<StageRefusal>();
  if (pending == 0) return std::nullopt;
  return inbox.wait(std::stop_token{});
}

// Les boîtes des autres threads. Chaque envoi peut échouer sur une borne pleine : false,
// le message n'est pas pris, la session le signale et continue.
class Stages {
 public:
  virtual ~Stages() = default;
  // Flush{epoch} à source, gpu, audio et subs, dans cet ordre.
  virtual bool flush(std::uint32_t epoch) = 0;
  // cmd::Seek au thread source, toujours après flush (protocole de media/source.h).
  virtual bool seek(const cmd::Seek& seek) = 0;
  // Lecture de l'horloge maître : Play depuis media_us (position de la première image de
  // l'époque), ou Pause.
  virtual bool play(std::uint32_t epoch, std::int64_t media_us) = 0;
  virtual bool pause() = 0;
  // Un slot revient au thread gpu (protocole de core/ring.h).
  virtual bool slot_free(const SlotFree& free) = 0;
  // Demande l'image de sous-titres d'un pts de sortie ; la réponse arrive dans MainInbox.
  virtual bool request_subtitles(const subs::RenderRequest& request) = 0;
  // Réglage du son (cmd::SetVolume, cmd::SetAudioDelay) au thread audio ; true sans piste audio.
  virtual bool audio(const Command& command) = 0;
  // Synchro sur l'écran : la vitesse de lecture qui cale la sortie sur les vsyncs. Autre que 1,
  // l'horloge hôte mène à cette vitesse et l'audio la suit ; 1, l'audio mène de nouveau. false :
  // la boîte audio est pleine, rien n'a changé.
  virtual bool set_speed(double /*speed*/) { return true; }
  // L'audio qui mène ne fournit plus d'horloge (piste illisible, trou, début tardif, fin avant la
  // vidéo) : l'horloge hôte mène à sa place, depuis media_us, jusqu'au prochain Flush. false :
  // rien n'a changé (pas d'audio, ou l'hôte mène déjà).
  virtual bool audio_lost(std::uint32_t /*epoch*/, std::int64_t /*media_us*/) { return false; }
  // cmd::Quit à tous les threads.
  virtual void quit() = 0;
};

// Taille de l'image de sortie, en pixels : la fenêtre, ou l'écran en plein écran.
struct OutputSize {
  int width = 0;
  int height = 0;
  friend bool operator==(const OutputSize&, const OutputSize&) = default;
};

// Ce qu'une présentation a coûté, mesuré par l'écran.
struct ScreenTiming {
  std::int64_t render_us = 0;   // composition (grade, sous-titres, OSD) soumise
  std::int64_t present_us = 0;  // instant hôte où l'image est partie à l'écran
  // Temps passé à attendre une place dans la file de l'écran : au-delà d'une fraction de
  // période, la file est pleine et l'image paraîtra après celles qui y sont déjà.
  std::int64_t queue_wait_us = 0;
  bool presented = true;  // false : fenêtre cachée ou réduite, rien de présenté
};

struct ShowRequest {
  const RingFrame* frame = nullptr;                // nullptr : rien encore, image noire
  const subs::SubtitleFrame* subtitles = nullptr;  // nullptr : aucun sous-titre
  std::span<const std::string> osd;                // lignes de l'OSD, en haut à gauche
  const ImDrawData* ui = nullptr;                  // interface (barre de lecture, menus), par-dessus tout
  int ui_bottom = 0;  // pixels pris en bas par l'interface : les sous-titres du bas passent au-dessus
  int ui_top = 0;     // pixels pris en haut par l'interface : l'OSD passe dessous
};

// L'image de l'interface pour ce tour (ui.h) : changed demande une présentation même sans image
// neuve. draw reste valable jusqu'à la recomposition suivante de l'interface.
struct UiFrame {
  const ImDrawData* draw = nullptr;  // nullptr : rien à dessiner
  bool changed = false;
  int bottom = 0;  // pixels pris en bas de l'écran (barre visible), 0 sinon
  int top = 0;     // pixels pris en haut (titre visible), 0 sinon
};

class Screen {
 public:
  virtual ~Screen() = default;
  virtual OutputSize size() const = 0;
  // Compose puis présente ; attend au plus la vsync suivante (FIFO).
  virtual std::expected<ScreenTiming, Refusal> show(const ShowRequest& request) = 0;
  // Le slot de frame ne sera plus lu par main : rend la SlotFree à envoyer à gpu. La
  // valeur est signalée côté GPU après le dernier rendu qui l'a lu, ou tout de suite après
  // ready_value pour une image jamais montrée.
  virtual SlotFree retire(const RingFrame& frame) = 0;
  // Le slot de frame est écrit (ready_value atteint), lu sans attendre. Montrée ou rendue avant,
  // l'image ferait attendre son écriture à la file de l'affichage, et derrière elle chaque
  // présentation suivante, interface et OSD compris. Faute de pouvoir le lire : vrai.
  virtual bool ready(const RingFrame& /*frame*/) const { return true; }
  virtual void set_fullscreen(bool on) = 0;
  virtual bool fullscreen() const = 0;
};

}  // namespace ajp::app::play
