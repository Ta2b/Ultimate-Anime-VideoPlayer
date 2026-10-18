// Logique du thread subs, sans thread : chaque message de sa boîte passe par
// handle(). Le worker possède la bibliothèque libass, le moteur de rendu, la
// piste choisie et la réserve d'atlas ; personne d'autre n'y touche.
#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <expected>
#include <memory>
#include <optional>

#include "core/types.h"
#include "subs/ass_library.h"
#include "subs/atlas_pool.h"
#include "subs/subs_messages.h"
#include "subs/subtitle_frame.h"
#include "subs/subtitle_renderer.h"
#include "subs/subtitle_track.h"

namespace ajp::subs {

class SubsWorker {
 public:
  // route : où rentrent les atlas relâchés ; pour le thread, sa propre boîte.
  // Le puits doit survivre au worker.
  static std::expected<std::unique_ptr<SubsWorker>, Refusal> create(const RendererOptions& options,
                                                                    SubtitleSink& sink, AtlasReturn route);

  SubsWorker(const SubsWorker&) = delete;
  SubsWorker& operator=(const SubsWorker&) = delete;
  ~SubsWorker() = default;

  // false : cmd::Quit, la boucle doit s'arrêter.
  bool handle(SubsInbox::Message&& message);

  // Demandes en attente d'un atlas libre.
  std::size_t waiting() const { return pending_.size(); }
  std::size_t atlases_in_flight() const { return pool_.in_flight(); }

 private:
  SubsWorker(AssLibrary library, SubtitleRenderer renderer, SubtitleSink& sink, AtlasReturn route);

  bool on(RenderRequest&& request);
  bool on(SelectTrack&& select);
  bool on(TrackPacket&& packet);
  bool on(AttachFont&& font);
  bool on(Command&& command);
  bool on(Flush&& flush);
  bool on(AtlasReturned&& returned);

  // Sert les demandes en attente, dans l'ordre, jusqu'à manquer d'atlas.
  void serve();
  void deliver(SubtitleFrame&& frame);
  void deliver_empty(const RenderRequest& request);
  void report(const Refusal& refusal);

  std::shared_ptr<spdlog::logger> log_;
  SubtitleSink& sink_;
  // Ordre de destruction : la piste, puis le moteur (qui relâche son atlas dans
  // la réserve encore vivante), puis la bibliothèque.
  AssLibrary library_;
  AtlasPool pool_;
  SubtitleRenderer renderer_;
  std::optional<SubtitleTrack> track_;
  std::uint32_t track_id_ = 0;
  std::deque<RenderRequest> pending_;  // au plus kMaxPendingRequests
  std::uint32_t epoch_ = 0;
};

}  // namespace ajp::subs
