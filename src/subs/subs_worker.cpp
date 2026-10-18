#include "subs/subs_worker.h"

#include <format>
#include <utility>
#include <variant>

#include <spdlog/logger.h>

#include "subs/log.h"

namespace ajp::subs {
namespace {

using OpenedTrack = std::expected<std::optional<SubtitleTrack>, Refusal>;

template <class T>
OpenedTrack opened(std::expected<T, Refusal>&& track) {
  if (!track) return std::unexpected(std::move(track.error()));
  return std::optional<SubtitleTrack>(std::move(*track));
}

OpenedTrack open_track(const AssLibrary& library, const TrackSource& source) {
  if (std::holds_alternative<NoTrack>(source)) return std::optional<SubtitleTrack>{};
  if (const auto* ass = std::get_if<AssStream>(&source)) {
    return opened(SubtitleTrack::from_header(library, ass->header));
  }
  if (std::holds_alternative<TextStream>(source)) return opened(SubtitleTrack::text(library));
  return opened(SubtitleTrack::load(library, std::get<ExternalFile>(source).path));
}

}  // namespace

SubsWorker::SubsWorker(AssLibrary library, SubtitleRenderer renderer, SubtitleSink& sink, AtlasReturn route)
    : log_(make_logger()),
      sink_(sink),
      library_(std::move(library)),
      pool_(std::move(route)),
      renderer_(std::move(renderer)) {}

std::expected<std::unique_ptr<SubsWorker>, Refusal> SubsWorker::create(const RendererOptions& options,
                                                                       SubtitleSink& sink,
                                                                       AtlasReturn route) {
  auto library = AssLibrary::create();
  if (!library) return std::unexpected(std::move(library.error()));
  auto renderer = SubtitleRenderer::create(*library, options);
  if (!renderer) return std::unexpected(std::move(renderer.error()));
  return std::unique_ptr<SubsWorker>(
      new SubsWorker(std::move(*library), std::move(*renderer), sink, std::move(route)));
}

bool SubsWorker::handle(SubsInbox::Message&& message) {
  return std::visit([this](auto&& m) { return on(std::forward<decltype(m)>(m)); }, std::move(message));
}

bool SubsWorker::on(RenderRequest&& request) {
  if (request.epoch < epoch_) return true;  // époque passée : render la jette aussi
  if (pending_.size() >= kMaxPendingRequests) {
    report({RefusalCode::SubtitleBacklog,
            std::format("demande à {} µs : {} déjà en attente d'un atlas", request.pts_us, pending_.size())});
    deliver_empty(request);
    return true;
  }
  pending_.push_back(request);
  if (!starved_) serve();  // sinon la tête attend un atlas : rien ne change avant son retour
  return true;
}

bool SubsWorker::on(SelectTrack&& select) {
  track_.reset();
  renderer_.forget();
  track_id_ = select.id;
  OpenedTrack track = open_track(library_, select.source);
  if (track) {
    track_ = std::move(*track);
    if (track_ && track_->embeds_fonts()) renderer_.fonts_changed();
  } else {
    report(track.error());
  }
  serve();
  return true;
}

bool SubsWorker::on(TrackPacket&& packet) {
  if (!track_ || packet.track != track_id_) {
    log_->debug("paquet de la piste {} ignoré : piste choisie {}", packet.track, track_id_);
    return true;
  }
  if (auto added = track_->add_packet(packet.packet); !added) report(added.error());
  return true;
}

bool SubsWorker::on(AttachFont&& font) {
  if (auto added = library_.add_font(font.name, font.data); !added) {
    report(added.error());
  } else {
    renderer_.fonts_changed();
  }
  return true;
}

bool SubsWorker::on(Command&& command) {
  return !std::holds_alternative<cmd::Quit>(command);
}

bool SubsWorker::on(Flush&& flush) {
  if (flush.epoch <= epoch_) {
    report({RefusalCode::FieldOutOfRange, std::format("Flush d'époque {} après {}", flush.epoch, epoch_)});
    return true;
  }
  epoch_ = flush.epoch;
  std::erase_if(pending_, [this](const RenderRequest& request) { return request.epoch < epoch_; });
  serve();
  return true;
}

bool SubsWorker::on(AtlasReturned&& returned) {
  pool_.give_back(std::move(returned.atlas));
  serve();
  return true;
}

void SubsWorker::serve() {
  starved_ = false;
  while (!pending_.empty()) {
    const RenderRequest request = pending_.front();
    if (!track_) {
      deliver_empty(request);
    } else {
      auto outcome = renderer_.render(*track_, request, pool_);
      if (!outcome) {
        report(outcome.error());
        deliver_empty(request);
      } else if (std::holds_alternative<WaitForAtlas>(*outcome)) {
        starved_ = true;  // la demande reste en tête jusqu'au prochain AtlasReturned
        return;
      } else {
        deliver(std::get<SubtitleFrame>(std::move(*outcome)));
      }
    }
    pending_.pop_front();
  }
}

void SubsWorker::deliver(SubtitleFrame&& frame) {
  const std::int64_t pts_us = frame.pts_us;
  if (!sink_.try_push(std::move(frame))) {
    log_->error("réponse à {} µs perdue : la boîte de render est pleine", pts_us);
  }
}

void SubsWorker::deliver_empty(const RenderRequest& request) {
  deliver({.pts_us = request.pts_us,
           .epoch = request.epoch,
           .width = request.target.width,
           .height = request.target.height});
}

void SubsWorker::report(const Refusal& refusal) {
  log_->warn("{} : {}", name(refusal.code), refusal.detail);
  sink_.refused(refusal);
}

}  // namespace ajp::subs
