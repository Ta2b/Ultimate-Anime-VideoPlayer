#include "media/source_loop.h"

#include <algorithm>
#include <format>
#include <utility>
#include <variant>
#include <vector>

#include "media/log.h"

namespace ajp::media {
namespace {

template <class... F>
struct Overloaded : F... {
  using F::operator()...;
};

// L'image visible à la cible est la dernière dont pts ≤ cible : on jette celles
// qui finissent avant. Sans durée connue, on jette celles qui commencent avant.
// Une image sans horodatage ne peut pas être visible à la cible : jetée aussi.
// Si la fin du flux arrive avant la cible, la dernière image jetée est livrée.
// Au plus tant d'images sans horodatage jetées de suite (environ 5 s à 24 i/s), A14-10.
constexpr int kMaxUntimedDrops = 120;
bool ends_before(const DecodedFrame& frame, std::int64_t target_us) {
  if (frame.pts_us == kNoPts) return true;
  const std::int64_t end = frame.pts_us + std::max<std::int64_t>(frame.duration_us, 1);
  return end <= target_us;
}

std::vector<PacketSink*> sinks_of(const SourceLinks& links) {
  std::vector<PacketSink*> sinks;
  sinks.reserve(links.packets.size());
  for (const PacketRoute& route : links.packets) sinks.push_back(route.sink);
  return sinks;
}

}  // namespace

SourceLoop::SourceLoop(SourceMedia media, SourceLinks links, SourceInbox& inbox)
    : media_(std::move(media)),
      links_(std::move(links)),
      inbox_(inbox),
      reserve_(sinks_of(links_), links_.reserve_bytes) {}

void SourceLoop::run(const std::stop_token& stop) {
  // L'arrêt du thread coupe aussi une lecture suspendue sur le fichier (partage réseau muet).
  const std::stop_callback interrupt(stop, [this] { media_.demuxer.interrupt(); });
  while (!stop.stop_requested() && !quit_) {
    // Le courrier d'abord, sans bloquer : ce thread est le seul à vider la boîte.
    // P4.13 : un verrou par message (try_pop), plus trois par paquet pour compter.
    while (!quit_) {
      auto message = inbox_.try_pop();
      if (!message) break;
      handle(*message);
    }
    if (quit_ || advance() == Progress::Moved) continue;
    auto message = inbox_.wait(stop);  // bloqué ou au repos : seul point d'attente
    if (!message) return;
    handle(*message);
  }
}

void SourceLoop::handle(SourceInbox::Message message) {
  std::visit(Overloaded{
                 [](const Credit&) {},  // réveil : advance() retente la livraison
                 [this](const Flush& flush) { on_flush(flush); },
                 [this](const Command& command) {
                   std::visit(Overloaded{
                                  [this](const cmd::Seek& seek) { on_seek(seek); },
                                  [this](const cmd::Quit&) { quit_ = true; },
                                  [](const auto&) {},  // pas l'affaire de la source
                              },
                              command);
                 },
             },
             message);
}

void SourceLoop::on_flush(const Flush& flush) {
  if (flush.epoch <= epoch_) {  // refusé et ignoré : la lecture continue
    links_.listener->failed(
        Refusal{.code = RefusalCode::FieldOutOfRange,
                .detail = std::format("Flush : époque {} après {}, elle doit croître", flush.epoch, epoch_),
                .recoverable = true});
    return;
  }
  // Ce qui est tenu appartient à l'époque passée : rien d'ancien ne part après.
  epoch_ = flush.epoch;
  forget_position();
  trim_before_.reset();
  awaiting_seek_ = true;
}

void SourceLoop::on_seek(const cmd::Seek& seek) {
  const bool flushed = std::exchange(awaiting_seek_, false);  // le Flush a jeté l'image et les paquets tenus
  const SeekTarget where = seek.exact ? SeekTarget::PreviousKeyframe : SeekTarget::NearestKeyframe;
  const std::uint32_t file_seeks = media_.demuxer.file_seeks();
  if (auto moved = media_.demuxer.seek(seek.target_us, media_.video_track, where); !moved) {
    Refusal refused = moved.error();
    refused.recoverable = true;  // la lecture reprend où elle était
    links_.listener->failed(refused);
    // Le fichier a bougé (une lecture a échoué après le repositionnement) : le décodeur ne tient plus sa
    // position, on revient à la dernière image tenue plutôt que de livrer des images corrompues. Refusé sans
    // toucher au fichier (cible hors bornes) : le Flush a déjà jeté l'image tenue et la réserve, la lecture
    // reprend donc elle aussi à la dernière image tenue, sans trou (A14-04). Sans Flush, ou sans image tenue
    // encore, le décodeur repart où il était.
    // B14-06 : le Flush a vidé la réserve et l'image tenue ; sans image tenue encore, la lecture reprend au
    // début plutôt que de laisser le décodeur sans recalage (des paquets audio du début seraient perdus).
    if (flushed && !last_pts_us_) last_pts_us_ = 0;
    if (media_.demuxer.file_seeks() != file_seeks || (flushed && last_pts_us_)) recover_position();
    return;
  }
  // B14-06 : la position sûre est dès maintenant la cible aboutie, pas l'image d'avant : un second seek
  // refusé avant la première image tenue y revient, et non à l'ancienne position.
  last_pts_us_ = seek.target_us;
  media_.decoder.flush();
  forget_position();
  drop_before_ = seek.exact ? std::optional(seek.target_us) : std::nullopt;
  trim_before_ = drop_before_;
  media_.decoder.skip_before(drop_before_);
  idle_ = false;
}

void SourceLoop::recover_position() {
  media_.decoder.flush();
  forget_position();
  idle_ = true;  // sans position sûre, rien ne part avant le prochain seek abouti
  trim_before_.reset();
  if (!last_pts_us_) return;
  const std::int64_t back_to = *last_pts_us_;
  if (auto back = media_.demuxer.seek(back_to, media_.video_track, SeekTarget::PreviousKeyframe); !back) {
    // Plus de position sûre : rien ne part avant un autre seek, que la session rejoue (A13-08).
    Refusal lost = back.error();
    lost.detail = std::format("position perdue : {}", lost.detail);
    lost.recoverable = true;
    links_.listener->failed(lost);
    return;
  }
  media_.decoder.flush();
  forget_position();
  drop_before_ = back_to;
  trim_before_ = back_to;
  media_.decoder.skip_before(drop_before_);
  idle_ = false;
}

// L'image et les paquets tenus viennent de l'ancienne position ou de l'ancienne
// époque : la réserve se vide avec eux, et l'amorçage recommence.
void SourceLoop::forget_position() {
  held_frame_.reset();
  last_dropped_.reset();
  reserve_.clear();
  primed_ = false;
  input_ended_ = false;
}

SourceLoop::Progress SourceLoop::advance() {
  if (awaiting_seek_) return Progress::Idle;
  // Les paquets tenus partent d'abord, sans jamais attendre ici.
  const bool side_clear = reserve_.offer(epoch_);
  if (held_frame_) {
    DecodedFrame copy = *held_frame_;  // une copie : un refus ne consomme rien
    copy.epoch = epoch_;
    if (!links_.frames->try_push(std::move(copy))) return Progress::Blocked;
    held_frame_.reset();
    primed_ = true;
    return Progress::Moved;
  }
  // Amorcée, l'époque reprend la contre-pression des files annexes, mais pas aussitôt : une file annexe
  // pleine laisse la vidéo avancer jusqu'à ce que gpu refuse une image (held_frame_ ci-dessus) ou que la
  // réserve atteigne la moitié de son plafond. Un fichier mal entrelacé (l'audio des secondes en avance) ne
  // prive plus la vidéo de paquets (A14-06).
  if (!side_clear && primed_ && reserve_.bytes() >= links_.reserve_bytes / 2) return Progress::Blocked;
  if (idle_) return Progress::Idle;
  return decode();
}

SourceLoop::Progress SourceLoop::decode() {
  auto frame = media_.decoder.receive();
  if (!frame) {
    fail(frame.error());
    return Progress::Idle;
  }
  if (*frame) {
    // A14-10 : une image sans horodatage ne peut être jugée ; passé kMaxUntimedDrops de suite, la cible ne
    // serait jamais atteinte (le reste du fichier jeté) : la lecture reprend sur elle.
    const bool untimed = (*frame)->pts_us == kNoPts;
    untimed_drops_ = untimed ? untimed_drops_ + 1 : 0;
    if (drop_before_ && ends_before(**frame, *drop_before_) && untimed_drops_ <= kMaxUntimedDrops) {
      last_dropped_ = std::move(**frame);
      return Progress::Moved;
    }
    if (drop_before_ && untimed_drops_ > kMaxUntimedDrops)
      make_logger()->warn(
          "seek exact : {} images sans horodatage de suite, la lecture reprend avant la cible",
          untimed_drops_);
    keep(std::move(**frame));
    return Progress::Moved;
  }
  if (media_.decoder.drained()) {
    if (drop_before_ && last_dropped_) {  // cible dans la dernière image : c'est elle qui s'affiche
      keep(std::move(*last_dropped_));
      return Progress::Moved;
    }
    // La fin du flux ne se signale qu'une fois tous les paquets livrés.
    if (!reserve_.empty()) return Progress::Blocked;
    idle_ = true;
    links_.listener->end_of_stream(epoch_);
    return Progress::Idle;
  }
  if (!input_ended_) {
    auto packet = media_.demuxer.read();
    if (!packet && !packet.error().recoverable) {
      fail(packet.error());
      return Progress::Idle;
    }
    if (packet && *packet) return route(std::move(**packet));
    // Fin du fichier, ou suite illisible : le décodeur rend ses dernières images, puis fin de flux.
    if (!packet) links_.listener->failed(packet.error());
    input_ended_ = true;
    if (auto ended = media_.decoder.send_end(); !ended) fail(ended.error());
    return idle_ ? Progress::Idle : Progress::Moved;
  }
  // Déjà signalée au décodeur : il finit de rendre ses images (receive() mène à drained()).
  return Progress::Moved;
}

SourceLoop::Progress SourceLoop::route(Packet packet) {
  if (packet.track == media_.video_track) {
    if (auto sent = media_.decoder.send(packet); !sent) {
      fail(sent.error());
      return Progress::Idle;
    }
    return Progress::Moved;
  }
  const auto it = std::ranges::find(links_.packets, packet.track, &PacketRoute::track);
  if (it == links_.packets.end()) return Progress::Moved;  // keep_only l'exclut déjà
  const auto index = static_cast<std::size_t>(it - links_.packets.begin());
  packet.discard_before_us = trim_before_;
  if (auto kept = reserve_.route(index, std::move(packet), epoch_); !kept) {
    fail(kept.error());
    return Progress::Idle;
  }
  return Progress::Moved;
}

void SourceLoop::keep(DecodedFrame frame) {
  drop_before_.reset();
  media_.decoder.skip_before(std::nullopt);
  untimed_drops_ = 0;
  last_dropped_.reset();
  held_frame_ = std::move(frame);
  if (held_frame_->pts_us != kNoPts) last_pts_us_ = held_frame_->pts_us;
  held_frame_->id = SourceId{next_id_++};
}

void SourceLoop::fail(const Refusal& refusal) {
  idle_ = true;
  links_.listener->failed(refusal);
}

}  // namespace ajp::media
