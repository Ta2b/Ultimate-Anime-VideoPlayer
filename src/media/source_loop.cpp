#include "media/source_loop.h"

#include <algorithm>
#include <format>
#include <utility>
#include <variant>
#include <vector>

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
  while (!stop.stop_requested() && !quit_) {
    // Le courrier d'abord, sans bloquer : ce thread est le seul à vider la boîte.
    while (has_mail() && !quit_) {
      auto message = inbox_.wait(stop);
      if (!message) return;
      handle(*message);
    }
    if (quit_ || advance() == Progress::Moved) continue;
    auto message = inbox_.wait(stop);  // bloqué ou au repos : seul point d'attente
    if (!message) return;
    handle(*message);
  }
}

bool SourceLoop::has_mail() const {
  return inbox_.pending<Credit>() + inbox_.pending<Command>() + inbox_.pending<Flush>() > 0;
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
        Refusal{RefusalCode::FieldOutOfRange,
                std::format("Flush : époque {} après {}, elle doit croître", flush.epoch, epoch_)});
    return;
  }
  // Ce qui est tenu appartient à l'époque passée : rien d'ancien ne part après.
  epoch_ = flush.epoch;
  forget_position();
  awaiting_seek_ = true;
}

void SourceLoop::on_seek(const cmd::Seek& seek) {
  awaiting_seek_ = false;
  const SeekTarget where = seek.exact ? SeekTarget::PreviousKeyframe : SeekTarget::NearestKeyframe;
  const std::uint32_t file_seeks = media_.demuxer.file_seeks();
  if (auto moved = media_.demuxer.seek(seek.target_us, media_.video_track, where); !moved) {
    links_.listener->failed(moved.error());
    // Refusé sans toucher au fichier (cible hors bornes) : la lecture reprend où elle était. Le
    // fichier a bougé (une lecture a échoué après le repositionnement) : le décodeur ne tient plus
    // sa position, on revient à la dernière image tenue plutôt que de livrer des images corrompues.
    if (media_.demuxer.file_seeks() != file_seeks) recover_position();
    return;
  }
  media_.decoder.flush();
  forget_position();
  drop_before_ = seek.exact ? std::optional(seek.target_us) : std::nullopt;
  idle_ = false;
}

void SourceLoop::recover_position() {
  media_.decoder.flush();
  forget_position();
  idle_ = true;  // sans position sûre, rien ne part avant le prochain seek abouti
  if (!last_pts_us_) return;
  const std::int64_t back_to = *last_pts_us_;
  if (auto back = media_.demuxer.seek(back_to, media_.video_track, SeekTarget::PreviousKeyframe); !back)
    return;
  media_.decoder.flush();
  forget_position();
  drop_before_ = back_to;
  idle_ = false;
}

// L'image et les paquets tenus viennent de l'ancienne position ou de l'ancienne
// époque : la réserve se vide avec eux, et l'amorçage recommence.
void SourceLoop::forget_position() {
  held_frame_.reset();
  last_dropped_.reset();
  reserve_.clear();
  primed_ = false;
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
  // Amorcée, l'époque reprend la contre-pression des files annexes.
  if (!side_clear && primed_) return Progress::Blocked;
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
    if (drop_before_ && ends_before(**frame, *drop_before_)) {
      last_dropped_ = std::move(**frame);
      return Progress::Moved;
    }
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
  auto packet = media_.demuxer.read();
  if (!packet) {
    fail(packet.error());
    return Progress::Idle;
  }
  if (!*packet) {
    if (auto ended = media_.decoder.send_end(); !ended) fail(ended.error());
    return idle_ ? Progress::Idle : Progress::Moved;
  }
  return route(std::move(**packet));
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
  if (auto kept = reserve_.route(index, std::move(packet), epoch_); !kept) {
    fail(kept.error());
    return Progress::Idle;
  }
  return Progress::Moved;
}

void SourceLoop::keep(DecodedFrame frame) {
  drop_before_.reset();
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
