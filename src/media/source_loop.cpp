#include "media/source_loop.h"

#include <algorithm>
#include <format>
#include <utility>
#include <variant>

namespace ajp::media {
namespace {

template <class... F>
struct Overloaded : F... {
  using F::operator()...;
};

// L'image visible à la cible est la dernière dont pts ≤ cible : on jette celles
// qui finissent avant. Sans durée connue, on jette celles qui commencent avant.
bool ends_before(const DecodedFrame& frame, std::int64_t target_us) {
  const std::int64_t end = frame.pts_us + std::max<std::int64_t>(frame.duration_us, 1);
  return end <= target_us;
}

}  // namespace

SourceLoop::SourceLoop(SourceMedia media, SourceLinks links, SourceInbox& inbox)
    : media_(std::move(media)), links_(std::move(links)), inbox_(inbox) {}

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
  held_frame_.reset();
  held_packet_.reset();
}

void SourceLoop::on_seek(const cmd::Seek& seek) {
  const SeekTarget where = seek.exact ? SeekTarget::PreviousKeyframe : SeekTarget::NearestKeyframe;
  if (auto moved = media_.demuxer.seek(seek.target_us, media_.video_track, where); !moved) {
    links_.listener->failed(moved.error());  // la position n'a pas changé
    return;
  }
  media_.decoder.flush();
  held_frame_.reset();
  held_packet_.reset();
  drop_before_ = seek.exact ? std::optional(seek.target_us) : std::nullopt;
  idle_ = false;
}

SourceLoop::Progress SourceLoop::advance() {
  if (held_frame_) {
    DecodedFrame copy = *held_frame_;  // une copie : un refus ne consomme rien
    copy.epoch = epoch_;
    if (!links_.frames->try_push(std::move(copy))) return Progress::Blocked;
    held_frame_.reset();
    return Progress::Moved;
  }
  if (held_packet_) {
    Packet copy = held_packet_->packet;
    copy.epoch = epoch_;
    if (!held_packet_->sink->try_push(std::move(copy))) return Progress::Blocked;
    held_packet_.reset();
    return Progress::Moved;
  }
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
    if (drop_before_ && ends_before(**frame, *drop_before_)) return Progress::Moved;
    drop_before_.reset();
    held_frame_ = std::move(**frame);
    held_frame_->id = SourceId{next_id_++};
    return Progress::Moved;
  }
  if (media_.decoder.drained()) {
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
  if (it != links_.packets.end()) held_packet_ = HeldPacket{it->sink, std::move(packet)};
  return Progress::Moved;
}

void SourceLoop::fail(const Refusal& refusal) {
  idle_ = true;
  links_.listener->failed(refusal);
}

}  // namespace ajp::media
