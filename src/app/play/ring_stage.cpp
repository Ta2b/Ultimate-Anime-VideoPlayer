#include "app/play/ring_stage.h"

#include <format>
#include <type_traits>
#include <utility>
#include <variant>

#include "core/thread.h"

namespace ajp::app::play {

RingStage::RingStage(VideoStageInit init, std::uint32_t slots, std::uint32_t generation)
    : init_(std::move(init)),
      generation_(generation),
      inbox_(StageInbox::Bounds{kMaxFrames, slots, 16, 4}),
      released_(slots, 0),
      free_(slots, true) {}

RingStage::~RingStage() {
  stop();
}

void RingStage::stop() {
  thread_.request_stop();
  if (thread_.joinable()) thread_.join();
}

std::expected<void, Refusal> RingStage::start(std::string_view name) {
  auto thread =
      start_thread(name, init_.report,
                   [this](const std::stop_token& stop, const ThreadReport& report) { run(stop, report); });
  if (!thread) return std::unexpected(std::move(thread.error()));
  thread_ = std::move(*thread);
  return {};
}

bool RingStage::try_push(DecodedFrame&& frame) {
  if (in_flight_.fetch_add(1) >= kMaxFrames || !inbox_.try_push(std::move(frame))) {
    --in_flight_;
    return false;
  }
  return true;
}

void RingStage::run(const std::stop_token& stop, const ThreadReport& report) {
  while (auto message = inbox_.wait(stop)) {
    bool quit = false;
    std::visit(
        [&](auto& m) {
          using M = std::decay_t<decltype(m)>;
          if constexpr (std::is_same_v<M, DecodedFrame>) {
            pending_.push_back(std::move(m));
          } else if constexpr (std::is_same_v<M, SlotFree>) {
            // Un SlotFree d'un autre ring, ou hors des slots, est ignoré (core/ring.h).
            if (m.generation == generation_ && m.slot < free_.size()) {
              free_[m.slot] = true;
              released_[m.slot] = m.semaphore_value;
            }
          } else if constexpr (std::is_same_v<M, Flush>) {
            epoch_ = m.epoch;
          } else {
            quit = std::holds_alternative<cmd::Quit>(m);
          }
        },
        *message);
    if (quit) return;
    emit(report);
  }
}

void RingStage::emit(const ThreadReport& report) {
  while (!pending_.empty()) {
    if (pending_.front().epoch < epoch_ || failed_) {  // d'avant le Flush, ou étage en échec : jetée
      pending_.pop_front();
      --in_flight_;
      init_.credit();
      continue;
    }
    std::uint32_t slot = 0;
    while (slot < free_.size() && !free_[slot]) ++slot;
    if (slot == free_.size()) return;  // attend un SlotFree
    const DecodedFrame& frame = pending_.front();
    const std::uint64_t ready = released_[slot] + 1;
    if (auto produced = produce(frame, slot, released_[slot], ready); !produced) {
      failed_ = true;  // une image refusée arrête la vidéo : l'OSD le dit, main arrête la lecture
      report(Refusal{RefusalCode::ThreadFailed,
                     std::format("{} {}", name(produced.error().code), produced.error().detail)});
      continue;
    }
    const RingFrame ring{.slot = slot,
                         .generation = generation_,
                         .pts_us = frame.pts_us,
                         .left = frame.id,
                         .right = frame.id,
                         .epoch = frame.epoch,
                         .ready_value = ready};
    free_[slot] = false;
    released_[slot] = ready;  // le slot est écrit : sa prochaine attente part de ready + 1
    if (!init_.ring->try_push(ring)) {
      // Boîte de main pleine : contraire au contrat (borne = slots). Le slot est perdu pour
      // ce ring ; on le dit plutôt que d'attendre.
      report(Refusal{RefusalCode::SequenceOrder, "boîte de main pleine : RingFrame perdu"});
    }
    pending_.pop_front();
    --in_flight_;
    init_.credit();
  }
}

}  // namespace ajp::app::play
