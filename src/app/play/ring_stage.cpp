#include "app/play/ring_stage.h"

#include <algorithm>
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
    const auto take = [&](StageInbox::Message& taken) {
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
              quit = quit || std::holds_alternative<cmd::Quit>(m);
            }
          },
          taken);
    };
    take(*message);
    // Tout ce qui attend déjà avant de produire : l'image suivante arrivée derrière la tête est
    // alors dans pending_, et outputs() peut l'ingérer et la mesurer en avance (next()).
    while (!quit) {
      auto more = inbox_.try_pop();
      if (!more) break;
      take(*more);
    }
    if (quit) return;
    emit(report);
  }
}

void RingStage::fail(const ThreadReport& report, const Refusal& refusal) {
  failed_ = true;  // une image refusée arrête la vidéo : l'OSD le dit, main arrête la lecture
  report(Refusal{RefusalCode::ThreadFailed, std::format("{} {}", name(refusal.code), refusal.detail)});
}

void RingStage::emit(const ThreadReport& report) {
  while (!pending_.empty()) {
    if (pending_.front().epoch < epoch_ || failed_) {  // d'avant le Flush, ou étage en échec : jetée
      pending_.pop_front();
      front_outputs_ = 0;
      --in_flight_;
      init_.credit();
      continue;
    }
    const DecodedFrame& frame = pending_.front();
    if (front_outputs_ == 0 && wants_next() && pending_.size() < 2) return;  // attend la suivante
    if (front_outputs_ == 0 && pending_.size() > 1 && pending_[1].epoch == frame.epoch &&
        passed(frame, pending_[1])) {
      pending_.pop_front();
      --in_flight_;
      init_.credit();
      continue;
    }
    if (front_outputs_ == 0) {
      auto count = outputs(frame);
      if (count && *count > kMaxOutputs) {
        count = std::unexpected(
            Refusal{RefusalCode::FieldOutOfRange,
                    std::format("{} sorties pour une image, au plus {}", *count, kMaxOutputs)});
      }
      if (!count) {
        fail(report, count.error());
        continue;
      }
      front_outputs_ = std::max(*count, 1U);
      front_done_ = 0;
    }
    while (front_done_ < front_outputs_) {
      std::uint32_t slot = 0;
      while (slot < free_.size() && !free_[slot]) ++slot;
      if (slot == free_.size()) return;  // attend un SlotFree
      const std::uint64_t ready = released_[slot] + 1;
      auto produced = produce(frame, front_done_, slot, released_[slot], ready);
      if (!produced) {
        fail(report, produced.error());
        break;
      }
      const RingFrame ring{.slot = slot,
                           .generation = generation_,
                           .pts_us = produced->pts_us,
                           .left = produced->left,
                           .right = produced->right,
                           .uninterpolated = produced->uninterpolated,
                           .epoch = frame.epoch,
                           .ready_value = ready};
      free_[slot] = false;
      released_[slot] = ready;  // le slot est écrit : sa prochaine attente part de ready + 1
      ++front_done_;
      if (!init_.ring->try_push(ring)) {
        // Boîte de main pleine : contraire au contrat (borne = slots). Le slot ne reviendra
        // jamais ; l'étage s'arrête en le disant plutôt que de se figer après quelques pertes.
        fail(report, Refusal{RefusalCode::SequenceOrder, "boîte de main pleine : RingFrame perdu"});
        break;
      }
    }
    if (failed_) continue;
    pending_.pop_front();
    front_outputs_ = 0;
    --in_flight_;
    init_.credit();
  }
}

}  // namespace ajp::app::play
