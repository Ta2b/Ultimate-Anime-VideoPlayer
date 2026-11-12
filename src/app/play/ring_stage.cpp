#include "app/play/ring_stage.h"

#include <algorithm>
#include <chrono>
#include <format>
#include <type_traits>
#include <utility>
#include <variant>

#include "app/play/links.h"
#include "core/clock.h"
#include "core/thread.h"
#include "core/thread_role.h"

namespace ajp::app::play {

RingStage::RingStage(VideoStageInit init, std::uint32_t slots, std::uint32_t generation)
    : init_(std::move(init)),
      generation_(generation),
      inbox_(StageInbox::Bounds{kMaxFrames, slots, 16, 4, 4}),
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
      start_thread(name, init_.report, [this](const std::stop_token& stop, const ThreadReport& report) {
        set_thread_role(ThreadRole::Playback);
        run(stop, report);
      });
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
  while (true) {
    // En attente d'un slot : réveillé toutes les kHealthAfterUs pour demander à la sous-classe si l'appareil
    // répond encore ; sinon, attente simple.
    auto message = slot_wait_since_us_ == 0
                       ? inbox_.wait(stop)
                       : inbox_.wait_for(stop, std::chrono::microseconds(kHealthAfterUs));
    if (!message) {
      if (stop.stop_requested()) return;
      if (slot_wait_since_us_ != 0 && !failed_ && host_now_us() - slot_wait_since_us_ >= kHealthAfterUs) {
        if (auto healthy = health(); !healthy) {
          fail(report, healthy.error());
          slot_wait_since_us_ = 0;
          emit(report);  // l'étage en échec jette ce qui attend
        }
      }
      continue;
    }
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
              ended_ = false;
            } else if constexpr (std::is_same_v<M, StreamEnd>) {
              ended_ = m.epoch == epoch_;
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
  if (refusal.code == RefusalCode::SourceResized) {  // main rouvre la chaîne à la nouvelle taille
    report(refusal);
    return;
  }
  // Fatal pour main par sa clé, le code gardé : DeviceLost dit « relancer » et déclenche la relance (A31-01).
  Refusal fatal = refusal;
  fatal.key = std::string(kFatalStageKey);
  report(fatal);
}

RingFrame RingStage::ring_frame(const Output& produced, std::uint32_t epoch, std::uint32_t slot,
                                std::uint64_t ready, std::int64_t submitted_us) const {
  return RingFrame{.slot = slot,
                   .generation = generation_,
                   .pts_us = produced.pts_us,
                   .left = produced.left,
                   .right = produced.right,
                   .repeated = produced.repeated,
                   .uninterpolated = produced.uninterpolated || gap_,
                   .planned_gap_us = planned_gap_us(front_done_),
                   .debanded = produced.debanded,
                   .epoch = epoch,
                   .ready_value = ready,
                   .submitted_us = submitted_us};
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
    if (front_outputs_ == 0 && wants_next() && pending_.size() < 2 &&
        !(ended_.load() && in_flight_.load() <= pending_.size()))
      return;  // attend la suivante
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
      const Fate verdict = fate(frame, front_done_);
      if (verdict == Fate::Same) {
        skip(frame, front_done_);
        ++front_done_;
        gap_ = true;
        continue;
      }
      if (verdict == Fate::Late) {
        if (auto ok = late(frame, front_done_); !ok) {
          fail(report, ok.error());
          break;
        }
        ++front_done_;
        continue;
      }
      // Tour à tour depuis le slot qui suit le dernier écrit : un slot rendu tôt n'est pas repris avant les
      // autres, et l'ordre ne dépend plus de l'instant où main rend les siens.
      const auto count = static_cast<std::uint32_t>(free_.size());
      std::uint32_t slot = count;
      for (std::uint32_t k = 0; k < count && slot == count; ++k) {
        if (free_[(next_slot_ + k) % count]) slot = (next_slot_ + k) % count;
      }
      if (slot == count) {  // attend un SlotFree
        if (slot_wait_since_us_ == 0) slot_wait_since_us_ = host_now_us();
        return;
      }
      slot_wait_since_us_ = 0;
      const std::uint64_t ready = released_[slot] + 1;
      const std::int64_t submitted = host_now_us();
      auto produced = produce(frame, front_done_, slot, released_[slot], ready);
      if (!produced) {
        fail(report, produced.error());
        break;
      }
      const RingFrame ring = ring_frame(*produced, frame.epoch, slot, ready, submitted);
      gap_ = false;
      free_[slot] = false;
      next_slot_ = (slot + 1) % count;
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
  slot_wait_since_us_ = 0;  // plus rien n'attend de slot
}

}  // namespace ajp::app::play
