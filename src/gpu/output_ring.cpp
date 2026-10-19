#include "gpu/output_ring.h"

#include <format>
#include <string>
#include <utility>

namespace ajp::gpu {
namespace {

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

// La première valeur impaire au-delà de wait : main n'attend que des valeurs impaires.
constexpr std::uint64_t next_odd(std::uint64_t wait) {
  return wait % 2 == 0 ? wait + 1 : wait + 2;
}

}  // namespace

std::expected<OutputRing, Refusal> OutputRing::create(std::span<const RingSlotDesc> slots) {
  if (slots.empty() || slots.size() > kMaxRingSlots) {
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("{} slots, de 1 à {}", slots.size(), kMaxRingSlots));
  }
  OutputRing ring;
  ring.size_ = slots.size();
  ring.generation_ = slots.front().generation;
  for (std::size_t i = 0; i < slots.size(); ++i) {
    const RingSlotDesc& desc = slots[i];
    const RingSlotDesc& first = slots.front();
    if (desc.index != i || desc.generation != ring.generation_ || desc.width <= 0 || desc.height <= 0 ||
        desc.width != first.width || desc.height != first.height || desc.format != first.format) {
      return refuse(RefusalCode::FieldOutOfRange,
                    std::format("slot {} : indice {}, génération {} (attendue {}), {}×{}", i, desc.index,
                                desc.generation, ring.generation_, desc.width, desc.height));
    }
    ring.slots_[i].desc = desc;
  }
  return ring;
}

bool OutputRing::fits(const core::Plan& plan) const {
  const RingSlotDesc& slot = slots_.front().desc;
  return slot.width == plan.output_width && slot.height == plan.output_height && slot.format == plan.ring;
}

std::size_t OutputRing::count(SlotState state) const {
  std::size_t n = 0;
  for (std::size_t i = 0; i < size_; ++i) n += slots_[i].state == state ? 1U : 0U;
  return n;
}

std::optional<OutputRing::Lease> OutputRing::acquire() {
  for (std::size_t tried = 0; tried < size_; ++tried) {
    const std::size_t index = (next_ + tried) % size_;
    Slot& slot = slots_[index];
    if (slot.state != SlotState::Free) continue;
    next_ = (index + 1) % size_;
    slot.state = SlotState::Writing;
    return Lease{.slot = static_cast<std::uint32_t>(index),
                 .desc = &slot.desc,
                 .sync = {.wait_value = slot.wait_value, .signal_value = next_odd(slot.wait_value)}};
  }
  return std::nullopt;
}

void OutputRing::abandon(std::uint32_t slot) {
  if (slot < size_ && slots_[slot].state == SlotState::Writing) slots_[slot].state = SlotState::Free;
}

void OutputRing::commit(std::uint32_t slot, RingFrame frame) {
  if (slot >= size_ || slots_[slot].state != SlotState::Writing) return;
  Slot& s = slots_[slot];
  s.ready_value = next_odd(s.wait_value);
  frame.slot = slot;
  frame.generation = generation_;
  frame.ready_value = s.ready_value;
  s.frame = frame;
  s.state = SlotState::Ready;
  ready_[(ready_head_ + ready_count_) % kMaxRingSlots] = slot;
  ++ready_count_;
}

std::size_t OutputRing::deliver(RingFrameSink& sink) {
  std::size_t delivered = 0;
  while (ready_count_ > 0) {
    Slot& slot = slots_[ready_[ready_head_]];
    if (!sink.try_push(RingFrame{slot.frame})) break;
    slot.state = SlotState::Shown;
    ready_head_ = (ready_head_ + 1) % kMaxRingSlots;
    --ready_count_;
    ++delivered;
  }
  return delivered;
}

std::expected<Release, Refusal> OutputRing::release(const SlotFree& message) {
  if (message.generation != generation_) return Release::OtherGeneration;
  if (message.slot >= size_) {
    return refuse(RefusalCode::SequenceOrder,
                  std::format("SlotFree : slot {} hors du ring de {}", message.slot, size_));
  }
  Slot& slot = slots_[message.slot];
  if (slot.state != SlotState::Shown || message.semaphore_value != slot.ready_value + 1) {
    return refuse(RefusalCode::SequenceOrder,
                  std::format("SlotFree : slot {} à la valeur {}, attendu chez main à {}", message.slot,
                              message.semaphore_value, slot.ready_value + 1));
  }
  slot.wait_value = message.semaphore_value;
  slot.state = SlotState::Free;
  return Release::Freed;
}

std::size_t OutputRing::drop_before(std::uint32_t epoch) {
  std::size_t dropped = 0;
  std::size_t kept = 0;
  for (std::size_t i = 0; i < ready_count_; ++i) {
    const std::uint32_t index = ready_[(ready_head_ + i) % kMaxRingSlots];
    Slot& slot = slots_[index];
    if (slot.frame.epoch < epoch) {
      // L'écriture a eu lieu : la prochaine attend sa propre valeur, sans passer par main.
      slot.wait_value = slot.ready_value;
      slot.state = SlotState::Free;
      ++dropped;
    } else {
      ready_[(ready_head_ + kept) % kMaxRingSlots] = index;
      ++kept;
    }
  }
  ready_count_ = kept;
  return dropped;
}

}  // namespace ajp::gpu
