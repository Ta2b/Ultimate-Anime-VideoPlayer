#include "media/packet.h"

#include <algorithm>
#include <format>
#include <utility>

extern "C" {
#include <libavcodec/packet.h>
}

namespace ajp::media {
namespace {

// Une durée inconnue ou négative ne compte pas : la borne en octets reste.
std::int64_t counted_duration(const Packet& packet) {
  return std::max<std::int64_t>(packet.duration_us, 0);
}

}  // namespace

std::span<const std::byte> Packet::data() const {
  if (!av || av->data == nullptr || av->size <= 0) return {};
  return std::as_bytes(std::span(av->data, static_cast<std::size_t>(av->size)));
}

std::size_t Packet::footprint() const {
  return data().size() + kPacketOverhead;
}

std::expected<std::unique_ptr<PacketQueue>, Refusal> PacketQueue::create(const PacketQueueLimits& limits,
                                                                         std::function<void()> on_space) {
  const std::optional<std::int64_t>& duration = limits.max_duration_us;
  if (duration && (*duration <= 0 || *duration > kMaxQueueDurationUs)) {
    return std::unexpected(
        Refusal{RefusalCode::FieldOutOfRange,
                std::format("PacketQueue : durée {} µs hors de ]0, {}]", *duration, kMaxQueueDurationUs)});
  }
  if (limits.max_bytes < kMinQueueBytes || limits.max_bytes > kMaxQueueBytes) {
    return std::unexpected(
        Refusal{RefusalCode::FieldOutOfRange, std::format("PacketQueue : {} octets hors de [{}, {}]",
                                                          limits.max_bytes, kMinQueueBytes, kMaxQueueBytes)});
  }
  return std::unique_ptr<PacketQueue>(new PacketQueue(limits, std::move(on_space)));
}

PacketQueue::PacketQueue(const PacketQueueLimits& limits, std::function<void()> on_space)
    : limits_(limits), on_space_(std::move(on_space)) {}

bool PacketQueue::try_push(Packet&& packet) {
  std::scoped_lock lock(mutex_);
  const std::int64_t duration = counted_duration(packet);
  const std::size_t bytes = packet.footprint();
  const bool too_long = limits_.max_duration_us && duration_us_ + duration > *limits_.max_duration_us;
  const bool too_big = bytes_ + bytes > limits_.max_bytes;
  if (!queue_.empty() && (too_long || too_big)) {
    refused_ = true;
    return false;
  }
  duration_us_ += duration;
  bytes_ += bytes;
  queue_.push_back(std::move(packet));
  return true;
}

void PacketQueue::forget(const Packet& packet) {
  duration_us_ -= counted_duration(packet);
  bytes_ -= packet.footprint();
}

std::optional<Packet> PacketQueue::try_pop() {
  std::optional<Packet> packet;
  bool wake_producer = false;
  {
    std::scoped_lock lock(mutex_);
    if (queue_.empty()) return std::nullopt;
    packet = std::move(queue_.front());
    queue_.pop_front();
    forget(*packet);
    wake_producer = std::exchange(refused_, false);
  }
  if (wake_producer && on_space_) on_space_();
  return packet;
}

void PacketQueue::discard_before(std::uint32_t epoch) {
  bool wake_producer = false;
  {
    std::scoped_lock lock(mutex_);
    const auto old = std::erase_if(queue_, [this, epoch](const Packet& p) {
      if (p.epoch >= epoch) return false;
      forget(p);
      return true;
    });
    wake_producer = old > 0 && std::exchange(refused_, false);
  }
  if (wake_producer && on_space_) on_space_();
}

std::size_t PacketQueue::size() const {
  std::scoped_lock lock(mutex_);
  return queue_.size();
}

std::int64_t PacketQueue::duration_us() const {
  std::scoped_lock lock(mutex_);
  return duration_us_;
}

std::size_t PacketQueue::bytes() const {
  std::scoped_lock lock(mutex_);
  return bytes_;
}

}  // namespace ajp::media
