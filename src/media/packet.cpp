#include "media/packet.h"

#include <format>
#include <utility>

extern "C" {
#include <libavcodec/packet.h>
}

namespace ajp::media {

std::span<const std::byte> Packet::data() const {
  if (!av || av->data == nullptr || av->size <= 0) return {};
  return std::as_bytes(std::span(av->data, static_cast<std::size_t>(av->size)));
}

std::expected<std::unique_ptr<PacketQueue>, Refusal> PacketQueue::create(std::size_t bound,
                                                                         std::function<void()> on_space) {
  if (bound == 0 || bound > kMaxPacketQueue) {
    return std::unexpected(
        Refusal{RefusalCode::FieldOutOfRange,
                std::format("PacketQueue : borne {} hors de [1, {}]", bound, kMaxPacketQueue)});
  }
  return std::unique_ptr<PacketQueue>(new PacketQueue(bound, std::move(on_space)));
}

PacketQueue::PacketQueue(std::size_t bound, std::function<void()> on_space)
    : bound_(bound), on_space_(std::move(on_space)) {}

bool PacketQueue::try_push(Packet&& packet) {
  std::scoped_lock lock(mutex_);
  if (queue_.size() >= bound_) {
    refused_ = true;
    return false;
  }
  queue_.push_back(std::move(packet));
  return true;
}

std::optional<Packet> PacketQueue::try_pop() {
  std::optional<Packet> packet;
  bool wake_producer = false;
  {
    std::scoped_lock lock(mutex_);
    if (queue_.empty()) return std::nullopt;
    packet = std::move(queue_.front());
    queue_.pop_front();
    wake_producer = std::exchange(refused_, false);
  }
  if (wake_producer && on_space_) on_space_();
  return packet;
}

void PacketQueue::discard_before(std::uint32_t epoch) {
  bool wake_producer = false;
  {
    std::scoped_lock lock(mutex_);
    const auto old = std::erase_if(queue_, [epoch](const Packet& p) { return p.epoch < epoch; });
    wake_producer = old > 0 && std::exchange(refused_, false);
  }
  if (wake_producer && on_space_) on_space_();
}

std::size_t PacketQueue::size() const {
  std::scoped_lock lock(mutex_);
  return queue_.size();
}

}  // namespace ajp::media
