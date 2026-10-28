// Consommateurs de test du thread source : un FrameSink borné qui joue gpu
// (il rend un Credit à chaque image retirée) et un SourceListener qui note tout.
#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <vector>

#include "core/frame.h"
#include "media/source.h"

namespace ajp::media::test {

using namespace std::chrono_literals;
inline constexpr auto kTimeout = 10s;

class TestSink final : public FrameSink {
 public:
  // capacity : borne comme celle de la boîte de gpu ; refuse_first : refus
  // inconditionnels avant d'accepter quoi que ce soit.
  explicit TestSink(std::size_t capacity, int refuse_first = 0)
      : capacity_(capacity), refuse_first_(refuse_first) {}

  bool try_push(DecodedFrame&& frame) override {
    {
      std::scoped_lock lock(mutex_);
      ++attempts_;
      if (refuse_first_ > 0 || queue_.size() >= capacity_) {
        if (refuse_first_ > 0) --refuse_first_;
        ++refusals_;
        changed_.notify_all();
        return false;
      }
      queue_.push_back(std::move(frame));
    }
    changed_.notify_all();
    return true;
  }

  // Retire une image comme gpu, puis rend un Credit à la source.
  std::optional<DecodedFrame> pop(SourceInbox& inbox, std::chrono::milliseconds timeout = kTimeout) {
    std::unique_lock lock(mutex_);
    if (!changed_.wait_for(lock, timeout, [&] { return !queue_.empty(); })) return std::nullopt;
    DecodedFrame frame = std::move(queue_.front());
    queue_.pop_front();
    lock.unlock();
    inbox.try_push(Credit{});
    return frame;
  }

  bool wait_refusals(int count) {
    std::unique_lock lock(mutex_);
    return changed_.wait_for(lock, kTimeout, [&] { return refusals_ >= count; });
  }

  int attempts() const {
    std::scoped_lock lock(mutex_);
    return attempts_;
  }
  std::size_t queued() const {
    std::scoped_lock lock(mutex_);
    return queue_.size();
  }

 private:
  mutable std::mutex mutex_;
  std::condition_variable changed_;
  std::deque<DecodedFrame> queue_;
  std::size_t capacity_;
  int refuse_first_;
  int attempts_ = 0;
  int refusals_ = 0;
};

class TestListener final : public SourceListener {
 public:
  void end_of_stream(std::uint32_t epoch) override {
    {
      std::scoped_lock lock(mutex_);
      ends_.push_back(epoch);
    }
    changed_.notify_all();
  }
  void failed(const Refusal& refusal) override {
    {
      std::scoped_lock lock(mutex_);
      failures_.push_back(refusal);
    }
    changed_.notify_all();
  }

  bool wait_end(std::uint32_t epoch) {
    std::unique_lock lock(mutex_);
    return changed_.wait_for(lock, kTimeout, [&] { return std::ranges::find(ends_, epoch) != ends_.end(); });
  }
  std::optional<Refusal> wait_failure() {
    std::unique_lock lock(mutex_);
    if (!changed_.wait_for(lock, kTimeout, [&] { return !failures_.empty(); })) return std::nullopt;
    return failures_.front();
  }
  std::size_t failure_count() const {
    std::scoped_lock lock(mutex_);
    return failures_.size();
  }

 private:
  mutable std::mutex mutex_;
  std::condition_variable changed_;
  std::vector<std::uint32_t> ends_;
  std::vector<Refusal> failures_;
};

}  // namespace ajp::media::test
