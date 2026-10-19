#include "gpu/gpu_thread.h"

#include <format>
#include <utility>
#include <variant>

namespace ajp::gpu {
namespace {

template <class... F>
struct Visitor : F... {
  using F::operator()...;
};

}  // namespace

GpuThread::GpuThread(LoadedBackend backend, GpuLinks links, GpuBounds bounds)
    : backend_(std::move(backend)),
      links_(std::move(links)),
      bounds_(bounds),
      inbox_({bounds.frames, bounds.slot_frees, bounds.rings, bounds.installs, bounds.commands,
              bounds.flushes}) {}

std::expected<std::unique_ptr<GpuThread>, Refusal> GpuThread::start(LoadedBackend backend, GpuLinks links,
                                                                    GpuBounds bounds) {
  if (links.frames == nullptr || !links.report) {
    return std::unexpected(
        Refusal{RefusalCode::FieldMissing, "thread gpu : puits des images ou rapporteur absent"});
  }
  if (bounds.frames == 0 || bounds.frames > GpuWorker::kMaxBacklog || bounds.slot_frees == 0 ||
      bounds.rings == 0 || bounds.installs == 0 || bounds.commands == 0 || bounds.flushes == 0) {
    return std::unexpected(Refusal{RefusalCode::FieldOutOfRange,
                                   std::format("thread gpu : bornes non nulles, images de 1 à {} (reçu {})",
                                               GpuWorker::kMaxBacklog, bounds.frames)});
  }
  ThreadReport report = links.report;
  std::unique_ptr<GpuThread> thread(new GpuThread(std::move(backend), std::move(links), bounds));
  auto started =
      start_thread("gpu", std::move(report),
                   [self = thread.get()](const std::stop_token& stop, const ThreadReport& reporter) {
                     self->run(stop, reporter);
                   });
  if (!started) return std::unexpected(std::move(started.error()));
  thread->thread_ = std::move(*started);
  return thread;
}

bool GpuThread::try_push(DecodedFrame&& frame) {
  {
    const std::scoped_lock lock(mutex_);
    if (held_ >= bounds_.frames) {
      refused_ = true;
      return false;
    }
    ++held_;
  }
  // La boîte a la même borne et ne compte que ce qui n'est pas encore sorti : elle a la place.
  if (inbox_.try_push(std::move(frame))) return true;
  const std::scoped_lock lock(mutex_);
  --held_;
  refused_ = true;
  return false;
}

GpuStats GpuThread::stats() const {
  const std::scoped_lock lock(mutex_);
  return stats_;
}

void GpuThread::run(const std::stop_token& stop, const ThreadReport& report) {
  GpuWorker worker(backend_.backend(), backend_.rife_input(), *links_.frames, report);
  while (std::optional<GpuInbox::Message> message = inbox_.wait(stop)) {
    std::visit(Visitor{[&](DecodedFrame& frame) {
                         if (!worker.frame(std::move(frame))) {
                           report({RefusalCode::SequenceOrder, "thread gpu : plus d'images que sa borne"});
                           release_frames(1);
                         }
                       },
                       [&](const SlotFree& free) { worker.slot_free(free); },
                       [&](const RingGeneration& ring) { worker.ring(ring); },
                       [&](const Install& install) { worker.install(install); }, [](const Command&) {},
                       [&](const Flush& flush) { worker.flush(flush); }},
               *message);
    worker.pump();
    release_frames(worker.take_consumed());
    const GpuStats stats = worker.stats();
    const std::scoped_lock lock(mutex_);
    stats_ = stats;
  }
}

void GpuThread::release_frames(std::size_t count) {
  if (count == 0) return;
  bool wake = false;
  {
    const std::scoped_lock lock(mutex_);
    held_ -= count;
    wake = std::exchange(refused_, false);
  }
  if (wake && links_.credit) links_.credit();
}

}  // namespace ajp::gpu
