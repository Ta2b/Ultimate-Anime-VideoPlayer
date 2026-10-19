#include "gpu/worker.h"

#include <algorithm>
#include <chrono>
#include <format>
#include <utility>

#include "gpu/log.h"

namespace ajp::gpu {

bool GpuWorker::frame(DecodedFrame frame) {
  ++stats_.frames;
  return backlog_.push(std::move(frame));
}

void GpuWorker::slot_free(const SlotFree& message) {
  if (!ring_) return;
  if (auto released = ring_->release(message); !released) report_(released.error());
}

void GpuWorker::ring(const RingGeneration& message) {
  auto ring = OutputRing::create(message.slots);
  if (!ring) {
    report_(ring.error());
    return;
  }
  const OutputRing* newest = next_ring_ ? &*next_ring_ : ring_ ? &*ring_ : nullptr;
  if (newest != nullptr && ring->generation() <= newest->generation()) {
    report_({RefusalCode::SequenceOrder,
             std::format("ring de génération {} après la {}", ring->generation(), newest->generation())});
    return;
  }
  next_ring_ = *ring;  // une génération plus récente reçue entre-temps remplace celle qui attendait
}

void GpuWorker::install(const Install& message) {
  const auto* request = dynamic_cast<const ProgramRequest*>(message.program.get());
  if (request == nullptr) {
    report_({RefusalCode::FieldType, "Install : le programme n'est pas une gpu::ProgramRequest"});
    return;
  }
  auto program = load_program(*request, *backend_, rife_);
  if (!program) {
    report_(program.error());
    return;
  }
  next_ = std::move(*program);  // un Install plus récent remplace celui qui attendait
}

void GpuWorker::flush(const Flush& message) {
  if (message.epoch < epoch_) {
    report_(
        {RefusalCode::SequenceOrder, std::format("Flush {} antérieur à l'époque {}", message.epoch, epoch_)});
    return;
  }
  epoch_ = message.epoch;
  // Rien n'attend côté hôte : les images d'avant sont jetées, le GPU finit ce qu'il a reçu.
  for (std::size_t i = backlog_.size(); i > 0; --i) {
    DecodedFrame frame = backlog_.pop();
    if (frame.epoch < epoch_) {
      ++stats_.dropped;
      ++consumed_;
    } else {
      backlog_.push(std::move(frame));
    }
  }
  if (executor_) {
    stats_.dropped += executor_->pending_outputs();
    executor_->flush(epoch_);
  }
  if (ring_) stats_.dropped += ring_->drop_before(epoch_);
  last_.reset();
  end_of_stream_ = false;
}

void GpuWorker::end_of_stream() {
  end_of_stream_ = true;
}

void GpuWorker::pump() {
  for (;;) {
    if (ring_) ring_->deliver(*sink_);
    adopt_ring();
    switch_program();
    bool progressed = write_outputs();
    const bool idle = executor_ && !executor_->has_output() && !next_ && writable();
    if (idle && !backlog_.empty()) {
      process(backlog_.pop());
      ++consumed_;
      progressed = true;
    } else if (idle && end_of_stream_) {
      // Après la dernière image reçue : les paires qui attendaient leur anticipation partent.
      end_of_stream_ = false;
      if (auto ok = executor_->finish(); !ok) report_(ok.error());
      progressed = true;
    }
    if (!progressed) return;
  }
}

void GpuWorker::adopt_ring() {
  // Les images déjà écrites dans l'ancienne génération partent d'abord : main les attend avant
  // de la détruire, et toute image de la nouvelle doit les suivre.
  if (!next_ring_ || (ring_ && ring_->count(SlotState::Ready) > 0)) return;
  ring_ = next_ring_;
  next_ring_.reset();
}

void GpuWorker::switch_program() {
  if (!next_ || !ring_ || next_ring_ || !ring_->fits(next_->plan())) return;  // attend son ring
  // Les images du Program en cours partent d'abord si le ring les reçoit encore ; la suite
  // reprend à la paire suivante. Sinon elles ne seront jamais écrites.
  if (executor_ && executor_->has_output()) {
    if (ring_->fits(executor_->program().plan())) return;
    stats_.dropped += executor_->pending_outputs();
  }
  retire_executor();
  executor_ = std::make_unique<Executor>(std::move(next_));
  next_.reset();
  stats_.plan_hash = executor_->program().plan_hash();
  make_logger()->info("Program {:016x} : {}", stats_.plan_hash, describe(executor_->program()));
  if (last_ && last_->epoch == epoch_) {
    if (auto ok = executor_->seed(*last_); !ok) report_(ok.error());
  }
}

bool GpuWorker::writable() const {
  return ring_ && !next_ring_ && executor_ && ring_->fits(executor_->program().plan());
}

bool GpuWorker::write_outputs() {
  if (!ring_ || !writable()) return false;
  OutputRing& ring = *ring_;
  bool progressed = false;
  while (executor_->has_output()) {
    const FinalImage& image = executor_->front();
    if (image.frame.epoch < epoch_) {
      executor_->pop();
      ++stats_.dropped;
      continue;
    }
    const std::optional<OutputRing::Lease> lease = ring.acquire();
    if (!lease) break;  // le thread attend un SlotFree
    auto written =
        executor_->program().kernels().write_ring(tensor_of(image.content->high), *lease->desc, lease->sync);
    if (written) {
      ring.commit(lease->slot, image.frame);
      ++stats_.written;
    } else {
      ring.abandon(lease->slot);
      report_(written.error());
      ++stats_.dropped;
    }
    executor_->pop();
    ring.deliver(*sink_);
    progressed = true;
  }
  return progressed;
}

void GpuWorker::process(const DecodedFrame& frame) {
  if (frame.epoch < epoch_) {
    ++stats_.dropped;
    return;
  }
  if (frame.epoch > epoch_) {  // une époque plus récente vaut un Flush
    epoch_ = frame.epoch;
    if (ring_) stats_.dropped += ring_->drop_before(epoch_);
  }
  const auto start = std::chrono::steady_clock::now();
  if (auto ok = executor_->push(frame); !ok) {
    report_(ok.error());
    ++stats_.dropped;
  }
  const auto spent =
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start);
  stats_.last_pair_us = spent.count();
  stats_.max_pair_us = std::max(stats_.max_pair_us, stats_.last_pair_us);
  last_ = frame;
}

void GpuWorker::retire_executor() {
  if (!executor_) return;
  const ExecutorCounters& done = executor_->counters();
  retired_.interpolated += done.interpolated;
  retired_.repeated += done.repeated;
  retired_.scene_cuts += done.scene_cuts;
  retired_.lost_measures += done.lost_measures;
  executor_.reset();
}

GpuStats GpuWorker::stats() const {
  GpuStats stats = stats_;
  const ExecutorCounters current = executor_ ? executor_->counters() : ExecutorCounters{};
  stats.interpolated = retired_.interpolated + current.interpolated;
  stats.repeated = retired_.repeated + current.repeated;
  stats.scene_cuts = retired_.scene_cuts + current.scene_cuts;
  stats.lost_measures = retired_.lost_measures + current.lost_measures;
  return stats;
}

}  // namespace ajp::gpu
