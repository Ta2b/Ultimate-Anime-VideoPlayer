#include "app/play/span_auto.h"

#include <algorithm>
#include <chrono>
#include <format>

#include "app/play/i18n.h"
#include "core/log.h"

namespace ajp::app::play {

std::string span_fp8_unavailable_reason(const SpanFp8Facts& facts) {
  if (facts.capability) return {};
  if (facts.backend_name != "tensorrt") {
    if (facts.tensorrt_present) {
      return i18n::tr_format(
          "TensorRT is installed but the player uses {}: pick TensorRT in Settings > Export and engines > "
          "Engines",
          facts.backend_name.empty() ? std::string(i18n::tr("another engine")) : facts.backend_name);
    }
    return i18n::tr("TensorRT is not installed: install it in Settings > Export and engines > Engines");
  }
  return i18n::tr_format(
      "{} is not an NVIDIA RTX 40 (Ada) card, or the engine was built without these kernels",
      facts.gpu_name.empty() ? std::string(i18n::tr("this card")) : facts.gpu_name);
}

SpanJob::~SpanJob() {
  if (note) stop_and_wait(*note);
}

void poll_prepare(PrepareNote& note) {
  if (!note.pending || note.pending->wait_for(std::chrono::seconds(0)) != std::future_status::ready) return;
  const std::expected<std::string, Refusal> done = note.pending->get();
  note.pending.reset();
  note.progress.reset();
  note.stop.reset();
  if (done) {
    note.text = *done;
  } else {
    // Le détail technique va au journal ; l'écran dit une phrase, dans la langue de l'interface.
    log_channel("app")->warn("SPAN Performance, preparation : {} {}",
                             name(done.error().code),  // i18n-ok : journal
                             done.error().detail);
    note.text = i18n::tr("Performance mode could not be prepared on this machine; playback stays in FP16.");
  }
  note.refused = !done;
  note.completed = true;
}

void stop_and_wait(PrepareNote& note) {
  if (note.stop) note.stop->request_stop();
  if (note.pending) note.pending->wait();  // le fil rend la main vite : les étapes regardent l'arrêt
  note.pending.reset();
}

std::string span_progress_text(float fraction) {
  return i18n::tr_format("Preparing Performance mode… {} %",
                         static_cast<int>(std::clamp(fraction, 0.0F, 1.0F) * 100.0F));
}

std::string span_failure_text(std::string_view detail) {
  static_cast<void>(detail);  // le détail est au journal (poll_prepare)
  return i18n::tr("Performance mode unavailable, playing in FP16.");
}

bool should_start_span_auto(PrecisionTier precision, const SpanPrepare& prepare, bool job_running,
                            std::string_view key, std::string_view attempted) {
  return precision == PrecisionTier::Performance && prepare.offered && !prepare.ready && !prepare.refused &&
         static_cast<bool>(prepare.run) && !job_running && key != attempted;
}

SpanAutoTick tick_span_auto(PrecisionTier precision, const SpanPrepare& prepare, SpanJob& job,
                            std::string_view key) {
  SpanAutoTick tick;
  PrepareNote& note = *job.note;
  poll_prepare(note);
  if (note.completed) {
    note.completed = false;
    if (note.refused) {
      tick.notify = span_failure_text(note.text);
    } else {
      tick.notify = i18n::tr("Performance mode ready: playback uses it now");
      tick.reopen = true;
    }
    return tick;
  }
  if (note.pending) {
    if (note.progress) tick.notify = span_progress_text(note.progress->get().first);
    return tick;
  }
  if (should_start_span_auto(precision, prepare, false, key, job.attempted)) {
    job.attempted = std::string(key);
    if (start_span_prepare(prepare, note)) tick.notify = span_progress_text(0.0F);
  }
  return tick;
}

}  // namespace ajp::app::play
