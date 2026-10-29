#include "app/play/span_auto.h"

#include <algorithm>
#include <chrono>
#include <format>

namespace ajp::app::play {

std::string span_fp8_unavailable_reason(const SpanFp8Facts& facts) {
  if (facts.capability) return {};
  if (facts.backend_name != "tensorrt") {
    if (facts.tensorrt_present) {
      return std::format(
          "le moteur TensorRT est installé mais le lecteur utilise {} : choisissez TensorRT dans Paramètres "
          "> "
          "Export et moteurs > Moteurs",
          facts.backend_name.empty() ? std::string("un autre moteur") : facts.backend_name);
    }
    return "moteur TensorRT non installé : installez-le dans Paramètres > Export et moteurs > Moteurs";
  }
  return std::format(
      "{} n'est pas une carte NVIDIA RTX 40 (Ada) ou le moteur a été construit sans ces noyaux",
      facts.gpu_name.empty() ? std::string("cette carte") : facts.gpu_name);
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
  note.text = done ? *done : done.error().detail;
  note.refused = !done;
  note.completed = true;
}

void stop_and_wait(PrepareNote& note) {
  if (note.stop) note.stop->request_stop();
  if (note.pending) note.pending->wait();  // le fil rend la main vite : les étapes regardent l'arrêt
  note.pending.reset();
}

std::string span_progress_text(float fraction) {
  return std::format("Préparation du mode Performance… {} %",
                     static_cast<int>(std::clamp(fraction, 0.0F, 1.0F) * 100.0F));
}

std::string span_failure_text(std::string_view detail) {
  return std::format("Mode Performance impossible, lecture en FP16 : {}", detail);
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
      tick.notify = "Mode Performance prêt : la lecture le prend maintenant";
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
