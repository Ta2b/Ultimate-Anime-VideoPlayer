#include "app/play/display_probe.h"

#include <utility>

#include "core/clock.h"

namespace ajp::app::play {

DisplayReaders system_display_readers() {
  return {
      .hz = [](SDL_Window* window, const DisplaySdl& sdl) { return display_refresh_hz(window, sdl); },
      .peak = [](SDL_Window* window, const DisplaySdl& sdl) { return peak_reading(peak_facts(window, sdl)); },
      .vrr = [](SDL_Window* window,
                const std::string& connector) { return window_vrr_armed(window, connector); },
      .facts = [](SDL_Window* window, const DisplaySdl& sdl) { return window_display(window, sdl.name); }};
}

DisplayProbe::DisplayProbe(SDL_Window* window, DisplayReaders readers, void (*wake)(void*), void* context)
    : window_(window), readers_(std::move(readers)), wake_(wake), context_(context) {
  thread_ = std::jthread([this](const std::stop_token& stop) { run(stop); });
}

DisplayProbe::~DisplayProbe() {
  thread_.request_stop();
  changed_.notify_all();
}

void DisplayProbe::ask(DisplaySdl sdl, bool peak, bool vrr, std::string connector, bool facts) {
  {
    const std::scoped_lock lock(mutex_);
    // Une demande encore en attente garde ce qu'elle voulait déjà relire.
    const bool had_peak = pending_ && pending_->peak;
    const bool had_vrr = pending_ && pending_->vrr;
    const bool had_facts = pending_ && pending_->facts;
    pending_ = Request{.sdl = std::move(sdl),
                       .peak = peak || had_peak,
                       .vrr = vrr || had_vrr || !connector.empty(),
                       .connector = std::move(connector),
                       .facts = facts || had_facts};
  }
  changed_.notify_one();
}

std::optional<DisplayReading> DisplayProbe::take() {
  const std::scoped_lock lock(mutex_);
  return std::exchange(done_, std::nullopt);
}

bool DisplayProbe::busy() const {
  const std::scoped_lock lock(mutex_);
  return pending_.has_value() || running_;
}

void DisplayProbe::run(const std::stop_token& stop) {
  while (true) {
    Request request;
    {
      std::unique_lock lock(mutex_);
      if (!changed_.wait(lock, stop, [&] { return pending_.has_value(); })) return;
      if (!pending_) continue;  // tenu par le prédicat ; dit pour l'analyse
      request = std::move(*pending_);
      pending_.reset();
      running_ = true;
    }
    const std::int64_t start = host_now_us();
    DisplayReading reading;
    if (readers_.hz) reading.hz = readers_.hz(window_, request.sdl);
    if (request.peak && readers_.peak) {
      reading.peak_read = true;
      const PeakReading peak = readers_.peak(window_, request.sdl);
      reading.peak_nits = peak.nits;
      reading.calibrated_peak = peak.calibrated;
    }
    if (request.facts && readers_.facts) {
      reading.facts_read = true;
      reading.facts = readers_.facts(window_, request.sdl);
    }
    if (request.vrr && readers_.vrr) {
      reading.vrr_read = true;
      reading.vrr_armed = readers_.vrr(window_, request.connector);
    }
    reading.took_us = host_now_us() - start;
    {
      const std::scoped_lock lock(mutex_);
      // Un relevé pas encore pris garde ce que celui-ci n'a pas relu.
      if (done_) {
        if (!reading.peak_read && done_->peak_read) {
          reading.peak_read = true;
          reading.peak_nits = done_->peak_nits;
          reading.calibrated_peak = done_->calibrated_peak;
        }
        if (!reading.facts_read && done_->facts_read) {
          reading.facts_read = true;
          reading.facts = std::move(done_->facts);
        }
        if (!reading.vrr_read && done_->vrr_read) {
          reading.vrr_read = true;
          reading.vrr_armed = done_->vrr_armed;
        }
      }
      done_ = reading;
      running_ = false;
    }
    if (wake_ != nullptr) wake_(context_);
  }
}

}  // namespace ajp::app::play
