#include "app/play/present_watch.h"

#include <format>
#include <fstream>
#include <utility>

#include <SDL3/SDL_messagebox.h>

#include "app/play/i18n.h"
#include "app/play/relaunch_target.h"
#include "core/clock.h"
#include "core/file_sync.h"
#include "core/log.h"
#include "core/thread.h"
#include "core/utf8_path.h"

namespace ajp::app::play {
namespace {

constexpr std::chrono::milliseconds kPresentWatchTick{250};

}  // namespace

std::string present_stall_key(std::string_view device, std::uint32_t driver_version) {
  return std::format("{} · {:08x}", device, driver_version);
}

std::string read_present_stall(const std::filesystem::path& file) {
  std::ifstream in(file, std::ios::binary);
  std::string line;
  if (!in || !std::getline(in, line)) return {};
  if (line.ends_with('\r')) line.pop_back();
  return line;
}

PresentWatch::PresentWatch(std::filesystem::path file, std::string key, bool tell)
    : file_(std::move(file)), key_(std::move(key)), tell_(tell) {
  // Sans thread (refus du système) : pas de garde, la présentation tourne comme avant.
  auto started = start_thread("garde de présentation", ThreadReport([](const Refusal&) {}),
                              [this](const std::stop_token& stop, const ThreadReport&) { run(stop); });
  if (started) thread_ = std::move(*started);
}

PresentWatch::~PresentWatch() = default;  // std::jthread demande l'arrêt et attend le thread

PresentWatch::Scope::Scope(PresentWatch* watch, const char* stage) : watch_(watch) {
  if (watch_ == nullptr) return;
  watch_->stage_.store(stage, std::memory_order_relaxed);
  watch_->since_us_.store(host_now_us(), std::memory_order_release);
}

PresentWatch::Scope::~Scope() {
  if (watch_ != nullptr) watch_->since_us_.store(0, std::memory_order_release);
}

void PresentWatch::run(const std::stop_token& stop) {
  std::unique_lock lock(mutex_);
  while (!wake_.wait_for(lock, stop, kPresentWatchTick, [&] { return stop.stop_requested(); })) {
    const std::int64_t since = since_us_.load(std::memory_order_acquire);
    if (since == 0) continue;
    const std::int64_t waited = host_now_us() - since;
    if (waited >= std::chrono::duration_cast<std::chrono::microseconds>(kPresentStallLimit).count())
      stalled(stage_.load(std::memory_order_relaxed), waited);
  }
}

void PresentWatch::stalled(const char* stage, std::int64_t for_us) {
  const auto log = log_channel("app");
  if (auto written = core::write_text_atomically(file_, key_ + "\n"); !written)
    log->warn("présentation : {} non écrit ({})", core::to_utf8(file_.filename()), written.error().detail);
  log->error(
      "présentation : le pilote ne rend pas la main depuis {} s ({}, {}) ; le prochain lancement présentera "
      "sans retour de présentation sur cette carte et ce pilote ({})",
      for_us / 1'000'000, stage != nullptr ? stage : "?", key_, core::to_utf8(file_.filename()));
  log->flush();
  if (tell_) {
    SDL_ShowSimpleMessageBox(
        SDL_MESSAGEBOX_ERROR, "ajp",
        i18n::tr("ajp stopped responding inside the graphics driver and will close. Start it again: it "
                 "will present frames without presentation feedback on this driver."),
        nullptr);
  }
  exit_stuck("présentation figée dans le pilote");
}

}  // namespace ajp::app::play
