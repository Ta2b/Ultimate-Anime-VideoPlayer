#include "app/play/latest_writer.h"

#include <utility>

#include "core/log.h"
#include "core/thread.h"

namespace ajp::app::play {
namespace {

void report(const std::expected<void, Refusal>& written) {
  if (!written) log_channel("app")->warn("{}", written.error().detail);
}

// Le job, dont une exception devient un refus : elle n'arrête ni le thread d'écriture ni l'affichage.
std::expected<void, Refusal> guarded(const LatestWriter::Job& job) noexcept {
  try {
    return job();
  } catch (...) {
    return std::unexpected(refusal_from_exception(RefusalCode::ThreadFailed, "écriture"));
  }
}

}  // namespace

LatestWriter::LatestWriter() {
  auto started =
      start_thread("écriture", ThreadReport([](const Refusal& refusal) { report(std::unexpected(refusal)); }),
                   [this](const std::stop_token& stop, const ThreadReport& /*report*/) { run(stop); });
  if (started) {
    thread_ = std::move(*started);
  } else {
    log_channel("app")->warn("{} : écritures sur le thread d'affichage", started.error().detail);
  }
}

LatestWriter::~LatestWriter() {
  flush();
  thread_.request_stop();
  wake_.notify_all();
}

void LatestWriter::post(const std::string& key, Job job) {
  if (!thread_.joinable()) {
    report(guarded(job));
    return;
  }
  {
    const std::scoped_lock lock(mutex_);
    pending_.insert_or_assign(key, std::move(job));
  }
  wake_.notify_one();
}

void LatestWriter::flush() {
  if (!thread_.joinable()) return;
  std::unique_lock lock(mutex_);
  done_.wait(lock, [this] { return pending_.empty() && running_ == 0; });
}

void LatestWriter::run(const std::stop_token& stop) {
  std::unique_lock lock(mutex_);
  while (true) {
    wake_.wait(lock, stop, [this] { return !pending_.empty(); });
    if (pending_.empty()) return;  // arrêt demandé, plus rien à écrire
    auto next = pending_.begin();
    Job job = std::move(next->second);
    pending_.erase(next);
    ++running_;
    lock.unlock();
    report(guarded(job));  // ne lève pas (P2.5) : l'écriture finit toujours, flush() ne l'attend pas en vain
    lock.lock();
    --running_;
    if (pending_.empty() && running_ == 0) done_.notify_all();
  }
}

}  // namespace ajp::app::play
