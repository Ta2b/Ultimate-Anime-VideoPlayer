#include "app/play/latest_writer.h"

#include <utility>
#include <vector>

#include "app/play/pumped_task.h"
#include "core/log.h"
#include "core/thread.h"

namespace ajp::app::play {
namespace {

// Les états des writers détruits pendant qu'une écriture durait : wait_abandoned les attend à la sortie.
struct Abandoned {
  std::mutex mutex;
  std::vector<std::weak_ptr<void>> states;
};

Abandoned& abandoned() {
  static Abandoned instance;
  return instance;
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

void LatestWriter::finish(State& state, const std::string& key, const std::expected<void, Refusal>& written) {
  if (!written) log_channel("app")->warn("{} : {}", key, written.error().detail);
  const std::scoped_lock lock(state.mutex);
  if (written)
    state.failed.erase(key);
  else
    state.failed.insert_or_assign(key, written.error());
}

LatestWriter::LatestWriter() : state_(std::make_shared<State>()) {
  auto started = start_thread(
      "écriture",
      ThreadReport([](const Refusal& refusal) { log_channel("app")->warn("{}", refusal.detail); }),
      [state = state_](const std::stop_token& /*stop*/, const ThreadReport& /*report*/) { run(state); });
  if (started) {
    thread_ = std::move(*started);
  } else {
    log_channel("app")->warn("{} : écritures sur le thread d'affichage", started.error().detail);
  }
}

LatestWriter::~LatestWriter() {
  const bool finished = flush();
  {
    const std::scoped_lock lock(state_->mutex);
    state_->stop = true;
  }
  state_->wake.notify_all();
  if (!thread_.joinable()) return;
  if (finished) return;  // le thread sort seul, jthread l'attend
  // Une écriture dure encore : le thread la finit sans nous (il tient l'état), le processus l'attend à sa
  // sortie.
  log_channel("app")->warn("écritures encore en cours après {} ms : elles finissent en arrière-plan",
                           kFlushLimit.count());
  {
    const std::scoped_lock lock(abandoned().mutex);
    abandoned().states.emplace_back(state_);
  }
  // Gardé au registre des fils abandonnés plutôt que détaché : à la sortie, settle_abandoned_threads sait
  // qu'il écrit encore et choisit quick_exit plutôt que les destructeurs statiques sous ses pieds (C30-04).
  abandon_thread(std::move(thread_), [weak = std::weak_ptr<State>(state_)] {
    const std::shared_ptr<State> state = weak.lock();
    if (!state) return true;
    const std::scoped_lock lock(state->mutex);
    return state->idle();
  });
}

void LatestWriter::post(const std::string& key, Job job) {
  if (!thread_.joinable()) {
    finish(*state_, key, guarded(job));
    return;
  }
  {
    const std::scoped_lock lock(state_->mutex);
    state_->pending.insert_or_assign(key, std::move(job));
  }
  state_->wake.notify_one();
}

bool LatestWriter::flush(std::chrono::milliseconds limit) {
  if (!thread_.joinable()) return true;
  std::unique_lock lock(state_->mutex);
  return state_->done.wait_for(lock, limit, [this] { return state_->idle(); });
}

std::vector<LatestWriter::Failure> LatestWriter::take_failures() {
  const std::scoped_lock lock(state_->mutex);
  std::vector<Failure> out;
  out.reserve(state_->failed.size());
  for (auto& [key, refusal] : state_->failed)
    out.push_back(Failure{.key = key, .refusal = std::move(refusal)});
  state_->failed.clear();
  return out;
}

void LatestWriter::wait_abandoned(std::chrono::milliseconds limit) {
  std::vector<std::weak_ptr<void>> states;
  {
    const std::scoped_lock lock(abandoned().mutex);
    states = std::exchange(abandoned().states, {});
  }
  const auto until = std::chrono::steady_clock::now() + limit;
  for (const std::weak_ptr<void>& weak : states) {
    // Le thread tient l'état tant qu'il tourne : verrouillé ici, il est encore là ; sinon il a fini.
    const std::shared_ptr<void> alive = weak.lock();
    if (!alive) continue;
    State& state = *static_cast<State*>(alive.get());
    std::unique_lock lock(state.mutex);
    state.done.wait_until(lock, until, [&state] { return state.idle(); });
  }
}

void LatestWriter::run(const std::shared_ptr<State>& state) {
  std::unique_lock lock(state->mutex);
  while (true) {
    state->wake.wait(lock, [&state] { return state->stop || !state->pending.empty(); });
    if (state->pending.empty()) return;  // arrêt demandé, plus rien à écrire
    auto next = state->pending.begin();
    std::string key = next->first;
    Job job = std::move(next->second);
    state->pending.erase(next);
    ++state->running;
    lock.unlock();
    const auto written = guarded(job);  // ne lève pas (P2.5) : l'écriture finit toujours
    finish(*state, key, written);
    lock.lock();
    --state->running;
    if (state->idle()) state->done.notify_all();
  }
}

}  // namespace ajp::app::play
