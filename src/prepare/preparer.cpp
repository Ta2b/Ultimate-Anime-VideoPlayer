#include "prepare/preparer.h"

#include <deque>
#include <exception>
#include <memory>
#include <utility>
#include <variant>

#include <spdlog/logger.h>

#include "prepare/log.h"

namespace ajp::prepare {
namespace {

// État du thread preparer, créé et détruit dans le thread.
class Worker {
 public:
  Worker(PreparerInbox& inbox, PreparerSink& sink) : inbox_(inbox), sink_(sink), log_(make_logger()) {}

  void handle(BuildRequest& request, const std::stop_token& stop) { deliver(build(request, stop)); }

  void handle(const Credit& /*credit*/, const std::stop_token& /*stop*/) { flush(); }

  // spdlog rattrape ses propres erreurs : journaliser ici ne relance rien.
  void log_escape() const { log_->error("exception rattrapée dans la boucle du preparer"); }

 private:
  BuildDone build(BuildRequest& request, const std::stop_token& stop) {
    BuildDone done{.id = request.id, .model = request.model.id, .result = {}};
    if (request.backend == nullptr) {
      done.result = std::unexpected(Refusal{RefusalCode::FieldMissing, "BuildRequest.backend absent"});
    } else if (request.cancel.stop_requested()) {
      done.result = std::unexpected(Refusal{RefusalCode::Cancelled, "demande annulée avant sa construction"});
    } else {
      done.result = run_backend(request, stop);
    }
    if (done.result) {
      log_->info("demande {} : {} prêt ({})", done.id, done.model, done.result->cache_key);
    } else {
      log_->warn("demande {} : {} refusé, {} : {}", done.id, done.model, name(done.result.error().code),
                 done.result.error().detail);
    }
    return done;
  }

  // Une exception d'un backend tiers devient un refus : elle ne traverse pas la boucle.
  std::expected<gpu::Artifact, Refusal> run_backend(BuildRequest& request, const std::stop_token& stop) {
    std::stop_source build_stop;
    const auto stop_build = [&build_stop] { build_stop.request_stop(); };
    const std::stop_callback on_shutdown(stop, stop_build);
    const std::stop_callback on_cancel(request.cancel, stop_build);
    gpu::Progress progress{
        .report =
            [this, &request](float fraction, const std::string& stage) {
              sink_.publish({.id = request.id,
                             .model = request.model.id,
                             .fraction = fraction,
                             .stage = stage,
                             .queued = inbox_.pending<BuildRequest>()});
            },
        .stop = build_stop.get_token(),
    };
    try {
      return request.backend->build(request.model, request.shapes, progress);
    } catch (const std::exception& error) {
      return std::unexpected(
          Refusal{RefusalCode::BuilderCrashed, std::string("exception du backend : ") + error.what()});
    } catch (...) {
      return std::unexpected(Refusal{RefusalCode::BuilderCrashed, "exception inconnue du backend"});
    }
  }

  // Les résultats partent dans l'ordre ; ceux que le puits refuse attendent un Credit.
  void deliver(BuildDone done) {
    undelivered_.push_back(std::move(done));
    flush();
  }

  void flush() {
    while (!undelivered_.empty() && sink_.deliver(std::move(undelivered_.front()))) undelivered_.pop_front();
  }

  PreparerInbox& inbox_;
  PreparerSink& sink_;
  std::shared_ptr<spdlog::logger> log_;
  std::deque<BuildDone> undelivered_;
};

// Inbox::wait rend encore les messages en attente après la demande d'arrêt : la
// boucle teste l'arrêt elle-même, pour abandonner la file au lieu de la construire.
void run(const std::stop_token& stop, PreparerInbox& inbox, PreparerSink& sink) {
  Worker worker(inbox, sink);
  while (!stop.stop_requested()) {
    auto message = inbox.wait(stop);
    if (!message) break;
    try {
      std::visit([&](auto& m) { worker.handle(m, stop); }, *message);
    } catch (...) {
      worker.log_escape();  // dernier filet : rien ne sort de la boucle du thread
    }
  }
}

}  // namespace

Preparer::Preparer(PreparerSink& sink, PreparerBounds bounds)
    : inbox_({bounds.requests, bounds.credits}),
      thread_([this, &sink](const std::stop_token& stop) { run(stop, inbox_, sink); }) {}

bool Preparer::submit(BuildRequest&& request) {
  return inbox_.try_push(std::move(request));
}

bool Preparer::credit(Credit credit) {
  return inbox_.try_push(credit);
}

}  // namespace ajp::prepare
