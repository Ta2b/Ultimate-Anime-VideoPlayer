#include "subs/subs_thread.h"

#include <exception>
#include <optional>
#include <utility>

#include <spdlog/logger.h>

#include "core/thread.h"
#include "subs/log.h"
#include "subs/subs_worker.h"

namespace ajp::subs {
namespace {

void run(const std::stop_token& stop, SubsWorker& worker, SubsInbox& inbox, spdlog::logger& log) {
  while (std::optional<SubsInbox::Message> message = inbox.wait(stop)) {
    // Aucune exception ne sort de la boucle : le message fautif est perdu, le thread continue.
    try {
      if (!worker.handle(std::move(*message))) return;
    } catch (const std::exception& error) {
      log.error("message perdu : {}", error.what());
    } catch (...) {
      log.error("message perdu : exception inconnue");
    }
  }
}

// Un atlas relâché rentre par la boîte. Elle ne déborde pas : il n'existe
// jamais plus de kMaxAtlasesInFlight atlas, et autant de places. Appelée depuis
// le destructeur du dernier lecteur, sur n'importe quel thread : elle ne lève rien.
AtlasReturn return_route(const std::shared_ptr<SubsInbox>& inbox) {
  return [weak = std::weak_ptr<SubsInbox>(inbox), log = make_logger()](std::unique_ptr<SubtitleAtlas> atlas) {
    const std::shared_ptr<SubsInbox> box = weak.lock();
    if (box == nullptr) return;  // le thread n'est plus : l'atlas est simplement détruit
    try {
      if (!box->try_push(AtlasReturned{std::move(atlas)})) log->error("atlas rendu détruit : boîte pleine");
    } catch (const std::exception& error) {
      log->error("atlas rendu détruit : {}", error.what());
    }
  };
}

}  // namespace

std::expected<std::unique_ptr<SubsThread>, Refusal> SubsThread::start(const RendererOptions& options,
                                                                      SubtitleSink& sink,
                                                                      std::shared_ptr<SubsInbox> inbox) {
  if (inbox == nullptr) return std::unexpected(Refusal{RefusalCode::FieldMissing, "subs : boîte absente"});
  auto worker = SubsWorker::create(options, sink, return_route(inbox));
  if (!worker) return std::unexpected(std::move(worker.error()));

  auto started = start_thread(
      "subs", ThreadReport([&sink](const Refusal& refusal) { sink.refused(refusal); }),
      [worker = std::move(*worker), inbox = std::move(inbox), log = make_logger()](
          const std::stop_token& stop, const ThreadReport& /*report*/) { run(stop, *worker, *inbox, *log); });
  if (!started) return std::unexpected(std::move(started.error()));
  std::unique_ptr<SubsThread> thread(new SubsThread());
  thread->thread_ = std::move(*started);
  return thread;
}

}  // namespace ajp::subs
