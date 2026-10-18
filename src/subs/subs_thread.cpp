#include "subs/subs_thread.h"

#include <exception>
#include <optional>
#include <utility>

#include <spdlog/logger.h>

#include "subs/log.h"
#include "subs/subs_worker.h"

namespace ajp::subs {
namespace {

void run(const std::stop_token& stop, SubsWorker& worker, SubsInbox& inbox, spdlog::logger& log) {
  while (std::optional<SubsInbox::Message> message = inbox.wait(stop)) {
    try {
      if (!worker.handle(std::move(*message))) return;
    } catch (const std::exception& error) {  // aucune exception ne sort de la boucle
      log.error("message perdu : {}", error.what());
    }
  }
}

}  // namespace

std::expected<std::unique_ptr<SubsThread>, Refusal> SubsThread::start(const RendererOptions& options,
                                                                      SubtitleSink& sink,
                                                                      std::shared_ptr<SubsInbox> inbox) {
  if (inbox == nullptr) return std::unexpected(Refusal{RefusalCode::FieldMissing, "subs : boîte absente"});
  // Un atlas relâché rentre par la boîte. Elle ne déborde pas : il n'existe
  // jamais plus de kMaxAtlasesInFlight atlas, et autant de places.
  AtlasReturn route = [weak = std::weak_ptr<SubsInbox>(inbox)](std::unique_ptr<SubtitleAtlas> atlas) {
    if (const std::shared_ptr<SubsInbox> box = weak.lock()) box->try_push(AtlasReturned{std::move(atlas)});
  };
  auto worker = SubsWorker::create(options, sink, std::move(route));
  if (!worker) return std::unexpected(std::move(worker.error()));

  std::unique_ptr<SubsThread> thread(new SubsThread());
  thread->thread_ = std::jthread([worker = std::move(*worker), inbox = std::move(inbox), log = make_logger()](
                                     const std::stop_token& stop) { run(stop, *worker, *inbox, *log); });
  return thread;
}

}  // namespace ajp::subs
