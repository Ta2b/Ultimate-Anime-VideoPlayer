#include "app/play/engine_build.h"

#include <cmath>
#include <format>
#include <utility>

#include "app/play/bench_events.h"
#include "core/file_sync.h"
#include "core/thread.h"
#include "core/thread_role.h"

namespace ajp::app::play {

std::expected<gpu::Artifact, Refusal> with_space_reason(std::expected<gpu::Artifact, Refusal> outcome,
                                                        const std::filesystem::path& cache_dir,
                                                        std::uintmax_t needed) {
  if (outcome) return outcome;
  const RefusalCode code = outcome.error().code;
  if (code != RefusalCode::BuilderFailed && code != RefusalCode::CacheIo &&
      code != RefusalCode::FileUnwritable && code != RefusalCode::IoFailed)
    return outcome;
  std::optional<Refusal> full = core::lacking_space(cache_dir, needed);
  if (!full) return outcome;
  full->detail += std::format(" ; {}", outcome.error().detail);
  return std::unexpected(std::move(*full));
}

std::expected<std::unique_ptr<EngineBuild>, Refusal> EngineBuild::start(EngineRequest request) {
  std::unique_ptr<EngineBuild> build(new EngineBuild());
  EngineBuild* self = build.get();
  // Copiés avant le déplacement de request : l'ordre d'évaluation des arguments n'est pas fixé.
  auto* const wake = request.wake;
  void* const context = request.wake_context;
  auto thread = start_thread(
      "preparer", ThreadReport([self, wake, context](const Refusal& refusal) {
        {
          const std::scoped_lock lock(self->mutex_);
          if (!self->result_) self->result_ = std::unexpected(refusal);
        }
        if (wake != nullptr) wake(context);
      }),
      [self, request = std::move(request)](const std::stop_token& stop, const ThreadReport&) mutable {
        set_thread_role(ThreadRole::Background);  // une reconstruction ne concurrence pas le rendu (C20-06)
        self->run(stop, request);
      });
  if (!thread) return std::unexpected(std::move(thread.error()));
  build->thread_ = std::move(*thread);
  return build;
}

EngineBuild::~EngineBuild() {
  thread_.request_stop();
  if (thread_.joinable()) thread_.join();
}

void EngineBuild::run(const std::stop_token& stop, const EngineRequest& request) {
  auto finish = [this, &request](std::expected<gpu::Artifact, Refusal> outcome) {
    {
      const std::scoped_lock lock(mutex_);
      result_ = std::move(outcome);
    }
    if (request.wake != nullptr) request.wake(request.wake_context);
  };
  // B08-03 : l'instance de l'ouverture, créée au premier moteur et reprise par les suivants (énumération des
  // appareils, contexte et runtime une fois par ouverture, pas une fois par moteur).
  BackendLibrary::Instance own;
  BackendLibrary::Instance* backend = request.builder ? request.builder.get() : &own;
  if (!*backend) {
    auto created = BackendLibrary::create(request.library, request.init);
    if (!created) {
      finish(std::unexpected(std::move(created.error())));
      return;
    }
    *backend = std::move(*created);
  }
  gpu::Progress progress{.report =
                             [this](float fraction, const std::string& stage) {
                               // Une vraie construction avance entre 0 et 1 ; un moteur en cache rend 1 seul.
                               if (fraction > 0.0F && fraction < 1.0F && !told_build_.exchange(true))
                                 tell_bench_parent({.event = "build"});
                               const std::scoped_lock lock(mutex_);
                               fraction_ = fraction;
                               stage_ = stage;
                             },
                         .stop = stop};
  finish(
      with_space_reason((*backend)->build(request.model, request.shapes, progress), request.init.cache_dir));
}

std::optional<std::expected<gpu::Artifact, Refusal>> EngineBuild::result() {
  const std::scoped_lock lock(mutex_);
  return result_;
}

EngineBuild::Status EngineBuild::status() const {
  const std::scoped_lock lock(mutex_);
  return Status{.fraction = fraction_, .stage = stage_};
}

std::string EngineBuild::progress() const {
  const std::scoped_lock lock(mutex_);
  return std::format("{} % : {}", std::lround(fraction_ * 100.0F), stage_);
}

}  // namespace ajp::app::play
