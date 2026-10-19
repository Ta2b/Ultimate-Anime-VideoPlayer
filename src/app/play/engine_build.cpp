#include "app/play/engine_build.h"

#include <cmath>
#include <format>
#include <utility>

#include "core/thread.h"

namespace ajp::app::play {

std::expected<std::unique_ptr<EngineBuild>, Refusal> EngineBuild::start(EngineRequest request) {
  std::unique_ptr<EngineBuild> build(new EngineBuild());
  EngineBuild* self = build.get();
  auto thread = start_thread(
      "preparer", ThreadReport([self](const Refusal& refusal) {
        const std::scoped_lock lock(self->mutex_);
        if (!self->result_) self->result_ = std::unexpected(refusal);
      }),
      [self, request = std::move(request)](const std::stop_token& stop, const ThreadReport&) mutable {
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
  auto finish = [this](std::expected<gpu::Artifact, Refusal> outcome) {
    const std::scoped_lock lock(mutex_);
    result_ = std::move(outcome);
  };
  auto backend = BackendLibrary::create(request.library, request.init);
  if (!backend) {
    finish(std::unexpected(std::move(backend.error())));
    return;
  }
  gpu::Progress progress{.report =
                             [this](float fraction, const std::string& stage) {
                               const std::scoped_lock lock(mutex_);
                               fraction_ = fraction;
                               stage_ = stage;
                             },
                         .stop = stop};
  finish((*backend)->build(request.model, request.shapes, progress));
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
