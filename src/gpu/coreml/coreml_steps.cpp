#include "gpu/coreml/coreml_steps.h"

#include <format>
#include <utility>
#include <variant>

#include "core/thread.h"

namespace ajp::gpu::coreml_rt {

DeferredSteps::~DeferredSteps() {
  if (!worker_.joinable()) return;
  worker_.request_stop();
  worker_.join();
}

std::expected<void, Refusal> DeferredSteps::push(std::span<const vk::net::HostBinding> inputs,
                                                 std::span<const vk::net::HostBinding> outputs,
                                                 std::function<std::expected<void, Refusal>()> ready,
                                                 std::function<void(std::expected<void, Refusal>)> done) {
  if (!ready || !done)
    return std::unexpected(
        Refusal{.code = RefusalCode::InputMismatch, .detail = "Core ML : étape sans rappels"});
  {
    const std::scoped_lock lock(mutex_);
    if (failed_) return std::unexpected(*failed_);
    if (!worker_.joinable()) {
      auto worker = start_thread(
          "coreml" + label_, ThreadReport([this](const Refusal& refusal) { note_failure(refusal); }),
          [this](const std::stop_token& stop, const ThreadReport& /*report*/) { serve(stop); });
      if (!worker) return std::unexpected(std::move(worker.error()));
      worker_ = std::move(*worker);
    }
  }
  if (!jobs_.try_push(Job{.inputs = {inputs.begin(), inputs.end()},
                          .outputs = {outputs.begin(), outputs.end()},
                          .ready = std::move(ready),
                          .done = std::move(done)})) {
    return std::unexpected(
        Refusal{.code = RefusalCode::ThreadUnavailable,
                .detail = std::format("Core ML : {} étapes en file, le fil n'en reprend pas", kStepBound)});
  }
  return {};
}

void DeferredSteps::note_failure(const Refusal& refusal) {
  const std::scoped_lock lock(mutex_);
  if (!failed_) failed_ = refusal;
}

void DeferredSteps::serve(const std::stop_token& stop) {
  while (std::optional<Inbox<Job>::Message> message = jobs_.wait(stop)) {
    Job& job = std::get<Job>(*message);
    std::expected<void, Refusal> result;
    try {
      result = job.ready();
      if (result) result = compute_(job.inputs, job.outputs);
    } catch (...) {
      result = std::unexpected(refusal_from_exception(RefusalCode::DeviceUnavailable, "Core ML, étape"));
    }
    job.done(std::move(result));
  }
}

}  // namespace ajp::gpu::coreml_rt
