// Le fil des étapes différées des runtimes de l'hôte (deferred_steps.h) : les étapes dans l'ordre
// des appels, l'attente des entrées avant le calcul, un refus ou une exception rendus à done, et
// chaque étape en file faite avant la destruction.
#include "gpu/vulkan/backend/deferred_steps.h"

#include <atomic>
#include <expected>
#include <future>
#include <mutex>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>

using namespace ajp;
using ajp::gpu::vk::net::DeferredSteps;
using ajp::gpu::vk::net::HostBinding;

namespace {

struct Log {
  std::mutex mutex;
  std::vector<int> computed;
  std::vector<std::expected<void, Refusal>> done;
};

DeferredSteps::Compute recording(Log& log) {
  return [&log](std::span<const HostBinding> inputs,
                std::span<const HostBinding> /*outputs*/) -> std::expected<void, Refusal> {
    if (inputs.empty()) throw std::runtime_error("sans entrée");
    const std::scoped_lock lock(log.mutex);
    log.computed.push_back(static_cast<int>(inputs[0].bytes));
    return {};
  };
}

std::function<void(std::expected<void, Refusal>)> into(Log& log) {
  return [&log](std::expected<void, Refusal> result) {
    const std::scoped_lock lock(log.mutex);
    log.done.push_back(std::move(result));
  };
}

std::function<std::expected<void, Refusal>()> ready_now() {
  return [] { return std::expected<void, Refusal>{}; };
}

}  // namespace

TEST_CASE("Étapes différées : dans l'ordre des appels, toutes faites avant la destruction",
          "[vk_backend][deferred]") {
  Log log;
  {
    DeferredSteps steps("test-steps", "Essai", recording(log));
    for (int i = 1; i <= 5; ++i) {
      const std::vector<HostBinding> in{HostBinding{.bytes = static_cast<std::uint64_t>(i)}};
      REQUIRE(steps.push(in, {}, ready_now(), into(log)));
    }
  }
  CHECK(log.computed == std::vector<int>{1, 2, 3, 4, 5});
  REQUIRE(log.done.size() == 5);
  for (const auto& result : log.done) CHECK(result.has_value());
}

TEST_CASE("Étapes différées : l'attente des entrées précède le calcul, ses refus vont à done",
          "[vk_backend][deferred]") {
  Log log;
  std::atomic<bool> waited{false};
  {
    DeferredSteps steps("test-steps", "Essai", recording(log));
    // Après le fil : un échec plus bas rompt la promesse, et l'attente de la première étape cède.
    std::promise<void> release;
    std::shared_future<void> gate = release.get_future().share();
    const std::vector<HostBinding> first{HostBinding{.bytes = 1}};
    REQUIRE(steps.push(
        first, {},
        [gate, &waited] {
          gate.wait();
          waited = true;
          return std::expected<void, Refusal>{};
        },
        into(log)));
    const std::vector<HostBinding> second{HostBinding{.bytes = 2}};
    REQUIRE(steps.push(
        second, {},
        [] {
          return std::expected<void, Refusal>(
              std::unexpected(Refusal{RefusalCode::DeviceUnavailable, "file perdue"}));
        },
        into(log)));
    // Une exception du calcul : un refus, jamais la fin du fil.
    REQUIRE(steps.push({}, {}, ready_now(), into(log)));
    {
      const std::scoped_lock lock(log.mutex);
      CHECK(log.computed.empty());  // la première attend encore ses entrées
    }
    release.set_value();
  }
  CHECK(waited);
  CHECK(log.computed == std::vector<int>{1});
  REQUIRE(log.done.size() == 3);
  CHECK(log.done[0].has_value());
  REQUIRE_FALSE(log.done[1].has_value());
  CHECK(log.done[1].error().detail == "file perdue");
  REQUIRE_FALSE(log.done[2].has_value());
  CHECK(log.done[2].error().code == RefusalCode::DeviceUnavailable);
}

TEST_CASE("Étapes différées : sans rappel, ou la file pleine, rien n'est lancé", "[vk_backend][deferred]") {
  Log log;
  {
    DeferredSteps steps("test-steps", "Essai", recording(log));
    std::promise<void> release;
    std::shared_future<void> gate = release.get_future().share();
    auto refused = steps.push({}, {}, nullptr, into(log));
    REQUIRE_FALSE(refused);
    CHECK(refused.error().code == RefusalCode::InputMismatch);
    CHECK(refused.error().detail.starts_with("Essai"));

    // La première étape tient le fil ; les suivantes remplissent la file.
    const std::vector<HostBinding> in{HostBinding{.bytes = 7}};
    REQUIRE(steps.push(
        in, {},
        [gate] {
          gate.wait();
          return std::expected<void, Refusal>{};
        },
        into(log)));
    std::size_t queued = 0;
    std::expected<void, Refusal> pushed;
    while ((pushed = steps.push(in, {}, ready_now(), into(log)))) ++queued;
    CHECK(pushed.error().code == RefusalCode::ThreadUnavailable);
    // Le fil a pu prendre la première avant les autres : au plus la borne, au moins la borne - 1.
    CHECK(queued + 1 >= DeferredSteps::kStepBound);
    CHECK(queued <= DeferredSteps::kStepBound);
    release.set_value();
  }
  const std::scoped_lock lock(log.mutex);
  CHECK(log.done.size() == log.computed.size());
  CHECK(log.computed.size() >= DeferredSteps::kStepBound);
}
