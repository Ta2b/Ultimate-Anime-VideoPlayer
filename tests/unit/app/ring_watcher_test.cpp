// Le veilleur d'écriture (ring_watcher.h) sur un vrai appareil Vulkan (lavapipe suffit) : un sémaphore
// timeline signalé par l'hôte depuis un autre thread tient lieu de l'écriture de gpu.
#include "app/play/ring_watcher.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>

#include <catch2/catch_test_macros.hpp>
#include <libplacebo/log.h>
#include <libplacebo/vulkan.h>

#include "core/clock.h"

namespace ajp::app::play {
namespace {

struct Device {
  pl_log log = pl_log_create(PL_API_VER, nullptr);
  pl_vk_inst inst = nullptr;
  pl_vulkan vk = nullptr;
  PFN_vkGetDeviceProcAddr proc = nullptr;
  Device() {
    pl_vk_inst_params inst_params{};
    inst = pl_vk_inst_create(log, &inst_params);
    if (inst == nullptr) return;
    pl_vulkan_params params = pl_vulkan_default_params;
    params.instance = inst->instance;
    params.get_proc_addr = inst->get_proc_addr;
    params.allow_software = true;
    vk = pl_vulkan_create(log, &params);
    if (vk != nullptr)
      proc =
          reinterpret_cast<PFN_vkGetDeviceProcAddr>(vk->get_proc_addr(vk->instance, "vkGetDeviceProcAddr"));
  }
  ~Device() {
    pl_vulkan_destroy(&vk);
    pl_vk_inst_destroy(&inst);
    pl_log_destroy(&log);
  }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
};

void wake(void* context) {
  static_cast<std::atomic<int>*>(context)->fetch_add(1);
}

// Un appareil simulé dont vkWaitSemaphores rend VK_ERROR_DEVICE_LOST (A10-03).
VKAPI_ATTR VkResult VKAPI_CALL lost_wait(VkDevice, const VkSemaphoreWaitInfo*, uint64_t) {
  return VK_ERROR_DEVICE_LOST;
}
VKAPI_ATTR VkResult VKAPI_CALL fake_counter(VkDevice, VkSemaphore, uint64_t* value) {
  *value = 0;
  return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL fake_signal(VkDevice, const VkSemaphoreSignalInfo*) {
  return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL fake_create(VkDevice, const VkSemaphoreCreateInfo*,
                                           const VkAllocationCallbacks*, VkSemaphore* out) {
  static int handle = 0;
  *out = reinterpret_cast<VkSemaphore>(&handle);  // jamais déréférencé par le veilleur
  return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL fake_destroy(VkDevice, VkSemaphore, const VkAllocationCallbacks*) {}

}  // namespace

TEST_CASE("veilleur d'écriture : l'écriture finie est datée et réveille main", "[app][session]") {
  Device device;
  if (device.vk == nullptr || device.proc == nullptr) SKIP("pas d'appareil Vulkan");
  pl_vulkan_sem_params sem_params{};
  sem_params.type = VK_SEMAPHORE_TYPE_TIMELINE;
  VkSemaphore semaphore = pl_vulkan_sem_create(device.vk->gpu, &sem_params);
  REQUIRE(semaphore != VK_NULL_HANDLE);
  const auto signal =
      reinterpret_cast<PFN_vkSignalSemaphore>(device.proc(device.vk->device, "vkSignalSemaphore"));
  REQUIRE(signal != nullptr);
  std::atomic<int> wakes{0};
  {
    auto watcher = RingWatcher::start(device.vk->device, device.proc, &wake, &wakes);
    REQUIRE(watcher);
    watcher->watch(semaphore, 1);
    CHECK_FALSE(watcher->take(semaphore, 1).has_value());  // pas encore écrite
    std::int64_t signaled_us = 0;
    std::thread writer([&] {
      std::this_thread::sleep_for(std::chrono::milliseconds(30));
      signaled_us = host_now_us();
      const VkSemaphoreSignalInfo info{
          .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO, .semaphore = semaphore, .value = 1};
      (void)signal(device.vk->device, &info);
    });
    writer.join();
    const std::int64_t deadline = host_now_us() + 1'000'000;
    while (wakes.load() == 0 && host_now_us() < deadline)
      std::this_thread::sleep_for(std::chrono::microseconds(200));
    CHECK(wakes.load() == 1);
    const auto at = watcher->take(semaphore, 1);
    REQUIRE(at.has_value());
    CHECK(*at >= signaled_us);
    // À la sortie de l'attente, bien avant son délai de 100 ms (marge pour une machine chargée).
    CHECK(*at - signaled_us < 30'000);
    CHECK_FALSE(watcher->take(semaphore, 1).has_value());  // rendue une fois
    // Oublier rend la main sans attendre la fin du délai de 100 ms.
    watcher->watch(semaphore, 2);
    const std::int64_t before = host_now_us();
    watcher->forget_all();
    CHECK(host_now_us() - before < 80'000);
  }
  pl_vulkan_sem_destroy(device.vk->gpu, &semaphore);
}

TEST_CASE("veilleur d'écriture : une erreur de vkWaitSemaphores le dit mort, forget_all rend la main",
          "[app][session]") {
  std::atomic<int> wakes{0};
  int device_storage = 0;
  auto* device = reinterpret_cast<VkDevice>(&device_storage);  // jamais déréférencé : l'Api est simulée
  const RingWatcher::Api api{.wait = lost_wait,
                             .counter = fake_counter,
                             .signal = fake_signal,
                             .create = fake_create,
                             .destroy = fake_destroy};
  auto watcher = RingWatcher::start_with(device, api, wake, &wakes);
  REQUIRE(watcher);
  int semaphore_storage = 0;
  watcher->watch(reinterpret_cast<VkSemaphore>(&semaphore_storage), 1);
  // Le thread sort à sa première attente.
  const std::int64_t start = host_now_us();
  while (watcher->alive() && host_now_us() - start < 2'000'000)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  CHECK_FALSE(watcher->alive());  // main sonde de nouveau l'anneau (Screen::wakes_on_write)
  // Avant : forget_all attendait pour toujours un thread fini.
  const std::int64_t asked = host_now_us();
  watcher->forget_all();
  CHECK(host_now_us() - asked < 100'000);
  CHECK(wakes.load() == 0);
}

}  // namespace ajp::app::play
