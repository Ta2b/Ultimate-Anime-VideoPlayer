#include "gpu/vulkan/compute/sync.h"

#include <format>
#include <utility>

namespace ajp::gpu::vk {

std::expected<VkSemaphore, Refusal> create_timeline(const DeviceFunctions& fn, VkDevice device,
                                                    std::uint64_t initial) {
  const VkSemaphoreTypeCreateInfo type{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                                       .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
                                       .initialValue = initial};
  const VkSemaphoreCreateInfo info{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &type};
  VkSemaphore semaphore = VK_NULL_HANDLE;
  if (auto ok = check(fn.vkCreateSemaphore(device, &info, nullptr, &semaphore), "vkCreateSemaphore"); !ok) {
    return std::unexpected(ok.error());
  }
  return semaphore;
}

std::expected<TimelineSemaphore, Refusal> TimelineSemaphore::create(const VkContext& context,
                                                                    std::uint64_t initial) {
  auto semaphore = create_timeline(context.fn(), context.device(), initial);
  if (!semaphore) return std::unexpected(semaphore.error());
  return TimelineSemaphore(&context.fn(), context.device(), *semaphore);
}

TimelineSemaphore::TimelineSemaphore(TimelineSemaphore&& other) noexcept
    : fn_(other.fn_), device_(other.device_), semaphore_(std::exchange(other.semaphore_, VK_NULL_HANDLE)) {}

TimelineSemaphore& TimelineSemaphore::operator=(TimelineSemaphore&& other) noexcept {
  if (this != &other) {
    destroy();
    fn_ = other.fn_;
    device_ = other.device_;
    semaphore_ = std::exchange(other.semaphore_, VK_NULL_HANDLE);
  }
  return *this;
}

TimelineSemaphore::~TimelineSemaphore() {
  destroy();
}

void TimelineSemaphore::destroy() {
  if (semaphore_ != VK_NULL_HANDLE) fn_->vkDestroySemaphore(device_, semaphore_, nullptr);
  semaphore_ = VK_NULL_HANDLE;
}

std::expected<std::uint64_t, Refusal> TimelineSemaphore::value() const {
  std::uint64_t current = 0;
  if (auto ok =
          check(fn_->vkGetSemaphoreCounterValue(device_, semaphore_, &current), "vkGetSemaphoreCounterValue");
      !ok) {
    return std::unexpected(ok.error());
  }
  return current;
}

Done TimelineSemaphore::signal(std::uint64_t value) {
  // Un signal qui ne dépasse pas la valeur courante est un usage invalide : refusé avant le pilote.
  auto current = this->value();
  if (!current) return std::unexpected(current.error());
  if (value <= *current) {
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("signal hôte de la valeur {}, le sémaphore est déjà à {}", value, *current));
  }
  const VkSemaphoreSignalInfo info{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO, .semaphore = semaphore_, .value = value};
  return check(fn_->vkSignalSemaphore(device_, &info), std::format("vkSignalSemaphore({})", value));
}

}  // namespace ajp::gpu::vk
