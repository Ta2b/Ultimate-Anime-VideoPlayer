// Adoption d'un appareil existant, réglages communs aux deux modes et destruction.
// La création d'un appareil propre est dans context_create.cpp, la soumission dans queue.cpp.
#include "gpu/vulkan/compute/context.h"

#include <format>
#include <utility>
#include <vector>

#include <spdlog/logger.h>

#include "gpu/vulkan/compute/log.h"
#include "gpu/vulkan/compute/sync.h"

namespace ajp::gpu::vk {
namespace {

std::string version_text(std::uint32_t version) {
  return std::format("{}.{}", VK_API_VERSION_MAJOR(version), VK_API_VERSION_MINOR(version));
}

// La famille adoptée existe et fait du calcul ; l'appareil y a créé count files, dont index.
// La famille peut en offrir davantage (16 sur la famille 0 d'une RTX) : seules les files créées
// existent, vkGetDeviceQueue au-delà est un usage invalide.
Done check_queue(const InstanceFunctions& fn, VkPhysicalDevice device, std::uint32_t family,
                 std::uint32_t index, std::uint32_t created) {
  std::uint32_t count = 0;
  fn.vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
  std::vector<VkQueueFamilyProperties> families(count);
  fn.vkGetPhysicalDeviceQueueFamilyProperties(device, &count, families.data());
  if (family >= count) {
    return refuse(RefusalCode::DeviceUnavailable,
                  std::format("famille de files {} adoptée, l'appareil en a {}", family, count));
  }
  const VkQueueFamilyProperties& props = families[family];
  if ((props.queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) {
    return refuse(RefusalCode::DeviceUnavailable,
                  std::format("la famille de files {} ne calcule pas", family));
  }
  if (created == 0 || created > props.queueCount) {
    return refuse(RefusalCode::DeviceUnavailable,
                  std::format("{} files créées dans la famille {}, de 1 à {} possibles", created, family,
                              props.queueCount));
  }
  if (index >= created) {
    return refuse(RefusalCode::DeviceUnavailable,
                  std::format("file {} adoptée, {} créées dans la famille {}", index, created, family));
  }
  return {};
}

}  // namespace

std::expected<std::unique_ptr<VkContext>, Refusal> VkContext::adopt(AdoptedDevice adopted) {
  if (adopted.get_instance_proc_addr == nullptr || adopted.instance == VK_NULL_HANDLE ||
      adopted.physical_device == VK_NULL_HANDLE || adopted.device == VK_NULL_HANDLE) {
    return refuse(
        RefusalCode::DeviceUnavailable,
        "appareil adopté incomplet : vkGetInstanceProcAddr, instance, appareil physique et VkDevice "
        "sont obligatoires");
  }
  // Un verrou sans son déverrou (ou l'inverse) laisserait la file partagée verrouillée.
  if (static_cast<bool>(adopted.lock_queue) != static_cast<bool>(adopted.unlock_queue)) {
    return refuse(RefusalCode::InputMismatch,
                  "appareil adopté : lock_queue et unlock_queue vont ensemble, ou aucun des deux");
  }
  if (adopted.api_version < kMinApiVersion) {
    return refuse(RefusalCode::FeatureUnavailable,
                  std::format("instance adoptée en Vulkan {}, {} au moins", version_text(adopted.api_version),
                              version_text(kMinApiVersion)));
  }
  std::unique_ptr<VkContext> ctx(new VkContext());
  ctx->get_instance_proc_addr_ = adopted.get_instance_proc_addr;
  ctx->instance_ = adopted.instance;
  ctx->physical_device_ = adopted.physical_device;
  ctx->device_ = adopted.device;
  ctx->queue_family_ = adopted.queue_family;
  ctx->queue_index_ = adopted.queue_index;
  ctx->lock_queue_ = std::move(adopted.lock_queue);
  ctx->unlock_queue_ = std::move(adopted.unlock_queue);

  auto instance_fn = load_instance(ctx->get_instance_proc_addr_, ctx->instance_);
  if (!instance_fn) return std::unexpected(instance_fn.error());
  ctx->instance_fn_ = *instance_fn;
  auto device_fn = load_device(ctx->instance_fn_, ctx->device_);
  if (!device_fn) return std::unexpected(device_fn.error());
  ctx->device_fn_ = *device_fn;

  ctx->caps_ = probe_device(ctx->instance_fn_, ctx->physical_device_, adopted.api_version);
  if (ctx->caps_.api_version < kMinApiVersion) {
    return refuse(RefusalCode::FeatureUnavailable,
                  std::format("{} : Vulkan {}, {} au moins", ctx->caps_.device_name,
                              version_text(ctx->caps_.api_version), version_text(kMinApiVersion)));
  }
  ctx->caps_.enabled = features_from_chain(adopted.features);
  ctx->caps_.extensions = std::move(adopted.extensions);
  if (!ctx->caps_.enabled.timeline_semaphore) {
    return refuse(RefusalCode::FeatureUnavailable,
                  "timelineSemaphore n'est pas activé sur l'appareil adopté");
  }
  if (auto ok = check_queue(ctx->instance_fn_, ctx->physical_device_, ctx->queue_family_, ctx->queue_index_,
                            adopted.queue_count);
      !ok) {
    return std::unexpected(ok.error());
  }
  if (auto ok = ctx->finish_setup(); !ok) return std::unexpected(ok.error());
  ctx->log().info("appareil adopté : {}", describe(ctx->caps_));
  return ctx;
}

Done VkContext::finish_setup() {
  if (!log_) log_ = make_logger();
  device_fn_.vkGetDeviceQueue(device_, queue_family_, queue_index_, &queue_);
  if (queue_ == VK_NULL_HANDLE) {
    return refuse(
        RefusalCode::DeviceUnavailable,
        std::format("vkGetDeviceQueue(famille {}, file {}) : aucune file", queue_family_, queue_index_));
  }
  const VkCommandPoolCreateInfo pool{.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                     .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                                     .queueFamilyIndex = queue_family_};
  if (auto ok =
          check(device_fn_.vkCreateCommandPool(device_, &pool, nullptr, &once_pool_), "vkCreateCommandPool");
      !ok) {
    return ok;
  }
  auto done = create_timeline(device_fn_, device_, 0);
  if (!done) return std::unexpected(done.error());
  once_done_ = *done;
  return {};
}

// Une création interrompue laisse des poignées nulles ou des fonctions non chargées : chaque
// destruction vérifie les deux.
VkContext::~VkContext() {
  const DeviceFunctions& f = device_fn_;
  if (device_ != VK_NULL_HANDLE) {
    // Fin de vie de l'appareil : tout travail encore soumis s'achève avant toute destruction.
    if (owns_device_ && f.vkDeviceWaitIdle != nullptr) f.vkDeviceWaitIdle(device_);
    // Un run_once dont l'attente a expiré peut être encore en vol. Sur un appareil adopté, que
    // l'on n'arrête pas, on attend sa fin ; s'il ne finit pas, pool et sémaphore restent en place.
    const bool idle = owns_device_ || wait_once_idle();
    if (idle && once_done_ != VK_NULL_HANDLE) f.vkDestroySemaphore(device_, once_done_, nullptr);
    if (idle && once_pool_ != VK_NULL_HANDLE) f.vkDestroyCommandPool(device_, once_pool_, nullptr);
    if (owns_device_ && f.vkDestroyDevice != nullptr) f.vkDestroyDevice(device_, nullptr);
  }
  destroy_messenger();
  if (owns_instance_ && instance_fn_.vkDestroyInstance != nullptr) {
    instance_fn_.vkDestroyInstance(instance_, nullptr);
  }
}

bool VkContext::wait_once_idle() const {
  if (once_count_ == 0 || once_done_ == VK_NULL_HANDLE || device_fn_.vkWaitSemaphores == nullptr) return true;
  const VkSemaphoreWaitInfo wait{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                                 .semaphoreCount = 1,
                                 .pSemaphores = &once_done_,
                                 .pValues = &once_count_};
  const auto timeout = static_cast<std::uint64_t>(std::chrono::nanoseconds(kTeardownTimeout).count());
  const VkResult result = device_fn_.vkWaitSemaphores(device_, &wait, timeout);
  if (result == VK_SUCCESS) return true;
  if (log_) {
    log_->error("fin du contexte : run_once {} inachevé ({}), son pool reste à l'appareil", once_count_,
                result_name(result));
  }
  return false;
}

}  // namespace ajp::gpu::vk
