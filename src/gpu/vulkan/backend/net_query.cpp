#include "gpu/vulkan/backend/net_query.h"

#include <vector>

namespace ajp::gpu::vk::net {

QueryFunctions query_functions(const VkContext& context) {
  const auto get = [&](const char* name) {
    return context.instance_functions().vkGetDeviceProcAddr(context.device(), name);
  };
  return {.create = reinterpret_cast<PFN_vkCreateQueryPool>(get("vkCreateQueryPool")),
          .destroy = reinterpret_cast<PFN_vkDestroyQueryPool>(get("vkDestroyQueryPool")),
          .reset = reinterpret_cast<PFN_vkCmdResetQueryPool>(get("vkCmdResetQueryPool")),
          .write = reinterpret_cast<PFN_vkCmdWriteTimestamp>(get("vkCmdWriteTimestamp")),
          .results = reinterpret_cast<PFN_vkGetQueryPoolResults>(get("vkGetQueryPoolResults"))};
}

TimestampSupport timestamp_support(const VkContext& context) {
  const InstanceFunctions& fn = context.instance_functions();
  VkPhysicalDeviceProperties properties{};
  fn.vkGetPhysicalDeviceProperties(context.physical_device(), &properties);
  std::uint32_t count = 0;
  fn.vkGetPhysicalDeviceQueueFamilyProperties(context.physical_device(), &count, nullptr);
  std::vector<VkQueueFamilyProperties> families(count);
  fn.vkGetPhysicalDeviceQueueFamilyProperties(context.physical_device(), &count, families.data());
  if (context.queue_family() >= count || families[context.queue_family()].timestampValidBits == 0) return {};
  const QueryFunctions q = query_functions(context);
  if (q.create == nullptr || q.destroy == nullptr || q.reset == nullptr || q.write == nullptr ||
      q.results == nullptr) {
    return {};
  }
  return {.period_ns = static_cast<double>(properties.limits.timestampPeriod),
          .valid_bits = families[context.queue_family()].timestampValidBits};
}

double timestamp_period(const VkContext& context) {
  return timestamp_support(context).period_ns;
}

double ticks_to_ns(std::uint64_t begin, std::uint64_t end, const TimestampSupport& support) {
  const std::uint64_t mask =
      support.valid_bits >= 64 ? ~std::uint64_t{0} : ((std::uint64_t{1} << support.valid_bits) - 1);
  return static_cast<double>((end - begin) & mask) * support.period_ns;
}

QueryPool::~QueryPool() {
  if (pool_ != VK_NULL_HANDLE) fn_.destroy(context_.device(), pool_, nullptr);
}

Done QueryPool::create() {
  const VkQueryPoolCreateInfo info{.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                                   .queryType = VK_QUERY_TYPE_TIMESTAMP,
                                   .queryCount = count_};
  return check(fn_.create(context_.device(), &info, nullptr, &pool_), "vkCreateQueryPool");
}

}  // namespace ajp::gpu::vk::net
