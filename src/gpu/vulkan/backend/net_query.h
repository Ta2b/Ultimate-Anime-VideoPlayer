// Requêtes d'horodatage du moteur de réseaux, partagées par l'autotuner (durée d'un noyau seul)
// et le profil des dispatchs (net_profile.h) : fonctions chargées hors des tables du socle,
// support de la file compute, pool détruit avec son objet.
#pragma once

#include <cstdint>

#include "gpu/vulkan/compute/context.h"
#include "gpu/vulkan/compute/refuse.h"
#include "gpu/vulkan/compute/vulkan.h"

namespace ajp::gpu::vk::net {

// Fonctions des requêtes d'horodatage, hors des tables du socle.
struct QueryFunctions {
  PFN_vkCreateQueryPool create = nullptr;
  PFN_vkDestroyQueryPool destroy = nullptr;
  PFN_vkCmdResetQueryPool reset = nullptr;
  PFN_vkCmdWriteTimestamp write = nullptr;
  PFN_vkGetQueryPoolResults results = nullptr;
};

QueryFunctions query_functions(const VkContext& context);

// Ce que la file compute offre en horodatage : période d'un tick en nanosecondes et nombre de
// bits valides d'une estampille. period_ns vaut 0 sans horodatage (file sans bits valides,
// fonctions absentes).
struct TimestampSupport {
  double period_ns = 0.0;
  std::uint32_t valid_bits = 0;
};
TimestampSupport timestamp_support(const VkContext& context);

// Période d'horodatage de la file compute en nanosecondes ; 0 sans horodatage.
double timestamp_period(const VkContext& context);

// Durée en nanosecondes entre deux estampilles de valid_bits bits (le compteur peut déborder
// entre les deux : la différence se fait modulo 2^valid_bits).
double ticks_to_ns(std::uint64_t begin, std::uint64_t end, const TimestampSupport& support);

// Pool de count requêtes, détruit avec l'objet.
class QueryPool {
 public:
  QueryPool(const VkContext& context, QueryFunctions fn, std::uint32_t count)
      : context_(context), fn_(fn), count_(count) {}
  QueryPool(const QueryPool&) = delete;
  QueryPool& operator=(const QueryPool&) = delete;
  QueryPool(QueryPool&&) = delete;
  QueryPool& operator=(QueryPool&&) = delete;
  ~QueryPool();
  Done create();
  VkQueryPool handle() const { return pool_; }
  std::uint32_t count() const { return count_; }
  const QueryFunctions& fn() const { return fn_; }

 private:
  const VkContext& context_;
  QueryFunctions fn_;
  std::uint32_t count_;
  VkQueryPool pool_ = VK_NULL_HANDLE;
};

}  // namespace ajp::gpu::vk::net
