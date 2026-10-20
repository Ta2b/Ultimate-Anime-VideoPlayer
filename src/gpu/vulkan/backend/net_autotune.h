// Autotuner du moteur Vulkan : pour chaque nœud qui a plusieurs variantes (convolutions), garde
// la plus rapide sur cet appareil. Ordre de recherche : table pré-réglée de la famille, cache
// (par GPU et pilote, modèle et résolution, signature de la couche), puis mesure : chaque
// candidat est chronométré seul, par requêtes d'horodatage autour de plusieurs dispatchs
// enchaînés, après échauffement, et la médiane de plusieurs passes compte.
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <map>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

#include "core/types.h"
#include "gpu/ir/ir.h"
#include "gpu/vulkan/backend/device_profile.h"
#include "gpu/vulkan/backend/family.h"
#include "gpu/vulkan/backend/net_program.h"
#include "gpu/vulkan/compute/context.h"
#include "gpu/vulkan/compute/memory.h"
#include "gpu/vulkan/compute/pipeline.h"

namespace ajp::gpu::vk::net {

// Signature d'une couche : opération, attributs et formes, types et dispositions de ses
// tenseurs, sans les valeurs des poids ni les noms.
std::uint64_t layer_signature(const ir::Graph& graph, std::size_t node);

// Réglages connus d'avance pour une famille. La structure est prête ; les tables sont vides
// tant qu'aucune mesure de référence ne les remplit.
struct PresetEntry {
  std::uint64_t signature = 0;
  std::string_view variant;
  std::uint32_t tuning0 = 0;
  std::uint32_t tuning1 = 0;
};
struct PresetTable {
  GpuFamily family = GpuFamily::Unknown;
  std::span<const PresetEntry> entries;
};
std::span<const PresetTable> preset_tables();

struct TunerOptions {
  std::uint32_t warmup = 2;   // passes non comptées
  std::uint32_t passes = 5;   // passes mesurées : la médiane compte
  std::uint32_t repeats = 3;  // dispatchs enchaînés par passe
};

// Une mesure : le choix et sa médiane, en microsecondes par exécution du nœud.
struct Measured {
  KernelChoice choice;
  double median_us = 0.0;
  double spread_us = 0.0;  // écart type des passes
};

class Autotuner {
 public:
  Autotuner(VkContext& context, MemoryAllocator& memory, PipelineCache& pipelines, DeviceProfile profile,
            std::filesystem::path cache_file, TunerOptions options = {});

  // Le meilleur choix du nœud parmi candidates (non vide). scope distingue le modèle et la
  // résolution. Refus : Cancelled (stop demandé), ceux de l'abaissement et de l'exécution.
  std::expected<KernelChoice, Refusal> choose(const ir::Graph& graph, std::size_t node,
                                              std::span<const KernelChoice> candidates,
                                              std::string_view scope, const std::stop_token& stop = {});

  // Mesure un choix seul (banc, tests). Refus : ceux de l'abaissement et de l'exécution.
  std::expected<Measured, Refusal> measure(const ir::Graph& graph, std::size_t node,
                                           const KernelChoice& choice);

  // Écrit le cache, une entrée par ligne. Refus CacheIo.
  std::expected<void, Refusal> save() const;

  std::size_t measured() const { return measured_; }  // choix tranchés par mesure
  std::size_t cached() const { return cached_; }      // choix lus dans le cache ou une table
  bool timestamps() const { return period_ns_ > 0.0; }

 private:
  void load_cache();

  VkContext& context_;
  MemoryAllocator& memory_;
  PipelineCache& pipelines_;
  DeviceProfile profile_;
  std::filesystem::path cache_file_;
  TunerOptions options_;
  double period_ns_ = 0.0;  // 0 : pas de requêtes d'horodatage, mesure hôte
  std::map<std::string, KernelChoice> cache_;
  std::size_t measured_ = 0;
  std::size_t cached_ = 0;
};

}  // namespace ajp::gpu::vk::net
