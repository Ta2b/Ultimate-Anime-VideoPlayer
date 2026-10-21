// Autotuner du moteur Vulkan : pour chaque nœud qui a plusieurs variantes (convolutions), garde
// la plus rapide sur cet appareil parmi celles dont la sortie concorde avec celle du noyau direct
// (un pilote dont les matrices coopératives se trompent ne casse pas l'image). Ordre de recherche : table
// pré-réglée de la famille, cache (par GPU et pilote, modèle et résolution, signature de la couche), puis
// mesure : chaque candidat est chronométré seul, sur des entrées pseudo-aléatoires, par requêtes
// d'horodatage : toutes ses passes (échauffement compris, plusieurs dispatchs enchaînés chacune) dans
// une soumission, une estampille entre deux, et la médiane des passes comptées l'emporte.
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <map>
#include <optional>
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
#include "gpu/vulkan/compute/buffer.h"
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

// Les entrées d'une mesure : une valeur pseudo-aléatoire de [-1, 1] par élément, hachage de son
// index modulo la période du type, arrondie au type (float16 ou float). Une période première : un
// voisin, une ligne ou un plan décalés n'y lisent pas la même valeur, si bien qu'un noyau qui se
// trompe de voisin, de remplissage ou de canal sort une image discordante. 0 pour les autres types.
inline constexpr std::uint32_t kPatternPeriodF16 = 32749;  // 65 498 octets
inline constexpr std::uint32_t kPatternPeriodF32 = 16381;  // 65 524 octets
float tuning_input(ir::DataType type, std::uint64_t index);

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
  // La sortie du nœud sur les entrées de tuning_input : choose() la compare à celle du noyau direct.
  std::vector<float> output;
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
  std::size_t rejected() const { return rejected_; }  // candidats écartés : sortie discordante
  bool timestamps() const { return period_ns_ > 0.0; }

 private:
  void load_cache();
  // Le motif de tuning_input du type, envoyé au premier appel puis copié sur chaque entrée ;
  // nullptr hors float16 et float.
  std::expected<const Buffer*, Refusal> pattern(ir::DataType type);

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
  std::size_t rejected_ = 0;
  std::optional<Buffer> pattern_f16_;
  std::optional<Buffer> pattern_f32_;
};

}  // namespace ajp::gpu::vk::net
