// Catalogue des cas du banc : un noyau de KernelRunner à des dimensions et une précision
// données, avec les octets qu'il doit au minimum lire et écrire en mémoire vidéo (ses entrées
// et ses sorties ; ni les intermédiaires, ni les tables de poids). Les entrées sont remplies
// une fois, avant toute mesure, de valeurs plausibles : gris moyen, tenseurs à 0,5, mouvement
// uniforme de quelques pixels (une translation : accès cohérents, comme un flot lisse).
#pragma once

#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "gpu/vulkan/compute/buffer.h"
#include "gpu/vulkan/compute/context.h"
#include "gpu/vulkan/compute/image.h"
#include "gpu/vulkan/compute/kernel_runner.h"
#include "gpu/vulkan/compute/memory.h"
#include "gpu/vulkan/compute/recording.h"

namespace ajp::bench {

struct Rig {
  gpu::vk::VkContext& context;
  gpu::vk::MemoryAllocator& memory;
  gpu::vk::KernelRunner& kernels;
};

// Un cas prêt à rejouer. L'ordre des membres compte : l'enregistrement, déclaré en dernier,
// est détruit en premier (il attend sa dernière exécution), avant les tampons qu'il lie.
struct Prepared {
  std::vector<gpu::vk::Buffer> buffers;
  std::optional<gpu::vk::StorageImage> image;
  std::uint64_t bytes = 0;  // octets lus et écrits au minimum par une exécution
  std::unique_ptr<gpu::vk::Recording> recording;
};

struct Case {
  std::string kernel;     // « ingest NV12 », « warp_blend t=0,5 »...
  std::string size;       // « 1920x1080 », « 1280x720 → 2560x1440 »
  std::string precision;  // « f32 », « f16 », ou « - » pour les noyaux sans tenseur
  // Refus : ceux des noyaux et de l'allocateur ; FeatureUnavailable quand l'appareil n'a pas
  // la précision demandée (le banc saute alors le cas, avec la raison).
  std::function<std::expected<Prepared, Refusal>(Rig&)> prepare;

  std::string label() const { return kernel + " " + size + " " + precision; }
};

// Tous les cas, dans l'ordre du rapport : le plancher de mesure, puis chaque noyau à 1080p et à
// 4K, le rééchantillonnage 720p → 1440p et 1080p → 2160p, et le warp de S3 (1440p, mouvement
// estimé à 720p).
std::vector<Case> catalog();

}  // namespace ajp::bench
