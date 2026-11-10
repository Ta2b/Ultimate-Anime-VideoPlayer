// Chargeur Vulkan ouvert à l'exécution : libvulkan.so.1, vulkan-1.dll ou libvulkan.1.dylib.
// Le lecteur ne se lie jamais à Vulkan ; le cas normal reçoit le vkGetInstanceProcAddr de
// libplacebo, et cette bibliothèque ne sert qu'aux tests, aux outils et au banc.
#pragma once

#include <expected>
#include <filesystem>
#include <utility>

#include "core/dynamic_library.h"
#include "core/types.h"
#include "gpu/vulkan/compute/vulkan.h"

namespace ajp::gpu::vk {

class VulkanLibrary {
 public:
  // Le chargeur du système, cherché par le chargeur dynamique sous son nom habituel.
  static std::expected<VulkanLibrary, Refusal> open_system();
  // Un chargeur précis : tests (chemin factice), outils. Refus LibraryUnavailable si la
  // bibliothèque ne s'ouvre pas ou n'exporte pas vkGetInstanceProcAddr.
  static std::expected<VulkanLibrary, Refusal> open(const std::filesystem::path& path);

  // Valide tant que la bibliothèque vit.
  PFN_vkGetInstanceProcAddr get_instance_proc_addr() const { return get_instance_proc_addr_; }

 private:
  VulkanLibrary(DynamicLibrary library, PFN_vkGetInstanceProcAddr entry)
      : library_(std::move(library)), get_instance_proc_addr_(entry) {}
  static std::expected<VulkanLibrary, Refusal> wrap(DynamicLibrary library);

  // L'ouverture, le déplacement et la fermeture sont ceux de core (A45-09) : une correction de l'un
  // (texte de l'erreur, dossiers de recherche) vaut pour l'autre.
  DynamicLibrary library_;
  PFN_vkGetInstanceProcAddr get_instance_proc_addr_ = nullptr;
};

}  // namespace ajp::gpu::vk
