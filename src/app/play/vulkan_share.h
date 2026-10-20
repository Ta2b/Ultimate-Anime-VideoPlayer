// L'appareil Vulkan de l'écran prêté à FFmpeg : un AVHWDeviceContext sur l'instance, l'appareil
// physique et le VkDevice de libplacebo, avec ses files (graphique, calcul, transfert, vidéo) et
// ses extensions. Le décodeur Vulkan Video y décode des images que le backend Vulkan, qui adopte
// le même appareil, lit en place (core/frame.h, VulkanImageAccess) ; l'encodeur Vulkan Video y
// reçoit les images que l'export rend directement dans leurs plans. Sans ce partage, FFmpeg crée
// son propre VkDevice et chaque image repasse par la mémoire hôte.
//
// FFmpeg ne détruit que les appareils qu'il a créés : celui-ci reste à libplacebo, et le partage
// doit partir avant lui (PlaceboScreen::Impl).
#pragma once

#include <expected>
#include <memory>

#include <libplacebo/vulkan.h>

#include "core/types.h"
#include "media/hw_api.h"

struct AVBufferRef;

namespace ajp::app::play {

class VulkanShare final : public media::HwDeviceSource {
 public:
  // Refus VendorCallFailed : FFmpeg n'accepte pas l'appareil (fonctions, files, extensions).
  static std::expected<std::unique_ptr<VulkanShare>, Refusal> create(pl_vk_inst inst, pl_vulkan vk);
  ~VulkanShare() override;
  VulkanShare(const VulkanShare&) = delete;
  VulkanShare& operator=(const VulkanShare&) = delete;

  // Nouvelle référence sur l'appareil partagé pour Vulkan ; nullptr pour toute autre API.
  AVBufferRef* device(media::HwApi api) const override;
  // La référence du partage, empruntée : pour l'encodeur de l'export.
  AVBufferRef* vulkan() const { return device_; }

 private:
  VulkanShare() = default;
  AVBufferRef* device_ = nullptr;
};

}  // namespace ajp::app::play
