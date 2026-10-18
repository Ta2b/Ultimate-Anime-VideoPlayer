// Appareils de décodage matériel : configuration du codec, création du
// contexte FFmpeg par API, flux CUDA de décodage. Interne au module.
#pragma once

#include <cstdint>
#include <expected>
#include <memory>
#include <optional>

#include "core/types.h"
#include "media/av_util.h"
#include "media/cuda_driver.h"
#include "media/hw_api.h"

namespace ajp::media {

// Flux CUDA non bloquant posé dans AVCUDADeviceContext::stream : la copie de
// sécurité de NVDEC s'y fait. Retiré du contexte, puis détruit, avec l'objet.
class CudaDecodeStream {
 public:
  static std::expected<std::unique_ptr<CudaDecodeStream>, Refusal> attach(std::shared_ptr<CudaDriver> driver,
                                                                          AVBufferRef& device);
  ~CudaDecodeStream();
  CudaDecodeStream(const CudaDecodeStream&) = delete;
  CudaDecodeStream& operator=(const CudaDecodeStream&) = delete;

  // Événement de fin de copie, à poser sur chaque image.
  std::expected<ReadyEvent, Refusal> record() const;

 private:
  CudaDecodeStream(std::shared_ptr<CudaDriver> driver, AVBufferRef& device, std::uintptr_t context,
                   std::uintptr_t stream);

  std::shared_ptr<CudaDriver> driver_;
  BufferPtr device_;
  std::uintptr_t context_;
  std::uintptr_t stream_;
};

struct HwDevice {
  HwApi api = HwApi::Software;
  const AVCodec* codec = nullptr;            // décodeur qui porte la configuration matérielle
  AVPixelFormat format = AV_PIX_FMT_NONE;    // format matériel attendu dans get_format
  BufferPtr device;                          // AVHWDeviceContext
  std::unique_ptr<CudaDecodeStream> stream;  // Cuda seulement
};

// Décodeur et appareil de cette API pour ce codec, ou refus nommé : codec sans
// cette API, bibliothèque ou appareil absent, UUID différent du rendu.
std::expected<HwDevice, Refusal> open_hw_device(AVCodecID codec, HwApi api, const DecoderOptions& options);

// UUID d'un appareil Vulkan (VkPhysicalDeviceIDProperties::deviceUUID).
std::optional<DeviceUuid> vulkan_device_uuid(const AVBufferRef& device);

}  // namespace ajp::media
