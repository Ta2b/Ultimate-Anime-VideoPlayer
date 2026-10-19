// API de décodage matériel et options du décodeur vidéo.
#pragma once

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include "core/gpu_device.h"
#include "core/types.h"

struct AVBufferRef;

namespace ajp::media {

enum class HwApi : std::uint8_t { Cuda, Vulkan, D3d11va, Vaapi, VideoToolbox, Software };

std::string_view name(HwApi api);

// UUID de l'appareil de rendu (pl_gpu.uuid) : le type unique de core/gpu_device.h.
using DeviceUuid = ajp::DeviceUuid;

// Contexte d'appareil déjà créé par un autre module, par exemple le VkDevice
// de libplacebo enveloppé en AVHWDeviceContext : le décodeur le partage au lieu
// d'en créer un. Appelé à l'ouverture, depuis le thread qui ouvre le décodeur.
class HwDeviceSource {
 public:
  virtual ~HwDeviceSource() = default;
  // Nouvelle référence sur l'AVHWDeviceContext de cette API, ou nullptr.
  virtual AVBufferRef* device(HwApi api) const = 0;
};

struct DecoderOptions {
  // Ordre d'essai ; sans Software, aucun repli logiciel n'est permis.
  std::vector<HwApi> attempts;
  std::optional<DeviceUuid> render_device;  // exigé par Cuda et, s'il est connu, vérifié par Vulkan
  const HwDeviceSource* devices = nullptr;
  int extra_hw_frames = 4;  // surfaces en plus pour l'anticipation de gpu, [0, 32]
  int threads = 0;          // décodage logiciel, 0 : automatique, [0, 64]
};

inline constexpr int kMaxExtraHwFrames = 32;
inline constexpr int kMaxDecoderThreads = 64;

// Ordre d'essai par défaut de la plateforme, logiciel en dernier.
std::vector<HwApi> default_attempts();

// Un essai écarté et pourquoi.
struct Attempt {
  HwApi api = HwApi::Software;
  Refusal why{};
};

}  // namespace ajp::media
