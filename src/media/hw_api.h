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
  // Décodage logiciel, [0, 64]. 0 : automatique, un thread par cœur et
  // kAutoDecoderThreads au plus. Chaque thread d'images retarde la première
  // image d'un paquet vidéo, donc de quelques paquets audio lus avant elle.
  int threads = 0;
  // Vulkan ne décode par défaut que les codecs Vulkan Video (H.264, HEVC, AV1,
  // VP9), sur la file vidéo du GPU. Les décodeurs Vulkan par compute shaders de
  // FFmpeg (ffv1, prores, prores_raw, dpx, apv) ne s'essaient que si ceci est vrai :
  // ils prennent le calcul que gpu réserve à l'upscaling et à l'interpolation, et
  // leur échec ne se voit qu'au premier point clé, après l'ouverture.
  bool vulkan_compute = false;
  // Chaque image matérielle copiée en mémoire hôte (av_hwframe_transfer_data) : le décodeur
  // matériel décharge le CPU même pour un backend qui n'importe pas ses surfaces (Vulkan avec
  // VA-API). Quelques Mo par image sur le bus, contre le décodage logiciel d'une HEVC 10 bits.
  // Sauf les images d'un appareil partagé (devices), que son propriétaire lit en place.
  bool download = false;
  // Images décodées que l'aval tient à la fois (SourceLoop, l'étage gpu, le backend), [0, 32].
  // Non nul sous NVDEC : les surfaces sortent sans copie de sécurité
  // (AV_HWACCEL_FLAG_UNSAFE_OUTPUT) et extra_hw_frames monte à ce nombre au moins. Une image
  // tenue garde alors une surface de sortie du décodeur, réservée d'avance (nvdec.c:417 : autant
  // que le pool, 64 au plus), au lieu d'une copie dans le pool CUDA ; une de plus ne se
  // mapperait pas, et l'image serait perdue. Le prix : environ 25 surfaces de plus qu'avec la
  // copie, ~80 Mo en 1080p 8 bits ; au-delà de kMaxDirectSurfacePixels, la copie reste.
  int direct_surfaces = 0;
};

inline constexpr int kMaxExtraHwFrames = 32;
// Au-delà de 2560 × 1440, les surfaces directes coûteraient plus de 300 Mo (10 bits) : la copie
// de sécurité, une par image, coûte moins qu'elles sur une carte de 6 Go.
inline constexpr std::int64_t kMaxDirectSurfacePixels = std::int64_t{2560} * 1440;
inline constexpr int kMaxDecoderThreads = 64;
inline constexpr int kAutoDecoderThreads = 8;

// Ordre d'essai par défaut de la plateforme, logiciel en dernier.
std::vector<HwApi> default_attempts();

// Un essai écarté et pourquoi.
struct Attempt {
  HwApi api = HwApi::Software;
  Refusal why{};
};

}  // namespace ajp::media
