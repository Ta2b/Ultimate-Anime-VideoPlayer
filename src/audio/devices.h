// Les appareils de sortie du système (SDL3) : la liste, leurs canaux, celui par défaut. Le lecteur ouvre l'appareil
// choisi dans les réglages (OutputRequest::device) ou celui du système, et relit le format quand le défaut change.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ajp::audio {

struct AudioDevice {
  std::uint32_t id = 0;  // SDL_AudioDeviceID
  std::string name;
  int channels = 0;      // ceux du format préféré de l'appareil ; 0 : inconnus
  int sample_rate = 0;
  bool is_default = false;
};

// Les appareils de lecture, le défaut du système marqué. Vide si le sous-système audio n'est pas prêt.
std::vector<AudioDevice> playback_devices();

// L'appareil de ce nom ; celui du système si name est vide ou n'existe plus.
std::optional<AudioDevice> resolve_device(const std::vector<AudioDevice>& devices, std::string_view name);

// Home cinéma n'a de sens que pour un appareil qui reçoit plus de 2 canaux.
inline bool multichannel(const AudioDevice& device) { return device.channels > 2; }

}  // namespace ajp::audio
