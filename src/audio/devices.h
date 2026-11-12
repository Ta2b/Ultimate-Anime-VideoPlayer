// Les appareils de sortie du système (SDL3) : la liste, leurs canaux, celui par défaut. Le lecteur ouvre
// l'appareil choisi dans les réglages (OutputRequest::device) ou celui du système. Le format de la file ne
// change pas quand le défaut change : SDL convertit, et le limiteur doux du rappel postmix couvre le mixage
// vers un autre nombre de canaux (B16-07).
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
  int channels = 0;  // ceux du format préféré de l'appareil ; 0 : inconnus
  int sample_rate = 0;
  bool is_default = false;
};

// Les appareils de lecture, le défaut du système marqué. Vide si le sous-système audio n'est pas prêt.
std::vector<AudioDevice> playback_devices();

// Marque comme défaut l'appareil de ce nom, un seul : le premier. Deux DAC identiques portent le même nom, et les
// marquer tous deux ferait choisir au hasard (C16-09). Sans appareil de ce nom, personne n'est marqué : le
// premier de la liste n'est pas le défaut du système.
void mark_default_device(std::vector<AudioDevice>& devices, std::string_view default_name);

// Le défaut du système tel que SDL le décrit (son format préféré), pour quand la liste ne le nomme pas.
std::optional<AudioDevice> system_default_device();

// L'appareil de ce nom ; celui du système si name est vide ou n'existe plus.
std::optional<AudioDevice> resolve_device(const std::vector<AudioDevice>& devices, std::string_view name);

}  // namespace ajp::audio
