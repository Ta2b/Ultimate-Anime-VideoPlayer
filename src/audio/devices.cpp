#include "audio/devices.h"

#include <SDL3/SDL_audio.h>
#include <SDL3/SDL_init.h>
#include <SDL3/SDL_stdinc.h>

namespace ajp::audio {

std::vector<AudioDevice> playback_devices() {
  std::vector<AudioDevice> out;
  if (!SDL_WasInit(SDL_INIT_AUDIO)) return out;
  int count = 0;
  SDL_AudioDeviceID* ids = SDL_GetAudioPlaybackDevices(&count);
  if (ids == nullptr) return out;
  SDL_AudioSpec preferred{};
  SDL_GetAudioDeviceFormat(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &preferred, nullptr);
  const char* default_name = nullptr;
  (void)default_name;
  for (int i = 0; i < count; ++i) {
    AudioDevice device{.id = ids[i]};
    if (const char* name = SDL_GetAudioDeviceName(ids[i])) device.name = name;
    SDL_AudioSpec spec{};
    if (SDL_GetAudioDeviceFormat(ids[i], &spec, nullptr)) {
      device.channels = spec.channels;
      device.sample_rate = spec.freq;
    }
    out.push_back(std::move(device));
  }
  SDL_free(ids);
  // Le défaut : SDL ne le nomme pas dans la liste ; le format du défaut le désigne quand l'appareil est
  // unique à l'avoir, sinon le premier.
  SDL_AudioSpec spec{};
  int frames = 0;
  if (SDL_GetAudioDeviceFormat(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, &frames) && !out.empty()) {
    const char* name = SDL_GetAudioDeviceName(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK);
    bool marked = false;
    if (name != nullptr) {
      for (AudioDevice& device : out) {
        if (device.name == name) {
          device.is_default = true;
          marked = true;
        }
      }
    }
    if (!marked) out.front().is_default = true;
  }
  return out;
}

std::optional<AudioDevice> resolve_device(const std::vector<AudioDevice>& devices, std::string_view name) {
  if (!name.empty()) {
    for (const AudioDevice& device : devices)
      if (device.name == name) return device;
  }
  for (const AudioDevice& device : devices)
    if (device.is_default) return device;
  return std::nullopt;
}

}  // namespace ajp::audio
