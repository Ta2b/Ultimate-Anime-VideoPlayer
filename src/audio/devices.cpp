#include "audio/devices.h"

#include <SDL3/SDL_audio.h>
#include <SDL3/SDL_init.h>
#include <SDL3/SDL_stdinc.h>

namespace ajp::audio {

void mark_default_device(std::vector<AudioDevice>& devices, std::string_view default_name) {
  for (AudioDevice& device : devices) device.is_default = false;
  if (default_name.empty()) return;
  for (AudioDevice& device : devices) {
    if (device.name == default_name) {
      device.is_default = true;
      return;
    }
  }
}

std::optional<AudioDevice> system_default_device() {
  if (!SDL_WasInit(SDL_INIT_AUDIO)) return std::nullopt;
  SDL_AudioSpec spec{};
  if (!SDL_GetAudioDeviceFormat(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, nullptr)) return std::nullopt;
  AudioDevice device{.id = SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, .channels = spec.channels, .sample_rate = spec.freq,
                     .is_default = true};
  if (const char* name = SDL_GetAudioDeviceName(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK)) device.name = name;
  return device;
}

std::vector<AudioDevice> playback_devices() {
  std::vector<AudioDevice> out;
  if (!SDL_WasInit(SDL_INIT_AUDIO)) return out;
  int count = 0;
  SDL_AudioDeviceID* ids = SDL_GetAudioPlaybackDevices(&count);
  if (ids == nullptr) return out;
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
  // Le défaut : SDL le nomme par son format préféré, hors de la liste.
  if (!out.empty()) {
    const char* name = SDL_GetAudioDeviceName(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK);
    mark_default_device(out, name != nullptr ? name : "");
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
