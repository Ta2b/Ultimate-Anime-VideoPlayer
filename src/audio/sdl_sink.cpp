// File SDL3 : un SDL_AudioStream lié au périphérique de sortie par défaut.
//
// La position jouée se mesure sans course : SDL appelle on_device_request
// verrou du flux tenu, juste avant de tirer des trames, et pushed_ ne change
// que sous ce même verrou. Trames consommées = poussées - encore en file.
#include <climits>
#include <cmath>
#include <format>
#include <string>
#include <string_view>

#include <SDL3/SDL_audio.h>
#include <SDL3/SDL_error.h>
#include <SDL3/SDL_init.h>
#include <SDL3/SDL_log.h>

#include "audio/refuse.h"
#include "audio/sink.h"
#include "core/clock.h"

namespace ajp::audio {
namespace {

std::unexpected<Refusal> refuse_sdl(std::string_view call) {
  return refuse(RefusalCode::AudioDeviceUnavailable, std::format("{} : {}", call, SDL_GetError()));
}

// Sous-système audio de SDL ; SDL compte lui-même les initialisations.
class AudioSubsystem {
 public:
  AudioSubsystem() : ready_(SDL_InitSubSystem(SDL_INIT_AUDIO)) {}
  ~AudioSubsystem() {
    if (ready_) SDL_QuitSubSystem(SDL_INIT_AUDIO);
  }
  AudioSubsystem(const AudioSubsystem&) = delete;
  AudioSubsystem& operator=(const AudioSubsystem&) = delete;
  bool ready() const { return ready_; }

 private:
  bool ready_;
};

class SdlSink final : public AudioSink {
 public:
  SdlSink() = default;
  SdlSink(const SdlSink&) = delete;
  SdlSink& operator=(const SdlSink&) = delete;
  // Détruire le flux ferme le périphérique et attend la fin d'un rappel en cours.
  ~SdlSink() override {
    if (stream_ != nullptr) SDL_DestroyAudioStream(stream_);
  }

  std::expected<void, Refusal> open(const OutputRequest& request);

  PcmFormat format() const override { return format_; }
  std::expected<void, Refusal> put(std::span<const float> samples) override;
  std::int64_t pushed_frames() const override { return pushed_; }
  DeviceReading reading() const override;
  std::int64_t latency_frames() const override;
  void set_paused(bool paused) override;
  void set_gain(float gain) override { SDL_SetAudioStreamGain(stream_, gain); }
  float gain() const override { return SDL_GetAudioStreamGain(stream_); }
  void set_ratio(double ratio) override {
    SDL_SetAudioStreamFrequencyRatio(stream_, static_cast<float>(ratio));
  }
  double ratio() const override { return SDL_GetAudioStreamFrequencyRatio(stream_); }
  std::int64_t clear() override;
  void on_demand(std::function<void(const DeviceReading&)> callback) override;

  // Thread du périphérique, verrou du flux tenu par SDL.
  void notify_demand() const;

 private:
  std::int64_t consumed_locked() const;

  AudioSubsystem subsystem_;
  SDL_AudioStream* stream_ = nullptr;
  PcmFormat format_;
  std::int64_t pushed_ = 0;  // écrit sous le verrou du flux, par le thread audio seul
  std::function<void(const DeviceReading&)> demand_;
};

void SDLCALL on_device_request(void* sink, SDL_AudioStream* /*stream*/, int /*additional*/, int /*total*/) {
  static_cast<const SdlSink*>(sink)->notify_demand();
}

std::expected<void, Refusal> SdlSink::open(const OutputRequest& request) {
  if (!subsystem_.ready()) return refuse_sdl("SDL_InitSubSystem(SDL_INIT_AUDIO)");
  SDL_AudioSpec preferred{};
  if (!SDL_GetAudioDeviceFormat(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &preferred, nullptr)) {
    return refuse_sdl("SDL_GetAudioDeviceFormat");
  }
  const auto format = checked({.sample_rate = request.sample_rate != 0 ? request.sample_rate : preferred.freq,
                               .channels = request.channels != 0 ? request.channels : preferred.channels});
  if (!format) return std::unexpected(format.error());
  const SDL_AudioSpec spec{
      .format = SDL_AUDIO_F32, .channels = format->channels, .freq = format->sample_rate};
  // Le périphérique s'ouvre en pause ; SDL convertit vers son format réel si besoin.
  stream_ = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, nullptr, nullptr);
  if (stream_ == nullptr) return refuse_sdl("SDL_OpenAudioDeviceStream");
  format_ = *format;
  return {};
}

std::expected<void, Refusal> SdlSink::put(std::span<const float> samples) {
  if (samples.empty()) return {};
  const auto channels = static_cast<std::size_t>(format_.channels);
  if (samples.size() % channels != 0 || samples.size_bytes() > static_cast<std::size_t>(INT_MAX)) {
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("{} échantillons pour {} canaux", samples.size(), channels));
  }
  SDL_LockAudioStream(stream_);
  const bool queued = SDL_PutAudioStreamData(stream_, samples.data(), static_cast<int>(samples.size_bytes()));
  if (queued) pushed_ += static_cast<std::int64_t>(samples.size() / channels);
  SDL_UnlockAudioStream(stream_);
  if (!queued) return refuse_sdl("SDL_PutAudioStreamData");
  return {};
}

DeviceReading SdlSink::reading() const {
  SDL_LockAudioStream(stream_);
  const std::int64_t consumed = consumed_locked();
  SDL_UnlockAudioStream(stream_);
  return {.consumed_frames = consumed, .host_us = host_now_us()};
}

std::int64_t SdlSink::latency_frames() const {
  SDL_AudioSpec device{};
  int buffer_frames = 0;
  if (!SDL_GetAudioDeviceFormat(SDL_GetAudioStreamDevice(stream_), &device, &buffer_frames) ||
      device.freq <= 0) {
    return 0;
  }
  // Le tampon se compte en trames du périphérique ; on le ramène aux trames de la file.
  const double scale = static_cast<double>(format_.sample_rate) / device.freq * ratio();
  return std::llround(buffer_frames * scale);
}

void SdlSink::set_paused(bool paused) {
  if (paused) {
    SDL_PauseAudioStreamDevice(stream_);
  } else {
    SDL_ResumeAudioStreamDevice(stream_);
  }
}

std::int64_t SdlSink::clear() {
  SDL_LockAudioStream(stream_);
  const std::int64_t consumed = consumed_locked();
  SDL_ClearAudioStream(stream_);
  SDL_UnlockAudioStream(stream_);
  return consumed;
}

void SdlSink::on_demand(std::function<void(const DeviceReading&)> callback) {
  SDL_SetAudioStreamGetCallback(stream_, nullptr, nullptr);  // attend la fin d'un rappel en cours
  demand_ = std::move(callback);
  if (demand_) SDL_SetAudioStreamGetCallback(stream_, &on_device_request, this);
}

void SdlSink::notify_demand() const {
  if (!demand_) return;
  try {
    demand_(DeviceReading{.consumed_frames = consumed_locked(), .host_us = host_now_us()});
  } catch (...) {
    // Aucune exception ne remonte dans le thread de SDL ; la mesure suivante remplacera celle-ci.
    SDL_LogWarn(SDL_LOG_CATEGORY_AUDIO, "ajp : mesure de la file audio perdue");
  }
}

std::int64_t SdlSink::consumed_locked() const {
  const int queued_bytes = SDL_GetAudioStreamQueued(stream_);
  const std::int64_t frame_bytes = std::int64_t{format_.channels} * std::int64_t{sizeof(float)};
  return pushed_ - (queued_bytes > 0 ? queued_bytes / frame_bytes : 0);
}

}  // namespace

std::expected<std::unique_ptr<AudioSink>, Refusal> open_sdl_sink(const OutputRequest& request) {
  auto sink = std::make_unique<SdlSink>();
  if (auto opened = sink->open(request); !opened) return std::unexpected(opened.error());
  return std::unique_ptr<AudioSink>(std::move(sink));
}

}  // namespace ajp::audio
