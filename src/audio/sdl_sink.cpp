// File SDL3 : un SDL_AudioStream lié au périphérique de sortie choisi (celui du système par défaut).
//
// La position jouée se mesure sans course : SDL appelle on_device_request
// verrou du flux tenu, juste avant de tirer des trames, et pushed_ ne change
// que sous ce même verrou. Trames consommées = poussées - encore en file.
#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <format>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>

#include <SDL3/SDL_audio.h>
#include <SDL3/SDL_error.h>
#include <SDL3/SDL_events.h>
#include <SDL3/SDL_hints.h>
#include <SDL3/SDL_init.h>
#include <SDL3/SDL_log.h>

#include "audio/converter.h"
#include "audio/devices.h"
#include "audio/log.h"
#include "audio/shaper.h"
#include "audio/sink.h"
#include "core/clock.h"
#include "core/types.h"

namespace ajp::audio {
namespace {

std::unexpected<Refusal> refuse_sdl(std::string_view call) {
  return refuse(RefusalCode::AudioDeviceUnavailable, std::format("{} : {}", call, SDL_GetError()));
}

// Sous-système audio de SDL ; SDL compte lui-même les initialisations.
class AudioSubsystem {
 public:
  // B16-06 : le flux est un film pour le système (PipeWire, PulseAudio), pas un jeu : rôle et routage justes.
  AudioSubsystem() {
    SDL_SetHint(SDL_HINT_AUDIO_DEVICE_STREAM_ROLE, "movie");
    ready_ = SDL_InitSubSystem(SDL_INIT_AUDIO);
  }
  ~AudioSubsystem() {
    if (ready_) SDL_QuitSubSystem(SDL_INIT_AUDIO);
  }
  AudioSubsystem(const AudioSubsystem&) = delete;
  AudioSubsystem& operator=(const AudioSubsystem&) = delete;
  bool ready() const { return ready_; }
  // B16-01 : un pilote qui n'a pas pu démarrer au lancement (PipeWire pas encore prêt) se réessaie.
  bool retry() {
    if (!ready_) ready_ = SDL_InitSubSystem(SDL_INIT_AUDIO);
    return ready_;
  }

 private:
  bool ready_ = false;
};

// Format de la file quand ni la demande ni l'appareil n'en disent un (file dormante, B16-01).
constexpr PcmFormat kDormantFormat{.sample_rate = 48'000, .channels = 2};

class SdlSink final : public AudioSink {
 public:
  SdlSink() = default;
  SdlSink(const SdlSink&) = delete;
  SdlSink& operator=(const SdlSink&) = delete;
  // Détruire le flux ferme le périphérique et attend la fin d'un rappel en cours.
  ~SdlSink() override {
    if (watching_) SDL_RemoveEventWatch(&on_event, this);
    if (stream_ != nullptr) SDL_DestroyAudioStream(stream_);
  }

  std::expected<void, Refusal> open(const OutputRequest& request);

  bool ready() const override { return stream_ != nullptr; }
  PcmFormat format() const override { return format_; }
  std::expected<void, Refusal> put(std::span<const float> samples) override;
  std::int64_t pushed_frames() const override { return pushed_; }
  DeviceReading reading() const override;
  std::int64_t latency_frames() const override;
  void set_paused(bool paused) override;
  // Le gain passe par le rappel postmix (rampe, limiteur doux au-delà de 1) ; sans lui, par celui de SDL.
  void set_gain(float gain) override {
    gain_.store(gain);
    if (!postmix_ && stream_ != nullptr) SDL_SetAudioStreamGain(stream_, gain);
  }
  float gain() const override { return gain_.load(); }
  void set_ratio(double ratio) override {
    ratio_ = ratio;
    if (stream_ != nullptr) SDL_SetAudioStreamFrequencyRatio(stream_, static_cast<float>(ratio));
  }
  double ratio() const override { return ratio_; }
  std::int64_t clear() override;
  void on_demand(std::function<void(const DeviceReading&)> callback) override;
  void on_removed(std::function<void()> callback) override {
    const std::scoped_lock lock(removed_mutex_);
    removed_ = std::move(callback);
  }
  void on_returned(std::function<void()> callback) override {
    const std::scoped_lock lock(removed_mutex_);
    returned_ = std::move(callback);
  }
  std::expected<void, Refusal> reopen_default() override;
  std::expected<void, Refusal> reopen_chosen() override;

  // Thread du périphérique, verrou du flux tenu par SDL.
  void notify_demand() const;
  // Thread du périphérique, juste avant l'envoi au matériel : gain, rampes, limiteur.
  void shape(const SDL_AudioSpec& spec, float* buffer, int bytes);

 private:
  std::int64_t consumed_locked() const;
  std::expected<void, Refusal> open_stream(SDL_AudioDeviceID device_id);
  // Remplace le flux par un neuf sur device_id ; l'ancien reste en place si le neuf ne s'ouvre pas.
  std::expected<void, Refusal> reopen(SDL_AudioDeviceID device_id);
  // Thread de SDL qui pose l'événement : SDL_EVENT_AUDIO_DEVICE_REMOVED du périphérique logique ouvert, ou
  // SDL_EVENT_AUDIO_DEVICE_ADDED d'un appareil qui porte le nom de l'appareil choisi.
  static bool SDLCALL on_event(void* sink, SDL_Event* event);
  void call_locked(const std::function<void()>& callback);

  AudioSubsystem subsystem_;
  OutputRequest request_;
  SDL_AudioStream* stream_ = nullptr;  // nul : file dormante (B16-01)
  PcmFormat format_;
  std::int64_t pushed_ = 0;  // écrit sous le verrou du flux, par le thread audio seul
  std::function<void(const DeviceReading&)> demand_;
  std::atomic<SDL_AudioDeviceID> device_{0};  // le périphérique logique du flux, lu par on_event
  std::atomic<float> gain_{1.0F};             // cible du gain, lue par le rappel postmix
  std::atomic<bool> fade_in_{false};          // le prochain tampon entre en fondu
  OutputShaper shaper_;                       // au rappel postmix seul
  bool postmix_ = false;                      // le rappel postmix est posé
  double ratio_ = 1.0;
  bool paused_ = true;
  bool on_default_ = true;  // le son passe sur la sortie par défaut (choisie, ou de repli)
  bool watching_ = false;
  std::mutex removed_mutex_;
  std::function<void()> removed_;
  std::function<void()> returned_;
};

void SdlSink::call_locked(const std::function<void()>& callback) {
  if (!callback) return;
  try {
    callback();
  } catch (...) {  // NOLINT(bugprone-empty-catch) : aucune exception ne remonte dans SDL
  }
}

bool SDLCALL SdlSink::on_event(void* sink, SDL_Event* event) {
  auto* self = static_cast<SdlSink*>(sink);
  const bool audio_event =
      event->type == SDL_EVENT_AUDIO_DEVICE_REMOVED || event->type == SDL_EVENT_AUDIO_DEVICE_ADDED;
  if (!audio_event || event->adevice.recording) return true;  // adevice n'a de sens que pour ces deux types
  if (event->type == SDL_EVENT_AUDIO_DEVICE_REMOVED && event->adevice.which == self->device_.load()) {
    const std::scoped_lock lock(self->removed_mutex_);
    self->call_locked(self->removed_);
  } else if (event->type == SDL_EVENT_AUDIO_DEVICE_ADDED && !self->request_.device.empty()) {
    // B31-16 : l'appareil choisi revient (casque rebranché, ampli rallumé).
    const char* name = SDL_GetAudioDeviceName(event->adevice.which);
    if (name != nullptr && self->request_.device == name) {
      const std::scoped_lock lock(self->removed_mutex_);
      self->call_locked(self->returned_);
    }
  }
  return true;  // l'événement suit son cours
}

void SDLCALL on_device_request(void* sink, SDL_AudioStream* /*stream*/, int /*additional*/, int /*total*/) {
  static_cast<const SdlSink*>(sink)->notify_demand();
}

void SDLCALL on_postmix(void* sink, const SDL_AudioSpec* spec, float* buffer, int bytes) {
  static_cast<SdlSink*>(sink)->shape(*spec, buffer, bytes);
}

// B16-03 : au-dessus de 100 % le gain est suivi du limiteur doux, au lieu de l'écrêtage dur du mixeur ;
// B16-07 : de même quand SDL doit mixer vers un autre nombre de canaux que celui de la file ; B16-08 : le
// gain change par une rampe de kRampMs, et un son qui repart (reprise, vidage, file neuve) entre en fondu.
void SdlSink::shape(const SDL_AudioSpec& spec, float* buffer, int bytes) {
  if (spec.channels <= 0 || bytes <= 0) return;
  shaper_.process({buffer, static_cast<std::size_t>(bytes) / sizeof(float)},
                  {.channels = spec.channels,
                   .sample_rate = spec.freq,
                   .gain = gain_.load(),
                   .fade_in = fade_in_.exchange(false),
                   .force_limit = spec.channels != format_.channels});
}

// Journal : la file dormante dit pourquoi, une fois ; le lecteur l'annonce et réessaie (B16-01).
std::expected<void, Refusal> SdlSink::open(const OutputRequest& request) {
  request_ = request;
  const std::shared_ptr<spdlog::logger> log = make_logger();
  // L'appareil choisi s'il existe encore, sinon le défaut du système (suivi par SDL quand il change).
  SDL_AudioDeviceID device_id = SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK;
  std::optional<Refusal> unavailable;
  SDL_AudioSpec preferred{};
  if (!subsystem_.ready()) {
    unavailable = refuse_sdl("SDL_InitSubSystem(SDL_INIT_AUDIO)").error();
  } else {
    if (!request.device.empty()) {
      if (const auto found = resolve_device(playback_devices(), request.device);
          found && found->name == request.device)
        device_id = found->id;
    }
    if (!SDL_GetAudioDeviceFormat(device_id, &preferred, nullptr)) {
      unavailable = refuse_sdl("SDL_GetAudioDeviceFormat").error();
      preferred = {};
    }
  }
  const auto format = checked({.sample_rate = request.sample_rate != 0 ? request.sample_rate
                                              : preferred.freq != 0    ? preferred.freq
                                                                       : kDormantFormat.sample_rate,
                               .channels = request.channels != 0     ? request.channels
                                           : preferred.channels != 0 ? preferred.channels
                                                                     : kDormantFormat.channels});
  if (!format) return std::unexpected(format.error());
  format_ = *format;
  on_default_ = device_id == SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK;
  if (!unavailable) {
    if (auto opened = open_stream(device_id); !opened) unavailable = opened.error();
  }
  if (unavailable) {
    log->warn("sortie audio indisponible au lancement ({}) : file dormante, réessai à intervalles réguliers",
              unavailable->detail);
  }
  watching_ = SDL_AddEventWatch(&on_event, this);
  return {};
}

std::expected<void, Refusal> SdlSink::open_stream(SDL_AudioDeviceID device_id) {
  const SDL_AudioSpec spec{
      .format = SDL_AUDIO_F32, .channels = format_.channels, .freq = format_.sample_rate};
  // Le périphérique s'ouvre en pause ; SDL convertit vers son format réel si besoin.
  SDL_AudioStream* opened = SDL_OpenAudioDeviceStream(device_id, &spec, nullptr, nullptr);
  if (opened == nullptr) return refuse_sdl("SDL_OpenAudioDeviceStream");
  stream_ = opened;
  const SDL_AudioDeviceID logical = SDL_GetAudioStreamDevice(stream_);
  device_.store(logical);
  paused_ = true;
  fade_in_.store(true);  // un flux neuf entre en fondu
  postmix_ = SDL_SetAudioPostmixCallback(logical, &on_postmix, this);
  if (!postmix_) {
    // Sans rappel postmix : gain de SDL, écrêtage dur au-delà de 1 (le comportement d'avant).
    SDL_SetAudioStreamGain(stream_, gain_.load());
    make_logger()->warn("SDL_SetAudioPostmixCallback : {} (gain sans limiteur ni rampe)", SDL_GetError());
  }
  if (ratio_ != 1.0) SDL_SetAudioStreamFrequencyRatio(stream_, static_cast<float>(ratio_));
  return {};
}

// Le flux neuf d'abord ; s'il ne s'ouvre pas, l'ancien reste en place (A16-01 : un périphérique retiré qui ne
// joue plus rien, au lieu d'un flux nul pour toute la vidéo). Le lecteur réessaie plus tard
// (PlayerConfig::reopen_retry_us).
std::expected<void, Refusal> SdlSink::reopen(SDL_AudioDeviceID device_id) {
  SDL_AudioStream* old = stream_;
  const SDL_AudioDeviceID old_device = device_.load();
  device_.store(0);
  const bool had_postmix = postmix_;
  const bool was_paused = paused_;
  if (old != nullptr) {
    SDL_SetAudioStreamGetCallback(old, nullptr, nullptr);  // attend la fin d'un rappel
    // C16-05 : l'ancien flux ne touche plus au façonneur (l'attente ci-dessus et celle-ci couvrent un rappel
    // en cours) et ne joue plus ce que le neuf va rejouer.
    SDL_PauseAudioStreamDevice(old);
    if (had_postmix) SDL_SetAudioPostmixCallback(old_device, nullptr, nullptr);
  }
  if (auto opened = open_stream(device_id); !opened) {
    device_.store(old_device);
    if (old != nullptr) {
      if (had_postmix) SDL_SetAudioPostmixCallback(old_device, &on_postmix, this);
      if (!was_paused) {
        fade_in_.store(true);
        SDL_ResumeAudioStreamDevice(old);
      }
      if (demand_) SDL_SetAudioStreamGetCallback(old, &on_device_request, this);
    }
    return opened;  // stream_ gardé
  }
  if (old != nullptr) {
    SDL_ClearAudioStream(old);
    SDL_DestroyAudioStream(old);
  }
  on_default_ = device_id == SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK;
  if (demand_) SDL_SetAudioStreamGetCallback(stream_, &on_device_request, this);
  return {};
}

std::expected<void, Refusal> SdlSink::reopen_default() {
  SDL_AudioDeviceID device_id = SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK;
  if (stream_ == nullptr) {  // file dormante : l'appareil choisi s'il est là, sinon le défaut
    if (!subsystem_.retry()) return refuse_sdl("SDL_InitSubSystem(SDL_INIT_AUDIO)");
    if (!request_.device.empty()) {
      if (const auto found = resolve_device(playback_devices(), request_.device);
          found && found->name == request_.device)
        device_id = found->id;
    }
  }
  return reopen(device_id);
}

// B31-16 : le son est sur la sortie par défaut parce que l'appareil choisi avait disparu ; il revient.
std::expected<void, Refusal> SdlSink::reopen_chosen() {
  if (request_.device.empty() || !on_default_ || stream_ == nullptr) {
    return std::unexpected(Refusal{RefusalCode::AudioDeviceUnavailable, "pas d'appareil choisi à rouvrir"});
  }
  const auto found = resolve_device(playback_devices(), request_.device);
  if (!found || found->name != request_.device) {
    return std::unexpected(Refusal{RefusalCode::AudioDeviceUnavailable, "appareil choisi absent"});
  }
  return reopen(found->id);
}

std::expected<void, Refusal> SdlSink::put(std::span<const float> samples) {
  if (samples.empty()) return {};
  if (stream_ == nullptr) return refuse(RefusalCode::AudioDeviceUnavailable, "file audio dormante");
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
  if (stream_ == nullptr) return {.consumed_frames = pushed_, .host_us = host_now_us()};
  SDL_LockAudioStream(stream_);
  const std::int64_t consumed = consumed_locked();
  SDL_UnlockAudioStream(stream_);
  return {.consumed_frames = consumed, .host_us = host_now_us()};
}

std::int64_t SdlSink::latency_frames() const {
  if (stream_ == nullptr) return 0;
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
  paused_ = paused;
  if (stream_ == nullptr) return;
  if (!paused) fade_in_.store(true);
  if (paused) {
    SDL_PauseAudioStreamDevice(stream_);
  } else {
    SDL_ResumeAudioStreamDevice(stream_);
  }
}

std::int64_t SdlSink::clear() {
  if (stream_ == nullptr) return pushed_;
  if (!paused_) fade_in_.store(true);  // le son coupé net repart en fondu
  SDL_LockAudioStream(stream_);
  const std::int64_t consumed = consumed_locked();
  SDL_ClearAudioStream(stream_);
  SDL_UnlockAudioStream(stream_);
  return consumed;
}

void SdlSink::on_demand(std::function<void(const DeviceReading&)> callback) {
  if (stream_ != nullptr)
    SDL_SetAudioStreamGetCallback(stream_, nullptr, nullptr);  // attend la fin d'un rappel
  demand_ = std::move(callback);
  if (demand_ && stream_ != nullptr) SDL_SetAudioStreamGetCallback(stream_, &on_device_request, this);
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
