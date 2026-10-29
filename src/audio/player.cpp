#include "audio/player.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <format>
#include <iterator>
#include <utility>
#include <variant>
#include <vector>

#include <spdlog/spdlog.h>

#include "audio/log.h"
#include "audio/refuse.h"

namespace ajp::audio {
namespace {

constexpr std::int64_t kMinQueueTargetUs = 20'000;
constexpr std::int64_t kMaxQueueTargetUs = 2'000'000;
constexpr std::int64_t kMinResyncUs = 10'000;
constexpr std::int64_t kMaxResyncUs = 1'000'000;

bool positive(double value) {
  return std::isfinite(value) && value > 0.0;
}

}  // namespace

std::expected<PlayerConfig, Refusal> checked(const PlayerConfig& config) {
  if (config.queue_target_us < kMinQueueTargetUs || config.queue_target_us > kMaxQueueTargetUs) {
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("audio.queue_target_us = {}, attendu de {} à {}", config.queue_target_us,
                              kMinQueueTargetUs, kMaxQueueTargetUs));
  }
  if (config.resync_us < kMinResyncUs || config.resync_us > kMaxResyncUs) {
    return refuse(RefusalCode::FieldOutOfRange, std::format("audio.resync_us = {}, attendu de {} à {}",
                                                            config.resync_us, kMinResyncUs, kMaxResyncUs));
  }
  const DriftGains& gains = config.drift;
  if (!positive(gains.proportional) || !positive(gains.integral) || !positive(gains.smoothing_s)) {
    return refuse(RefusalCode::FieldOutOfRange,
                  "audio.drift : gains et constante de temps positifs attendus");
  }
  return config;
}

AudioPlayer::AudioPlayer(std::unique_ptr<AudioOutput> output, const PlayerConfig& config, PlayerEvents events)
    : output_(std::move(output)),
      config_(config),
      events_(std::move(events)),
      log_(make_logger()),
      clock_(std::make_shared<AudioClock>()),
      regulator_(config.drift),
      mixer_(MixConfig{.listen = config.mix.listen,
                       .tuning = config.mix.tuning,
                       .faithful = config.mix.faithful,
                       .sink_channels = output_->format().channels,
                       .sample_rate = output_->format().sample_rate,
                       .hrir = config.mix.hrir}) {}

AudioPlayer::~AudioPlayer() = default;

void AudioPlayer::handle(AudioPacket&& packet) {
  if (packet.epoch < epoch_) {  // reste d'avant un seek
    release(1);
    return;
  }
  if (packet.epoch > epoch_) start_epoch(packet.epoch);  // ses paquets ont devancé le Flush
  if (packet.discard_before_us) trim_before_us_ = packet.discard_before_us;
  held_.push_back(std::move(packet));
  pump();
}

void AudioPlayer::handle(const AudioTrack& track) {
  track_ = track;
  open_decoder(track);
  pump();
}

// Le décodeur rend les canaux que le mixeur veut de cette piste (mixer.h : decoder_channels).
void AudioPlayer::open_decoder(const AudioTrack& track) {
  const PcmFormat wanted{.sample_rate = output_->format().sample_rate,
                         .channels = mixer_.decoder_channels(track.codec.channels)};
  auto decoder = AudioDecoder::open(track.codec, wanted);
  if (!decoder) {
    decoder_.reset();
    report(decoder.error());
    return;
  }
  decoder_ = std::move(*decoder);
  log_->info("piste « {} » ({} Hz, {} canaux) vers {} Hz, {} canaux, écoute {}{}", track.codec.codec,
             track.codec.sample_rate, track.codec.channels, wanted.sample_rate, wanted.channels,
             static_cast<int>(mixer_.config().listen), mixer_.faithful() ? " fidèle" : "");
}

// Les blocs décodés passent par le mixeur avant la file de sortie.
void AudioPlayer::queue(std::vector<PcmBlock>&& blocks) {
  for (PcmBlock& block : blocks) ready_.push_back(mixer_.process(std::move(block)));
}

void AudioPlayer::handle(const EndOfStream& end) {
  if (end.epoch < epoch_) return;                  // fin d'avant un seek
  if (end.epoch > epoch_) start_epoch(end.epoch);  // seek après le dernier paquet : a devancé son Flush
  end_pending_ = true;
  pump();
}

void AudioPlayer::handle(const Command& command) {
  if (std::holds_alternative<cmd::Play>(command)) {
    resume();
  } else if (std::holds_alternative<cmd::Pause>(command)) {
    pause();
  } else if (std::holds_alternative<cmd::TogglePause>(command)) {
    if (paused_) {
      resume();
    } else {
      pause();
    }
  } else if (const auto* delay = std::get_if<cmd::SetAudioDelay>(&command)) {
    set_delay(delay->delay_us);
  } else if (const auto* volume = std::get_if<cmd::SetVolume>(&command)) {
    if (auto set = output_->set_volume(volume->gain); !set) report(set.error());
  } else if (const auto* listening = std::get_if<cmd::SetListening>(&command)) {
    const int before = track_ ? mixer_.decoder_channels(track_->codec.channels) : 0;
    MixConfig mix = mixer_.config();
    mix.listen = static_cast<Listen>(std::min<std::uint8_t>(listening->listen, 2));
    mix.faithful = listening->faithful;
    if (listening->tuned) mix.tuning = tuning_from_wire(listening->tuning);
    if (listening->hrir_set) mix.hrir = listening->hrir;  // absent : le modèle paramétrique
    mixer_ = Mixer(mix);
    log_->info(
        "écoute {}{} : égaliseur graves {:+.1f} voix {:+.1f} aigus {:+.1f} dB, nuit {:.2f}, pièce {:.2f}",
        static_cast<int>(mix.listen), mix.faithful ? " fidèle" : "", mix.tuning.eq_bass_db,
        mix.tuning.eq_voice_db, mix.tuning.eq_treble_db, mix.tuning.night, mix.tuning.room);
    // Les canaux voulus du décodeur ont changé : il est rouvert, ce qui était décodé est refait au prochain
    // paquet.
    if (track_ && mixer_.decoder_channels(track_->codec.channels) != before) {
      open_decoder(*track_);
      ready_.clear();
      output_->clear();
      aligning_ = master_ != nullptr;
    }
  }
  // Les autres commandes ne concernent pas l'audio ; le seek arrive par Flush.
}

void AudioPlayer::handle(const Flush& flush) {
  if (flush.epoch > epoch_) start_epoch(flush.epoch);  // sinon déjà appliqué par un paquet
}

void AudioPlayer::handle(const DeviceReading& reading) {
  if (!paused_) {
    if (const auto heard = output_->position(reading)) {
      publish(*heard, reading);
      follow(*heard, reading);
    }
  }
  pump();
}

void AudioPlayer::handle(const SyncTo& sync) {
  master_ = sync.master;
  regulator_.reset();
  apply_ratio(1.0);
  if (master_) {
    realign();
  } else {
    aligning_ = false;
  }
  pump();
}

void AudioPlayer::start_epoch(std::uint32_t epoch) {
  epoch_ = epoch;
  release(held_.size());
  held_.clear();
  ready_.clear();
  if (decoder_) decoder_->flush();
  mixer_.reset();
  output_->clear();
  trim_before_us_.reset();
  end_pending_ = false;
  ended_ = false;
  aligning_ = master_ != nullptr;
  regulator_.restart();
  last_follow_us_.reset();
  clock_->publish({.epoch = epoch, .valid = false});
}

// Garde la file du périphérique remplie jusqu'à sa cible, sans plus.
void AudioPlayer::pump() {
  while (output_->queued_us() < config_.queue_target_us) {
    if (!ready_.empty()) {
      if (!push(ready_.front())) return;  // attend que l'horloge maître soit valide
      ready_.pop_front();
    } else if (!held_.empty()) {
      decode_next();
    } else if (end_pending_) {
      finish_stream();
    } else {
      return;
    }
  }
}

void AudioPlayer::decode_next() {
  const AudioPacket packet = std::move(held_.front());
  held_.pop_front();
  if (decoder_) {
    if (auto blocks = decoder_->decode(packet)) {
      queue(std::move(*blocks));
    } else {
      log_->warn("{} : {}", name(blocks.error().code), blocks.error().detail);  // paquet perdu, on continue
    }
  }
  release(1);
}

void AudioPlayer::finish_stream() {
  end_pending_ = false;
  ended_ = true;
  if (!decoder_) return;
  if (auto blocks = decoder_->drain()) {
    queue(std::move(*blocks));
  } else {
    log_->warn("{} : {}", name(blocks.error().code), blocks.error().detail);
  }
}

// false : le bloc attend, rien n'est poussé.
bool AudioPlayer::push(PcmBlock& block) {
  if (trim_before_us_) block.drop_before(*trim_before_us_, output_->format().sample_rate);
  if (block.frames() == 0) return true;  // entièrement avant la cible du seek : rien à caler
  if (aligning_ && !align(block)) return false;
  if (block.frames() == 0) return true;
  if (auto pushed = output_->push(block.pts_us, block.samples); !pushed) {
    report(pushed.error());
    return true;
  }
  // Premières trames de l'époque : la position est connue avant d'être entendue.
  if (!clock_->snapshot().valid) {
    clock_->publish({.media_us = block.pts_us + output_->delay_us(),
                     .host_us = output_->reading().host_us,
                     .epoch = epoch_,
                     .valid = true});
  }
  return true;
}

// Coupe le début du bloc ou le fait précéder d'un silence, pour qu'il soit
// entendu quand l'horloge maître atteint son temps média (délai compris).
// Le silence part par morceaux d'au plus la cible de la file : une avance de
// plusieurs minutes n'occupe pas la mémoire, et chaque morceau se recalcule sur
// l'horloge maître du moment. false : le bloc attend le morceau suivant.
bool AudioPlayer::align(PcmBlock& block) {
  const ClockSnapshot master = master_->snapshot();
  if (!master.valid || master.epoch != epoch_) return false;  // l'écran n'a pas encore d'image de l'époque
  const int rate = output_->format().sample_rate;
  // Ce qui sera poussé maintenant s'entendra après la file et le tampon du périphérique.
  const std::int64_t heard_at = output_->reading().host_us + output_->queued_us() + output_->latency_us();
  const std::int64_t target_us = master.media_at(heard_at) - output_->delay_us();
  const std::int64_t lead_us = block.pts_us - target_us;
  if (lead_us > 0) {
    const std::int64_t silence_us = std::min(lead_us, config_.queue_target_us);
    if (auto silence = output_->push_silence(us_to_frames(silence_us, rate)); !silence) {
      report(silence.error());
    }
    if (silence_us < lead_us) return false;
  } else {
    block.drop_before(target_us, rate);
  }
  aligning_ = block.frames() == 0;  // bloc entièrement en retard : le suivant se cale à son tour
  return true;
}

void AudioPlayer::publish(std::int64_t heard_us, const DeviceReading& reading) {
  const bool starving = reading.consumed_frames >= output_->pushed_frames();
  if (starving && ended_) {
    // Fin de piste : plus rien ne viendra, l'horloge continue sur l'hôte. Si
    // elle s'était arrêtée (manque de trames avant la fin, pause), elle repart
    // d'où elle était ; sinon elle continue déjà.
    const ClockSnapshot now = clock_->snapshot();
    if (now.valid && now.rate == 0.0) {
      clock_->publish({.media_us = now.media_us,
                       .host_us = reading.host_us,
                       .rate = 1.0,
                       .epoch = epoch_,
                       .valid = true});
    }
    return;
  }
  // En manque de trames, l'audio maître s'arrête : l'image l'attend.
  clock_->publish({.media_us = heard_us,
                   .host_us = reading.host_us,
                   .rate = starving ? 0.0 : output_->ratio(),
                   .epoch = epoch_,
                   .valid = true});
}

void AudioPlayer::follow(std::int64_t heard_us, const DeviceReading& reading) {
  if (!master_ || aligning_) return;
  const ClockSnapshot master = master_->snapshot();
  if (!master.valid || master.epoch != epoch_ || master.rate == 0.0) return;
  const std::int64_t error_us = heard_us - master.media_at(reading.host_us);
  if (std::abs(error_us) > config_.resync_us) {
    log_->info("écart de {} ms avec l'écran : recalage", error_us / 1000);
    realign();
    return;
  }
  const std::int64_t elapsed_us = last_follow_us_ ? reading.host_us - *last_follow_us_ : 0;
  last_follow_us_ = reading.host_us;
  apply_ratio(regulator_.update(error_us, elapsed_us));
}

// L'audio pas encore consommé revient en tête, pour se caler avec le reste.
void AudioPlayer::realign() {
  std::vector<PcmBlock> back = output_->take_back();
  ready_.insert(ready_.begin(), std::make_move_iterator(back.begin()), std::make_move_iterator(back.end()));
  aligning_ = true;
  regulator_.restart();
  last_follow_us_.reset();
}

void AudioPlayer::pause() {
  if (paused_) return;
  paused_ = true;
  output_->pause();
  const ClockSnapshot now = clock_->snapshot();
  const std::int64_t host = output_->reading().host_us;
  clock_->publish({.media_us = now.media_at(host), .host_us = host, .epoch = now.epoch, .valid = now.valid});
}

// L'horloge reste arrêtée jusqu'à la première mesure : le périphérique n'a
// encore rien joué, et une horloge repartie tout de suite reculerait à cette
// mesure du temps de démarrage du périphérique.
void AudioPlayer::resume() {
  if (!paused_) return;
  paused_ = false;
  output_->resume();
  pump();
}

void AudioPlayer::set_delay(std::int64_t delay_us) {
  const std::int64_t before = output_->delay_us();
  if (auto set = output_->set_delay(delay_us); !set) {
    report(set.error());
    return;
  }
  if (master_) {  // l'écran ne bouge pas : c'est l'audio qui se recale
    realign();
    pump();
    return;
  }
  ClockSnapshot now = clock_->snapshot();  // l'audio maître décale l'horloge, tout de suite
  if (!now.valid) return;
  now.media_us += delay_us - before;
  clock_->publish(now);
}

void AudioPlayer::apply_ratio(double ratio) {
  if (auto set = output_->set_ratio(ratio); !set) report(set.error());
}

void AudioPlayer::release(std::size_t count) const {
  if (!events_.packet_released) return;
  for (std::size_t i = 0; i < count; ++i) events_.packet_released();
}

void AudioPlayer::report(const Refusal& refusal) {
  log_->warn("{} : {}", name(refusal.code), refusal.detail);
  if (events_.refused) events_.refused(refusal);
}

}  // namespace ajp::audio
