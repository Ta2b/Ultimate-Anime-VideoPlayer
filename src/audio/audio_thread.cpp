#include "audio/audio_thread.h"

#include <exception>
#include <format>
#include <variant>

#include <spdlog/spdlog.h>

#include "audio/refuse.h"

namespace ajp::audio {
namespace {

constexpr std::size_t kMaxPacketBound = 4096;

// Paquets, pistes, fins de flux, commandes, Flush, mesures, choix du maître.
// Une seule mesure attend : la suivante, refusée, n'apporterait rien de plus.
AudioInbox::Bounds bounds(std::size_t max_packets) {
  return {max_packets, 2, 2, 32, 4, 1, 2};
}

}  // namespace

std::expected<std::unique_ptr<AudioThread>, Refusal> AudioThread::start(std::unique_ptr<AudioOutput> output,
                                                                        AudioThreadOptions options) {
  if (!output) return refuse(RefusalCode::AudioDeviceUnavailable, "aucune sortie audio");
  if (options.max_packets < 1 || options.max_packets > kMaxPacketBound) {
    return refuse(RefusalCode::FieldOutOfRange, std::format("audio.max_packets = {}, attendu de 1 à {}",
                                                            options.max_packets, kMaxPacketBound));
  }
  if (auto config = checked(options.player); !config) return std::unexpected(config.error());
  return std::unique_ptr<AudioThread>(new AudioThread(std::move(output), std::move(options)));
}

AudioThread::AudioThread(std::unique_ptr<AudioOutput> output, AudioThreadOptions options)
    : max_packets_(options.max_packets),
      credit_(std::move(options.credit)),
      inbox_(bounds(options.max_packets)) {
  // Thread du périphérique : poser la mesure, rien d'autre. Si une mesure
  // attend déjà, celle-ci est de trop et se perd sans dommage.
  output->on_demand([this](const DeviceReading& reading) { inbox_.try_push(DeviceReading{reading}); });
  player_ = std::make_unique<AudioPlayer>(
      std::move(output), options.player,
      PlayerEvents{.packet_released = [this] { release_packet(); }, .refused = std::move(options.refused)});
  clock_ = player_->clock();
  thread_ = std::jthread([this](const std::stop_token& stop) { run(stop); });
}

AudioThread::~AudioThread() = default;

bool AudioThread::push_packet(AudioPacket&& packet) {
  if (in_flight_.fetch_add(1) >= max_packets_ || !inbox_.try_push(std::move(packet))) {
    in_flight_.fetch_sub(1);
    return false;
  }
  return true;
}

void AudioThread::run(const std::stop_token& stop) {
  while (auto message = inbox_.wait(stop)) {
    // Aucune exception ne quitte la boucle : le message fautif est perdu, le thread continue.
    try {
      std::visit([this](auto&& m) { player_->handle(std::forward<decltype(m)>(m)); }, std::move(*message));
    } catch (const std::exception& error) {
      spdlog::default_logger()->error("audio : message perdu ({})", error.what());
    } catch (...) {
      spdlog::default_logger()->error("audio : message perdu (exception inconnue)");
    }
  }
}

// Thread audio. La borne était peut-être atteinte : le producteur attend
// peut-être un crédit. Un crédit de trop ne coûte qu'un essai.
void AudioThread::release_packet() {
  if (in_flight_.fetch_sub(1) >= max_packets_ && credit_) credit_(Credit{});
}

}  // namespace ajp::audio
