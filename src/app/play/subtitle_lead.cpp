#include "app/play/subtitle_lead.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <stop_token>
#include <utility>

#include "app/play/bridges.h"
#include "app/play/pumped_task.h"
#include "core/log.h"
#include "core/utf8_path.h"
#include "media/demuxer.h"

namespace ajp::app::play {

namespace {

// Boîte de subs pleine : on retente après ce temps, tant qu'aucun seek ne rend le paquet inutile.
constexpr auto kRetry = std::chrono::milliseconds(5);

}  // namespace

struct SubtitleLead::Core {
  static constexpr std::int64_t kNothingRead = std::numeric_limits<std::int64_t>::min();
  struct Job {
    int track = 0;
    std::optional<std::int64_t> place;  // où replacer le fichier avant de lire
    std::uint64_t generation = 0;
  };

  explicit Core(SubtitleLeadConfig c) : config(std::move(c)) {}

  void run(const std::stop_token& stop);
  bool wanted() const;  // sous mutex
  bool prepare(const Job& job, const std::stop_token& stop);
  void read_one(const Job& job, const std::stop_token& stop);
  bool push(subs::TrackPacket packet, std::uint64_t generation, const std::stop_token& stop);
  void mark_stopped();

  SubtitleLeadConfig config;
  std::unique_ptr<media::Demuxer> demuxer;  // écrit par le thread sous mutex, interrompu par le destructeur
  std::optional<int> kept;                  // thread : la piste que garde le démultiplexeur
  bool unusable = false;                    // thread : le fichier ne s'est pas ouvert une seconde fois

  mutable std::mutex mutex;
  std::condition_variable_any wake;
  std::optional<int> track;
  std::optional<std::int64_t> place;  // seek reçu, pas encore fait
  std::int64_t want_us = kNothingRead;
  std::int64_t read_us = kNothingRead;  // pts du dernier paquet lu depuis le dernier placement
  std::int64_t delay_us = 0;
  std::uint64_t generation = 0;  // change à chaque seek ou piste : un paquet d'avant est jeté
  std::uint64_t sent = 0;
  bool ended = false;  // fin du fichier ou lecture refusée : rien jusqu'au prochain seek
  bool busy = false;
  bool stopped = false;  // le thread a rendu la main (sous mutex)
  std::condition_variable stopped_cv;
};

SubtitleLead::SubtitleLead(SubtitleLeadConfig config)
    : core_(std::make_shared<Core>(std::move(config))), thread_([core = core_](const std::stop_token& stop) {
        core->run(stop);
        core->mark_stopped();
      }) {}

SubtitleLead::~SubtitleLead() {
  thread_.request_stop();
  {
    const std::scoped_lock lock(core_->mutex);
    if (core_->demuxer)
      core_->demuxer->interrupt();  // une lecture suspendue à un partage réseau rend la main
  }
  core_->wake.notify_all();
  // B30-01 : un read() bloqué dans le noyau (partage réseau muet) ne voit pas le drapeau d'interruption.
  // Passé kSubtitleStopLimit, le thread est laissé à son sort au lieu de figer la fenêtre.
  {
    std::unique_lock lock(core_->mutex);
    if (core_->stopped_cv.wait_for(lock, kSubtitleStopLimit, [this] { return core_->stopped; })) return;
  }
  const auto log = log_channel("app");
  log->warn("sous-titres en avance : pas d'arrêt en {} ms, laissés à leur sort", kSubtitleStopLimit.count());
  abandon_thread(std::move(thread_), [core = core_] {
    const std::scoped_lock lock(core->mutex);
    return core->stopped;
  });
}

void SubtitleLead::Core::mark_stopped() {
  {
    const std::scoped_lock lock(mutex);
    stopped = true;
  }
  stopped_cv.notify_all();
}

void SubtitleLead::select(std::optional<int> track) {
  Core& c = *core_;
  {
    const std::scoped_lock lock(c.mutex);
    if (c.track == track) return;
    c.track = track;
    ++c.generation;
    // La piste nouvelle se lit au seek qui suit le choix (Pipeline::select_subtitles).
    c.place.reset();
    c.ended = true;
  }
  c.wake.notify_all();
}

void SubtitleLead::seek(std::int64_t target_us) {
  Core& c = *core_;
  {
    const std::scoped_lock lock(c.mutex);
    ++c.generation;
    c.place = std::max<std::int64_t>(target_us - kSubtitlePrerollUs, 0);
    c.want_us = target_us + std::max<std::int64_t>(-c.delay_us, 0) + kSubtitleLeadUs;
    c.read_us = Core::kNothingRead;
    c.ended = false;
  }
  c.wake.notify_all();
}

void SubtitleLead::follow(std::int64_t pts_us, std::int64_t delay_us) {
  Core& c = *core_;
  {
    const std::scoped_lock lock(c.mutex);
    c.delay_us = delay_us;
    if (delay_us >= 0) return;  // la source livre à temps : rien à lire en avance
    const std::int64_t want = pts_us - delay_us + kSubtitleLeadUs;
    if (want <= c.want_us) return;
    c.want_us = want;
    // Délai devenu négatif en pleine lecture, ou rien lu encore : la lecture reprend à l'instant demandé,
    // pas là où elle s'était arrêtée (tout le fichier d'entre les deux).
    if (!c.place && c.read_us < pts_us - kSubtitleLeadUs) {
      ++c.generation;
      c.place = pts_us;
      c.read_us = Core::kNothingRead;
      c.ended = false;
    }
  }
  c.wake.notify_all();
}

std::uint64_t SubtitleLead::sent() const {
  const std::scoped_lock lock(core_->mutex);
  return core_->sent;
}

bool SubtitleLead::idle() const {
  const std::scoped_lock lock(core_->mutex);
  return !core_->busy && !core_->wanted();
}

bool SubtitleLead::Core::wanted() const {
  return track.has_value() && (place.has_value() || (!ended && read_us < want_us));
}

void SubtitleLead::Core::run(const std::stop_token& stop) {
  while (!stop.stop_requested()) {
    Job job;
    {
      std::unique_lock lock(mutex);
      busy = false;
      if (!wake.wait(lock, stop, [this] { return wanted(); }) || !track) return;
      busy = true;
      job = Job{.track = *track, .place = std::exchange(place, std::nullopt), .generation = generation};
    }
    if (!prepare(job, stop)) continue;
    read_one(job, stop);
  }
}

// Ouvre le fichier au premier besoin, garde la piste choisie et place le fichier. false : rien à lire
// jusqu'au prochain seek.
bool SubtitleLead::Core::prepare(const Job& job, const std::stop_token& stop) {
  const auto log = log_channel("app");
  const auto stop_reading = [this, &job] {
    const std::scoped_lock lock(mutex);
    if (generation == job.generation) ended = true;
    return false;
  };
  if (unusable) return stop_reading();
  if (!demuxer) {
    // B30-01 : l'arrêt atteint aussi l'ouverture, qui se faisait avant que le destructeur puisse
    // l'interrompre.
    auto opened = media::Demuxer::open(config.file, {.attachments = false}, stop);
    if (!opened) {
      if (stop.stop_requested()) return false;
      log->warn("sous-titres en avance : {} ne s'ouvre pas une seconde fois ({})",
                core::to_utf8(config.file.filename()), opened.error().detail);
      unusable = true;
      return stop_reading();
    }
    const std::scoped_lock lock(mutex);
    demuxer = std::make_unique<media::Demuxer>(std::move(*opened));
    if (stop.stop_requested()) demuxer->interrupt();  // arrêté entre l'ouverture et ce verrou
  }
  if (kept != job.track) {
    const std::array<int, 1> keep{job.track};
    if (auto done = demuxer->keep_only(keep); !done) {
      log->debug("sous-titres en avance : piste {} : {}", job.track, done.error().detail);
      kept.reset();
      return stop_reading();
    }
    kept = job.track;
  }
  if (job.place) {
    if (auto placed = demuxer->seek_file(*job.place, config.reference); !placed) {
      log->debug("sous-titres en avance : {}", placed.error().detail);
      return stop_reading();
    }
  }
  return true;
}

void SubtitleLead::Core::read_one(const Job& job, const std::stop_token& stop) {
  auto packet = demuxer->read();
  if (!packet || !*packet) {
    if (!packet) log_channel("app")->debug("sous-titres en avance : {}", packet.error().detail);
    const std::scoped_lock lock(mutex);
    if (generation == job.generation) ended = true;
    return;
  }
  const media::Packet& read = **packet;
  if (read.track != job.track || read.pts_us == kNoPts) return;
  const bool pushed =
      push(subs_packet(read, std::ranges::contains(config.mov_text, read.track)), job.generation, stop);
  const std::scoped_lock lock(mutex);
  if (generation != job.generation) return;
  read_us = std::max(read_us, read.pts_us);
  if (pushed) ++sent;
}

// Retente tant que la boîte de subs est pleine ; false si un seek, un changement de piste ou l'arrêt rend
// le paquet inutile.
bool SubtitleLead::Core::push(subs::TrackPacket packet, std::uint64_t at_generation,
                              const std::stop_token& stop) {
  while (!config.inbox->try_push(std::move(packet))) {  // NOLINT(bugprone-use-after-move) : rendu intact
    std::unique_lock lock(mutex);
    wake.wait_for(lock, stop, kRetry, [this, at_generation] { return generation != at_generation; });
    if (stop.stop_requested() || generation != at_generation) return false;
  }
  return true;
}

}  // namespace ajp::app::play
