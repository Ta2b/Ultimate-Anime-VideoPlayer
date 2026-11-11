#include "app/play/subtitle_lead.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <utility>

#include "app/play/bridges.h"
#include "core/log.h"
#include "core/utf8_path.h"

namespace ajp::app::play {

namespace {

// Boîte de subs pleine : on retente après ce temps, tant qu'aucun seek ne rend le paquet inutile.
constexpr auto kRetry = std::chrono::milliseconds(5);

}  // namespace

SubtitleLead::SubtitleLead(SubtitleLeadConfig config)
    : config_(std::move(config)), thread_([this](const std::stop_token& stop) { run(stop); }) {}

SubtitleLead::~SubtitleLead() {
  thread_.request_stop();
  {
    std::scoped_lock lock(mutex_);
    if (demuxer_) demuxer_->interrupt();  // une lecture suspendue à un partage réseau rend la main
  }
  wake_.notify_all();
  thread_.join();
}

void SubtitleLead::select(std::optional<int> track) {
  {
    std::scoped_lock lock(mutex_);
    if (track_ == track) return;
    track_ = track;
    ++generation_;
    // La piste nouvelle se lit au seek qui suit le choix (Pipeline::select_subtitles).
    place_.reset();
    ended_ = true;
  }
  wake_.notify_all();
}

void SubtitleLead::seek(std::int64_t target_us) {
  {
    std::scoped_lock lock(mutex_);
    ++generation_;
    place_ = std::max<std::int64_t>(target_us - kSubtitlePrerollUs, 0);
    want_us_ = target_us + std::max<std::int64_t>(-delay_us_, 0) + kSubtitleLeadUs;
    read_us_ = kNothingRead;
    ended_ = false;
  }
  wake_.notify_all();
}

void SubtitleLead::follow(std::int64_t pts_us, std::int64_t delay_us) {
  {
    std::scoped_lock lock(mutex_);
    delay_us_ = delay_us;
    if (delay_us >= 0) return;  // la source livre à temps : rien à lire en avance
    const std::int64_t want = pts_us - delay_us + kSubtitleLeadUs;
    if (want <= want_us_) return;
    want_us_ = want;
    // Délai devenu négatif en pleine lecture, ou rien lu encore : la lecture reprend à l'instant demandé,
    // pas là où elle s'était arrêtée (tout le fichier d'entre les deux).
    if (!place_ && read_us_ < pts_us - kSubtitleLeadUs) {
      ++generation_;
      place_ = pts_us;
      read_us_ = kNothingRead;
      ended_ = false;
    }
  }
  wake_.notify_all();
}

std::uint64_t SubtitleLead::sent() const {
  std::scoped_lock lock(mutex_);
  return sent_;
}

bool SubtitleLead::idle() const {
  std::scoped_lock lock(mutex_);
  return !busy_ && !wanted();
}

bool SubtitleLead::wanted() const {
  return track_.has_value() && (place_.has_value() || (!ended_ && read_us_ < want_us_));
}

void SubtitleLead::run(const std::stop_token& stop) {
  while (!stop.stop_requested()) {
    Job job;
    {
      std::unique_lock lock(mutex_);
      busy_ = false;
      if (!wake_.wait(lock, stop, [this] { return wanted(); }) || !track_) return;
      busy_ = true;
      job = Job{.track = *track_, .place = std::exchange(place_, std::nullopt), .generation = generation_};
    }
    if (!prepare(job)) continue;
    read_one(job, stop);
  }
}

// Ouvre le fichier au premier besoin, garde la piste choisie et place le fichier. false : rien à lire
// jusqu'au prochain seek.
bool SubtitleLead::prepare(const Job& job) {
  const auto log = log_channel("app");
  const auto stop_reading = [this, &job] {
    std::scoped_lock lock(mutex_);
    if (generation_ == job.generation) ended_ = true;
    return false;
  };
  if (unusable_) return stop_reading();
  if (!demuxer_) {
    auto opened = media::Demuxer::open(config_.file, {.attachments = false});
    if (!opened) {
      log->warn("sous-titres en avance : {} ne s'ouvre pas une seconde fois ({})",
                core::to_utf8(config_.file.filename()), opened.error().detail);
      unusable_ = true;
      return stop_reading();
    }
    std::scoped_lock lock(mutex_);
    demuxer_ = std::make_unique<media::Demuxer>(std::move(*opened));
  }
  if (kept_ != job.track) {
    const std::array<int, 1> keep{job.track};
    if (auto kept = demuxer_->keep_only(keep); !kept) {
      log->debug("sous-titres en avance : piste {} : {}", job.track, kept.error().detail);
      kept_.reset();
      return stop_reading();
    }
    kept_ = job.track;
  }
  if (job.place) {
    if (auto placed = demuxer_->seek_file(*job.place, config_.reference); !placed) {
      log->debug("sous-titres en avance : {}", placed.error().detail);
      return stop_reading();
    }
  }
  return true;
}

void SubtitleLead::read_one(const Job& job, const std::stop_token& stop) {
  auto packet = demuxer_->read();
  if (!packet || !*packet) {
    if (!packet) log_channel("app")->debug("sous-titres en avance : {}", packet.error().detail);
    std::scoped_lock lock(mutex_);
    if (generation_ == job.generation) ended_ = true;
    return;
  }
  const media::Packet& read = **packet;
  if (read.track != job.track || read.pts_us == kNoPts) return;
  const bool sent =
      push(subs_packet(read, std::ranges::contains(config_.mov_text, read.track)), job.generation, stop);
  std::scoped_lock lock(mutex_);
  if (generation_ != job.generation) return;
  read_us_ = std::max(read_us_, read.pts_us);
  if (sent) ++sent_;
}

// Retente tant que la boîte de subs est pleine ; false si un seek, un changement de piste ou l'arrêt rend
// le paquet inutile.
bool SubtitleLead::push(subs::TrackPacket packet, std::uint64_t generation, const std::stop_token& stop) {
  while (!config_.inbox->try_push(std::move(packet))) {  // NOLINT(bugprone-use-after-move) : rendu intact
    std::unique_lock lock(mutex_);
    wake_.wait_for(lock, stop, kRetry, [this, generation] { return generation_ != generation; });
    if (stop.stop_requested() || generation_ != generation) return false;
  }
  return true;
}

}  // namespace ajp::app::play
