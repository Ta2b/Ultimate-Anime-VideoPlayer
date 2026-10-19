#include <climits>
#include <format>
#include <optional>
#include <system_error>
#include <variant>

#include "export/av_util.h"
#include "export/encoder.h"
#include "export/export.h"
#include "export/log.h"
#include "export/mux.h"
#include "export/remux.h"

namespace ajp::exporter {
namespace {

struct ParametersFree {
  void operator()(AVCodecParameters* par) const { avcodec_parameters_free(&par); }
};
using ParametersPtr = std::unique_ptr<AVCodecParameters, ParametersFree>;

std::filesystem::path with_suffix(std::filesystem::path path, std::string_view suffix) {
  path += suffix;
  return path;
}

std::unexpected<Refusal> mismatch(std::string detail) {
  return refuse(RefusalCode::InputMismatch, std::move(detail));
}

// Une image hôte doit avoir la taille de sortie et un tampon qui couvre toutes ses lignes.
std::expected<void, Refusal> check_frame(const HostFrame& frame, const ExportJob& job) {
  if (frame.width != job.width || frame.height != job.height) {
    return mismatch(
        std::format("image {}x{}, sortie {}x{}", frame.width, frame.height, job.width, job.height));
  }
  const auto row = static_cast<std::size_t>(frame.width) * 8;
  if (frame.stride < row || frame.stride % 2 != 0 || frame.stride > static_cast<std::size_t>(INT_MAX)) {
    return mismatch(std::format("image : stride {} pour {} octets par ligne", frame.stride, row));
  }
  const std::size_t needed = (static_cast<std::size_t>(frame.height) - 1) * frame.stride + row;
  if (frame.rgba.size_bytes() < needed) {
    return mismatch(std::format("image : tampon de {} octets, {} attendus", frame.rgba.size_bytes(), needed));
  }
  if (frame.pts_us == kNoPts) return mismatch("image : pts inconnu");
  return {};
}

// Première passe : l'encodeur, le fichier vidéo seul et la mesure de lumière.
class FirstPass {
 public:
  FirstPass(VideoEncoder encoder, MuxOutput video) : encoder_(std::move(encoder)), video_(std::move(video)) {}

  std::expected<void, Refusal> encode(const HostFrame& frame, std::int64_t pts_us) {
    meter_.add(frame);
    if (auto ok = encoder_.send(frame, pts_us); !ok) return ok;
    return drain();
  }

  std::expected<void, Refusal> finish() {
    if (auto ok = encoder_.flush(); !ok) return ok;
    if (auto ok = drain(); !ok) return ok;
    return video_.finish();
  }

  LightLevel light() const { return meter_.level(); }

 private:
  std::expected<void, Refusal> drain() {
    if (!packet_) return refuse(RefusalCode::EncodeFailed, "av_packet_alloc : mémoire insuffisante");
    while (true) {
      auto got = encoder_.receive(*packet_);
      if (!got) return std::unexpected(got.error());
      if (!*got) return {};
      packet_->stream_index = 0;
      if (auto ok = video_.write(*packet_, kMicroseconds); !ok) return ok;
    }
  }

  VideoEncoder encoder_;
  MuxOutput video_;
  LightMeter meter_;
  PacketPtr packet_{av_packet_alloc()};
};

std::expected<FirstPass, Refusal> open_first_pass(const ExportJob& job,
                                                  const std::filesystem::path& video_path) {
  const EncodeGeometry geometry{.width = job.width,
                                .height = job.height,
                                .frame_rate = AVRational{job.frame_rate.num, job.frame_rate.den}};
  auto encoder = VideoEncoder::open(job.encoder, geometry, job.mastering, true);
  if (!encoder) return std::unexpected(encoder.error());
  ParametersPtr par(avcodec_parameters_alloc());
  if (!par) return refuse(RefusalCode::MuxFailed, "avcodec_parameters_alloc : mémoire insuffisante");
  if (const int error = avcodec_parameters_from_context(par.get(), &encoder->context()); error < 0) {
    return refuse_av(RefusalCode::MuxFailed, "avcodec_parameters_from_context", error);
  }
  // Le conteneur final doit porter ce codec : refusé ici, avant la première image.
  auto final_output = MuxOutput::create(job.output, job.container);
  if (!final_output) return std::unexpected(final_output.error());
  if (auto stream = final_output->add_stream(*par, kMicroseconds, "vidéo exportée"); !stream) {
    return std::unexpected(stream.error());
  }
  // Le fichier de travail est toujours en MP4 : il garde le dts de chaque paquet (images B)
  // et la microseconde des pts, que Matroska arrondirait à la milliseconde.
  auto video = MuxOutput::create(video_path, Container::Mp4);
  if (!video) return std::unexpected(video.error());
  if (auto stream = video->add_stream(*par, kMicroseconds, "vidéo exportée"); !stream) {
    return std::unexpected(stream.error());
  }
  if (auto ok = video->write_header(); !ok) return std::unexpected(ok.error());
  return FirstPass(std::move(*encoder), std::move(*video));
}

void publish(const ExportJob& job, const ExportProgress& progress) {
  if (job.on_progress) job.on_progress(progress);
}

}  // namespace

std::expected<void, Refusal> validate_job(const ExportJob& job) {
  if (job.output.empty()) return refuse(RefusalCode::FieldMissing, "export.output : chemin de sortie absent");
  const FrameRate rate = job.frame_rate;
  if (rate.num <= 0 || rate.den <= 0 || rate.num / rate.den >= kMaxFrameRate) {
    return refuse(RefusalCode::FieldOutOfRange, std::format("export.frame_rate : {}/{} hors de ]0, {}[ im/s",
                                                            rate.num, rate.den, kMaxFrameRate));
  }
  // La sortie remplace son chemin à la fin : elle ne doit pas écraser le fichier source.
  std::error_code ignored;
  if (!job.copy_from.empty() && (job.output.lexically_normal() == job.copy_from.lexically_normal() ||
                                 std::filesystem::equivalent(job.output, job.copy_from, ignored))) {
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("export.output : {} est le fichier source", job.output.string()));
  }
  if (job.frames_total && *job.frames_total < 0) {
    return refuse(RefusalCode::FieldOutOfRange,
                  std::format("export.frames_total : {} négatif", *job.frames_total));
  }
  if (auto spec = validate_settings(job.encoder, job.width, job.height); !spec)
    return std::unexpected(spec.error());
  return validate_mastering(job.mastering);
}

std::expected<ExportSummary, Refusal> run_export(const ExportJob& job) {
  if (auto ok = validate_job(job); !ok) return std::unexpected(ok.error());
  if (job.source == nullptr)
    return refuse(RefusalCode::FieldMissing, "export.source : aucune source d'images");
  const auto logger = make_logger();
  const auto part = with_suffix(job.output, ".part");
  const auto video_path = with_suffix(job.output, ".video.part");
  RemuxRequest request{.video = video_path,
                       .source = job.copy_from,
                       .copy = job.copy,
                       .container = job.container,
                       .mastering = job.mastering,
                       .tags = job.tags,
                       .output = part,
                       .stop = job.stop};
  // Une piste que le conteneur ne porte pas se refuse avant d'encoder la première image.
  if (!job.copy_from.empty()) {
    if (auto ok = check_copy(request); !ok) return std::unexpected(ok.error());
  }

  const RemoveOnExit remove_video(video_path);
  RemoveOnExit remove_part(part);
  auto pass = open_first_pass(job, video_path);
  if (!pass) return std::unexpected(pass.error());
  logger->info("export : {} vers {} ({}x{}, {})", job.encoder.encoder, job.output.string(), job.width,
               job.height, name(job.encoder.format));

  ProgressMeter progress(job.frames_total ? job.frames_total : job.source->total_frames());
  progress.start(ExportClock::now());
  std::optional<std::int64_t> first_pts;
  std::int64_t last_pts = 0;
  while (true) {
    if (job.stop.stop_requested()) return refuse(RefusalCode::Cancelled, "export annulé");
    auto next = job.source->next();
    if (!next) return std::unexpected(next.error());
    if (!*next) break;
    if (job.stop.stop_requested()) return refuse(RefusalCode::Cancelled, "export annulé");
    const auto* frame = std::get_if<HostFrame>(&**next);
    if (frame == nullptr) {
      return refuse(RefusalCode::FormatUnsupported,
                    "export : voie appareil (P010 sur le GPU) pas encore branchée");
    }
    if (auto ok = check_frame(*frame, job); !ok) return std::unexpected(ok.error());
    if (first_pts && frame->pts_us <= last_pts) {
      return mismatch(std::format("image : pts {} µs après {} µs", frame->pts_us, last_pts));
    }
    if (!first_pts) first_pts = frame->pts_us;
    last_pts = frame->pts_us;
    if (auto ok = pass->encode(*frame, frame->pts_us - *first_pts); !ok) return std::unexpected(ok.error());
    publish(job, progress.frame_done(ExportClock::now()));
  }
  if (!first_pts) return mismatch("export : la source n'a rendu aucune image");
  if (auto ok = pass->finish(); !ok) return std::unexpected(ok.error());

  ExportProgress finishing = progress.at(ExportClock::now());
  finishing.finishing = true;
  publish(job, finishing);
  if (job.stop.stop_requested()) return refuse(RefusalCode::Cancelled, "export annulé");

  request.begin_us = *first_pts;
  request.end_us =
      last_pts + av_rescale_q(1, AVRational{job.frame_rate.den, job.frame_rate.num}, kMicroseconds);
  request.light = pass->light();
  if (auto ok = remux(request); !ok) return std::unexpected(ok.error());
  std::error_code error;
  std::filesystem::rename(part, job.output, error);
  if (error) {
    return refuse(RefusalCode::FileUnwritable,
                  std::format("rename {} vers {} : {}", part.string(), job.output.string(), error.message()));
  }
  remove_part.release();

  ExportSummary summary{.output = job.output,
                        .frames = progress.frames_done(),
                        .light = request.light,
                        .elapsed = finishing.elapsed,
                        .frames_per_second = finishing.frames_per_second};
  logger->info("export : {} images en {} ms ({:.1f} im/s), MaxCLL {} MaxFALL {}", summary.frames,
               summary.elapsed.count(), summary.frames_per_second, summary.light.max_cll,
               summary.light.max_fall);
  return summary;
}

}  // namespace ajp::exporter
