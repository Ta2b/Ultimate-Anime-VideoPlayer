#include "export/export_writer.h"

#include <format>
#include <utility>

#include "core/log.h"
#include "core/thread.h"
#include "export/av_handles.h"
#include "export/muxer.h"
#include "export/video_encoder.h"

namespace ajp::exporter {

struct ExportWriter::State {
  Muxer muxer;
  VideoEncoder encoder;
  AvColor color;
  int width = 0;
  int height = 0;
  std::int64_t last_pts = 0;
  bool any = false;  // une image déjà encodée : last_pts vaut
};

namespace {

// La mémoire d'une Picture vue par FFmpeg : release appelé quand la dernière référence part.
void release_picture(void* opaque, std::uint8_t* /*data*/) {
  auto* release = static_cast<std::function<void()>*>(opaque);
  if (*release) (*release)();
  delete release;  // NOLINT(cppcoreguidelines-owning-memory) : rendu par av_buffer_create
}

// L'image FFmpeg lit la mémoire de la Picture, hôte ou CUDA ; release part avec sa dernière
// référence. Un refus rend la Picture (release) : l'image reste vide.
std::expected<void, Refusal> bind_picture(AVFrame& frame, Picture& picture, const ExportWriter::State& s) {
  frame.format = pixel_format(s.encoder.layout());
  if (picture.device) {
    // Des adresses CUDA : l'image porte le format et le contexte d'images de l'encodeur, qui
    // enregistre la mémoire auprès de NVENC la première fois et la lit en place.
    if (s.encoder.upload() != Upload::Cuda || s.encoder.frames_context() == nullptr) {
      if (picture.release) picture.release();
      return std::unexpected(
          Refusal{RefusalCode::EncodeFailed, "image en mémoire CUDA pour un encodeur hôte"});
    }
    frame.format = s.encoder.context().pix_fmt;
    frame.hw_frames_ctx = av_buffer_ref(s.encoder.frames_context());
    if (frame.hw_frames_ctx == nullptr) {
      if (picture.release) picture.release();
      return std::unexpected(Refusal{RefusalCode::EncodeFailed, "image : mémoire"});
    }
  }
  frame.width = s.width;
  frame.height = s.height;
  auto* release = new std::function<void()>(std::move(picture.release));  // NOLINT : à FFmpeg
  frame.buf[0] =
      av_buffer_create(picture.planes[0], picture.bytes, release_picture, release, AV_BUFFER_FLAG_READONLY);
  if (frame.buf[0] == nullptr) {
    release_picture(release, nullptr);
    return std::unexpected(Refusal{RefusalCode::EncodeFailed, "image : mémoire"});
  }
  for (std::size_t p = 0; p < picture.planes.size(); ++p) {
    frame.data[p] = picture.planes[p];
    frame.linesize[p] = picture.pitches[p];
  }
  return {};
}

}  // namespace

std::expected<std::unique_ptr<ExportWriter>, Refusal> ExportWriter::open(const WriterConfig& config) {
  auto muxer = Muxer::open(config.source, config.output);
  if (!muxer) return std::unexpected(std::move(muxer.error()));
  const ColorInfo color = config.color.value_or(muxer->source_color());
  auto encoder = VideoEncoder::open(EncoderSetup{.width = config.width,
                                                 .height = config.height,
                                                 .fps_num = config.fps_num,
                                                 .fps_den = config.fps_den,
                                                 .sample_aspect = muxer->source_aspect(),
                                                 .color = color,
                                                 .codec = config.codec,
                                                 .quality = config.quality,
                                                 .vendor = config.vendor,
                                                 .cuda_ordinal = config.cuda_ordinal,
                                                 .vulkan_device = config.vulkan_device});
  if (!encoder) return std::unexpected(std::move(encoder.error()));
  if (auto begun = muxer->begin(encoder->context()); !begun) return std::unexpected(std::move(begun.error()));
  std::unique_ptr<ExportWriter> writer(new ExportWriter());
  writer->layout_ = encoder->layout();
  writer->upload_ = encoder->upload();
  writer->color_ = color;
  writer->encoder_ = std::format("{} · {}", label(config.codec), encoder->label());
  writer->output_ = config.output;
  writer->state_ = std::make_unique<State>(State{.muxer = std::move(*muxer),
                                                 .encoder = std::move(*encoder),
                                                 .color = av_color(color),
                                                 .width = config.width,
                                                 .height = config.height});
  ExportWriter* self = writer.get();
  auto thread =
      start_thread("export", ThreadReport([self](const Refusal& refusal) {
                     const std::scoped_lock lock(self->mutex_);
                     if (!self->failure_) self->failure_ = refusal;
                     self->done_ = true;
                     self->wake_.notify_all();
                   }),
                   [self](const std::stop_token& stop, const ThreadReport& /*report*/) { self->run(stop); });
  if (!thread) return std::unexpected(std::move(thread.error()));
  writer->thread_ = std::move(*thread);
  return writer;
}

ExportWriter::~ExportWriter() {
  if (thread_.joinable()) {
    thread_.request_stop();
    wake_.notify_all();
    thread_.join();
  }
  for (Picture& picture : queue_) {
    if (picture.release) picture.release();
    av_frame_free(&picture.frame);
  }
  queue_.clear();
  state_.reset();  // l'encodeur rend ses dernières images ; un mux non fini supprime le fichier
}

bool ExportWriter::try_push(Picture&& picture) {
  {
    const std::scoped_lock lock(mutex_);
    if (done_ || finishing_ || failure_ || queue_.size() >= kQueueDepth) return false;
    queue_.push_back(std::move(picture));
  }
  wake_.notify_all();
  return true;
}

std::expected<void, Refusal> ExportWriter::finish() {
  {
    std::unique_lock lock(mutex_);
    finishing_ = true;
    wake_.notify_all();
    wake_.wait(lock, [this] { return done_; });
  }
  if (thread_.joinable()) thread_.join();
  const std::scoped_lock lock(mutex_);
  if (failure_) return std::unexpected(*failure_);
  return {};
}

std::optional<Refusal> ExportWriter::failure() const {
  const std::scoped_lock lock(mutex_);
  return failure_;
}

AVBufferRef* ExportWriter::vulkan_frames() const {
  const VideoEncoder& encoder = state_->encoder;
  if (encoder.upload() != Upload::Vulkan || !encoder.shared_device()) return nullptr;
  return encoder.frames_context();
}

WriterProgress ExportWriter::progress() const {
  return WriterProgress{.frames = frames_.load(), .bytes = bytes_.load()};
}

void ExportWriter::run(const std::stop_token& stop) {
  State& s = *state_;
  const auto log = log_channel("export");
  const auto sink = [&](AVPacket& packet) -> std::expected<void, Refusal> {
    auto written = s.muxer.write_video(packet);
    bytes_.store(s.muxer.bytes());
    return written;
  };
  const auto fail = [&](Refusal refusal) {
    log->error("export arrêté : {} {}", name(refusal.code), refusal.detail);
    const std::scoped_lock lock(mutex_);
    if (!failure_) failure_ = std::move(refusal);
    done_ = true;
    wake_.notify_all();
  };
  FramePtr frame(av_frame_alloc());
  if (!frame) return fail(Refusal{RefusalCode::EncodeFailed, "image : mémoire"});
  while (true) {
    Picture picture;
    {
      std::unique_lock lock(mutex_);
      wake_.wait(lock, stop, [this] { return !queue_.empty() || finishing_; });
      if (stop.stop_requested()) return;  // abandon : le destructeur rend les images et supprime le fichier
      if (queue_.empty()) break;          // finish() : tout est encodé
      picture = std::move(queue_.front());
      queue_.pop_front();
    }
    av_frame_unref(frame.get());
    if (picture.frame != nullptr) {
      // Une image de l'encodeur, rendue sur le GPU : prise telle quelle, avec sa synchronisation.
      av_frame_move_ref(frame.get(), picture.frame);
      av_frame_free(&picture.frame);
    } else if (auto bound = bind_picture(*frame, picture, s); !bound) {
      return fail(std::move(bound.error()));
    }
    frame->color_primaries = s.color.primaries;
    frame->color_trc = s.color.transfer;
    frame->colorspace = s.color.matrix;
    frame->color_range = s.color.range;
    frame->chroma_location = s.color.siting;
    // Sur la ligne du conteneur, comme les pistes copiées ; strictement croissant pour l'encodeur.
    std::int64_t pts = picture.pts_us + s.muxer.start_us();
    if (s.any && pts <= s.last_pts) pts = s.last_pts + 1;
    s.last_pts = pts;
    s.any = true;
    frame->pts = pts;
    auto encoded = s.encoder.encode(frame.get(), sink);
    av_frame_unref(frame.get());
    if (!encoded) return fail(std::move(encoded.error()));
    frames_.fetch_add(1);
  }
  if (auto flushed = s.encoder.encode(nullptr, sink); !flushed) return fail(std::move(flushed.error()));
  if (auto closed = s.muxer.finish(); !closed) return fail(std::move(closed.error()));
  bytes_.store(s.muxer.bytes());
  log->info("export écrit : {} images, {} Mo, {}", frames_.load(), bytes_.load() / 1'000'000, utf8(output_));
  const std::scoped_lock lock(mutex_);
  done_ = true;
  wake_.notify_all();
}

}  // namespace ajp::exporter
