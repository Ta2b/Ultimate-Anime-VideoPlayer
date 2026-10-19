#include "export/mux.h"

#include <format>
#include <system_error>

extern "C" {
#include <libavutil/dict.h>
}

namespace ajp::exporter {
namespace {

const char* format_name(Container container) {
  switch (container) {
    case Container::Matroska:
      return "matroska";
    case Container::Mp4:
      return "mp4";
  }
  return nullptr;
}

}  // namespace

void MuxOutput::Closer::operator()(AVFormatContext* ctx) const {
  if (ctx->pb != nullptr && (ctx->oformat->flags & AVFMT_NOFILE) == 0) avio_closep(&ctx->pb);
  avformat_free_context(ctx);
}

std::expected<MuxOutput, Refusal> MuxOutput::create(const std::filesystem::path& path, Container container) {
  AVFormatContext* raw = nullptr;
  const std::string text = path.string();
  if (const int error = avformat_alloc_output_context2(&raw, nullptr, format_name(container), text.c_str());
      error < 0) {
    return refuse_av(RefusalCode::MuxFailed,
                     std::format("avformat_alloc_output_context2 {}", name(container)), error);
  }
  MuxOutput output;
  output.ctx_.reset(raw);
  output.path_ = path;
  return output;
}

std::expected<AVStream*, Refusal> MuxOutput::add_stream(const AVCodecParameters& par, AVRational time_base,
                                                        std::string_view what, bool query_codec) {
  // 1 : le conteneur porte ce codec ; 0 : il ne le porte pas ; < 0 : il ne sait pas dire, et
  // l'écriture de l'en-tête tranchera.
  if (query_codec && avformat_query_codec(ctx_->oformat, par.codec_id, FF_COMPLIANCE_NORMAL) == 0) {
    return refuse(RefusalCode::FormatUnsupported,
                  std::format("{} : {} ne porte pas le codec {}", what, ctx_->oformat->name,
                              avcodec_get_name(par.codec_id)));
  }
  AVStream* stream = avformat_new_stream(ctx_.get(), nullptr);
  if (stream == nullptr) return refuse(RefusalCode::MuxFailed, std::format("avformat_new_stream {}", what));
  if (const int error = avcodec_parameters_copy(stream->codecpar, &par); error < 0) {
    return refuse_av(RefusalCode::MuxFailed, std::format("avcodec_parameters_copy {}", what), error);
  }
  stream->codecpar->codec_tag = 0;  // le conteneur de sortie choisit son étiquette
  stream->time_base = time_base;
  return stream;
}

std::expected<void, Refusal> MuxOutput::write_header() {
  const std::string text = path_.string();
  if ((ctx_->oformat->flags & AVFMT_NOFILE) == 0) {
    if (const int error = avio_open(&ctx_->pb, text.c_str(), AVIO_FLAG_WRITE); error < 0) {
      return refuse_av(RefusalCode::FileUnwritable, std::format("avio_open {}", text), error);
    }
  }
  AVDictionary* options = nullptr;
  // MP4 : l'index (moov) en tête du fichier, pour une lecture qui démarre sans tout lire.
  if (std::string_view(ctx_->oformat->name).starts_with("mp4"))
    av_dict_set(&options, "movflags", "+faststart", 0);
  const int error = avformat_write_header(ctx_.get(), &options);
  av_dict_free(&options);
  if (error < 0)
    return refuse_av(RefusalCode::MuxFailed, std::format("avformat_write_header {}", text), error);
  return {};
}

std::expected<void, Refusal> MuxOutput::write(AVPacket& packet, AVRational time_base) {
  const AVStream* stream = ctx_->streams[packet.stream_index];
  av_packet_rescale_ts(&packet, time_base, stream->time_base);
  if (const int error = av_interleaved_write_frame(ctx_.get(), &packet); error < 0) {
    return refuse_av(RefusalCode::MuxFailed,
                     std::format("av_interleaved_write_frame flux {}", packet.stream_index), error);
  }
  return {};
}

std::expected<void, Refusal> MuxOutput::finish() {
  if (const int error = av_write_trailer(ctx_.get()); error < 0) {
    return refuse_av(RefusalCode::MuxFailed, std::format("av_write_trailer {}", path_.string()), error);
  }
  if ((ctx_->oformat->flags & AVFMT_NOFILE) == 0) {
    if (const int error = avio_closep(&ctx_->pb); error < 0) {
      return refuse_av(RefusalCode::FileUnwritable, std::format("avio_closep {}", path_.string()), error);
    }
  }
  return {};
}

RemoveOnExit::~RemoveOnExit() {
  if (path_.empty()) return;
  std::error_code ignored;
  std::filesystem::remove(path_, ignored);
}

}  // namespace ajp::exporter
