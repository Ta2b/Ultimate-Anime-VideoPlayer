#include "media/deinterlace.h"

#include <format>
#include <string>

extern "C" {
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/frame.h>
#include <libavutil/mathematics.h>
}

namespace ajp::media {

namespace {

struct GraphFree {
  void operator()(AVFilterGraph* graph) const { avfilter_graph_free(&graph); }
};

std::unexpected<Refusal> refused(std::string_view what, int err) {
  return std::unexpected(Refusal{RefusalCode::DecodeFailed, std::format("bwdif : {} ({})", what, err)});
}

}  // namespace

struct Deinterlacer::Graph {
  std::unique_ptr<AVFilterGraph, GraphFree> graph;
  AVFilterContext* source = nullptr;
  AVFilterContext* sink = nullptr;
  int width = 0;
  int height = 0;
  int format = -1;
};

Deinterlacer::Deinterlacer() = default;
Deinterlacer::~Deinterlacer() = default;
Deinterlacer::Deinterlacer(Deinterlacer&&) noexcept = default;
Deinterlacer& Deinterlacer::operator=(Deinterlacer&&) noexcept = default;

std::expected<void, Refusal> Deinterlacer::push(FramePtr frame) {
  if ((frame->flags & AV_FRAME_FLAG_INTERLACED) != 0) active_ = true;
  if (!graph_ || graph_->width != frame->width || graph_->height != frame->height ||
      graph_->format != frame->format) {
    if (graph_) {
      // Taille ou format changés : les images retenues sortent d'abord, puis le graphe est refait.
      if (auto flushed = flush(); !flushed) return flushed;
    }
    auto fresh = std::make_unique<Graph>();
    fresh->graph.reset(avfilter_graph_alloc());
    if (!fresh->graph) return refused("graphe", AVERROR(ENOMEM));
    fresh->width = frame->width;
    fresh->height = frame->height;
    fresh->format = frame->format;
    const std::string args = std::format(
        "video_size={}x{}:pix_fmt={}:time_base=1/1000000:pixel_aspect={}/{}", frame->width, frame->height,
        frame->format, frame->sample_aspect_ratio.num > 0 ? frame->sample_aspect_ratio.num : 1,
        frame->sample_aspect_ratio.den > 0 ? frame->sample_aspect_ratio.den : 1);
    AVFilterContext* bwdif = nullptr;
    if (const int err = avfilter_graph_create_filter(&fresh->source, avfilter_get_by_name("buffer"), "in",
                                                     args.c_str(), nullptr, fresh->graph.get());
        err < 0)
      return refused("source", err);
    if (const int err = avfilter_graph_create_filter(&bwdif, avfilter_get_by_name("bwdif"), "bwdif",
                                                     "mode=send_frame:parity=auto:deint=interlaced", nullptr,
                                                     fresh->graph.get());
        err < 0)
      return refused("filtre", err);
    if (const int err = avfilter_graph_create_filter(&fresh->sink, avfilter_get_by_name("buffersink"), "out",
                                                     nullptr, nullptr, fresh->graph.get());
        err < 0)
      return refused("sortie", err);
    if (int err = avfilter_link(fresh->source, 0, bwdif, 0); err < 0) return refused("lien", err);
    if (int err = avfilter_link(bwdif, 0, fresh->sink, 0); err < 0) return refused("lien", err);
    if (int err = avfilter_graph_config(fresh->graph.get(), nullptr); err < 0) return refused("graphe", err);
    graph_ = std::move(fresh);
  }
  if (const int err = av_buffersrc_add_frame_flags(graph_->source, frame.get(), AV_BUFFERSRC_FLAG_KEEP_REF);
      err < 0)
    return refused("image", err);
  return {};
}

std::expected<void, Refusal> Deinterlacer::flush() {
  if (!graph_) return {};
  if (const int err = av_buffersrc_add_frame_flags(graph_->source, nullptr, 0); err < 0 && err != AVERROR_EOF)
    return refused("fin du flux", err);
  return {};
}

std::expected<FramePtr, Refusal> Deinterlacer::pull() {
  if (!graph_) return FramePtr();
  FramePtr out(av_frame_alloc());
  if (!out) return refused("image", AVERROR(ENOMEM));
  const int err = av_buffersink_get_frame(graph_->sink, out.get());
  if (err == AVERROR(EAGAIN) || err == AVERROR_EOF) return FramePtr();
  if (err < 0) return refused("sortie", err);
  // bwdif double la base de temps (une image par champ possible) : retour aux microsecondes.
  const AVRational base = av_buffersink_get_time_base(graph_->sink);
  if (out->pts != AV_NOPTS_VALUE) out->pts = av_rescale_q(out->pts, base, kMicroseconds);
  if (out->duration > 0) out->duration = av_rescale_q(out->duration, base, kMicroseconds);
  out->best_effort_timestamp = out->pts;
  return out;
}

void Deinterlacer::reset() {
  graph_.reset();
  active_ = false;
}

bool Deinterlacer::active() const {
  return active_;
}

}  // namespace ajp::media
