#include "media/deinterlace.h"

#include <format>
#include <string>
#include <utility>
#include <vector>

extern "C" {
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/frame.h>
#include <libavutil/mathematics.h>
}

#include "media/log.h"

namespace ajp::media {

namespace {

struct GraphFree {
  void operator()(AVFilterGraph* graph) const { avfilter_graph_free(&graph); }
};

std::unexpected<Refusal> refused(std::string_view what, int err) {
  return std::unexpected(Refusal{RefusalCode::DecodeFailed, std::format("bwdif : {} ({})", what, err)});
}

struct Step {
  const char* filter;
  const char* args;
};

constexpr Step kBwdif{"bwdif", "mode=send_frame:parity=auto:deint=interlaced"};

// Les filtres du mode, entre la source et la sortie. Sans fieldmatch ou decimate (un FFmpeg construit sans
// eux), le télécinéma retombe sur bwdif seul, comme avant.
std::vector<Step> steps_of(FieldMode mode) {
  if (mode == FieldMode::InverseTelecine) {
    if (avfilter_get_by_name("fieldmatch") != nullptr && avfilter_get_by_name("decimate") != nullptr)
      return {{"fieldmatch", "order=auto:combmatch=full"}, kBwdif, {"decimate", ""}};
    make_logger()->warn("fieldmatch ou decimate absent de FFmpeg : télécinéma désentrelacé par bwdif seul");
  }
  return {kBwdif};
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

Deinterlacer::Deinterlacer(FieldMode mode) : mode_(mode) {}
Deinterlacer::~Deinterlacer() = default;
Deinterlacer::Deinterlacer(Deinterlacer&&) noexcept = default;
Deinterlacer& Deinterlacer::operator=(Deinterlacer&&) noexcept = default;

std::expected<void, Refusal> Deinterlacer::rebuild(const AVFrame& frame) {
  auto fresh = std::make_unique<Graph>();
  fresh->graph.reset(avfilter_graph_alloc());
  if (!fresh->graph) return refused("graphe", AVERROR(ENOMEM));
  fresh->width = frame.width;
  fresh->height = frame.height;
  fresh->format = frame.format;
  // decimate demande une fréquence constante : celle du télécinéma, 29,97.
  const std::string args = std::format("video_size={}x{}:pix_fmt={}:time_base=1/1000000:pixel_aspect={}/{}{}",
                                       frame.width, frame.height, frame.format,
                                       frame.sample_aspect_ratio.num > 0 ? frame.sample_aspect_ratio.num : 1,
                                       frame.sample_aspect_ratio.den > 0 ? frame.sample_aspect_ratio.den : 1,
                                       mode_ == FieldMode::InverseTelecine ? ":frame_rate=30000/1001" : "");
  if (const int err = avfilter_graph_create_filter(&fresh->source, avfilter_get_by_name("buffer"), "in",
                                                   args.c_str(), nullptr, fresh->graph.get());
      err < 0)
    return refused("source", err);
  AVFilterContext* previous = fresh->source;
  for (const Step& step : steps_of(mode_)) {
    AVFilterContext* filter = nullptr;
    if (const int err = avfilter_graph_create_filter(&filter, avfilter_get_by_name(step.filter), step.filter,
                                                     step.args, nullptr, fresh->graph.get());
        err < 0)
      return refused(step.filter, err);
    if (const int err = avfilter_link(previous, 0, filter, 0); err < 0) return refused("lien", err);
    previous = filter;
  }
  if (const int err = avfilter_graph_create_filter(&fresh->sink, avfilter_get_by_name("buffersink"), "out",
                                                   nullptr, nullptr, fresh->graph.get());
      err < 0)
    return refused("sortie", err);
  if (const int err = avfilter_link(previous, 0, fresh->sink, 0); err < 0) return refused("lien", err);
  if (const int err = avfilter_graph_config(fresh->graph.get(), nullptr); err < 0)
    return refused("graphe", err);
  graph_ = std::move(fresh);
  return {};
}

std::expected<void, Refusal> Deinterlacer::push(FramePtr frame) {
  if ((frame->flags & AV_FRAME_FLAG_INTERLACED) != 0) active_ = true;
  // A15-05 : l'horodatage que FFmpeg a deviné (pts absent d'un AVI ou d'un MPEG-PS) passe par le graphe.
  if (frame->best_effort_timestamp != AV_NOPTS_VALUE) frame->pts = frame->best_effort_timestamp;
  if (!graph_ || graph_->width != frame->width || graph_->height != frame->height ||
      graph_->format != frame->format) {
    if (graph_) {
      // Taille ou format changés : les images retenues sortent d'abord, puis le graphe est refait (A15-04).
      if (auto flushed = flush(); !flushed) return flushed;
      while (true) {
        auto held = take();
        if (!held) return std::unexpected(held.error());
        if (!*held) break;
        drained_.push_back(std::move(*held));
      }
    }
    if (auto built = rebuild(*frame); !built) return built;
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

std::expected<FramePtr, Refusal> Deinterlacer::take() {
  if (!graph_) return FramePtr();
  FramePtr out(av_frame_alloc());
  if (!out) return refused("image", AVERROR(ENOMEM));
  const int err = av_buffersink_get_frame(graph_->sink, out.get());
  if (err == AVERROR(EAGAIN) || err == AVERROR_EOF) return FramePtr();
  if (err < 0) return refused("sortie", err);
  // bwdif double la base de temps (une image par champ possible), decimate la change : retour aux
  // microsecondes.
  const AVRational base = av_buffersink_get_time_base(graph_->sink);
  if (out->pts != AV_NOPTS_VALUE) out->pts = av_rescale_q(out->pts, base, kMicroseconds);
  if (mode_ == FieldMode::InverseTelecine) {
    // decimate garde la durée d'une image à 29,97 : celle de la sortie vient de sa fréquence.
    const AVRational rate = av_buffersink_get_frame_rate(graph_->sink);
    out->duration = rate.num > 0 ? av_rescale(1'000'000, rate.den, rate.num) : 0;
  } else if (out->duration > 0) {
    out->duration = av_rescale_q(out->duration, base, kMicroseconds);
  }
  out->best_effort_timestamp = out->pts;
  return out;
}

std::expected<FramePtr, Refusal> Deinterlacer::pull() {
  if (!drained_.empty()) {
    FramePtr front = std::move(drained_.front());
    drained_.pop_front();
    return front;
  }
  return take();
}

void Deinterlacer::reset() {
  graph_.reset();
  drained_.clear();
  active_ = false;
}

bool Deinterlacer::active() const {
  return active_;
}

}  // namespace ajp::media
