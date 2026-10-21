#include "audio/converter.h"

#include <algorithm>
#include <array>
#include <format>

extern "C" {
#include <libswresample/swresample.h>

#include <libavutil/channel_layout.h>
#include <libavutil/frame.h>
#include <libavutil/samplefmt.h>
}

#include "audio/av_error.h"
#include "audio/refuse.h"

namespace ajp::audio {
namespace {

constexpr std::int64_t kMicroseconds = 1'000'000;

// Disposition de l'entrée : celle de la trame, ou la disposition par défaut si l'ordre est inconnu.
AVChannelLayout input_layout(const AVFrame& frame) {
  AVChannelLayout layout{};
  if (frame.ch_layout.order == AV_CHANNEL_ORDER_UNSPEC) {
    av_channel_layout_default(&layout, frame.ch_layout.nb_channels);
  } else {
    av_channel_layout_copy(&layout, &frame.ch_layout);
  }
  return layout;
}

// Deux trames de même description passent par le même rééchantillonneur.
std::string describe(const AVFrame& frame) {
  AVChannelLayout layout = input_layout(frame);
  std::array<char, 128> text{};
  av_channel_layout_describe(&layout, text.data(), text.size());
  av_channel_layout_uninit(&layout);
  const char* format = av_get_sample_fmt_name(static_cast<AVSampleFormat>(frame.format));
  return std::format("{} {} Hz {}", format != nullptr ? format : "?", frame.sample_rate, text.data());
}

}  // namespace

// 5.1 se range en FL FR FC LFE SL SR : SDL3 accepte SL SR comme BL BR, et la
// plupart des sources 5.1 sont « 5.1(side) », qui passent ainsi sans remixage.
std::uint64_t sdl_channel_mask(int channels) {
  switch (channels) {
    case 1:
      return AV_CH_LAYOUT_MONO;
    case 2:
      return AV_CH_LAYOUT_STEREO;  // FL FR
    case 3:
      return AV_CH_LAYOUT_2POINT1;  // FL FR LFE
    case 4:
      return AV_CH_LAYOUT_QUAD;  // FL FR BL BR
    case 5:
      return AV_CH_LAYOUT_2POINT1 | AV_CH_BACK_LEFT | AV_CH_BACK_RIGHT;  // FL FR LFE BL BR
    case 6:
      return AV_CH_LAYOUT_5POINT1;  // FL FR FC LFE SL SR
    case 7:
      return AV_CH_LAYOUT_6POINT1;  // FL FR FC LFE BC SL SR
    case 8:
      return AV_CH_LAYOUT_7POINT1;  // FL FR FC LFE BL BR SL SR
    default:
      return 0;
  }
}

void PcmConverter::SwrFree::operator()(SwrContext* swr) const {
  swr_free(&swr);
}

PcmConverter::PcmConverter(PcmFormat output) : output_(output) {}
PcmConverter::PcmConverter(PcmConverter&&) noexcept = default;
PcmConverter& PcmConverter::operator=(PcmConverter&&) noexcept = default;
PcmConverter::~PcmConverter() = default;

std::expected<PcmBlock, Refusal> PcmConverter::convert(const AVFrame& frame,
                                                       std::optional<std::int64_t> pts_us) {
  PcmBlock block{.channels = output_.channels};
  if (const std::string input = describe(frame); input != input_) {
    block = drain();  // la queue de l'ancien format passe avant le nouveau
    if (auto configured = configure(frame); !configured) return std::unexpected(configured.error());
    input_ = input;
  }
  if (block.frames() == 0) {
    // La première trame rendue précède l'entrée de ce que le rééchantillonneur retient encore.
    const std::int64_t held_us = swr_get_delay(swr_.get(), kMicroseconds);
    block.pts_us = pts_us ? *pts_us - held_us : next_pts_us_.value_or(0);
  }
  if (const int error = append(block, &frame); error < 0) {
    return refuse(RefusalCode::AudioDecodeFailed, std::format("swr_convert : {}", av_error_text(error)));
  }
  next_pts_us_ = block.pts_us + frames_to_us(block.frames(), output_.sample_rate);
  return block;
}

PcmBlock PcmConverter::drain() {
  PcmBlock block{.pts_us = next_pts_us_.value_or(0), .channels = output_.channels};
  if (!swr_) return block;
  append(block, nullptr);
  next_pts_us_ = block.pts_us + frames_to_us(block.frames(), output_.sample_rate);
  return block;
}

void PcmConverter::reset() {
  swr_.reset();
  input_.clear();
  next_pts_us_.reset();
}

std::expected<void, Refusal> PcmConverter::configure(const AVFrame& frame) {
  if (frame.ch_layout.nb_channels < 1 || frame.sample_rate < 1) {
    return refuse(RefusalCode::AudioDecodeFailed,
                  std::format("trame audio sans format : {} canaux à {} Hz", frame.ch_layout.nb_channels,
                              frame.sample_rate));
  }
  AVChannelLayout in_layout = input_layout(frame);
  AVChannelLayout out_layout{};
  av_channel_layout_from_mask(&out_layout, sdl_channel_mask(output_.channels));
  SwrContext* swr = nullptr;
  int error = swr_alloc_set_opts2(&swr, &out_layout, AV_SAMPLE_FMT_FLT, output_.sample_rate, &in_layout,
                                  static_cast<AVSampleFormat>(frame.format), frame.sample_rate, 0, nullptr);
  av_channel_layout_uninit(&in_layout);
  av_channel_layout_uninit(&out_layout);
  swr_.reset(swr);
  if (error >= 0) error = swr_init(swr_.get());
  if (error < 0) {
    swr_.reset();
    input_.clear();
    return refuse(RefusalCode::AudioDecodeFailed,
                  std::format("libswresample refuse « {} » : {}", describe(frame), av_error_text(error)));
  }
  return {};
}

// Ajoute au bloc la conversion de frame, ou la queue du rééchantillonneur si frame est nul.
// Rend le code d'erreur de libswresample, négatif en cas d'échec.
int PcmConverter::append(PcmBlock& block, const AVFrame* frame) {
  const int in_frames = frame != nullptr ? frame->nb_samples : 0;
  const int capacity = swr_get_out_samples(swr_.get(), in_frames);
  if (capacity <= 0) return capacity;
  const std::size_t start = block.samples.size();
  const auto channels = static_cast<std::size_t>(output_.channels);
  block.samples.resize(start + static_cast<std::size_t>(capacity) * channels);
  std::array<std::uint8_t*, 1> out{reinterpret_cast<std::uint8_t*>(block.samples.data() + start)};
  const std::uint8_t* const* in = frame != nullptr ? frame->extended_data : nullptr;
  const int converted = swr_convert(swr_.get(), out.data(), capacity, in, in_frames);
  block.samples.resize(start + static_cast<std::size_t>(std::max(converted, 0)) * channels);
  return converted;
}

}  // namespace ajp::audio
