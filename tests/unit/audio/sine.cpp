#include "sine.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <string>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/channel_layout.h>
#include <libavutil/mathematics.h>
}

#include <catch2/catch_test_macros.hpp>

namespace ajp::audio::test {
namespace {

struct ContextFree {
  void operator()(AVCodecContext* context) const { avcodec_free_context(&context); }
};
struct PacketFree {
  void operator()(AVPacket* packet) const { av_packet_free(&packet); }
};

double sine_at(std::int64_t index, int sample_rate, double hz, double amplitude) {
  return amplitude * std::sin(2.0 * std::numbers::pi * hz * static_cast<double>(index) / sample_rate);
}

void write_sample(AVFrame& frame, int channel_index, int index, double value) {
  const auto format = static_cast<AVSampleFormat>(frame.format);
  const bool planar = av_sample_fmt_is_planar(format) != 0;
  const int slot = planar ? index : index * frame.ch_layout.nb_channels + channel_index;
  std::uint8_t* plane = frame.extended_data[planar ? channel_index : 0];
  switch (av_get_packed_sample_fmt(format)) {
    case AV_SAMPLE_FMT_S16:
      reinterpret_cast<std::int16_t*>(plane)[slot] = static_cast<std::int16_t>(std::lround(value * 32767.0));
      break;
    case AV_SAMPLE_FMT_S32:
      reinterpret_cast<std::int32_t*>(plane)[slot] =
          static_cast<std::int32_t>(std::lround(value * 2147483647.0));
      break;
    case AV_SAMPLE_FMT_FLT:
      reinterpret_cast<float*>(plane)[slot] = static_cast<float>(value);
      break;
    default:
      FAIL("format d'échantillon non prévu par le test");
  }
}

AudioPacket to_packet(const AVPacket& packet, const SineSpec& spec, AVRational time_base) {
  return AudioPacket{.data = std::vector<std::uint8_t>(packet.data, packet.data + packet.size),
                     .pts_us = spec.start_us + av_rescale_q(packet.pts, time_base, AVRational{1, 1'000'000}),
                     .epoch = spec.epoch};
}

}  // namespace

FramePtr sine_frame(AVSampleFormat format, int sample_rate, int channels, int frames, double hz,
                    double amplitude, std::int64_t first) {
  FramePtr frame(av_frame_alloc());
  REQUIRE(frame);
  frame->format = format;
  frame->sample_rate = sample_rate;
  frame->nb_samples = frames;
  av_channel_layout_default(&frame->ch_layout, channels);
  REQUIRE(av_frame_get_buffer(frame.get(), 0) >= 0);
  for (int i = 0; i < frames; ++i) {
    const double value = sine_at(first + i, sample_rate, hz, amplitude);
    for (int c = 0; c < channels; ++c) write_sample(*frame, c, i, value);
  }
  return frame;
}

EncodedClip encode_sine(const SineSpec& spec) {
  const AVCodec* encoder = avcodec_find_encoder_by_name(std::string(spec.encoder).c_str());
  REQUIRE(encoder != nullptr);
  std::unique_ptr<AVCodecContext, ContextFree> context(avcodec_alloc_context3(encoder));
  REQUIRE(context);
  context->sample_rate = spec.sample_rate;
  context->time_base = AVRational{1, spec.sample_rate};
  av_channel_layout_default(&context->ch_layout, spec.channels);
  const void* formats = nullptr;
  int format_count = 0;
  avcodec_get_supported_config(context.get(), encoder, AV_CODEC_CONFIG_SAMPLE_FORMAT, 0, &formats,
                               &format_count);
  context->sample_fmt = format_count > 0 ? static_cast<const AVSampleFormat*>(formats)[0] : AV_SAMPLE_FMT_S16;
  REQUIRE(avcodec_open2(context.get(), encoder, nullptr) >= 0);

  EncodedClip clip;
  std::unique_ptr<AVPacket, PacketFree> packet(av_packet_alloc());
  const auto receive = [&] {
    while (avcodec_receive_packet(context.get(), packet.get()) >= 0) {
      clip.packets.push_back(to_packet(*packet, spec, context->time_base));
      av_packet_unref(packet.get());
    }
  };
  const std::int64_t total = std::llround(spec.seconds * spec.sample_rate);
  const int frame_size = context->frame_size > 0 ? context->frame_size : spec.frame_size;
  for (std::int64_t start = 0; start < total; start += frame_size) {
    const int frames = static_cast<int>(std::min<std::int64_t>(frame_size, total - start));
    FramePtr frame = sine_frame(context->sample_fmt, spec.sample_rate, spec.channels, frames, spec.hz,
                                spec.amplitude, start);
    frame->pts = start;
    REQUIRE(avcodec_send_frame(context.get(), frame.get()) >= 0);
    receive();
  }
  REQUIRE(avcodec_send_frame(context.get(), nullptr) >= 0);
  receive();

  clip.codec = AudioCodecInfo{.codec = avcodec_get_name(context->codec_id),
                              .sample_rate = context->sample_rate,
                              .channels = context->ch_layout.nb_channels,
                              .block_align = context->block_align,
                              .bits_per_coded_sample = context->bits_per_coded_sample,
                              .bit_rate = context->bit_rate};
  if (context->extradata_size > 0) {
    clip.codec.extradata.assign(context->extradata, context->extradata + context->extradata_size);
  }
  return clip;
}

std::vector<float> concat(const std::vector<PcmBlock>& blocks) {
  std::vector<float> all;
  for (const PcmBlock& block : blocks) all.insert(all.end(), block.samples.begin(), block.samples.end());
  return all;
}

std::vector<float> channel(std::span<const float> interleaved, int channels, int index) {
  std::vector<float> mono;
  for (std::size_t i = static_cast<std::size_t>(index); i < interleaved.size();
       i += static_cast<std::size_t>(channels)) {
    mono.push_back(interleaved[i]);
  }
  return mono;
}

double tone_amplitude(std::span<const float> mono, int sample_rate, double hz) {
  const double step = 2.0 * std::numbers::pi * hz / sample_rate;
  double re = 0.0;
  double im = 0.0;
  for (std::size_t i = 0; i < mono.size(); ++i) {
    re += mono[i] * std::cos(step * static_cast<double>(i));
    im -= mono[i] * std::sin(step * static_cast<double>(i));
  }
  return mono.empty() ? 0.0 : 2.0 * std::hypot(re, im) / static_cast<double>(mono.size());
}

double peak(std::span<const float> mono) {
  float highest = 0.0F;
  for (const float sample : mono) highest = std::max(highest, std::abs(sample));
  return highest;
}

}  // namespace ajp::audio::test
