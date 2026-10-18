#include "audio/decoder.h"

#include <bit>
#include <climits>
#include <cstring>
#include <format>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/channel_layout.h>
#include <libavutil/mem.h>
}

#include "audio/av_error.h"
#include "audio/refuse.h"

namespace ajp::audio {
namespace {

// Les horodatages traversent libavcodec en microsecondes.
constexpr AVRational kMicrosecondBase{1, 1'000'000};
constexpr int kMaxCodedChannels = 64;
// libavcodec compte en int, remplissage compris.
constexpr std::size_t kMaxBytes = static_cast<std::size_t>(INT_MAX - AV_INPUT_BUFFER_PADDING_SIZE);

// Recopie les paramètres du flux dans le contexte du décodeur.
std::expected<void, Refusal> describe_stream(AVCodecContext& context, const AudioCodecInfo& codec) {
  if (codec.sample_rate < 0 || codec.channels < 0 || codec.channels > kMaxCodedChannels) {
    return refuse(RefusalCode::FieldOutOfRange, std::format("flux « {} » : {} canaux à {} Hz", codec.codec,
                                                            codec.channels, codec.sample_rate));
  }
  context.sample_rate = codec.sample_rate;
  if (codec.channel_mask != 0 && std::popcount(codec.channel_mask) == codec.channels) {
    av_channel_layout_from_mask(&context.ch_layout, codec.channel_mask);
  } else if (codec.channels > 0) {
    av_channel_layout_default(&context.ch_layout, codec.channels);
  }
  context.block_align = codec.block_align;
  context.bits_per_coded_sample = codec.bits_per_coded_sample;
  context.bit_rate = codec.bit_rate;
  context.pkt_timebase = kMicrosecondBase;
  if (codec.extradata.empty()) return {};
  if (codec.extradata.size() > kMaxBytes) {
    return refuse(RefusalCode::FieldOutOfRange, "extradata audio trop grand");
  }
  // libavcodec exige un remplissage nul après les données, et les libère lui-même.
  context.extradata =
      static_cast<std::uint8_t*>(av_mallocz(codec.extradata.size() + AV_INPUT_BUFFER_PADDING_SIZE));
  if (context.extradata == nullptr) return refuse(RefusalCode::AudioDecodeFailed, "mémoire insuffisante");
  std::memcpy(context.extradata, codec.extradata.data(), codec.extradata.size());
  context.extradata_size = static_cast<int>(codec.extradata.size());
  return {};
}

}  // namespace

void AudioDecoder::Free::operator()(AVCodecContext* context) const {
  avcodec_free_context(&context);
}
void AudioDecoder::Free::operator()(AVPacket* packet) const {
  av_packet_free(&packet);
}
void AudioDecoder::Free::operator()(AVFrame* frame) const {
  av_frame_free(&frame);
}

AudioDecoder::AudioDecoder(PcmFormat output) : converter_(output) {}
AudioDecoder::AudioDecoder(AudioDecoder&&) noexcept = default;
AudioDecoder& AudioDecoder::operator=(AudioDecoder&&) noexcept = default;
AudioDecoder::~AudioDecoder() = default;

std::expected<AudioDecoder, Refusal> AudioDecoder::open(const AudioCodecInfo& codec, PcmFormat output) {
  const auto format = checked(output);
  if (!format) return std::unexpected(format.error());
  const AVCodecDescriptor* descriptor = avcodec_descriptor_get_by_name(codec.codec.c_str());
  const AVCodec* decoder = descriptor != nullptr ? avcodec_find_decoder(descriptor->id) : nullptr;
  if (decoder == nullptr || descriptor->type != AVMEDIA_TYPE_AUDIO) {
    return refuse(RefusalCode::AudioCodecUnavailable,
                  std::format("aucun décodeur audio pour « {} »", codec.codec));
  }

  AudioDecoder result(*format);
  result.context_.reset(avcodec_alloc_context3(decoder));
  result.packet_.reset(av_packet_alloc());
  result.frame_.reset(av_frame_alloc());
  if (!result.context_ || !result.packet_ || !result.frame_) {
    return refuse(RefusalCode::AudioDecodeFailed, "mémoire insuffisante");
  }
  if (auto described = describe_stream(*result.context_, codec); !described) {
    return std::unexpected(described.error());
  }
  if (const int error = avcodec_open2(result.context_.get(), decoder, nullptr); error < 0) {
    return refuse(RefusalCode::AudioDecodeFailed,
                  std::format("avcodec_open2 « {} » : {}", codec.codec, av_error_text(error)));
  }
  return result;
}

std::expected<std::vector<PcmBlock>, Refusal> AudioDecoder::decode(const AudioPacket& packet) {
  if (packet.data.empty()) return std::vector<PcmBlock>{};  // un paquet vide signifierait « fin de flux »
  if (packet.data.size() > kMaxBytes) {
    return refuse(RefusalCode::FieldOutOfRange, std::format("paquet audio de {} octets", packet.data.size()));
  }
  av_packet_unref(packet_.get());
  if (av_new_packet(packet_.get(), static_cast<int>(packet.data.size())) < 0) {
    return refuse(RefusalCode::AudioDecodeFailed, "mémoire insuffisante");
  }
  std::memcpy(packet_->data, packet.data.data(), packet.data.size());
  packet_->pts = packet.pts_us.value_or(AV_NOPTS_VALUE);
  packet_->time_base = kMicrosecondBase;
  if (const int error = avcodec_send_packet(context_.get(), packet_.get()); error < 0) {
    return refuse(RefusalCode::AudioDecodeFailed,
                  std::format("paquet « {} » refusé : {}", context_->codec->name, av_error_text(error)));
  }
  return receive();
}

std::expected<std::vector<PcmBlock>, Refusal> AudioDecoder::drain() {
  if (const int error = avcodec_send_packet(context_.get(), nullptr); error < 0 && error != AVERROR_EOF) {
    return refuse(RefusalCode::AudioDecodeFailed, std::format("fin de flux : {}", av_error_text(error)));
  }
  auto blocks = receive();
  avcodec_flush_buffers(context_.get());  // le décodeur reste utilisable après un seek
  if (!blocks) return blocks;
  if (PcmBlock tail = converter_.drain(); tail.frames() > 0) blocks->push_back(std::move(tail));
  return blocks;
}

void AudioDecoder::flush() {
  avcodec_flush_buffers(context_.get());
  converter_.reset();
}

std::expected<std::vector<PcmBlock>, Refusal> AudioDecoder::receive() {
  std::vector<PcmBlock> blocks;
  for (;;) {
    const int error = avcodec_receive_frame(context_.get(), frame_.get());
    if (error == AVERROR(EAGAIN) || error == AVERROR_EOF) return blocks;
    if (error < 0) {
      return refuse(RefusalCode::AudioDecodeFailed,
                    std::format("décodage « {} » : {}", context_->codec->name, av_error_text(error)));
    }
    std::optional<std::int64_t> pts;
    if (frame_->pts != AV_NOPTS_VALUE) pts = frame_->pts;
    auto block = converter_.convert(*frame_, pts);
    av_frame_unref(frame_.get());
    if (!block) return std::unexpected(block.error());
    if (block->frames() > 0) blocks.push_back(std::move(*block));
  }
}

}  // namespace ajp::audio
