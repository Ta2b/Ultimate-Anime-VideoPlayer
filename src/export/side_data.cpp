#include "export/side_data.h"

#include <cmath>
#include <cstring>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/mastering_display_metadata.h>
}

#include "export/av_util.h"

namespace ajp::exporter {
namespace {

// Dénominateurs des champs SEI de HEVC : chromaticité en 1/50000, luminance en 1/10000 cd/m².
constexpr int kChromaDen = 50000;
constexpr int kLumaDen = 10000;

AVRational chroma(double value) {
  return AVRational{static_cast<int>(std::lround(value * kChromaDen)), kChromaDen};
}
AVRational luma(double nits) {
  return AVRational{static_cast<int>(std::lround(nits * kLumaDen)), kLumaDen};
}

void fill(AVMasteringDisplayMetadata& out, const Hdr10Mastering& in) {
  const Chromaticity primaries[3] = {in.red, in.green, in.blue};
  for (int i = 0; i < 3; ++i) {
    out.display_primaries[i][0] = chroma(primaries[i].x);
    out.display_primaries[i][1] = chroma(primaries[i].y);
  }
  out.white_point[0] = chroma(in.white.x);
  out.white_point[1] = chroma(in.white.y);
  out.min_luminance = luma(in.min_nits);
  out.max_luminance = luma(in.max_nits);
  out.has_primaries = 1;
  out.has_luminance = 1;
}

std::unexpected<Refusal> no_memory(const char* where) {
  return refuse(RefusalCode::EncodeFailed, std::string(where) + " : mémoire insuffisante");
}

}  // namespace

std::expected<void, Refusal> set_encoder_mastering(AVCodecContext& encoder, const Hdr10Mastering& mastering) {
  AVFrameSideData* entry = av_frame_side_data_new(
      &encoder.decoded_side_data, &encoder.nb_decoded_side_data, AV_FRAME_DATA_MASTERING_DISPLAY_METADATA,
      sizeof(AVMasteringDisplayMetadata), AV_FRAME_SIDE_DATA_FLAG_REPLACE);
  if (entry == nullptr) return no_memory("av_frame_side_data_new (mastering)");
  AVMasteringDisplayMetadata metadata{};
  fill(metadata, mastering);
  std::memcpy(entry->data, &metadata, sizeof(metadata));
  return {};
}

std::expected<void, Refusal> set_stream_hdr10(AVCodecParameters& par, const Hdr10Mastering& mastering,
                                              const LightLevel& light) {
  av_packet_side_data_remove(par.coded_side_data, &par.nb_coded_side_data,
                             AV_PKT_DATA_MASTERING_DISPLAY_METADATA);
  av_packet_side_data_remove(par.coded_side_data, &par.nb_coded_side_data, AV_PKT_DATA_CONTENT_LIGHT_LEVEL);

  AVPacketSideData* mdcv =
      av_packet_side_data_new(&par.coded_side_data, &par.nb_coded_side_data,
                              AV_PKT_DATA_MASTERING_DISPLAY_METADATA, sizeof(AVMasteringDisplayMetadata), 0);
  if (mdcv == nullptr) return no_memory("av_packet_side_data_new (mastering)");
  AVMasteringDisplayMetadata metadata{};
  fill(metadata, mastering);
  std::memcpy(mdcv->data, &metadata, sizeof(metadata));

  AVPacketSideData* clli =
      av_packet_side_data_new(&par.coded_side_data, &par.nb_coded_side_data, AV_PKT_DATA_CONTENT_LIGHT_LEVEL,
                              sizeof(AVContentLightMetadata), 0);
  if (clli == nullptr) return no_memory("av_packet_side_data_new (niveaux de lumière)");
  const AVContentLightMetadata levels{.MaxCLL = light.max_cll, .MaxFALL = light.max_fall};
  std::memcpy(clli->data, &levels, sizeof(levels));
  return {};
}

}  // namespace ajp::exporter
