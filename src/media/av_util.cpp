#include "media/av_util.h"

#include <array>
#include <cerrno>
#include <format>
#include <limits>

extern "C" {
#include <libavutil/error.h>
#include <libavutil/mathematics.h>
#include <libavutil/pixdesc.h>
}

namespace ajp::media {

std::unexpected<Refusal> refuse_av(RefusalCode code, std::string_view where, int error) {
  return refuse(code, std::format("{} : {}", where, av_error_text(error)));
}

namespace {

constexpr std::int64_t kMax = std::numeric_limits<std::int64_t>::max();
constexpr std::int64_t kMin = std::numeric_limits<std::int64_t>::min();

// a + b, borné à [INT64_MIN, INT64_MAX] (std::add_sat n'arrive qu'en C++26).
std::int64_t add_saturated(std::int64_t a, std::int64_t b) {
  if (b > 0 && a > kMax - b) return kMax;
  if (b < 0 && a < kMin - b) return kMin;
  return a + b;
}

}  // namespace

// P2.29 : av_rescale_q rend INT64_MIN (AV_NOPTS_VALUE) quand le résultat déborde ; la soustraction qui
// suivait était alors indéfinie. Un débordement reste « sans horodatage », le décalage est saturé.
std::int64_t to_us(std::int64_t ts, AVRational time_base, std::int64_t start_us) {
  if (ts == AV_NOPTS_VALUE) return kNoPts;
  const std::int64_t us = av_rescale_q(ts, time_base, kMicroseconds);
  if (us == AV_NOPTS_VALUE) return kNoPts;
  const std::int64_t shifted = add_saturated(us, start_us == kNoPts ? kMax : -start_us);
  return shifted == kNoPts ? kNoPts + 1 : shifted;  // une valeur saturée n'est pas « sans horodatage »
}

std::int64_t from_us(std::int64_t us, AVRational time_base, std::int64_t start_us) {
  const auto down = static_cast<AVRounding>(AV_ROUND_DOWN | AV_ROUND_PASS_MINMAX);
  return av_rescale_q_rnd(add_saturated(us, start_us), kMicroseconds, time_base, down);
}

std::int64_t duration_to_us(std::int64_t duration, AVRational time_base) {
  if (duration == AV_NOPTS_VALUE || duration <= 0) return 0;
  return av_rescale_q(duration, time_base, kMicroseconds);
}

bool chroma_beyond_420(const AVCodecParameters& av) {
  if (const AVPixFmtDescriptor* format = av_pix_fmt_desc_get(static_cast<AVPixelFormat>(av.format)))
    return format->nb_components >= 3 && (format->log2_chroma_w != 1 || format->log2_chroma_h != 1);
  switch (av.codec_id) {
    case AV_CODEC_ID_HEVC:
      return av.profile == AV_PROFILE_HEVC_REXT;
    case AV_CODEC_ID_VP9:
      return av.profile == AV_PROFILE_VP9_1 || av.profile == AV_PROFILE_VP9_3;
    case AV_CODEC_ID_AV1:
      return av.profile == AV_PROFILE_AV1_HIGH || av.profile == AV_PROFILE_AV1_PROFESSIONAL;
    default:
      return false;
  }
}

}  // namespace ajp::media
