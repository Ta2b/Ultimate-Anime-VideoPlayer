#include "media/av_util.h"

#include <array>
#include <cerrno>
#include <format>
#include <limits>

extern "C" {
#include <libavutil/error.h>
#include <libavutil/mathematics.h>
}

namespace ajp::media {

std::string av_error_text(int error) {
  // Les refus qu'on voit le plus à l'ouverture d'un fichier, en français (le reste garde le texte de FFmpeg).
  switch (error) {
    case AVERROR_INVALIDDATA:
      return "format non reconnu ou fichier abîmé (Invalid data found when processing input)";
    case AVERROR(ENOENT):
      return "fichier introuvable (No such file or directory)";
    case AVERROR(EACCES):
      return "accès refusé (Permission denied)";
    case AVERROR(EIO):
      return "fichier tronqué ou illisible (Input/output error)";
    default:
      break;
  }
  std::array<char, AV_ERROR_MAX_STRING_SIZE> text{};
  if (av_strerror(error, text.data(), text.size()) < 0) return std::format("erreur FFmpeg {}", error);
  return text.data();
}

std::unexpected<Refusal> refuse(RefusalCode code, std::string detail) {
  return std::unexpected(Refusal{code, std::move(detail)});
}

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

BufferPtr ref_buffer(const AVBufferRef* ref) {
  if (ref == nullptr) return nullptr;
  return BufferPtr(av_buffer_ref(ref));
}

}  // namespace ajp::media
