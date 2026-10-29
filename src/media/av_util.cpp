#include "media/av_util.h"

#include <array>
#include <cerrno>
#include <format>

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

std::int64_t to_us(std::int64_t ts, AVRational time_base, std::int64_t start_us) {
  if (ts == AV_NOPTS_VALUE) return kNoPts;
  return av_rescale_q(ts, time_base, kMicroseconds) - start_us;
}

std::int64_t from_us(std::int64_t us, AVRational time_base, std::int64_t start_us) {
  const auto down = static_cast<AVRounding>(AV_ROUND_DOWN | AV_ROUND_PASS_MINMAX);
  return av_rescale_q_rnd(us + start_us, kMicroseconds, time_base, down);
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
