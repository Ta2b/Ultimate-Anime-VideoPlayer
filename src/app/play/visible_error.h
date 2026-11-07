// Une erreur dite à l'utilisateur : une phrase dans la langue de l'interface, jamais le détail technique d'un
// refus (écrit en français, au journal). Le détail reste au journal : l'appelant le journalise.
#pragma once

#include <string>
#include <string_view>

#include "app/play/i18n.h"
#include "core/types.h"

namespace ajp::app::play {

// Des refus à dire en clair à l'ouverture (P3.26, P3.28) : la clé du refus les distingue.
inline constexpr std::string_view kDolbyVision5Key = "dolby-vision-5";
inline constexpr std::string_view kAudioOnlyKey = "audio-only";

inline std::string visible_error(const Refusal& refusal) {
  if (refusal.key == kDolbyVision5Key)
    return i18n::tr("Dolby Vision profile 5 is not supported (no HDR10 or SDR picture in this file)");
  if (refusal.key == kAudioOnlyKey) return i18n::tr("this file has sound only, no picture to play");
  switch (refusal.code) {
    case RefusalCode::MediaUnreadable:
    case RefusalCode::DecoderUnavailable:
    case RefusalCode::DecodeFailed:
    case RefusalCode::HwDecodeUnavailable:
    case RefusalCode::TrackUnavailable:
    case RefusalCode::FormatUnsupported:
      return i18n::tr("the file cannot be read (unknown format, damaged or missing)");
    case RefusalCode::FileUnreadable:
    case RefusalCode::FileUnwritable:
    case RefusalCode::IoFailed:
    case RefusalCode::CacheIo:
      return i18n::tr("a file could not be read or written");
    case RefusalCode::InsufficientSpace:
      return i18n::tr("not enough disk space");
    case RefusalCode::GpuMemoryExhausted:
      return i18n::tr("not enough GPU memory");
    case RefusalCode::AudioDeviceUnavailable:
      return i18n::tr("the audio output changed or is unavailable");
    case RefusalCode::DeviceLost:
      return i18n::tr("the graphics card stopped responding: restart the player");
    case RefusalCode::Cancelled:
      return i18n::tr("cancelled");
    default:
      return i18n::tr_format("error {} (details in the log)", name(refusal.code));
  }
}

}  // namespace ajp::app::play
