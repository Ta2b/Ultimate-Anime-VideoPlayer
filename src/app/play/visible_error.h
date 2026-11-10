// Une erreur dite à l'utilisateur : une phrase dans la langue de l'interface, jamais le détail technique d'un
// refus (écrit en français, au journal). Le détail reste au journal : l'appelant le journalise.
#pragma once

#include <array>
#include <cstdint>
#include <format>
#include <string>
#include <string_view>

#include "app/play/i18n.h"
#include "core/types.h"

namespace ajp::app::play {

// Des refus à dire en clair à l'ouverture (P3.26, P3.28) : la clé du refus les distingue.
inline constexpr std::string_view kDolbyVision5Key = "dolby-vision-5";
inline constexpr std::string_view kAudioOnlyKey = "audio-only";

// Les familles de refus sans phrase propre (A24-09) : avant, 56 codes sur 71 s'affichaient par leur nom
// anglais (« erreur OnnxExternalData »). Chaque code a maintenant une phrase ; son nom la suit entre
// parenthèses, pour qui cherche dans le journal.
enum class ErrorFamily : std::uint8_t {
  Settings,
  Model,
  Hdr,
  Seek,
  Sound,
  Subtitles,
  Overload,
  Engine,
  Resized,
  Graphics,
  Encoder,
  None
};

struct FamilyOf {
  RefusalCode code;
  ErrorFamily family;
};

inline constexpr std::array kErrorFamilies{
    FamilyOf{RefusalCode::ProfileUnreadable, ErrorFamily::Settings},
    FamilyOf{RefusalCode::ProfileVersion, ErrorFamily::Settings},
    FamilyOf{RefusalCode::FieldMissing, ErrorFamily::Settings},
    FamilyOf{RefusalCode::FieldType, ErrorFamily::Settings},
    FamilyOf{RefusalCode::FieldOutOfRange, ErrorFamily::Settings},
    FamilyOf{RefusalCode::TooManySteps, ErrorFamily::Settings},
    FamilyOf{RefusalCode::FactorProductTooLarge, ErrorFamily::Settings},
    FamilyOf{RefusalCode::StepOrder, ErrorFamily::Settings},
    FamilyOf{RefusalCode::PacingForbidden, ErrorFamily::Settings},
    FamilyOf{RefusalCode::SourceUnknown, ErrorFamily::Settings},
    FamilyOf{RefusalCode::LookaheadMissing, ErrorFamily::Settings},
    FamilyOf{RefusalCode::SequenceOrder, ErrorFamily::Settings},
    FamilyOf{RefusalCode::FormatVersion, ErrorFamily::Settings},
    FamilyOf{RefusalCode::ImportUnsupported, ErrorFamily::Settings},
    FamilyOf{RefusalCode::KeyUnknown, ErrorFamily::Settings},
    FamilyOf{RefusalCode::KeyConflict, ErrorFamily::Settings},
    FamilyOf{RefusalCode::OptionUnknown, ErrorFamily::Settings},
    FamilyOf{RefusalCode::ModelUnavailable, ErrorFamily::Model},
    FamilyOf{RefusalCode::ModelUnreadable, ErrorFamily::Model},
    FamilyOf{RefusalCode::ModelMismatch, ErrorFamily::Model},
    FamilyOf{RefusalCode::OnnxMalformed, ErrorFamily::Model},
    FamilyOf{RefusalCode::OnnxExternalData, ErrorFamily::Model},
    FamilyOf{RefusalCode::OnnxUnsupported, ErrorFamily::Model},
    FamilyOf{RefusalCode::OperatorUnknown, ErrorFamily::Model},
    FamilyOf{RefusalCode::GraphInvalid, ErrorFamily::Model},
    FamilyOf{RefusalCode::ShapeInvalid, ErrorFamily::Model},
    FamilyOf{RefusalCode::HashMismatch, ErrorFamily::Model},
    FamilyOf{RefusalCode::ReceiptInvalid, ErrorFamily::Model},
    FamilyOf{RefusalCode::InputMismatch, ErrorFamily::Model},
    FamilyOf{RefusalCode::HdrUnavailable, ErrorFamily::Hdr},
    FamilyOf{RefusalCode::SeekFailed, ErrorFamily::Seek},
    FamilyOf{RefusalCode::AudioCodecUnavailable, ErrorFamily::Sound},
    FamilyOf{RefusalCode::AudioDecodeFailed, ErrorFamily::Sound},
    FamilyOf{RefusalCode::SubtitleUnreadable, ErrorFamily::Subtitles},
    FamilyOf{RefusalCode::SubtitleFormat, ErrorFamily::Subtitles},
    FamilyOf{RefusalCode::SubtitleAtlasFull, ErrorFamily::Subtitles},
    FamilyOf{RefusalCode::SubtitleBacklog, ErrorFamily::Subtitles},
    FamilyOf{RefusalCode::LibassFailed, ErrorFamily::Subtitles},
    FamilyOf{RefusalCode::ThreadUnavailable, ErrorFamily::Overload},
    FamilyOf{RefusalCode::ThreadFailed, ErrorFamily::Overload},
    FamilyOf{RefusalCode::PacketBacklog, ErrorFamily::Overload},
    FamilyOf{RefusalCode::CacheBusy, ErrorFamily::Overload},
    FamilyOf{RefusalCode::BuilderFailed, ErrorFamily::Engine},
    FamilyOf{RefusalCode::BuilderCrashed, ErrorFamily::Engine},
    FamilyOf{RefusalCode::BuilderTimeout, ErrorFamily::Engine},
    FamilyOf{RefusalCode::SdkRejected, ErrorFamily::Engine},
    FamilyOf{RefusalCode::LibraryUnavailable, ErrorFamily::Engine},
    FamilyOf{RefusalCode::VendorCallFailed, ErrorFamily::Engine},
    FamilyOf{RefusalCode::AbiMismatch, ErrorFamily::Engine},
    FamilyOf{RefusalCode::SourceResized, ErrorFamily::Resized},
    FamilyOf{RefusalCode::DeviceMismatch, ErrorFamily::Graphics},
    FamilyOf{RefusalCode::VulkanCallFailed, ErrorFamily::Graphics},
    FamilyOf{RefusalCode::DeviceUnavailable, ErrorFamily::Graphics},
    FamilyOf{RefusalCode::FeatureUnavailable, ErrorFamily::Graphics},
    FamilyOf{RefusalCode::EncoderUnavailable, ErrorFamily::Encoder},
    FamilyOf{RefusalCode::EncodeFailed, ErrorFamily::Encoder},
};

constexpr ErrorFamily error_family(RefusalCode code) {
  for (const FamilyOf& entry : kErrorFamilies)
    if (entry.code == code) return entry.family;
  return ErrorFamily::None;
}

inline std::string family_phrase(ErrorFamily family) {
  switch (family) {
    case ErrorFamily::Settings:
      return i18n::tr("a settings file or an option is not valid");
    case ErrorFamily::Model:
      return i18n::tr("a model is missing, damaged or not supported");
    case ErrorFamily::Hdr:
      return i18n::tr("HDR is not available on this screen");
    case ErrorFamily::Seek:
      return i18n::tr("this file cannot be seeked at that point");
    case ErrorFamily::Sound:
      return i18n::tr("the sound of this file cannot be decoded");
    case ErrorFamily::Subtitles:
      return i18n::tr("the subtitles cannot be shown");
    case ErrorFamily::Overload:
      return i18n::tr("the player ran out of resources");
    case ErrorFamily::Engine:
      return i18n::tr("the engine could not be prepared or loaded");
    case ErrorFamily::Resized:
      return i18n::tr("the picture size changes inside this file");
    case ErrorFamily::Graphics:
      return i18n::tr("the graphics card or its driver refused an operation");
    case ErrorFamily::Encoder:
      return i18n::tr("the export encoder is unavailable or failed");
    case ErrorFamily::None:
      break;
  }
  return {};
}

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
      break;
  }
  if (const std::string phrase = family_phrase(error_family(refusal.code)); !phrase.empty())
    return std::format("{} ({})", phrase, name(refusal.code));
  return i18n::tr_format("error {} (details in the log)", name(refusal.code));
}

}  // namespace ajp::app::play
