// Une erreur dite à l'utilisateur : une phrase dans la langue de l'interface, jamais le détail technique d'un
// refus (écrit en français, au journal). Le détail reste au journal : l'appelant le journalise.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

#include "app/play/i18n.h"
#include "core/types.h"

namespace ajp::app::play {

// Des refus à dire en clair à l'ouverture (P3.26, P3.28) : la clé du refus les distingue.
inline constexpr std::string_view kDolbyVision5Key = "dolby-vision-5";
inline constexpr std::string_view kAudioOnlyKey = "audio-only";
inline constexpr std::string_view kImageSubtitlesKey = "image-subtitles";  // A17-02

namespace detail {

// Les codes de refus par famille, une phrase chacune (A20-09, A24-09) : une table plutôt qu'un switch de
// soixante cas.
enum class ErrorFamily : std::uint8_t {
  Other,
  Unreadable,
  FileIo,
  DiskFull,
  GpuMemory,
  AudioOutput,
  DeviceLost,
  Cancelled,
  Setting,
  Damaged,
  Model,
  Hdr,
  Seek,
  Audio,
  Subtitles,
  CacheBusy,
  Builder,
  Component,
  Driver,
  Internal,
  Encode,
};

inline constexpr std::array kErrorFamilies{
    std::pair{RefusalCode::MediaUnreadable, ErrorFamily::Unreadable},
    std::pair{RefusalCode::DecoderUnavailable, ErrorFamily::Unreadable},
    std::pair{RefusalCode::DecodeFailed, ErrorFamily::Unreadable},
    std::pair{RefusalCode::HwDecodeUnavailable, ErrorFamily::Unreadable},
    std::pair{RefusalCode::TrackUnavailable, ErrorFamily::Unreadable},
    std::pair{RefusalCode::FormatUnsupported, ErrorFamily::Unreadable},
    std::pair{RefusalCode::SourceResized, ErrorFamily::Unreadable},
    std::pair{RefusalCode::FileUnreadable, ErrorFamily::FileIo},
    std::pair{RefusalCode::FileUnwritable, ErrorFamily::FileIo},
    std::pair{RefusalCode::IoFailed, ErrorFamily::FileIo},
    std::pair{RefusalCode::CacheIo, ErrorFamily::FileIo},
    std::pair{RefusalCode::InsufficientSpace, ErrorFamily::DiskFull},
    std::pair{RefusalCode::GpuMemoryExhausted, ErrorFamily::GpuMemory},
    std::pair{RefusalCode::AudioDeviceUnavailable, ErrorFamily::AudioOutput},
    std::pair{RefusalCode::DeviceLost, ErrorFamily::DeviceLost},
    std::pair{RefusalCode::Cancelled, ErrorFamily::Cancelled},
    std::pair{RefusalCode::ProfileUnreadable, ErrorFamily::Setting},
    std::pair{RefusalCode::ProfileVersion, ErrorFamily::Setting},
    std::pair{RefusalCode::FieldMissing, ErrorFamily::Setting},
    std::pair{RefusalCode::FieldType, ErrorFamily::Setting},
    std::pair{RefusalCode::FieldOutOfRange, ErrorFamily::Setting},
    std::pair{RefusalCode::FormatVersion, ErrorFamily::Setting},
    std::pair{RefusalCode::KeyUnknown, ErrorFamily::Setting},
    std::pair{RefusalCode::KeyConflict, ErrorFamily::Setting},
    std::pair{RefusalCode::OptionUnknown, ErrorFamily::Setting},
    std::pair{RefusalCode::TooManySteps, ErrorFamily::Setting},
    std::pair{RefusalCode::FactorProductTooLarge, ErrorFamily::Setting},
    std::pair{RefusalCode::StepOrder, ErrorFamily::Setting},
    std::pair{RefusalCode::PacingForbidden, ErrorFamily::Setting},
    std::pair{RefusalCode::ImportUnsupported, ErrorFamily::Setting},
    std::pair{RefusalCode::SourceUnknown, ErrorFamily::Setting},
    std::pair{RefusalCode::LookaheadMissing, ErrorFamily::Setting},
    std::pair{RefusalCode::SequenceOrder, ErrorFamily::Setting},
    std::pair{RefusalCode::HashMismatch, ErrorFamily::Damaged},
    std::pair{RefusalCode::ModelUnavailable, ErrorFamily::Model},
    std::pair{RefusalCode::ModelUnreadable, ErrorFamily::Model},
    std::pair{RefusalCode::ModelMismatch, ErrorFamily::Model},
    std::pair{RefusalCode::OnnxMalformed, ErrorFamily::Model},
    std::pair{RefusalCode::OnnxExternalData, ErrorFamily::Model},
    std::pair{RefusalCode::OnnxUnsupported, ErrorFamily::Model},
    std::pair{RefusalCode::InputMismatch, ErrorFamily::Model},
    std::pair{RefusalCode::OperatorUnknown, ErrorFamily::Model},
    std::pair{RefusalCode::GraphInvalid, ErrorFamily::Model},
    std::pair{RefusalCode::ShapeInvalid, ErrorFamily::Model},
    std::pair{RefusalCode::HdrUnavailable, ErrorFamily::Hdr},
    std::pair{RefusalCode::SeekFailed, ErrorFamily::Seek},
    std::pair{RefusalCode::AudioCodecUnavailable, ErrorFamily::Audio},
    std::pair{RefusalCode::AudioDecodeFailed, ErrorFamily::Audio},
    std::pair{RefusalCode::SubtitleUnreadable, ErrorFamily::Subtitles},
    std::pair{RefusalCode::SubtitleFormat, ErrorFamily::Subtitles},
    std::pair{RefusalCode::SubtitleAtlasFull, ErrorFamily::Subtitles},
    std::pair{RefusalCode::SubtitleBacklog, ErrorFamily::Subtitles},
    std::pair{RefusalCode::LibassFailed, ErrorFamily::Subtitles},
    std::pair{RefusalCode::CacheBusy, ErrorFamily::CacheBusy},
    std::pair{RefusalCode::BuilderFailed, ErrorFamily::Builder},
    std::pair{RefusalCode::BuilderCrashed, ErrorFamily::Builder},
    std::pair{RefusalCode::BuilderTimeout, ErrorFamily::Builder},
    std::pair{RefusalCode::ReceiptInvalid, ErrorFamily::Builder},
    std::pair{RefusalCode::SdkRejected, ErrorFamily::Builder},
    std::pair{RefusalCode::LibraryUnavailable, ErrorFamily::Component},
    std::pair{RefusalCode::AbiMismatch, ErrorFamily::Component},
    std::pair{RefusalCode::VendorCallFailed, ErrorFamily::Driver},
    std::pair{RefusalCode::VulkanCallFailed, ErrorFamily::Driver},
    std::pair{RefusalCode::DeviceMismatch, ErrorFamily::Driver},
    std::pair{RefusalCode::DeviceUnavailable, ErrorFamily::Driver},
    std::pair{RefusalCode::FeatureUnavailable, ErrorFamily::Driver},
    std::pair{RefusalCode::ThreadUnavailable, ErrorFamily::Internal},
    std::pair{RefusalCode::ThreadFailed, ErrorFamily::Internal},
    std::pair{RefusalCode::PacketBacklog, ErrorFamily::Internal},
    std::pair{RefusalCode::EncoderUnavailable, ErrorFamily::Encode},
    std::pair{RefusalCode::EncodeFailed, ErrorFamily::Encode},
};

inline ErrorFamily error_family(RefusalCode code) {
  for (const auto& [known, family] : kErrorFamilies)
    if (known == code) return family;
  return ErrorFamily::Other;
}

}  // namespace detail

inline std::string visible_error(const Refusal& refusal) {
  if (refusal.key == kDolbyVision5Key)
    return i18n::tr("Dolby Vision profile 5 is not supported (no HDR10 or SDR picture in this file)");
  if (refusal.key == kAudioOnlyKey) return i18n::tr("this file has sound only, no picture to play");
  if (refusal.key == kImageSubtitlesKey)
    return i18n::tr("image subtitles (PGS, VobSub) are not supported: no subtitles");
  switch (detail::error_family(refusal.code)) {
    case detail::ErrorFamily::Unreadable:
      return i18n::tr("the file cannot be read (unknown format, damaged or missing)");
    case detail::ErrorFamily::FileIo:
      return i18n::tr("a file could not be read or written");
    case detail::ErrorFamily::DiskFull:
      return i18n::tr("not enough disk space");
    case detail::ErrorFamily::GpuMemory:
      return i18n::tr("not enough GPU memory");
    case detail::ErrorFamily::AudioOutput:
      return i18n::tr("the audio output changed or is unavailable");
    case detail::ErrorFamily::DeviceLost:
      return i18n::tr("the graphics card stopped responding: restart the player");
    case detail::ErrorFamily::Cancelled:
      return i18n::tr("cancelled");
    case detail::ErrorFamily::Setting:
      return i18n::tr("a setting or option has a value the player does not accept");
    case detail::ErrorFamily::Damaged:
      return i18n::tr("a downloaded file is damaged: try again");
    case detail::ErrorFamily::Model:
      return i18n::tr("the model file is missing or cannot be used");
    case detail::ErrorFamily::Hdr:
      return i18n::tr("HDR is not available on this display");
    case detail::ErrorFamily::Seek:
      return i18n::tr("this position cannot be reached in the file");
    case detail::ErrorFamily::Audio:
      return i18n::tr("the audio track cannot be played");
    case detail::ErrorFamily::Subtitles:
      return i18n::tr("the subtitles cannot be shown");
    case detail::ErrorFamily::CacheBusy:
      return i18n::tr("another player window is preparing the same engine: try again in a moment");
    case detail::ErrorFamily::Builder:
      return i18n::tr("the GPU engine could not be prepared");
    case detail::ErrorFamily::Component:
      return i18n::tr("a component of the player is missing or out of date: reinstall it");
    case detail::ErrorFamily::Driver:
      return i18n::tr("the graphics card or its driver refused an operation");
    case detail::ErrorFamily::Internal:
      return i18n::tr("an internal task failed: restart the player");
    case detail::ErrorFamily::Encode:
      return i18n::tr("the export could not encode the video");
    case detail::ErrorFamily::Other:
      break;
  }
  return i18n::tr_format("error {} (details in the log)", name(refusal.code));
}

}  // namespace ajp::app::play
