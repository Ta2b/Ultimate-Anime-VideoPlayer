#include "export/encoder_settings.h"

#include <algorithm>
#include <array>
#include <format>

namespace ajp::exporter {
namespace {

using enum PixelFormat;
using enum RateControl;

constexpr std::array kFfv1Formats{Rgb16, Yuv420p10};
constexpr std::array kNvencFormats{P010};
constexpr std::array kSvtFormats{Yuv420p10};
constexpr std::array kVulkanFormats{P010};

constexpr std::array kLosslessOnly{Lossless};
constexpr std::array kLossyRates{ConstantQuality, ConstantQp, Bitrate};
constexpr std::array kQpOrBitrate{ConstantQp, Bitrate};
constexpr std::array kQualityOrBitrate{ConstantQuality, Bitrate};

constexpr IntRange kNone{};

// Plages des options libavcodec de FFmpeg 9.0 (nvenc_hevc.c, nvenc_av1.c, libsvtav1.c,
// vulkan_encode_*.c). Une qualité 0 veut dire « automatique » pour NVENC : elle est refusée,
// le réglage doit être explicite.
constexpr std::array kTable{
    EncoderSpec{.name = "ffv1",
                .family = EncoderFamily::Ffv1,
                .formats = kFfv1Formats,
                .rates = kLosslessOnly,
                .quality = kNone,
                .qp = kNone,
                .preset = kNone},
    EncoderSpec{.name = "hevc_nvenc",
                .family = EncoderFamily::Nvenc,
                .formats = kNvencFormats,
                .rates = kLossyRates,
                .quality = {1, 51},
                .qp = {0, 51},
                .preset = {1, 7},
                .hardware = true},
    EncoderSpec{.name = "av1_nvenc",
                .family = EncoderFamily::Nvenc,
                .formats = kNvencFormats,
                .rates = kLossyRates,
                .quality = {1, 63},
                .qp = {0, 255},
                .preset = {1, 7},
                .hardware = true},
    EncoderSpec{.name = "libsvtav1",
                .family = EncoderFamily::SvtAv1,
                .formats = kSvtFormats,
                .rates = kQualityOrBitrate,
                .quality = {1, 63},
                .qp = kNone,
                .preset = {0, 13}},
    EncoderSpec{.name = "hevc_vulkan",
                .family = EncoderFamily::Vulkan,
                .formats = kVulkanFormats,
                .rates = kQpOrBitrate,
                .quality = kNone,
                .qp = {0, 51},
                .preset = kNone,
                .hardware = true},
    EncoderSpec{.name = "av1_vulkan",
                .family = EncoderFamily::Vulkan,
                .formats = kVulkanFormats,
                .rates = kQpOrBitrate,
                .quality = kNone,
                .qp = {0, 255},
                .preset = kNone,
                .hardware = true},
};

std::unexpected<Refusal> out_of_range(std::string detail) {
  return std::unexpected(Refusal{RefusalCode::FieldOutOfRange, std::move(detail)});
}

template <class T>
bool contains(std::span<const T> values, T value) {
  return std::ranges::find(values, value) != values.end();
}

// Une option sans objet dans le mode choisi doit rester à zéro : une valeur posée là serait
// ignorée en silence.
std::expected<void, Refusal> check_unused(const EncoderSettings& s) {
  const auto where = std::format("export.encoder {} ({})", s.encoder, name(s.rate));
  const bool uses_quality = s.rate == ConstantQuality || s.rate == ConstantQp;
  if (!uses_quality && s.quality != 0) return out_of_range(std::format("{} : quality sans objet", where));
  if (s.rate != Bitrate && (s.bitrate_kbps != 0 || s.max_bitrate_kbps != 0)) {
    return out_of_range(std::format("{} : débit sans objet", where));
  }
  return {};
}

std::expected<void, Refusal> check_rate(const EncoderSpec& spec, const EncoderSettings& s) {
  const auto where = std::format("export.encoder {}", s.encoder);
  if (s.rate == ConstantQuality && !spec.quality.contains(s.quality)) {
    return out_of_range(std::format("{} : quality {} hors de [{}, {}]", where, s.quality, spec.quality.min,
                                    spec.quality.max));
  }
  if (s.rate == ConstantQp && !spec.qp.contains(s.quality)) {
    return out_of_range(
        std::format("{} : qp {} hors de [{}, {}]", where, s.quality, spec.qp.min, spec.qp.max));
  }
  if (s.rate == Bitrate) {
    if (s.bitrate_kbps < kMinBitrateKbps || s.bitrate_kbps > kMaxBitrateKbps) {
      return out_of_range(std::format("{} : bitrate_kbps {} hors de [{}, {}]", where, s.bitrate_kbps,
                                      kMinBitrateKbps, kMaxBitrateKbps));
    }
    if (s.max_bitrate_kbps != 0 &&
        (s.max_bitrate_kbps < s.bitrate_kbps || s.max_bitrate_kbps > kMaxBitrateKbps)) {
      return out_of_range(std::format("{} : max_bitrate_kbps {} hors de [{}, {}]", where, s.max_bitrate_kbps,
                                      s.bitrate_kbps, kMaxBitrateKbps));
    }
  }
  return {};
}

}  // namespace

std::span<const EncoderSpec> encoder_table() {
  return kTable;
}

std::expected<const EncoderSpec*, Refusal> find_encoder_spec(std::string_view name) {
  // Un itérateur de std::array n'est pas un pointeur partout (MSVC) : l'adresse se prend.
  const auto found = std::ranges::find(kTable, name, &EncoderSpec::name);
  if (found == kTable.end()) {
    return std::unexpected(
        Refusal{RefusalCode::EncoderUnavailable, std::format("export.encoder {} : encodeur inconnu", name)});
  }
  return &*found;
}

std::expected<const EncoderSpec*, Refusal> validate_settings(const EncoderSettings& s, int width,
                                                             int height) {
  auto spec = find_encoder_spec(s.encoder);
  if (!spec) return spec;
  const auto where = std::format("export.encoder {}", s.encoder);

  if (!kSizeRange.contains(width) || !kSizeRange.contains(height)) {
    return out_of_range(std::format("{} : taille {}x{} hors de [{}, {}]", where, width, height,
                                    kSizeRange.min, kSizeRange.max));
  }
  if (!contains((*spec)->formats, s.format)) {
    return std::unexpected(
        Refusal{RefusalCode::FormatUnsupported, std::format("{} : format {} refusé", where, name(s.format))});
  }
  if (is_420(s.format) && (width % 2 != 0 || height % 2 != 0)) {
    return out_of_range(
        std::format("{} : taille {}x{} impaire en 4:2:0 ({})", where, width, height, name(s.format)));
  }
  if (!contains((*spec)->rates, s.rate)) {
    return std::unexpected(
        Refusal{RefusalCode::FormatUnsupported, std::format("{} : mode {} refusé", where, name(s.rate))});
  }
  if (auto unused = check_unused(s); !unused) return std::unexpected(unused.error());
  if (auto rate = check_rate(**spec, s); !rate) return std::unexpected(rate.error());
  if (s.preset && !(*spec)->preset.contains(*s.preset)) {
    if ((*spec)->preset.empty()) return out_of_range(std::format("{} : preset sans objet", where));
    return out_of_range(std::format("{} : preset {} hors de [{}, {}]", where, *s.preset, (*spec)->preset.min,
                                    (*spec)->preset.max));
  }
  if (s.keyframe_interval && !kKeyframeRange.contains(*s.keyframe_interval)) {
    return out_of_range(std::format("{} : keyframe_interval {} hors de [{}, {}]", where, *s.keyframe_interval,
                                    kKeyframeRange.min, kKeyframeRange.max));
  }
  return spec;
}

std::string_view name(Container container) {
  switch (container) {
    case Container::Matroska:
      return "matroska";
    case Container::Mp4:
      return "mp4";
  }
  return "?";
}

std::string_view name(PixelFormat format) {
  switch (format) {
    case Yuv420p10:
      return "yuv420p10";
    case P010:
      return "p010";
    case Rgb16:
      return "rgb16";
  }
  return "?";
}

std::string_view name(RateControl rate) {
  switch (rate) {
    case Lossless:
      return "lossless";
    case ConstantQuality:
      return "constant_quality";
    case ConstantQp:
      return "constant_qp";
    case Bitrate:
      return "bitrate";
  }
  return "?";
}

}  // namespace ajp::exporter
