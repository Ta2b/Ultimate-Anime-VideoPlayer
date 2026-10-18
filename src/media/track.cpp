#include "media/track.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <string_view>

extern "C" {
#include <libavcodec/codec_par.h>
}

namespace ajp::media {
namespace {

// Types MIME des polices vus dans les MKV d'anime, comme les accepte libass.
constexpr std::array<std::string_view, 9> kFontMimeTypes{
    "application/x-truetype-font",
    "application/x-font-ttf",
    "application/x-font-otf",
    "application/vnd.ms-opentype",
    "application/font-sfnt",
    "font/ttf",
    "font/otf",
    "font/sfnt",
    "font/collection",
};
constexpr std::array<std::string_view, 3> kFontExtensions{".ttf", ".otf", ".ttc"};

std::string lowercase(std::string_view text) {
  std::string out(text);
  std::ranges::transform(out, out.begin(),
                         [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return out;
}

}  // namespace

std::expected<std::shared_ptr<const CodecParameters>, Refusal> CodecParameters::copy_of(
    const AVCodecParameters& source) {
  AVCodecParameters* params = avcodec_parameters_alloc();
  if (params == nullptr || avcodec_parameters_copy(params, &source) < 0) {
    avcodec_parameters_free(&params);
    return std::unexpected(
        Refusal{RefusalCode::MediaUnreadable, "paramètres de codec : mémoire insuffisante"});
  }
  return std::shared_ptr<const CodecParameters>(new CodecParameters(params));
}

CodecParameters::~CodecParameters() {
  avcodec_parameters_free(&params_);
}

std::span<const std::byte> CodecParameters::extradata() const {
  if (params_->extradata == nullptr || params_->extradata_size <= 0) return {};
  return std::as_bytes(std::span(params_->extradata, static_cast<std::size_t>(params_->extradata_size)));
}

bool is_font(const TrackInfo& track) {
  if (track.kind != TrackKind::Attachment) return false;
  const std::string mime = lowercase(track.mime_type);
  if (std::ranges::find(kFontMimeTypes, mime) != kFontMimeTypes.end()) return true;
  const std::string name = lowercase(track.filename);
  return std::ranges::any_of(kFontExtensions, [&](std::string_view ext) { return name.ends_with(ext); });
}

}  // namespace ajp::media
