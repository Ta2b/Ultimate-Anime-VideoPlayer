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

// Types MIME et extensions des polices jointes, repris de the-database-mpv
// sub/sd_ass.c:151-165 (font_mimetypes, font_exts). Comportement fixé par
// tests/unit/media/demuxer_test.cpp.
constexpr std::array<std::string_view, 10> kFontMimeTypes{
    "application/x-truetype-font",
    "application/vnd.ms-opentype",
    "application/x-font-otf",
    "application/x-font-ttf",
    "application/x-font",  // probablement faux, mais présent dans des MKV
    "application/font-sfnt",
    "font/collection",
    "font/otf",
    "font/sfnt",
    "font/ttf",
};
constexpr std::array<std::string_view, 4> kFontExtensions{".ttf", ".ttc", ".otf", ".otc"};

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

// Comme attachment_is_font (sd_ass.c:166-186) : nom et données exigés, puis le
// type MIME, puis l'extension pour les MKV mal étiquetés.
bool is_font(const TrackInfo& track) {
  if (track.kind != TrackKind::Attachment || track.filename.empty()) return false;
  if (!track.parameters || track.parameters->extradata().empty()) return false;
  const std::string mime = lowercase(track.mime_type);
  if (std::ranges::find(kFontMimeTypes, mime) != kFontMimeTypes.end()) return true;
  const std::string name = lowercase(track.filename);
  return std::ranges::any_of(kFontExtensions, [&](std::string_view ext) { return name.ends_with(ext); });
}

}  // namespace ajp::media
