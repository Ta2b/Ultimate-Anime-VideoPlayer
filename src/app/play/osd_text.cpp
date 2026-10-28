#include "app/play/osd_text.h"

#include <algorithm>
#include <cstddef>
#include <format>
#include <utility>
#include <variant>

namespace ajp::app::play {
namespace {

// Neutralise ce qu'ASS interprète dans un texte de dialogue : accolades (en parenthèses) et barre
// oblique ; les fins de ligne, qui couperaient la ligne Dialogue, sont retirées.
std::string escaped(std::string_view line) {
  std::string out;
  out.reserve(line.size());
  for (const char c : line) {
    if (c == '{') {
      out += '(';
    } else if (c == '}') {
      out += ')';
    } else if (c == '\\') {
      out += '/';
    } else if (c != '\n' && c != '\r') {
      out += c;
    }
  }
  return out;
}

}  // namespace

std::string osd_script(std::span<const std::string> lines, OutputSize size) {
  const int font = std::max(12, size.height / 42);
  // Une coupure entre deux lignes, même vides : une ligne vide en tête reste un saut de ligne.
  std::string text;
  for (std::size_t i = 0; i < lines.size(); ++i) {
    if (i > 0) text += "\\N";
    text += escaped(lines[i]);
  }
  return std::format(
      "[Script Info]\nScriptType: v4.00+\nPlayResX: {}\nPlayResY: {}\nWrapStyle: 2\nScaledBorderAndShadow: "
      "yes\n\n"
      "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
      "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
      "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
      "Style: "
      "Osd,sans-serif,{},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,0,7,{},{},{},"
      "1\n\n"
      "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
      "Dialogue: 0,0:00:00.00,9:59:59.99,Osd,,0,0,0,,{}\n",
      size.width, size.height, font, font, font, font, text);
}

OsdText::OsdText(subs::AssLibrary library, subs::SubtitleRenderer renderer)
    : library_(std::move(library)), renderer_(std::move(renderer)) {
  // Tout vit sur le thread main : un atlas relâché revient directement dans la réserve.
  pool_ = std::make_unique<subs::AtlasPool>(
      [pool = &pool_](std::unique_ptr<subs::SubtitleAtlas> atlas) { (*pool)->give_back(std::move(atlas)); });
}

OsdText::~OsdText() {
  frame_ = {};
  renderer_.forget();
}

std::expected<std::unique_ptr<OsdText>, Refusal> OsdText::create() {
  auto library = subs::AssLibrary::create();
  if (!library) return std::unexpected(std::move(library.error()));
  auto renderer = subs::SubtitleRenderer::create(*library, subs::RendererOptions{});
  if (!renderer) return std::unexpected(std::move(renderer.error()));
  return std::unique_ptr<OsdText>(new OsdText(std::move(*library), std::move(*renderer)));
}

std::expected<const subs::SubtitleFrame*, Refusal> OsdText::render(std::span<const std::string> lines,
                                                                   OutputSize size) {
  const bool same = size == size_ && std::ranges::equal(lines, lines_);
  if (!same) {
    auto track = subs::SubtitleTrack::from_ass(library_, osd_script(lines, size));
    if (!track) return std::unexpected(std::move(track.error()));
    track_.emplace(std::move(*track));
    lines_.assign(lines.begin(), lines.end());
    size_ = size;
    renderer_.forget();
    frame_ = {};
    auto outcome = renderer_.render(
        *track_, subs::RenderRequest{.target = {.width = size.width, .height = size.height}}, *pool_);
    if (!outcome) return std::unexpected(std::move(outcome.error()));
    if (auto* frame = std::get_if<subs::SubtitleFrame>(&*outcome)) frame_ = std::move(*frame);
  }
  return frame_.atlas ? &frame_ : nullptr;
}

}  // namespace ajp::app::play
