#include "app/play/controls_draw.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <optional>
#include <string_view>

#include <imgui.h>

#include "app/play/controls_icons.h"

namespace ajp::app::play {
namespace {

using icons::accent;
using icons::black;
using icons::icon_audio_track;
using icons::icon_chapter;
using icons::icon_fullscreen;
using icons::icon_gear;
using icons::icon_pause;
using icons::icon_play;
using icons::icon_skip;
using icons::icon_subtitles;
using icons::icon_volume;
using icons::Pen;
using icons::white;

// Géométrie de la barre, en pixels à l'échelle 1.
constexpr float kPad = 18.0F;
constexpr float kRow = 46.0F;
constexpr float kBarZone = 22.0F;
constexpr float kPanelExtra = 8.0F;
constexpr float kTitleSize = 17.0F * 1.1F;  // taille du titre (police de l'interface × 1,1)

// Menu d'une liste de pistes, au-dessus de son bouton. none_label : l'entrée « aucune » (ou rien).
void track_popup(const char* id, const char* heading, const std::vector<TrackEntry>& entries,
                 std::optional<int> selected, const char* none_label, float right, float bottom, float scale,
                 std::vector<Input>& actions, bool subtitles) {
  ImGui::SetNextWindowPos({right, bottom}, ImGuiCond_Always, {1.0F, 1.0F});
  ImGui::SetNextWindowSizeConstraints({260.0F * scale, 0.0F}, {720.0F * scale, 560.0F * scale});
  if (!ImGui::BeginPopup(id)) return;
  ImGui::TextDisabled("%s", heading);
  ImGui::Separator();
  if (none_label != nullptr && ImGui::Selectable(none_label, !selected))
    actions.emplace_back(SelectSubtitles{});
  for (const TrackEntry& entry : entries) {
    ImGui::PushID(entry.index);
    if (ImGui::Selectable(entry.label.c_str(), selected == entry.index)) {
      if (subtitles) {
        actions.emplace_back(SelectSubtitles{.track = entry.index});
      } else {
        actions.emplace_back(SelectAudio{.track = entry.index});
      }
    }
    ImGui::PopID();
  }
  ImGui::EndPopup();
}

// Bouton rond invisible : un disque clair au survol ; true au clic.
bool round_button(const Pen& pen, const char* id, ImVec2 center, float size) {
  ImGui::SetCursorScreenPos({center.x - size * 0.5F, center.y - size * 0.5F});
  const bool pressed = ImGui::InvisibleButton(id, {size, size});
  if (ImGui::IsItemHovered() || ImGui::IsItemActive())
    pen.list->AddCircleFilled(center, size * 0.44F, white(0.16F * pen.opacity));
  return pressed;
}

void tooltip(std::string_view text) {
  if (ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip))
    ImGui::SetTooltip("%.*s", static_cast<int>(text.size()), text.data());
}

// Bulle de texte centrée sur x, posée au-dessus de bottom, gardée entre min_x et max_x. Au
// premier plan : elle dépasse le haut de la fenêtre de la barre.
void bubble(const Pen& pen, float x, float bottom, std::string_view text, float min_x, float max_x) {
  ImDrawList* list = ImGui::GetForegroundDrawList();
  const ImVec2 size = ImGui::CalcTextSize(text.data(), text.data() + text.size());
  const ImVec2 pad{9.0F * pen.scale, 4.0F * pen.scale};
  const float width = size.x + 2.0F * pad.x;
  const float x0 = std::clamp(x - width * 0.5F, min_x, std::max(min_x, max_x - width));
  const ImVec2 p0{x0, bottom - size.y - 2.0F * pad.y};
  list->AddRectFilled(p0, {x0 + width, bottom}, black(0.82F * pen.opacity), 6.0F * pen.scale);
  list->AddText({p0.x + pad.x, p0.y + pad.y}, white(pen.opacity), text.data(), text.data() + text.size());
}

void draw_title(const Pen& pen, std::string_view title, float width) {
  if (title.empty()) return;
  const float pad = 18.0F * pen.scale;
  const float size = ImGui::GetFontSize() * 1.25F;
  pen.list->PushClipRect({pad, 0.0F}, {width - pad, size * 3.0F});
  // Gras (style Showtime) : le texte repassé à un demi-pixel.
  for (const float dx : {0.0F, 0.6F * pen.scale}) {
    pen.list->AddText(ImGui::GetFont(), size, {pad + dx, pad}, white(0.97F * pen.opacity), title.data(),
                      title.data() + title.size());
  }
  pen.list->PopClipRect();
}

void draw_toast(const Controls& controls, float width, float scale, std::int64_t now_us) {
  const auto text = controls.toast(now_us);
  if (!text) return;
  const float opacity = controls.toast_opacity(now_us);
  ImDrawList* list = ImGui::GetForegroundDrawList();
  const ImVec2 size = ImGui::CalcTextSize(text->data(), text->data() + text->size());
  const ImVec2 pad{16.0F * scale, 8.0F * scale};
  const ImVec2 p0{width - size.x - 2.0F * pad.x - kPad * scale, kPad * scale};  // en haut à droite
  list->AddRectFilled(p0, {p0.x + size.x + 2.0F * pad.x, p0.y + size.y + 2.0F * pad.y},
                      black(0.72F * opacity), (size.y + 2.0F * pad.y) * 0.5F);
  list->AddText({p0.x + pad.x, p0.y + pad.y}, white(opacity), text->data(), text->data() + text->size());
}

}  // namespace

float controls_height(float scale) {
  return (kPad + kRow + kBarZone + kPanelExtra) * scale;
}

float title_height(float scale) {
  return (kPad * 1.5F + kTitleSize) * scale;
}

bool draw_controls(Controls& controls, const ControlsView& view, const UiLook& look, std::int64_t now_us,
                   std::vector<Input>& actions) {
  const ImGuiIO& io = ImGui::GetIO();
  const float width = io.DisplaySize.x;
  const float height = io.DisplaySize.y;
  const float s = look.scale;
  if (look.keep_visible) controls.set_held(true);  // visible dès ce tour
  const float opacity = controls.opacity(now_us, view.paused);
  bool held = look.keep_visible;

  if (opacity > 0.0F) {
    const float pad = kPad * s;
    const float row = kRow * s;
    const float bar_zone = kBarZone * s;
    const float panel = controls_height(s);

    // Dégradés sous la barre et sous le titre : le texte reste lisible sur une image claire.
    ImDrawList* back = ImGui::GetBackgroundDrawList();
    const float shade = panel * 2.0F;
    back->AddRectFilledMultiColor({0.0F, height - shade}, {width, height}, black(0.0F), black(0.0F),
                                  black(0.74F * opacity), black(0.74F * opacity));
    back->AddRectFilledMultiColor({0.0F, 0.0F}, {width, 96.0F * s}, black(0.55F * opacity),
                                  black(0.55F * opacity), black(0.0F), black(0.0F));
    draw_title(Pen{back, s, opacity}, look.title, width);

    ImGui::SetNextWindowPos({0.0F, height - panel});
    ImGui::SetNextWindowSize({width, panel});
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, opacity);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0.0F, 0.0F});
    constexpr ImGuiWindowFlags kPanel =
        ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav |
        ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoScrollWithMouse;
    ImGui::Begin("##controls", nullptr, kPanel);
    ImGui::PopStyleVar();  // marges : seulement la fenêtre de la barre, pas ses menus
    const Pen pen{ImGui::GetWindowDrawList(), s, opacity};

    // Barre de progression : fine, plus épaisse au survol, bulle du temps sous la souris.
    const float x0 = pad;
    const float x1 = width - pad;
    const float bar_y = height - pad - row - bar_zone * 0.5F;
    ImGui::SetCursorScreenPos({x0, bar_y - bar_zone * 0.5F});
    ImGui::InvisibleButton("##seek", {x1 - x0, bar_zone});
    const bool bar_hovered = ImGui::IsItemHovered();
    const bool bar_active = ImGui::IsItemActive();
    const double pointer = std::clamp(static_cast<double>((io.MousePos.x - x0) / (x1 - x0)), 0.0, 1.0);
    const bool seekable = view.duration_us > 0;
    if (seekable) {
      if (ImGui::IsItemActivated()) {
        controls.press(pointer);
      } else if (bar_active && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 2.0F * s)) {
        if (auto seek = controls.drag_to(pointer, view.duration_us, now_us))
          actions.emplace_back(Command{*seek});
      }
      if (ImGui::IsItemDeactivated() && controls.dragging())
        actions.emplace_back(Command{controls.release(pointer, view.duration_us)});
    }
    const double played = controls.drag_fraction().value_or(fraction_of(view.position_us, view.duration_us));
    const bool lit = (bar_hovered || bar_active) && seekable;
    const float thick = (lit ? 6.0F : 3.5F) * s;
    const float xp = x0 + static_cast<float>(played) * (x1 - x0);
    pen.list->AddRectFilled({x0, bar_y - thick * 0.5F}, {x1, bar_y + thick * 0.5F}, white(0.28F * opacity),
                            thick * 0.5F);
    if (lit) {
      const float xh = x0 + static_cast<float>(pointer) * (x1 - x0);
      if (xh > xp)
        pen.list->AddRectFilled({xp, bar_y - thick * 0.5F}, {xh, bar_y + thick * 0.5F}, white(0.3F * opacity),
                                thick * 0.5F);
    }
    pen.list->AddRectFilled({x0, bar_y - thick * 0.5F}, {std::max(xp, x0 + thick), bar_y + thick * 0.5F},
                            accent(opacity), thick * 0.5F);
    // Le début de chaque chapitre : une encoche dans la barre.
    for (const ChapterMark& chapter : view.chapters) {
      if (!seekable || chapter.start_us <= 0 || chapter.start_us >= view.duration_us) continue;
      const float xc = x0 + static_cast<float>(fraction_of(chapter.start_us, view.duration_us)) * (x1 - x0);
      pen.list->AddRectFilled({xc - 1.5F * s, bar_y - thick * 0.5F - 1.0F * s},
                              {xc + 1.5F * s, bar_y + thick * 0.5F + 1.0F * s}, black(0.75F * opacity));
    }
    if (lit) {
      pen.list->AddCircleFilled({xp, bar_y}, 7.5F * s, accent(opacity));
      const double shown = controls.dragging() ? played : pointer;
      const std::int64_t shown_us = time_at(shown, view.duration_us);
      std::string text = format_clock(shown_us, view.duration_us);
      if (const auto chapter = chapter_at(view.chapters, shown_us);
          chapter && !view.chapters[*chapter].title.empty())
        text += " · " + view.chapters[*chapter].title;
      bubble(pen, x0 + static_cast<float>(shown) * (x1 - x0), bar_y - 12.0F * s, text, x0, x1);
    }

    // Rangée de boutons : lecture, volume, temps à gauche ; réglages et plein écran à droite.
    const float cy = height - pad - row * 0.5F;
    const float icon = 22.0F * s;
    float x = pad + row * 0.5F;
    // Avec des chapitres, la lecture est entourée de « précédent » et « suivant ».
    const bool chaptered = !view.chapters.empty();
    if (chaptered) {
      if (round_button(pen, "##chapter_back", {x, cy}, row))
        actions.emplace_back(StepChapter{.direction = -1});
      tooltip("Chapitre précédent (Page bas)");
      icon_chapter(pen, {x, cy}, icon * 0.9F, -1);
      x += row;
    }
    // La lecture/pause est au centre de l'image (commandes rondes) : pas de doublon dans la rangée.
    if (chaptered) {
      if (round_button(pen, "##chapter_next", {x, cy}, row))
        actions.emplace_back(StepChapter{.direction = 1});
      tooltip("Chapitre suivant (Page haut)");
      icon_chapter(pen, {x, cy}, icon * 0.9F, 1);
      x += row;
    }
    if (round_button(pen, "##mute", {x, cy}, row)) actions.emplace_back(ToggleMute{});
    tooltip(controls.muted() ? "Rétablir le son (m)" : "Couper le son (m)");
    icon_volume(pen, {x, cy}, icon, controls.volume(), controls.muted());

    const float vx0 = x + row * 0.5F + 2.0F * s;
    const float vw = 100.0F * s;
    ImGui::SetCursorScreenPos({vx0, cy - row * 0.5F});
    ImGui::InvisibleButton("##volume", {vw, row});
    if (ImGui::IsItemActive()) {
      const float wanted = std::clamp((io.MousePos.x - vx0) / vw, 0.0F, 1.0F) * kMaxVolume;
      if (std::abs(wanted - controls.volume()) > 0.004F || controls.muted())
        actions.emplace_back(Command{controls.set_volume(wanted)});
    }
    const bool volume_lit = ImGui::IsItemHovered() || ImGui::IsItemActive();
    const float level = controls.muted() ? 0.0F : controls.volume() / kMaxVolume;
    const float track = (volume_lit ? 4.0F : 3.0F) * s;
    pen.list->AddRectFilled({vx0, cy - track * 0.5F}, {vx0 + vw, cy + track * 0.5F}, white(0.3F * opacity),
                            track * 0.5F);
    pen.list->AddRectFilled({vx0, cy - track * 0.5F}, {vx0 + vw * level, cy + track * 0.5F}, white(opacity),
                            track * 0.5F);
    const float unity = vx0 + vw / kMaxVolume;  // repère des 100 %
    pen.list->AddLine({unity, cy - 5.0F * s}, {unity, cy + 5.0F * s}, white(0.45F * opacity), 1.0F * s);
    pen.list->AddCircleFilled({vx0 + vw * level, cy}, (volume_lit ? 7.0F : 5.5F) * s, white(opacity));
    if (volume_lit) {
      const std::string percent = controls.muted()
                                      ? std::string("Muet")
                                      : std::format("{} %", std::lround(controls.volume() * 100.0F));
      bubble(pen, vx0 + vw * level, cy - 12.0F * s, percent, x0, x1);
    }

    std::string clock = std::format("{} / {}", format_clock(view.position_us, view.duration_us),
                                    format_clock(view.duration_us, view.duration_us));
    if (const auto chapter = chapter_at(view.chapters, view.position_us);
        chapter && !view.chapters[*chapter].title.empty())
      clock += " · " + view.chapters[*chapter].title;
    const ImVec2 clock_size = ImGui::CalcTextSize(clock.c_str());
    pen.list->AddText({vx0 + vw + 18.0F * s, cy - clock_size.y * 0.5F}, white(0.95F * opacity),
                      clock.c_str());

    float rx = width - pad - row * 0.5F;
    if (round_button(pen, "##fullscreen", {rx, cy}, row)) actions.emplace_back(ToggleFullscreen{});
    tooltip(view.fullscreen ? "Quitter le plein écran (f, Échap)" : "Plein écran (f)");
    icon_fullscreen(pen, {rx, cy}, icon, view.fullscreen);
    rx -= row;
    if (round_button(pen, "##settings", {rx, cy}, row)) actions.emplace_back(ToggleSettings{});
    tooltip("Paramètres (p)");
    icon_gear(pen, {rx, cy}, icon);
    const TrackMenu& tracks = look.tracks;
    if (!tracks.audio.empty()) {
      rx -= row;
      if (round_button(pen, "##audio", {rx, cy}, row)) ImGui::OpenPopup("##audio_menu");
      tooltip("Piste audio (a)");
      icon_audio_track(pen, {rx, cy}, icon);
      track_popup("##audio_menu", "Piste audio", tracks.audio, tracks.audio_selected, nullptr,
                  rx + row * 0.5F, height - panel - 4.0F * s, s, actions, false);
    }
    if (!tracks.subtitles.empty()) {
      rx -= row;
      if (round_button(pen, "##subtitles", {rx, cy}, row)) ImGui::OpenPopup("##subtitles_menu");
      tooltip("Sous-titres (j)");
      icon_subtitles(pen, {rx, cy}, icon, tracks.subtitle_selected.has_value());
      track_popup("##subtitles_menu", "Sous-titres", tracks.subtitles, tracks.subtitle_selected, "Désactivés",
                  rx + row * 0.5F, height - panel - 4.0F * s, s, actions, true);
    }

    held = look.keep_visible || ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem) ||
           bar_active || controls.dragging() || ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId);
    ImGui::End();

    // Commandes rondes au centre de l'image (style Showtime) : retour de 10 s, lecture/pause en grand, avance
    // de 10 s.
    {
      const float big = 72.0F * s;
      const float small = 48.0F * s;
      const ImVec2 mid{width * 0.5F, (height - panel) * 0.5F};
      ImGui::SetNextWindowPos({mid.x - 1.7F * big, mid.y - big * 0.6F});
      ImGui::SetNextWindowSize({3.4F * big, big * 1.2F});
      ImGui::Begin("##transport", nullptr, kPanel);
      const Pen center{ImGui::GetWindowDrawList(), s, opacity};
      const ImVec2 left{mid.x - 1.4F * big, mid.y};
      const ImVec2 right{mid.x + 1.4F * big, mid.y};
      if (round_button(center, "##back10", left, small))
        actions.emplace_back(Command{cmd::SeekRelative{.delta_us = -10'000'000}});
      tooltip("Reculer de 10 s (flèche gauche)");
      icon_skip(center, left, small * 0.62F, -1);
      if (round_button(center, "##forward10", right, small))
        actions.emplace_back(Command{cmd::SeekRelative{.delta_us = 10'000'000}});
      tooltip("Avancer de 10 s (flèche droite)");
      icon_skip(center, right, small * 0.62F, 1);
      center.list->AddCircleFilled(mid, big * 0.5F, black(0.38F * opacity));
      if (round_button(center, "##play_big", mid, big)) actions.emplace_back(Command{cmd::TogglePause{}});
      tooltip(view.paused ? "Lecture (espace)" : "Pause (espace)");
      if (view.paused) {
        icon_play(center, mid, big * 0.42F);
      } else {
        icon_pause(center, mid, big * 0.42F);
      }
      held = held || ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem);
      ImGui::End();
    }
    ImGui::PopStyleVar();
  }
  controls.set_held(held);
  draw_toast(controls, width, s, now_us);
  return opacity > 0.0F || controls.toast_opacity(now_us) > 0.0F;
}

void draw_preparation(const PrepareView& view, const UiLook& look, std::int64_t now_us) {
  const ImGuiIO& io = ImGui::GetIO();
  const float width = io.DisplaySize.x;
  const float height = io.DisplaySize.y;
  const float s = look.scale;
  ImDrawList* list = ImGui::GetBackgroundDrawList();
  const float panel = controls_height(s);
  list->AddRectFilledMultiColor({0.0F, height - panel * 2.0F}, {width, height}, black(0.0F), black(0.0F),
                                black(0.6F), black(0.6F));
  draw_title(Pen{list, s, 1.0F}, look.title, width);

  // La barre à la place de la barre de progression.
  const float pad = kPad * s;
  const float x0 = pad;
  const float x1 = width - pad;
  const float bar_y = height - pad - kRow * s - kBarZone * s * 0.5F;
  const float thick = 3.5F * s;
  list->AddRectFilled({x0, bar_y - thick * 0.5F}, {x1, bar_y + thick * 0.5F}, white(0.22F), thick * 0.5F);
  std::string text = view.label;
  if (view.fraction) {
    const float f = std::clamp(*view.fraction, 0.0F, 1.0F);
    list->AddRectFilled({x0, bar_y - thick * 0.5F},
                        {std::max(x0 + thick, x0 + f * (x1 - x0)), bar_y + thick * 0.5F}, accent(1.0F),
                        thick * 0.5F);
    text += std::format(" · {} %", static_cast<int>(std::floor(f * 100.0F)));
  } else {
    // Durée inconnue : un segment qui parcourt la barre.
    const float span = (x1 - x0) * 0.18F;
    const float t = static_cast<float>(now_us % 1'600'000) / 1'600'000.0F;
    const float start = x0 + t * (x1 - x0 + span) - span;
    list->AddRectFilled({std::max(start, x0), bar_y - thick * 0.5F},
                        {std::min(start + span, x1), bar_y + thick * 0.5F}, accent(1.0F), thick * 0.5F);
  }
  const float cy = height - pad - kRow * s * 0.5F;
  const ImVec2 size = ImGui::CalcTextSize(text.c_str());
  list->AddText({x0, cy - size.y * 0.5F}, white(0.95F), text.c_str());
  if (view.show_details && !view.details.empty()) {
    const float line = ImGui::GetTextLineHeightWithSpacing();
    float y = bar_y - 14.0F * s - line * static_cast<float>(view.details.size());
    for (const std::string& detail : view.details) {
      list->AddText({x0, y}, white(0.7F), detail.c_str());
      y += line;
    }
  }
}

}  // namespace ajp::app::play
