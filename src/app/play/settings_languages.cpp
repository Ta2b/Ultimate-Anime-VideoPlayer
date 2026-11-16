#include "app/play/settings_languages.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <string_view>

#include <imgui.h>

#include "app/play/i18n.h"
#include "app/play/panel_style.h"
#include "app/play/tracks.h"

namespace ajp::app::play {

using panel::note;
using panel::section;

namespace {

std::string language_label(std::string_view tag) {
  if (tag == media::kOriginalAudio) return i18n::tr("Original (marked by the file)");
  std::string name = language_name(tag);
  return name.empty() ? std::string(tag) : name;  // une balise écrite à la main dans play.toml
}

// Une langue au choix : `none_label` pour la valeur vide (audio : « Selon le fichier », sous-titres :
// « Aucun » ; nul : pas de valeur vide), l'originale marquée par le fichier si `original`, sinon une des
// langues du menu. Vrai quand le choix change.
bool language_combo(const char* label, const char* none_label, std::string& tag, bool original) {
  const std::string current = tag.empty() && none_label != nullptr ? none_label : language_label(tag);
  if (!panel::combo(label, current.c_str())) return false;
  const std::string before = tag;
  if (none_label != nullptr && ImGui::Selectable(none_label, tag.empty())) tag.clear();
  if (original && ImGui::Selectable(i18n::tr("Original (marked by the file)"), tag == media::kOriginalAudio))
    tag = std::string(media::kOriginalAudio);
  for (const Language& language : menu_languages()) {
    if (ImGui::Selectable(i18n::tr(language.name), same_language(tag, language.code)))
      tag = std::string(language.code);
  }
  ImGui::EndCombo();
  return tag != before;
}

const char* version_label(LanguageVersion version) {
  switch (version) {
    case LanguageVersion::Original:
      return i18n::tr("Original version, subtitled");
    case LanguageVersion::Dubbed:
      return i18n::tr("Dubbed in my language");
    case LanguageVersion::Custom:
      return i18n::tr("Custom (Advanced)");
  }
  return "";
}

const char* rank_title(std::size_t rank) {
  static_assert(limits::kLanguagePresets == 4, "un titre par choix");
  switch (rank) {
    case 0:
      return i18n::tr("1st choice");
    case 1:
      return i18n::tr("2nd choice");
    case 2:
      return i18n::tr("3rd choice");
    default:
      return i18n::tr("4th choice");
  }
}

}  // namespace

void simple_languages(std::string& own, LanguageVersion& version,
                      std::vector<media::LanguagePreset>& presets) {
  const bool own_changed = language_combo(i18n::tr("My language"), nullptr, own, false);
  panel::tip(i18n::tr("My language"),
             i18n::tr("The language of your subtitles, or of the dub. The system language by default."));
  LanguageVersion picked = version;
  if (panel::combo(i18n::tr("I prefer"), version_label(version))) {
    for (const LanguageVersion choice : {LanguageVersion::Original, LanguageVersion::Dubbed}) {
      if (ImGui::Selectable(version_label(choice), choice == version)) picked = choice;
    }
    ImGui::EndCombo();
  }
  panel::tip(i18n::tr("I prefer"),
             i18n::tr("Original version: the audio the file marks as original, else Japanese, else Chinese, "
                      "with subtitles in your language; your dub when none of them has those subtitles.\n"
                      "Dubbed: your language without subtitles, else the original version.\n"
                      "Recommended: original version."));
  if (picked != version || (own_changed && version != LanguageVersion::Custom)) {
    version = picked;
    presets = language_presets_for(own, version);
  }
  if (version == LanguageVersion::Custom)
    note(
        i18n::tr("Tracks chosen one by one in Advanced. Pick a version to go back to the automatic choice."));
}

void track_choices(std::vector<media::LanguagePreset>& presets, LanguageVersion& version) {
  presets.resize(limits::kLanguagePresets);
  for (std::size_t rank = 0; rank < presets.size(); ++rank) {
    ImGui::PushID(static_cast<int>(rank));
    section(rank_title(rank));
    bool changed =
        language_combo(i18n::tr("Audio language"), i18n::tr("From the file"), presets[rank].audio, true);
    panel::tip(
        i18n::tr("Audio language"),
        i18n::tr("Picked when a file opens: the first choice whose audio and subtitles are both in the "
                 "file. Change the current track from the bar."));
    changed |=
        language_combo(i18n::tr("Subtitle language"), i18n::tr("None"), presets[rank].subtitles, false);
    panel::tip(i18n::tr("Subtitle language"),
               i18n::tr("Picked when a file opens. Change the current track from the bar."));
    if (changed) version = LanguageVersion::Custom;
    panel::end_group();  // l'encadré de la section ouvre sa propre fenêtre fille : il se ferme avant le PopID
    ImGui::PopID();
  }
}

bool languages_page(PlaySettings& d) {
  section(i18n::tr("Languages"));
  simple_languages(d.own_language, d.language_version, d.language_presets);
  section(i18n::tr("Subtitles"));
  int percent = static_cast<int>(std::lround(d.subtitle_scale * 100.0F));
  if (ImGui::SliderInt(panel::left(i18n::tr("Size")).c_str(), &percent,
                       static_cast<int>(limits::kSubtitleScale.lo * 100.0F),
                       static_cast<int>(limits::kSubtitleScale.hi * 100.0F), "%d %%",
                       ImGuiSliderFlags_AlwaysClamp)) {
    d.subtitle_scale = static_cast<float>(percent) / 100.0F;
  }
  panel::live_preview();
  panel::tip(i18n::tr("Size"), i18n::tr("Scales all subtitle text. Recommended: 100 %."));
  ImGui::SliderInt(panel::left(i18n::tr("Delay (ms)")).c_str(), &d.subtitle_delay_ms,
                   limits::kSubtitleDelaySlider.lo, limits::kSubtitleDelaySlider.hi, "%d",
                   ImGuiSliderFlags_None);
  const bool held = ImGui::IsItemActive();
  panel::live_preview();
  d.subtitle_delay_ms =
      std::clamp(d.subtitle_delay_ms, limits::kSubtitleDelayMs.lo, limits::kSubtitleDelayMs.hi);
  static_assert(limits::kSubtitleDelayMs.hi == 30'000, "le texte de l'infobulle dit ±30,000 ms");
  panel::tip(i18n::tr("Delay (ms)"),
             i18n::tr("Positive: subtitles later. Keys z and x: 100 ms. Ctrl+click to type a value (up to "
                      "±30,000 ms). This session only."));
  if (ImGui::Button(i18n::tr("Reset size and delay"))) {
    d.subtitle_scale = 1.0F;
    d.subtitle_delay_ms = 0;
  }
  panel::advanced_header();
  track_choices(d.language_presets, d.language_version);
  return held;
}

}  // namespace ajp::app::play
