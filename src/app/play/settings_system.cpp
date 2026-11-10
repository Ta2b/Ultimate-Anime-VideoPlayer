#include "app/play/settings_system.h"

#include <algorithm>
#include <array>
#include <ctime>
#include <expected>
#include <format>

#include <SDL3/SDL_misc.h>
#include <SDL3/SDL_video.h>
#include <imgui.h>

#include "app/play/auto_bench.h"
#include "app/play/auto_bench_run.h"
#include "app/play/auto_config.h"
#include "app/play/i18n.h"
#include "app/play/panel_style.h"
#include "app/play/paths.h"
#include "app/play/problem_report.h"
#include "app/play/span_auto.h"
#include "app/play/updater.h"
#include "app/play/visible_error.h"
#include "core/log.h"
#include "core/thread.h"
#include "core/utf8_path.h"

namespace ajp::app::play {

using panel::kWarn;
using panel::note;
using panel::section;

void default_player_page(const PlayerPlace& place, SystemNote& last, bool already_default) {
  section(i18n::tr("Default player"));
  panel::wrapped(i18n::tr("Videos (mkv, mp4, webm, avi, mov…) open in AJP from the file manager."));
  if (already_default) note(i18n::tr("AJP is already the default video player."));
#ifdef __APPLE__
  static_cast<void>(last);
  note(register_player(place).value_or(std::string()));
#else
  if (last.pending && last.pending->wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
    const std::expected<std::string, Refusal> done = last.pending->get();
    if (!done) log_channel("app")->warn("inscription : {} {}", name(done.error().code), done.error().detail);
    last =
        done ? SystemNote{.text = *done} : SystemNote{.text = visible_error(done.error()), .refused = true};
  }
  const bool busy = last.pending != nullptr;
  const auto start = [&](std::expected<std::string, Refusal> (*action)(const PlayerPlace&)) {
    last.pending = start_task<std::string>("registration", [action, place] { return action(place); });
    last.text.clear();
    last.refused = false;
  };
  ImGui::BeginDisabled(busy);
  if (ImGui::Button(i18n::tr("Make this the default player"))) start(&choose_as_default);
  panel::same_line_if_fits(i18n::tr("Add to \"Open with\""));
  if (ImGui::Button(i18n::tr("Add to \"Open with\""))) start(&register_player);
  panel::same_line_if_fits(i18n::tr("Remove from \"Open with\""));
  if (ImGui::Button(i18n::tr("Remove from \"Open with\""))) start(&unregister_player);
  ImGui::EndDisabled();
  if (busy) {
    note(i18n::tr("Working…"));
  } else if (last.refused) {
    ImGui::PushStyleColor(ImGuiCol_Text, kWarn);
    panel::wrapped(last.text.c_str());
    ImGui::PopStyleColor();
  } else if (!last.text.empty()) {
    note(last.text);
  }
#endif
}

void language_section(PlaySettings& d) {
  section(i18n::tr("Language"));
  const char* shown = d.language == "fr" ? "Français" : "English";  // i18n-ok : noms des langues
  if (ImGui::BeginCombo(i18n::tr("Interface language"), shown)) {
    if (ImGui::Selectable("English", d.language == "en")) d.language = "en";   // i18n-ok
    if (ImGui::Selectable("Français", d.language == "fr")) d.language = "fr";  // i18n-ok
    ImGui::EndCombo();
  }
  i18n::set_language(i18n::language_from_code(d.language));
}

void gpu_section(PlaySettings& d, const std::vector<std::string>& gpus) {
  if (gpus.size() < 2) return;
  section(i18n::tr("Graphics card"));
  const char* shown = d.gpu.empty() ? i18n::tr("Automatic (dedicated card first)") : d.gpu.c_str();
  if (ImGui::BeginCombo(i18n::tr("Card for playback"), shown)) {
    if (ImGui::Selectable(i18n::tr("Automatic (dedicated card first)"), d.gpu.empty())) d.gpu.clear();
    for (const std::string& name : gpus)
      if (ImGui::Selectable(name.c_str(), d.gpu == name)) d.gpu = name;
    ImGui::EndCombo();
  }
  note(i18n::tr("Applies at next launch."));
}

namespace {

// A23-04 : update.toml relu à l'ouverture de la page, après une vérification, puis toutes les deux secondes
// (le fil des mises à jour peut l'écrire), et non plus à chaque image.
const UpdateState& page_state(UpdateNote& note, const std::filesystem::path& file) {
  const auto now = std::chrono::steady_clock::now();
  if (!note.state || now - note.read_at > std::chrono::seconds(2)) {
    note.state = load_update_state(file);
    note.read_at = now;
  }
  return *note.state;
}

}  // namespace

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void updates_page(PlaySettings& d, const PlayerPlace& place, UpdateNote& result) {
  const std::filesystem::path state_file = paths::update_state(place.app_data);
  if (result.pending && result.pending->wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
    const std::expected<std::string, Refusal> done = result.pending->get();
    if (!done) log_channel("app")->warn("maj : {} {}", name(done.error().code), done.error().detail);
    result.pending.reset();
    result.text = done ? *done : visible_error(done.error());
    result.refused = !done;
    result.state.reset();  // la vérification vient de réécrire update.toml
  }
  const UpdateState& state = page_state(result, state_file);

  // Une simple notification (updater.h) : une version existe, et où la prendre ; rien ne s'installe.
  section(i18n::tr("Updates"));
  struct Mode {
    UpdateMode key;
    const char* label;
    const char* detail;
  };
  constexpr std::array<Mode, 2> kModes{
      {{UpdateMode::Notify, "Notify me",
        "A notification with the link to the new version; you install it yourself."},
       {UpdateMode::Never, "Never", "No checks."}}};
  for (const Mode& mode : kModes) {
    if (ImGui::RadioButton(i18n::tr(mode.label), d.update_mode == mode.key)) d.update_mode = mode.key;
    panel::tip(i18n::tr(mode.detail));
  }
  if (std::string_view(update_public_key()).empty())
    note(i18n::tr("This package has no signing key: updates cannot be verified."));

  const char* frequency = d.update_frequency == UpdateFrequency::Launch   ? i18n::tr("At every launch")
                          : d.update_frequency == UpdateFrequency::Weekly ? i18n::tr("Weekly")
                                                                          : i18n::tr("Daily");
  if (ImGui::BeginCombo(i18n::tr("Frequency"), frequency)) {
    if (ImGui::Selectable(i18n::tr("At every launch"), d.update_frequency == UpdateFrequency::Launch))
      d.update_frequency = UpdateFrequency::Launch;
    if (ImGui::Selectable(i18n::tr("Daily"), d.update_frequency == UpdateFrequency::Daily))
      d.update_frequency = UpdateFrequency::Daily;
    if (ImGui::Selectable(i18n::tr("Weekly"), d.update_frequency == UpdateFrequency::Weekly))
      d.update_frequency = UpdateFrequency::Weekly;
    ImGui::EndCombo();
  }
  note(i18n::tr_format("Installed version: {}", installed_version()));
  if (!state.latest_version.empty()) note(i18n::tr_format("Available: {}", state.latest_version));
  if (state.last_check_unix > 0) {
    const std::time_t at = static_cast<std::time_t>(state.last_check_unix);
    std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &at);
#else
    localtime_r(&at, &local);
#endif
    std::array<char, 32> text{};
    std::strftime(text.data(), text.size(), "%Y-%m-%d %H:%M", &local);
    note(i18n::tr_format("Last check: {}", text.data()));
  }

  const bool busy = result.pending != nullptr;
  ImGui::BeginDisabled(busy);
  if (ImGui::Button(i18n::tr("Check now"))) {
    result.pending = start_task<std::string>(
        "settings", [state_file, stop = result.stop] -> std::expected<std::string, Refusal> {
          return check_updates_now(state_file, stop->get_token());
        });
    result.text.clear();
    result.refused = false;
  }
  ImGui::EndDisabled();
  // La page de la release (ses notes, du manifeste signé, en https), sinon celle des releases : l'utilisateur
  // y prend l'archive ou l'installeur.
  if (!state.latest_version.empty()) {
    ImGui::SameLine();
    if (ImGui::Button(i18n::tr("Download page"))) {
      const std::string url =
          state.notes_url.starts_with("https://") ? state.notes_url : releases_page(release_repo());
      SDL_OpenURL(url.c_str());
    }
  }
  if (busy) {
    note(i18n::tr("Working…"));
  } else if (result.refused) {
    ImGui::PushStyleColor(ImGuiCol_Text, kWarn);
    panel::wrapped(result.text.c_str());
    ImGui::PopStyleColor();
  } else if (!result.text.empty()) {
    note(result.text);
  }
  if (!state.last_error.empty() && result.text.empty())
    note(i18n::tr_format("The last check failed: {}", state.last_error));
}

bool start_span_prepare(const SpanPrepare& prepare, PrepareNote& state) {
  if (!prepare.offered || !prepare.run || state.pending) return false;
  auto progress = std::make_shared<SpanDeriveProgress>();
  auto stop = std::make_shared<std::stop_source>();
  state.progress = progress;
  state.stop = stop;
  state.pending = start_task<std::string>(
      "settings", [run = prepare.run, progress, stop] -> std::expected<std::string, Refusal> {
        try {
          return run(*progress, stop->get_token());
        } catch (const std::exception& error) {  // un thread qui lève ne doit pas emporter le lecteur
          return std::unexpected(Refusal{RefusalCode::ThreadFailed, error.what()});
        }
      });
  state.text.clear();
  state.refused = false;
  state.completed = false;
  return true;
}

void span_prepare_section(const SpanPrepare& prepare, PrepareNote& state) {
  section(i18n::tr("SPAN Performance mode (FP8)"));
  panel::wrapped(
      i18n::tr("With SPAN's Performance precision, the player prepares this mode by itself for the open "
               "video (about a minute, FP16 meanwhile). Redo it by hand here."));
  poll_prepare(state);
  const bool busy = state.pending != nullptr;
  note(prepare.note);
  if (prepare.offered) {
    ImGui::BeginDisabled(busy);
    if (ImGui::Button(i18n::tr("Prepare again for this video"))) start_span_prepare(prepare, state);
    ImGui::EndDisabled();
    if (busy) {
      ImGui::SameLine();
      if (ImGui::Button(i18n::tr("Cancel##span_prepare")) && state.stop) state.stop->request_stop();
    }
  }
  if (busy && state.progress) {
    const auto [fraction, stage] = state.progress->get();
    ImGui::ProgressBar(fraction, ImVec2(-1.0F, 0.0F), stage.c_str());
  } else if (state.refused) {
    ImGui::PushStyleColor(ImGuiCol_Text, kWarn);
    panel::wrapped(state.text.c_str());
    ImGui::PopStyleColor();
  } else if (!state.text.empty()) {
    note(state.text);
  }
}

void report_problem_section(const PlayerPlace& place, const std::string& gpu_name) {
  constexpr const char* kIssueUrl =
      "https://github.com/Ta2b/Ultimate-Anime-VideoPlayer/issues/new?template=bug.yml";
  // Ce que le dernier clic a donné, montré sous le bouton.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables) : le seul thread de l'interface
  static std::string last;
  section(i18n::tr("Report a problem"));
  if (ImGui::Button(i18n::tr("Report a problem"))) {
    if (!gpu_name.empty()) note_report_fact("GPU", gpu_name);
    if (const char* driver = SDL_GetCurrentVideoDriver(); driver != nullptr)
      note_report_fact("SDL video", driver);
    ImGui::SetClipboardText(current_report(core::to_utf8(place.home)).c_str());
    last = SDL_OpenURL(kIssueUrl) ? i18n::tr("Report copied. Paste it into the page that opened.")
                                  : i18n::tr_format("Report copied. Open {} and paste it there.", kIssueUrl);
  }
  if (!last.empty()) panel::wrapped(last.c_str());
  panel::tip(i18n::tr(
      "Copies the version, the system, the graphics card, the preset and the end of the log (and the one "
      "before it after a crash), with only the file names of your videos, then opens the report page. "
      "Paste the report there."));
}

}  // namespace ajp::app::play
